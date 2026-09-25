"""Ingest a folder of NCLaw-generated trajectories into our dump schema.

NCLaw's own simulator writes one torch pickle per saved step,

    <state_root>/0000.pt, 0001.pt, ...   dict(x, v, C, F, stress, sections, types)

with x and v of shape (N, 3), C, F and stress of shape (N, 3, 3), all in THEIR
y-up frame on the unit box (experiments/eval.py lines 104 and 128 of their tree,
and the same dict in their dataset script). This module turns such a folder into
one schema-valid npz that ``ident`` reads, so identification and cross-engine
rollout consume their data with no change to the solve code.

What their files do NOT carry, and where it comes from:

  particle volume and density
      Not in the state. Their MPMInitData sets one scalar volume per group:
      a cube gets prod(size) / N (uniform seeding) and a mesh gets
      mesh.volume / N * prod(size). rho is 1e3 for all four materials of the
      comparison. The manifest carries it; see MANIFEST_SCHEMA below.
  the Cauchy stress
      Their ``stress`` channel is the MLS-MPM stress their P2G consumes,
      2 mu (F - R) F^T + lambda J (J - 1) I for fixed corotated, which is the
      KIRCHHOFF stress tau = J sigma. Our schema carries Cauchy, so the ingest
      divides by J = det(F). ``stress_kind`` in the manifest overrides.
  the stress-to-state pairing
      Their loop computes stress = elasticity(F) BEFORE the step and saves it
      next to the post-step x, v, C, F, so the saved stress lags the saved
      state by one simulator step (and their frame 0 carries the zero-filled
      initial stress). ``stress_lag_steps`` (default 1) shifts the channel back
      into alignment and drops the last frame, which also removes the zero
      first frame. Alignment is only exact when skip_frame is 1; with a coarser
      cadence the ingest records the residual lag.
  the velocity gradient convention
      Their G2P accumulates new_C += 4 w inv_dx^2 outer(v, dpos), which reads
      as C_ij = dv_i/dx_j, our L. The ingest does not assume this: it runs the
      acceleration-consistency probe and transposes C only if the measurement
      says so.

Degradation tiers. A folder may be missing channels (a renderer's point cloud,
a trimmed release). Every channel is then either measured, derived, or absent,
and the per-channel verdict is recorded in the dump metadata:

  v absent      central finite differences of x in time
  C absent      moving-least-squares velocity gradient over the k nearest
                neighbours, the neighbour sets fixed in the REFERENCE frame
  F absent      the same least squares between reference and current offsets,
                which is the deformation-gradient analogue
  stress absent no oracle pressure. The elastic legs still run (their columns
                are built from F); the Drucker-Prager leg must refuse or state
                a pressure closure, and the metadata says so explicitly.

CLI:
  .venv/bin/python -m experiments.nclaw.ingest <nclaw_dir> --manifest m.json \\
      --out out/nclaw_suite/dumps/<name>.npz
  .venv/bin/python -m experiments.nclaw.ingest --roundtrip <our_dump.npz> \\
      --material jelly
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.conventions import (
    LOG10_I_TABLE_MAX,
    LOG10_I_TABLE_MIN,
    MU_TABLE_POINTS,
)
from experiments.nclaw.dump_writer import (
    COORDINATE_CONVENTION,
    IN_PLANE_AXES,
    OUT_OF_PLANE_AXIS,
)
from experiments.nclaw.probe_l_convention import (
    L_CONVENTION_STRING,
    _local_velocity_gradient,
)
from ident.io.schema import SCHEMA_VERSION

INGEST_VERSION = "nclaw-ingest-1.0"

# Their y-up frame to our z-up frame; the same proper rotation the suite uses to
# rotate their throw and gravity. Imported lazily so this module works without a
# clone of their repository present.
def _rotation() -> np.ndarray:
    from experiments.nclaw.suite import R_YUP_TO_ZUP
    return R_YUP_TO_ZUP


MANIFEST_SCHEMA: dict[str, str] = {
    # required
    "material": "jelly | plasticine | sand | water; picks the law tag and the identify leg",
    "rho": "particle density, kg/m^3 (their configs: 1e3 for all four materials)",
    "dt": "their sim.dt, seconds (their sim/low.yaml and high.yaml: 5e-4)",
    "skip_frame": "their sim.skip_frame; frame_dt = dt * skip_frame",
    "num_grids": "their sim.num_grids, the grid the data was produced on (low 20, high 32)",
    "particle_volume|total_volume|shape": (
        "one volume source is required: particle_volume (m^3, their MPMInitData vol), "
        "total_volume (m^3, divided by the particle count), or "
        'shape={"kind": "cube", "size": [sx, sy, sz]} for prod(size) / N'
    ),
    # optional, with defaults
    "grid_lim": "box side, default 1.0 (their unit box)",
    "bound": "their sim.bound in cells, default 3; sets our slip-plane clearance",
    "gravity_yup": "default [0.0, -9.8, 0.0], their sim.gravity",
    "frame_convention": '"yup" (default, theirs) or "zup" (already ours, no rotation)',
    "stress_kind": '"kirchhoff" (default, theirs) or "cauchy"',
    "stress_lag_steps": "default 1: their eval.py saves elasticity(F_{k-1}) beside state k",
    "group": "int; with multi-material folders, keep only particles whose types == group",
    "mls_k": "neighbour count for derived channels, default 24 (the probe's k)",
    "mls_ridge": "relative ridge on the local normal equations, default 1e-8",
    "law": "dump law tag; default from material",
    "law_params": "dict recorded verbatim in the dump metadata (their truth, for the record)",
    "name": "short scene name, default the folder name",
}

REQUIRED_MANIFEST = ("material", "rho", "dt", "skip_frame", "num_grids")
VOLUME_KEYS = ("particle_volume", "total_volume", "shape")

LAW_FOR_MATERIAL = {
    "jelly": "corotated",
    "plasticine": "vonmises",
    "sand": "drucker_prager",
    "water": "eos_fluid",
}


class ManifestError(ValueError):
    """Raised when the manifest is missing an entry the ingest cannot invent."""


@dataclass
class IngestResult:
    path: Path
    meta: dict[str, Any]
    provenance: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

CONFIG_CANDIDATES = ("hydra.yaml", ".hydra/config.yaml", "config.yaml")


def _config_path(nclaw_dir: Path) -> Path | None:
    """Their resolved config, if the run left one where eval.py puts it.

    Their eval.py writes ``OmegaConf.save(cfg, exp_root / 'hydra.yaml')``, one
    level above the state folder, so state_root/../hydra.yaml is the normal
    location. Hydra's own output_subdir is disabled in their
    configs/default.yaml, so there is no .hydra folder; that path is kept for a
    run made with an override.
    """
    for rel in CONFIG_CANDIDATES:
        for base in (nclaw_dir, nclaw_dir.parent):
            c = base / rel
            if c.is_file():
                return c
    return None


def _read_hydra_config(nclaw_dir: Path, log=print) -> dict[str, Any]:
    """What their resolved config provides; the manifest always wins over it.

    Their layout is cfg.sim.{dt, skip_frame, num_grids, bound, gravity} and
    cfg.env.blob.{name, rho, shape}, the blob group nesting under env (their own
    scripts address it as env.blob.material.ckpt). Reading it needs pyyaml,
    which this repository does not depend on; without pyyaml the config is
    skipped with a message and the manifest carries everything.
    """
    c = _config_path(nclaw_dir)
    if c is None:
        return {}
    try:
        import yaml
    except ImportError:                                       # pragma: no cover
        log(f"[ingest] {c} found but pyyaml is not installed; the manifest must "
            "carry material, rho, dt, skip_frame, num_grids and the volume source")
        return {}
    cfg = yaml.safe_load(c.read_text()) or {}
    sim = cfg.get("sim") or {}
    env = cfg.get("env") or {}
    blob = env.get("blob") or env
    shape = blob.get("shape") or {}
    out: dict[str, Any] = {"_config_file": str(c)}
    for key, val in (("dt", sim.get("dt")), ("skip_frame", sim.get("skip_frame")),
                     ("num_grids", sim.get("num_grids")), ("bound", sim.get("bound")),
                     ("gravity_yup", sim.get("gravity")), ("rho", blob.get("rho")),
                     ("material", blob.get("name"))):
        if val is not None:
            out[key] = val
    if shape.get("size") is not None and shape.get("type") == "cube":
        out["shape"] = {"kind": "cube", "size": shape["size"]}
    return out


def resolve_manifest(nclaw_dir: Path, manifest: dict | str | Path | None,
                     n_particles: int) -> dict[str, Any]:
    """Merge a hydra config (if any) with the manifest, the manifest winning."""
    if manifest is None:
        man: dict[str, Any] = {}
    elif isinstance(manifest, (str, Path)):
        man = json.loads(Path(manifest).read_text())
    else:
        man = dict(manifest)

    cfg = _read_hydra_config(nclaw_dir)
    merged: dict[str, Any] = {**cfg, **man}

    missing = [k for k in REQUIRED_MANIFEST if merged.get(k) is None]
    vol_key = "particle_volume|total_volume|shape"
    if not any(merged.get(k) is not None for k in VOLUME_KEYS):
        missing.append(vol_key)
    if missing:
        lines = [f"  {k}: {MANIFEST_SCHEMA.get(k, MANIFEST_SCHEMA[vol_key])}"
                 for k in missing]
        raise ManifestError(
            f"manifest for {nclaw_dir} is missing required entries. Add to the json:\n"
            + "\n".join(lines)
            + "\nTheir values live in experiments/configs/sim/*.yaml (dt, skip_frame, "
              "num_grids, bound, gravity) and configs/env/blob/*.yaml (rho, shape)."
        )

    if merged.get("particle_volume") is None:
        if merged.get("total_volume") is not None:
            merged["particle_volume"] = float(merged["total_volume"]) / n_particles
            merged["particle_volume_source"] = "total_volume / N"
        else:
            shape = merged["shape"]
            size = np.asarray(shape.get("size", 1.0), dtype=float).reshape(-1)
            if shape.get("kind", "cube") != "cube":
                raise ManifestError(
                    "shape-derived volume is implemented for kind 'cube' only "
                    "(their cube seeding gives vol = prod(size) / N); for a mesh give "
                    "particle_volume or total_volume, which their MPMInitData computes "
                    "as mesh.volume / N * prod(size)")
            merged["particle_volume"] = float(np.prod(size)) / n_particles
            merged["particle_volume_source"] = "prod(shape.size) / N"
    else:
        merged["particle_volume"] = float(merged["particle_volume"])
        merged["particle_volume_source"] = "manifest particle_volume"

    merged.setdefault("grid_lim", 1.0)
    merged.setdefault("bound", 3)
    merged.setdefault("gravity_yup", [0.0, -9.8, 0.0])
    merged.setdefault("frame_convention", "yup")
    merged.setdefault("stress_kind", "kirchhoff")
    merged.setdefault("stress_lag_steps", 1)
    merged.setdefault("mls_k", 24)
    merged.setdefault("mls_ridge", 1.0e-8)
    merged.setdefault("law", LAW_FOR_MATERIAL.get(str(merged["material"]), "corotated"))
    merged.setdefault("law_params", {})
    merged.setdefault("name", nclaw_dir.name)
    if merged["frame_convention"] not in ("yup", "zup"):
        raise ManifestError("frame_convention must be 'yup' (theirs) or 'zup' (ours)")
    if merged["stress_kind"] not in ("kirchhoff", "cauchy"):
        raise ManifestError("stress_kind must be 'kirchhoff' (theirs) or 'cauchy'")
    return merged


# ---------------------------------------------------------------------------
# Frame files
# ---------------------------------------------------------------------------

def frame_files(nclaw_dir: Path) -> list[Path]:
    """Their frame pickles in step order, from 0000.pt upward."""
    files = sorted(Path(nclaw_dir).glob("*.pt"), key=lambda p: int(p.stem))
    stems = [int(f.stem) for f in files]
    gaps = set(b - a for a, b in zip(stems, stems[1:]))
    if len(gaps) > 1:
        raise ManifestError(
            f"frame files are not uniformly spaced (gaps {sorted(gaps)}); a "
            "missing file would silently corrupt the time axis")
    if len(files) < 3:
        raise ManifestError(
            f"{nclaw_dir} holds {len(files)} '*.pt' frame files; the ingest needs at "
            "least 3. Point it at their state_root (the folder of 0000.pt, 0001.pt, ...)")
    return files


def _torch_load(path: Path, allow_pickle: bool) -> dict:
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_pickle:
            raise ManifestError(
                f"{path} did not load under weights_only=True ({exc}); rerun with "
                "allow_pickle=True (--allow-pickle) if you trust the folder") from exc
        return torch.load(path, map_location="cpu", weights_only=False)


def read_frames(nclaw_dir: Path, group: int | None = None,
                allow_pickle: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    """Stack their per-frame dicts into (T, N, ...) arrays, still in their frame."""
    files = frame_files(nclaw_dir)
    keys = ("x", "v", "C", "F", "stress")
    stacks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    present: set[str] = set()
    sel: np.ndarray | None = None
    info: dict[str, Any] = {"n_frame_files": len(files), "frame_stems": [files[0].stem,
                                                                        files[-1].stem]}
    for i, f in enumerate(files):
        d = _torch_load(f, allow_pickle)
        if not isinstance(d, dict) or "x" not in d:
            raise ManifestError(f"{f} is not one of their state dicts (no 'x' key)")
        if i == 0:
            present = {k for k in keys if d.get(k) is not None}
            if "types" in d and d["types"] is not None:
                types = np.asarray(d["types"]).reshape(-1)
                info["n_groups"] = int(types.max()) + 1 if types.size else 1
                if group is not None:
                    sel = types == int(group)
                    info["group"] = int(group)
                    info["n_selected"] = int(sel.sum())
            if "sections" in d and d["sections"] is not None:
                info["sections"] = [int(s) for s in np.asarray(d["sections"]).reshape(-1)]
        for k in present:
            a = np.asarray(d[k].detach().cpu().numpy() if hasattr(d[k], "detach")
                           else d[k], dtype=np.float32)
            stacks[k].append(a[sel] if sel is not None else a)
    out = {k: np.stack(stacks[k]) for k in present}
    info["channels_in_files"] = sorted(present)
    return out, info


# ---------------------------------------------------------------------------
# Derived channels
# ---------------------------------------------------------------------------

def _neighbors(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of every point, self excluded."""
    k = int(min(k, len(X) - 1))
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(X).query(X, k=k + 1, workers=-1)
        return np.asarray(idx)[:, 1:]
    except ImportError:                                        # pragma: no cover
        out = np.empty((len(X), k), dtype=int)
        for lo in range(0, len(X), 2048):
            hi = min(lo + 2048, len(X))
            d2 = ((X[lo:hi, None, :] - X[None, :, :]) ** 2).sum(-1)
            out[lo:hi] = np.argsort(d2, axis=1)[:, 1:k + 1]
        return out


def _mls_tensor(dref: np.ndarray, dtar: np.ndarray, ridge: float) -> np.ndarray:
    """Least-squares T with dtar_i ~ T_ij dref_j, per particle, with a ridge."""
    A = np.einsum("pki,pkj->pij", dref, dref)
    B = np.einsum("pki,pkj->pij", dref, dtar)
    tr = np.trace(A, axis1=1, axis2=2) / 3.0
    A = A + (ridge * np.maximum(tr, 1e-30))[:, None, None] * np.eye(3)[None]
    return np.swapaxes(np.linalg.solve(A, B), 1, 2)


def fd_velocity(x: np.ndarray, frame_dt: float) -> np.ndarray:
    """Central finite differences in time, one-sided at the two ends."""
    v = np.empty_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * frame_dt)
    v[0] = (x[1] - x[0]) / frame_dt
    v[-1] = (x[-1] - x[-2]) / frame_dt
    return v


def mls_velocity_gradient(x: np.ndarray, v: np.ndarray, nbr: np.ndarray,
                          ridge: float) -> np.ndarray:
    """L_ij = dv_i/dx_j from a local least squares at the CURRENT positions."""
    T = x.shape[0]
    L = np.empty((T, x.shape[1], 3, 3), dtype=np.float64)
    for f in range(T):
        dref = x[f][nbr] - x[f][:, None, :]
        dtar = v[f][nbr] - v[f][:, None, :]
        L[f] = _mls_tensor(dref, dtar, ridge)
    return L


def mls_deformation_gradient(x: np.ndarray, nbr: np.ndarray, ridge: float) -> np.ndarray:
    """F_ij = dx_i/dX_j from the same least squares against frame-0 offsets."""
    T = x.shape[0]
    dref = x[0][nbr] - x[0][:, None, :]
    F = np.empty((T, x.shape[1], 3, 3), dtype=np.float64)
    for f in range(T):
        F[f] = _mls_tensor(dref, x[f][nbr] - x[f][:, None, :], ridge)
    return F


# ---------------------------------------------------------------------------
# The L convention, measured on the ingested data
# ---------------------------------------------------------------------------

def l_convention_from_arrays(x: np.ndarray, v: np.ndarray, C: np.ndarray,
                             k: int = 24, n_sample: int = 200, seed: int = 0,
                             frame_index: int | None = None) -> dict:
    """Does their C read as dv_i/dx_j, or as its transpose?

    The same measurement ``probe_l_convention.probe`` makes on a dump, run on
    arrays so the ingest can settle the convention before it writes anything:
    fit the spatial velocity gradient from the neighbourhood of each sampled
    particle and compare C and C^T against the fit.
    """
    speed = np.linalg.norm(v, axis=2).mean(axis=1)
    fi = int(np.argmax(speed)) if frame_index is None else int(frame_index)
    x3, v3, C3 = np.asarray(x[fi], float), np.asarray(v[fi], float), np.asarray(C[fi], float)
    rng = np.random.default_rng(seed)
    pool = np.where(np.linalg.norm(v3, axis=1) > 0.05)[0]
    if len(pool) < 30:
        pool = np.arange(len(x3))
    sample = rng.choice(pool, size=min(n_sample, len(pool)), replace=False)
    G = _local_velocity_gradient(x3, v3, sample, k=k)
    Cs = C3[sample]
    scale = np.maximum(np.linalg.norm(G.reshape(len(sample), -1), axis=1), 1e-9)
    err_L = np.linalg.norm((Cs - G).reshape(len(sample), -1), axis=1) / scale
    err_LT = np.linalg.norm((np.swapaxes(Cs, 1, 2) - G).reshape(len(sample), -1),
                            axis=1) / scale
    med_L, med_LT = float(np.median(err_L)), float(np.median(err_LT))
    return {
        "frame_index": fi,
        "median_err_vs_L": med_L,
        "median_err_vs_LT": med_LT,
        "verdict": ("L == dv_i/dx_j (a = dv/dt + L@v)" if med_L < med_LT
                    else "L == (dv_i/dx_j)^T (a = dv/dt + L^T@v)"),
        "matches_expected": bool(med_L < med_LT and med_L < 0.25),
        "k": int(k),
        "n_sample": len(sample),
    }


# ---------------------------------------------------------------------------
# The ingest
# ---------------------------------------------------------------------------

def _rotate_vec(R: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.einsum("ij,fpj->fpi", R, a)


def _rotate_position(R: np.ndarray, x: np.ndarray, grid_lim: float) -> np.ndarray:
    """Rotate ABSOLUTE box positions about the box centre, not the origin.

    ``_rotate_vec`` is exact for free vectors (v, gravity, the throw), which
    are translation-invariant. Positions are not: the box is [0, grid_lim]^3,
    not centred at the origin, so R @ x alone walks particles out of the box
    (measured: (x, -z, y) sends z in [0, 1] to y in [-1, 0], entirely outside
    the destination grid). Rotating about the box centre keeps [0, grid_lim]^3
    mapped to itself; ``_rotate_vec`` remains correct for every other channel.
    """
    c = grid_lim / 2.0
    return _rotate_vec(R, x - c) + c


def _rotate_tensor(R: np.ndarray, a: np.ndarray) -> np.ndarray:
    """R A R^T on a (T, P, 3, 3) field: both legs of a two-point tensor rotate."""
    return np.einsum("ij,fpjk,lk->fpil", R, a, R)


def _rotate_tensor_frame(R: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.einsum("ij,pjk,lk->pil", R, a, R)


def _manifest_dict(manifest: dict | str | Path | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    if isinstance(manifest, (str, Path)):
        return json.loads(Path(manifest).read_text())
    return dict(manifest)


def _peek_particles(nclaw_dir: Path, group: int | None,
                    allow_pickle: bool) -> int:
    """Particle count of the group to be ingested, from the first frame alone."""
    d = _torch_load(frame_files(nclaw_dir)[0], allow_pickle)
    n = int(np.asarray(d["x"]).shape[0])
    if group is None or d.get("types") is None:
        return n
    return int((np.asarray(d["types"]).reshape(-1) == int(group)).sum())


def read_nclaw_dir(nclaw_dir: str | Path, manifest: dict | str | Path | None,
                   out_path: str | Path, allow_pickle: bool = True,
                   log=print) -> IngestResult:
    """One folder of their trajectories to one schema-valid npz of ours."""
    nclaw_dir = Path(nclaw_dir)
    out_path = Path(out_path)
    man_in = _manifest_dict(manifest)
    group = None if man_in.get("group") is None else int(man_in["group"])
    n_sel = _peek_particles(nclaw_dir, group, allow_pickle)
    man = resolve_manifest(nclaw_dir, man_in, n_sel)
    raw, info = read_frames(nclaw_dir, group=group, allow_pickle=allow_pickle)

    T, P = raw["x"].shape[0], raw["x"].shape[1]
    frame_dt = float(man["dt"]) * int(man["skip_frame"])
    prov: dict[str, str] = {"x": "measured"}
    notes: list[str] = []

    R = np.eye(3) if man["frame_convention"] == "zup" else _rotation()
    x = _rotate_position(R, raw["x"].astype(np.float64), float(man["grid_lim"]))

    # stress alignment first: it decides the frame count everything else keeps
    lag = int(man["stress_lag_steps"]) if "stress" in raw else 0
    n_drop = 0
    if lag > 0 and int(man["skip_frame"]) == 1:
        n_drop = lag
        raw["stress"] = raw["stress"][lag:]
        for k in ("x", "v", "C", "F"):
            if k in raw:
                raw[k] = raw[k][:T - lag]
        x = x[:T - lag]
        T -= lag
        notes.append(f"stress shifted back {lag} step(s) to pair with the state it drove; "
                     f"the last {lag} frame(s) dropped so every channel stays measured")
    elif lag > 0:
        notes.append(f"stress lags the state by {lag} simulator step(s) and skip_frame is "
                     f"{man['skip_frame']}, so the lag is not a whole saved frame and is "
                     "left uncorrected")

    if "v" in raw:
        v = _rotate_vec(R, raw["v"].astype(np.float64))
        prov["v"] = "measured"
    else:
        v = fd_velocity(x, frame_dt)
        prov["v"] = "derived"
        notes.append("v absent: central finite differences of x in time")

    nbr: np.ndarray | None = None
    if "C" not in raw or "F" not in raw:
        nbr = _neighbors(x[0], int(man["mls_k"]))

    if "F" in raw:
        F = _rotate_tensor(R, raw["F"].astype(np.float64))
        prov["F"] = "measured"
    else:
        F = mls_deformation_gradient(x, nbr, float(man["mls_ridge"]))
        prov["F"] = "derived"
        notes.append(f"F absent: moving least squares of current against reference "
                     f"offsets over k={int(man['mls_k'])} reference-frame neighbours")

    probe: dict[str, Any]
    if "C" in raw:
        C = _rotate_tensor(R, raw["C"].astype(np.float64))
        probe = l_convention_from_arrays(x, v, C, k=int(man["mls_k"]))
        best = min(probe["median_err_vs_L"], probe["median_err_vs_LT"])
        if best > 0.15 and not man.get("l_convention"):
            raise ManifestError(
                f"both residuals are poor ({probe['median_err_vs_L']:.3f} vs "
                f"{probe['median_err_vs_LT']:.3f}); pin l_convention in the "
                "manifest")
        transposed = probe["median_err_vs_LT"] < probe["median_err_vs_L"]
        L = np.swapaxes(C, 2, 3) if transposed else C
        prov["L"] = "measured"
        probe["C_transposed_on_ingest"] = bool(transposed)
    else:
        L = mls_velocity_gradient(x, v, nbr, float(man["mls_ridge"]))
        probe = l_convention_from_arrays(x, v, L, k=int(man["mls_k"]))
        probe["C_transposed_on_ingest"] = False
        probe["note"] = ("no C channel: L is the MLS fit itself, so this probe is a "
                         "self-consistency check of the fit only")
        prov["L"] = "derived"
        notes.append(f"C absent: MLS velocity gradient over k={int(man['mls_k'])} "
                     "reference-frame neighbours")

    J = np.linalg.det(F)
    J = np.where(np.abs(J) < 1e-12, 1.0, J)
    vol0 = np.full(P, float(man["particle_volume"]), dtype=np.float32)
    mass = (np.float32(float(man["rho"])) * vol0).astype(np.float32)
    volume = (J * vol0[None, :].astype(np.float64))
    prov["volume"] = "derived"
    prov["mass"] = "derived"
    prov["volume0"] = "manifest"

    if "stress" in raw:
        stress = _rotate_tensor(R, raw["stress"].astype(np.float64))
        if man["stress_kind"] == "kirchhoff":
            stress = stress / J[:, :, None, None]
            notes.append("stress channel read as Kirchhoff tau = J sigma (their P2G "
                         "convention) and divided by J = det(F) to give Cauchy")
        prov["stress"] = "measured"
        pressure_source = "true_mpm_trace"
    else:
        stress = np.zeros((T, P, 3, 3))
        prov["stress"] = "absent"
        pressure_source = "absent"
        notes.append("stress absent: there is NO oracle pressure. The elastic and "
                     "volumetric legs still run from F; the Drucker-Prager friction leg "
                     "must refuse or name a pressure closure")

    g3 = R @ np.asarray(man["gravity_yup"], dtype=float).reshape(3)
    extra = {
        "material": man["material"],
        "shape": man["name"],
        "n_grid": int(man["num_grids"]),
        "grid_lim": float(man["grid_lim"]),
        "dt": float(man["dt"]),
        "substeps_per_frame": int(man["skip_frame"]),
        "gravity": [float(t) for t in g3],
        "vel_name": "nclaw_ingested",
        "bound_cells": int(man["bound"]),
        "collider_bc": "slip",
        "recovered": False,
        "source": "nclaw_ingest",
        "ingest_version": INGEST_VERSION,
        "nclaw_dir": str(nclaw_dir),
        "nclaw_manifest": {k: v for k, v in man.items() if k != "law_params"},
        "nclaw_frame_info": info,
        "channel_provenance": prov,
        "degradation_notes": notes,
        "frames_dropped_for_stress_lag": int(n_drop),
        "l_convention_probe": probe,
        "has_oracle_pressure": pressure_source == "true_mpm_trace",
    }

    log10I = np.linspace(LOG10_I_TABLE_MIN, LOG10_I_TABLE_MAX, MU_TABLE_POINTS)
    edges = np.logspace(LOG10_I_TABLE_MIN, LOG10_I_TABLE_MAX, 41)
    arrays = dict(
        schema_version=np.array(SCHEMA_VERSION),
        coordinate_convention=np.array(COORDINATE_CONVENTION),
        in_plane_axes=np.asarray(IN_PLANE_AXES, dtype=int),
        out_of_plane_axis=np.array(OUT_OF_PLANE_AXIS, dtype=int),
        L_convention=np.array(L_CONVENTION_STRING),
        frame_dt=np.array(frame_dt),
        grain_diameter=np.array(1.0e-3),
        rho_s=np.array(float(man["rho"])),
        rho_bulk=np.array(float(man["rho"])),
        packing_fraction=np.array(1.0),
        gravity_inplane=np.asarray([g3[0], g3[2]], dtype=float),
        pressure_source=np.array(pressure_source),
        law=np.array(str(man["law"])),
        mu_table_log10I=log10I,
        mu_table_mu=np.zeros_like(log10I),
        flowing_I_hist_edges=edges,
        flowing_I_hist_counts=np.zeros(40),
        meta_json=np.array(json.dumps(
            {"law_params": man["law_params"],
             "units": {"x": "m", "v": "m/s", "L": "1/s", "stress": "Pa",
                       "volume": "m^3", "mass": "kg", "times": "s"},
             **extra}, default=float)),
        times=np.arange(T, dtype=float) * frame_dt,
        x=x.astype(np.float32),
        v=v.astype(np.float32),
        L=L.astype(np.float32).reshape(T, P, 9),
        stress=stress.astype(np.float32).reshape(T, P, 9),
        volume=volume.astype(np.float32),
        mass=mass,
        active=np.ones((T, P), dtype=bool),
        F=F.astype(np.float32).reshape(T, P, 9),
        volume0=vol0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    log(f"[ingest] {nclaw_dir} -> {out_path.name}: {T} frames x {P} particles, "
        f"frame_dt={frame_dt:g}s, provenance={prov}")
    for n in notes:
        log(f"[ingest]   note: {n}")
    log(f"[ingest]   L convention: {probe['verdict']} "
        f"(median residual {probe['median_err_vs_L']:.3f} against L, "
        f"{probe['median_err_vs_LT']:.3f} against L^T)")
    return IngestResult(path=out_path, meta=extra, provenance=prov)


# ---------------------------------------------------------------------------
# Export: one of OUR dumps in THEIR format, for the round-trip validation
# ---------------------------------------------------------------------------

def export_to_nclaw(dump_npz: str | Path, out_dir: str | Path,
                    stress_kind: str = "kirchhoff", frames: int | None = None,
                    stress_lag_steps: int = 0, log=print) -> dict:
    """Write one of our dumps as their state folder: 0000.pt, 0001.pt, ...

    Their format exactly: one torch pickle per frame holding
    dict(x, v, C, F, stress, sections, types) in float32 in their y-up frame,
    with our L playing C and our Cauchy stress converted to the Kirchhoff
    stress their P2G consumes. The round-trip test exports a dump, ingests it
    back, and checks the inverse holds. This lets the ingest path be tested
    before any of their data arrives.

    ``stress_lag_steps`` reproduces the one-step offset their eval loop leaves
    in the channel: file f then carries the stress of frame f - 1, and file 0
    carries zeros the way their zero-initialised state does. The default 0
    writes an aligned folder, since our dumps are aligned.
    """
    import torch
    dump_npz, out_dir = Path(dump_npz), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(dump_npz)
    T = int(d["x"].shape[0]) if frames is None else min(int(frames), int(d["x"].shape[0]))
    P = int(d["x"].shape[1])
    Rt = _rotation().T                      # z-up back to y-up
    grid_lim = float(json.loads(str(d["meta_json"]))["grid_lim"])
    c = grid_lim / 2.0
    vol0 = d["volume0"] if "volume0" in d.files else None
    sections = [P]
    types = torch.zeros(P, dtype=torch.int)
    lag = int(stress_lag_steps)
    for f in range(T):
        # position is an affine rotation about the box centre (see
        # _rotate_position); v is a free vector, plain linear rotation
        x = np.einsum("ij,pj->pi", Rt, d["x"][f].astype(np.float64) - c) + c
        v = np.einsum("ij,pj->pi", Rt, d["v"][f].astype(np.float64))
        L = d["L"][f].astype(np.float64).reshape(P, 3, 3)
        F = d["F"][f].astype(np.float64).reshape(P, 3, 3)
        fs = f - lag
        if fs < 0:
            sig = np.zeros((P, 3, 3))
        else:
            sig = d["stress"][fs].astype(np.float64).reshape(P, 3, 3)
            if stress_kind == "kirchhoff":
                Js = np.linalg.det(d["F"][fs].astype(np.float64).reshape(P, 3, 3))
                Js = np.where(np.abs(Js) < 1e-12, 1.0, Js)
                sig = sig * Js[:, None, None]
        ckpt = {
            "x": torch.from_numpy(x.astype(np.float32)),
            "v": torch.from_numpy(v.astype(np.float32)),
            "C": torch.from_numpy(_rotate_tensor_frame(Rt, L).astype(np.float32)),
            "F": torch.from_numpy(_rotate_tensor_frame(Rt, F).astype(np.float32)),
            "stress": torch.from_numpy(_rotate_tensor_frame(Rt, sig).astype(np.float32)),
            "sections": sections,
            "types": types,
        }
        torch.save(ckpt, out_dir / f"{f:04d}.pt")
    meta = {"n_frames": T, "n_particles": P, "stress_kind": stress_kind,
            "stress_lag_steps": lag,
            "particle_volume": (float(vol0[0]) if vol0 is not None else None),
            "frame_dt": float(d["frame_dt"]),
            "n_grid": int(json.loads(str(d["meta_json"]))["n_grid"]),
            "grid_lim": float(json.loads(str(d["meta_json"]))["grid_lim"]),
            "rho": float(d["rho_s"])}
    log(f"[export] {dump_npz.name} -> {out_dir} as {T} frames of their format")
    return meta


def manifest_for_export(dump_npz: str | Path, material: str,
                        stress_kind: str = "kirchhoff") -> dict:
    """The manifest that describes an ``export_to_nclaw`` folder.

    Its stress channel is already paired with the state that produced it, so
    ``stress_lag_steps`` is 0 here where a folder written by their eval.py needs
    the default 1.
    """
    d = np.load(dump_npz)
    meta = json.loads(str(d["meta_json"]))
    return {
        "material": material,
        "rho": float(d["rho_s"]),
        "dt": float(d["frame_dt"]),
        "skip_frame": 1,
        "num_grids": int(meta["n_grid"]),
        "grid_lim": float(meta["grid_lim"]),
        "bound": int(meta.get("bound_cells", 3)),
        "particle_volume": float(d["volume0"][0]),
        "stress_kind": stress_kind,
        "stress_lag_steps": 0,
        "name": Path(dump_npz).stem,
        "law_params": meta.get("law_params", {}),
    }


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------

def roundtrip_check(dump_npz: str | Path, material: str, workdir: str | Path,
                    identify: bool = True, keep: bool = False,
                    stress_kind: str = "kirchhoff", degraded: bool = False,
                    log=print) -> dict:
    """Export one of our dumps to their format, ingest it back, compare.

    Returns the measured agreement: positions exactly, the rotated tensors to
    float32 round-off, the L-convention verdict on both files, and the
    identified theta on both paths.
    """
    import shutil
    dump_npz, workdir = Path(dump_npz), Path(workdir)
    state_dir = workdir / f"{dump_npz.stem}_nclaw_format"
    back = workdir / f"{dump_npz.stem}_roundtrip.npz"
    exp = export_to_nclaw(dump_npz, state_dir, stress_kind=stress_kind, log=log)
    man = manifest_for_export(dump_npz, material, stress_kind=stress_kind)
    res = read_nclaw_dir(state_dir, man, back, log=log)

    a, b = np.load(dump_npz), np.load(back)
    T = exp["n_frames"]

    def rel(name: str) -> float:
        u = a[name][:T].astype(np.float64)
        w = b[name][:T].astype(np.float64)
        s = max(float(np.abs(u).max()), 1e-300)
        return float(np.abs(u - w).max() / s)

    out: dict[str, Any] = {
        "dump": str(dump_npz), "material": material, "n_frames": T,
        "n_particles": exp["n_particles"], "stress_kind": stress_kind,
        "x_max_abs_diff": float(np.abs(a["x"][:T].astype(np.float64)
                                       - b["x"][:T].astype(np.float64)).max()),
        "x_bitwise_identical": bool(np.array_equal(a["x"][:T], b["x"][:T])),
        "v_bitwise_identical": bool(np.array_equal(a["v"][:T], b["v"][:T])),
        "rel_max_diff": {k: rel(k) for k in ("v", "L", "F", "stress", "volume")},
        "mass_bitwise_identical": bool(np.array_equal(a["mass"], b["mass"])),
        "volume0_bitwise_identical": bool(np.array_equal(a["volume0"], b["volume0"])),
        "provenance": res.provenance,
        "probe_roundtrip": res.meta["l_convention_probe"],
    }
    from experiments.nclaw.probe_l_convention import probe as probe_dump
    out["probe_original_dump"] = probe_dump(dump_npz)
    out["probe_verdicts_agree"] = (
        out["probe_original_dump"]["verdict"] == res.meta["l_convention_probe"]["verdict"])

    if identify:
        from experiments.nclaw import suite
        direct = suite.stage_identify(material, dump=dump_npz, tag="rt_direct", log=log)
        rt = suite.stage_identify(material, dump=back, tag="rt_ingested", log=log)
        out["theta_direct"] = direct["theta_engine"]
        out["theta_roundtrip"] = rt["theta_engine"]
        out["theta_rel_diff"] = {
            k: (abs(rt["theta_engine"][k] / v - 1.0) if v not in (0.0, None) else None)
            for k, v in direct["theta_engine"].items()}
        out["refused_direct"] = direct["refused_parameters"]
        out["refused_roundtrip"] = rt["refused_parameters"]

    if degraded:
        strip_dir = workdir / f"{dump_npz.stem}_x_only_format"
        x_only = workdir / f"{dump_npz.stem}_x_only.npz"
        strip_to_positions(state_dir, strip_dir, log=log)
        xres = read_nclaw_dir(strip_dir, {**man, "stress_lag_steps": 0}, x_only, log=log)
        from experiments.nclaw import suite
        xid = suite.stage_identify(material, dump=x_only, tag="rt_x_only", log=log)
        out["x_only"] = {
            "provenance": xres.provenance,
            "degradation_notes": xres.meta["degradation_notes"],
            "theta_engine": xid["theta_engine"],
            "refused_parameters": xid["refused_parameters"],
            "elastic": {k: xid.get("elastic", {}).get(k)
                        for k in ("mu", "lam", "E", "nu", "n_rows", "cond_AtA",
                                  "residual_rel", "refused", "reason")},
            "friction": {k: xid.get("friction", {}).get(k)
                         for k in ("refused", "reason", "pressure_source")},
        }
        if not keep:
            import shutil
            shutil.rmtree(strip_dir, ignore_errors=True)
            x_only.unlink(missing_ok=True)
    if not keep:
        shutil.rmtree(state_dir, ignore_errors=True)
        back.unlink(missing_ok=True)
    return out


def strip_to_positions(state_dir: str | Path, out_dir: str | Path,
                       log=print) -> Path:
    """Copy a state folder keeping only x: the kinematics-only degradation tier."""
    import torch
    state_dir, out_dir = Path(state_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in frame_files(state_dir):
        d = _torch_load(f, True)
        torch.save({"x": d["x"], "sections": d.get("sections", [d["x"].shape[0]]),
                    "types": d.get("types", torch.zeros(d["x"].shape[0], dtype=torch.int))},
                   out_dir / f.name)
    log(f"[strip] {state_dir} -> {out_dir}: x only")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nclaw_dir", nargs="?", help="their state_root: 0000.pt, 0001.pt, ...")
    ap.add_argument("--manifest", help="json manifest; see MANIFEST_SCHEMA in this module")
    ap.add_argument("--out", help="output npz path")
    ap.add_argument("--no-pickle", action="store_true",
                    help="fail instead of falling back to weights_only=False")
    ap.add_argument("--roundtrip", help="validate the chain on one of OUR dumps instead")
    ap.add_argument("--material", default="jelly")
    ap.add_argument("--workdir", default=None, help="scratch folder for --roundtrip")
    ap.add_argument("--degraded", action="store_true",
                    help="--roundtrip: also measure the x-only tier on the same dump")
    ap.add_argument("--keep", action="store_true",
                    help="--roundtrip: keep the exported folder and the ingested npz")
    ap.add_argument("--out-json", default=None, help="write the --roundtrip result here")
    ap.add_argument("--print-schema", action="store_true")
    a = ap.parse_args(argv)

    if a.print_schema:
        print(json.dumps(MANIFEST_SCHEMA, indent=2))
        return
    if a.roundtrip:
        wd = Path(a.workdir) if a.workdir else Path(a.roundtrip).parent / "roundtrip"
        wd.mkdir(parents=True, exist_ok=True)
        res = roundtrip_check(a.roundtrip, a.material, wd, degraded=a.degraded,
                              keep=a.keep)
        text = json.dumps(res, indent=2, default=float)
        print(text)
        if a.out_json:
            Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out_json).write_text(text + "\n")
        return
    if not a.nclaw_dir or not a.out:
        raise SystemExit("give <nclaw_dir> with --out, or --roundtrip <our_dump.npz>")
    read_nclaw_dir(a.nclaw_dir, a.manifest, a.out, allow_pickle=not a.no_pickle)


if __name__ == "__main__":
    main()

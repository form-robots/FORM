"""Ingest NCLaw's own truth trajectories, every scenario of every material,
into the truth dumps the cross-engine comparisons read.

Writes out/nclaw_cross_generalize/dumps/<material>_<tag>_truth.npz, the
trajectories experiments/nclaw/compare.py and experiments/fe_ls/cross.py
identify from and score against. Ingest only: no identification, no rollout.

Layout expected under --log-root (NCLaw's experiments/log):

  <material>/dataset/state/0000.pt, 0001.pt, ...    (+ ../hydra.yaml)
  <material>/time/state/...
  <material>/vel/0001/state/...  vel/0007/state/...  vel/0008/state/...
  <material>/shape/<mesh>/state/...

and the tags they are written under: dataset, time, vel_<seed>, shape_<mesh>.

Cube scenes (dataset, time, vel/*) resolve their manifest entirely from
hydra.yaml (dt, skip_frame, num_grids, rho, material, cube volume). Mesh
scenes (shape/*) need a particle volume hydra.yaml cannot give; it is read
from NCLaw's own precompute cache (--nclaw-dir, nclaw/assets/<mesh>_<res>_
<mode>.npz), the exact ``vol`` their MPMInitData.get_mesh caches before
scaling by prod(size) -- see nclaw/sim/mpm.py.

Needs pyyaml (for hydra.yaml), which is not a declared dependency of this
repo: ``uv pip install pyyaml`` into .venv first.

Run:
  .venv/bin/python -m experiments.nclaw.cross_generalize --material all \\
      --log-root /path/to/NCLaw/experiments/log \\
      --nclaw-dir /path/to/NCLaw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nclaw import ingest as ing

OUT = ROOT / "out" / "nclaw_cross_generalize"
DUMPS = OUT / "dumps"
MATERIALS = ("jelly", "plasticine", "sand", "water")


# ---------------------------------------------------------------------------
# Manifest: hydra.yaml for everything, the NCLaw asset cache for mesh volume
# ---------------------------------------------------------------------------

def _hydra_blob(state_dir: Path) -> dict:
    """Their resolved env.blob.* section, from the hydra.yaml ingest.py itself
    looks up (state_root/../hydra.yaml)."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is needed to read NCLaw's hydra.yaml: "
                         "uv pip install pyyaml into .venv") from None
    cfg_path = ing._config_path(state_dir)
    if cfg_path is None:
        raise SystemExit(f"no hydra.yaml found near {state_dir}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    env = cfg.get("env") or {}
    return env.get("blob") or env


def mesh_particle_volume(assets_dir: Path, shape: dict) -> float:
    """Their own per-particle volume for a mesh blob, read from the SAME
    precompute cache MPMInitData.get_mesh reads (nclaw/sim/mpm.py):
    ``vol = mesh.volume / N`` is cached per-particle, THEN scaled by
    prod(size); replaying that scale here reproduces their number exactly, no
    mesh reload needed.
    """
    name, resolution, mode = shape["name"], shape["resolution"], shape["mode"]
    cache = assets_dir / f"{name}_{resolution}_{mode}.npz"
    if not cache.is_file():
        raise SystemExit(
            f"no precompute cache at {cache}; NCLaw writes this the first time "
            f"it renders the {name!r} mesh at resolution {resolution}, mode {mode!r}")
    d = np.load(cache)
    size = np.asarray(shape["size"], dtype=float)
    return float(d["vol"]) * float(np.prod(size))


def build_manifest(state_dir: Path, material: str, assets_dir: Path | None) -> dict:
    """The manifest for one scenario folder: hydra.yaml supplies everything
    except a mesh blob's volume, which needs the NCLaw asset cache."""
    blob = _hydra_blob(state_dir)
    shape = blob.get("shape") or {}
    man: dict = {"material": material}
    if shape.get("type") == "mesh":
        if assets_dir is None:
            raise SystemExit(
                f"{state_dir} is a mesh scene (shape={shape.get('name')!r}); pass "
                "--nclaw-dir so its assets/ precompute cache can give the volume")
        man["particle_volume"] = mesh_particle_volume(assets_dir, shape)
    return man


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------

def discover(log_root: Path, material: str) -> dict[str, Path]:
    """tag -> state folder, for one material's dataset/time/vel/*/shape/*."""
    base = log_root / material
    out: dict[str, Path] = {}
    for tag, rel in (("dataset", "dataset/state"), ("time", "time/state")):
        p = base / rel
        if p.is_dir():
            out[tag] = p
    for p in sorted((base / "vel").glob("*")) if (base / "vel").is_dir() else []:
        if (p / "state").is_dir():
            out[f"vel_{p.name}"] = p / "state"
    for p in sorted((base / "shape").glob("*")) if (base / "shape").is_dir() else []:
        if (p / "state").is_dir():
            out[f"shape_{p.name}"] = p / "state"
    return out


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_material(material: str, log_root: Path, assets_dir: Path | None,
                    force: bool = False, log=print) -> dict[str, Path]:
    """Every scenario folder of one material to its truth npz; tag -> path."""
    scenes = discover(log_root, material)
    if "dataset" not in scenes:
        raise SystemExit(f"no dataset/state under {log_root / material}; "
                         "the identify leg downstream needs their training throw")
    log(f"[cross-gen] {material}: {len(scenes)} scenario(s): {sorted(scenes)}")

    DUMPS.mkdir(parents=True, exist_ok=True)
    truths: dict[str, Path] = {}
    for tag, state_dir in scenes.items():
        out_path = DUMPS / f"{material}_{tag}_truth.npz"
        if out_path.exists() and not force:
            log(f"[cross-gen] reuse {out_path.name}")
        else:
            man = build_manifest(state_dir, material, assets_dir)
            ing.read_nclaw_dir(state_dir, man, out_path, log=log)
        truths[tag] = out_path
    return truths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--material", default="all",
                    help="comma list of jelly,plasticine,sand,water, or 'all'")
    ap.add_argument("--log-root", required=True,
                    help="NCLaw's experiments/log (holds <material>/dataset, "
                         "time, vel/*, shape/*)")
    ap.add_argument("--nclaw-dir", default=None,
                    help="the NCLaw clone, for shape/* mesh volumes: "
                         "<nclaw-dir>/nclaw/assets/<mesh>_<res>_<mode>.npz")
    ap.add_argument("--force", action="store_true",
                    help="re-ingest scenes whose dumps already exist")
    a = ap.parse_args(argv)

    materials = list(MATERIALS) if a.material == "all" \
        else [m.strip() for m in a.material.split(",") if m.strip()]
    log_root = Path(a.log_root)
    assets_dir = Path(a.nclaw_dir) / "nclaw" / "assets" if a.nclaw_dir else None
    written = {m: ingest_material(m, log_root, assets_dir, force=a.force)
               for m in materials}
    print()
    for m, truths in written.items():
        print(f"[cross-gen] {m}: {', '.join(sorted(truths))} -> {DUMPS}")


if __name__ == "__main__":
    main()

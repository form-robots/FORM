"""The NCLaw ingestion path, validated before any NCLaw data exists.

The chain under test is: one of our dumps -> their exact state format
(0000.pt, 0001.pt, ... holding dict(x, v, C, F, stress, sections, types) in
their y-up frame) -> back through experiments/nclaw/ingest.py -> a schema-valid
dump of ours. Because the export is the inverse of the ingest by construction,
every disagreement is a bug in one of the two, which is what makes the
round trip a real test of the path their data will travel.

The trajectory here is manufactured rather than simulated: a smooth,
spatially varying displacement field with an analytic velocity, velocity
gradient and deformation gradient, and a fixed-corotated Cauchy stress. No
simulator runs in this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from experiments.nclaw import ingest as ing
from experiments.nclaw import suite

from ident.io.schema import validate_dump_schema
from ident.weakform.elastic_grid import E_nu_to_moduli, corotated_cauchy_columns

N_GRID = 12
GRID_LIM = 1.0
FRAME_DT = 4.0e-3
N_FRAMES = 20
RHO = 1000.0
E_TRUE, NU_TRUE = 1.0e5, 0.2
GRAVITY = np.array([0.0, 0.0, -9.8])
K_WAVE = 2.0 * np.pi * 0.75
AMP = 0.02


def _field(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A smooth non-uniform displacement field u(X) and its gradient du_i/dX_j."""
    k = K_WAVE
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    u = np.stack([0.5 * np.sin(k * x) + 0.3 * np.sin(k * z),
                  0.4 * np.cos(k * y),
                  0.35 * np.sin(k * (x + z))], axis=1)
    G = np.zeros((len(X), 3, 3))
    G[:, 0, 0] = 0.5 * k * np.cos(k * x)
    G[:, 0, 2] = 0.3 * k * np.cos(k * z)
    G[:, 1, 1] = -0.4 * k * np.sin(k * y)
    G[:, 2, 0] = 0.35 * k * np.cos(k * (x + z))
    G[:, 2, 2] = 0.35 * k * np.cos(k * (x + z))
    return u, G


def synthetic_dump(path: Path, n_frames: int = N_FRAMES) -> Path:
    """A schema-valid dump of a manufactured elastic motion, float handling and
    all keys matching what the suite's own DumpWriter produces."""
    dx = GRID_LIM / N_GRID
    h = dx / 2.0
    ax = np.arange(-0.25, 0.25 + 0.5 * h, h)
    X = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + 0.5
    P = len(X)
    u, G = _field(X)
    t = np.arange(n_frames) * FRAME_DT
    t_tot = (n_frames - 1) * FRAME_DT
    s = np.sin(np.pi * t / t_tot)
    ds = (np.pi / t_tot) * np.cos(np.pi * t / t_tot)

    xs = X[None] + AMP * s[:, None, None] * u[None]
    F = np.eye(3)[None, None] + AMP * s[:, None, None, None] * G[None]
    vs = AMP * ds[:, None, None] * u[None]
    # v is Lagrangian, so the spatial gradient goes through the inverse map:
    # L = dv/dx = (dv/dX) (dX/dx) = amp s'(t) G F^{-1}
    L = AMP * ds[:, None, None, None] * np.einsum("pij,fpjk->fpik", G, np.linalg.inv(F))

    mu, lam = E_nu_to_moduli(E_TRUE, NU_TRUE)
    stress = np.empty_like(F)
    for f in range(n_frames):
        s_mu, s_lam = corotated_cauchy_columns(F[f])
        stress[f] = mu * s_mu + lam * s_lam

    vol0 = np.full(P, h ** 3, dtype=np.float32)
    mass = (np.float32(RHO) * vol0).astype(np.float32)
    F32 = F.astype(np.float32)
    J = np.linalg.det(F32.astype(np.float64))
    volume = (J * vol0[None, :].astype(np.float64)).astype(np.float32)

    log10I = np.linspace(-4.0, 0.0, 256)
    meta = {
        "law_params": {"E": E_TRUE, "nu": NU_TRUE},
        "units": {"x": "m", "v": "m/s", "L": "1/s", "stress": "Pa"},
        "material": "jelly", "shape": "manufactured", "n_grid": N_GRID,
        "grid_lim": GRID_LIM, "dt": FRAME_DT, "substeps_per_frame": 1,
        "gravity": [float(g) for g in GRAVITY], "vel_name": "manufactured",
        "bound_cells": 3, "collider_bc": "slip", "recovered": False,
    }
    np.savez_compressed(
        path,
        schema_version=np.array("trackeuclid-dump-1.0"),
        coordinate_convention=np.array("configA_x0_z2_yout1"),
        in_plane_axes=np.asarray([0, 2], dtype=int),
        out_of_plane_axis=np.array(1, dtype=int),
        L_convention=np.array(ing.L_CONVENTION_STRING),
        frame_dt=np.array(FRAME_DT), grain_diameter=np.array(1.0e-3),
        rho_s=np.array(RHO), rho_bulk=np.array(RHO),
        packing_fraction=np.array(1.0),
        gravity_inplane=np.asarray([GRAVITY[0], GRAVITY[2]], dtype=float),
        pressure_source=np.array("true_mpm_trace"), law=np.array("corotated"),
        mu_table_log10I=log10I, mu_table_mu=np.zeros_like(log10I),
        flowing_I_hist_edges=np.logspace(-4, 0, 41),
        flowing_I_hist_counts=np.zeros(40),
        meta_json=np.array(json.dumps(meta)),
        times=t, x=xs.astype(np.float32), v=vs.astype(np.float32),
        L=L.astype(np.float32).reshape(n_frames, P, 9),
        stress=stress.astype(np.float32).reshape(n_frames, P, 9),
        volume=volume, mass=mass, active=np.ones((n_frames, P), dtype=bool),
        F=F32.reshape(n_frames, P, 9), volume0=vol0,
    )
    return path


@pytest.fixture(scope="module")
def dump(tmp_path_factory) -> Path:
    return synthetic_dump(tmp_path_factory.mktemp("nclaw_rt") / "manufactured.npz")


@pytest.fixture(scope="module")
def exported(dump: Path) -> tuple[Path, dict]:
    """Our dump in their format, plus the manifest that describes the folder."""
    out = dump.parent / "state_root"
    ing.export_to_nclaw(dump, out, log=lambda *_: None)
    return out, ing.manifest_for_export(dump, "jelly")


@pytest.fixture(scope="module")
def ingested(exported, dump: Path) -> Path:
    state_dir, man = exported
    back = dump.parent / "roundtrip.npz"
    ing.read_nclaw_dir(state_dir, man, back, log=lambda *_: None)
    return back


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_names_every_missing_entry(exported):
    state_dir, _ = exported
    with pytest.raises(ing.ManifestError) as e:
        ing.resolve_manifest(state_dir, {}, 100)
    msg = str(e.value)
    for k in ("material", "rho", "dt", "skip_frame", "num_grids", "particle_volume"):
        assert k in msg, f"the failure message must name {k}"


def test_manifest_derives_particle_volume_three_ways(exported):
    state_dir, _ = exported
    base = {"material": "jelly", "rho": 1e3, "dt": 5e-4, "skip_frame": 1,
            "num_grids": 20}
    direct = ing.resolve_manifest(state_dir, {**base, "particle_volume": 1.25e-4}, 1000)
    total = ing.resolve_manifest(state_dir, {**base, "total_volume": 0.125}, 1000)
    shape = ing.resolve_manifest(
        state_dir, {**base, "shape": {"kind": "cube", "size": [0.5, 0.5, 0.5]}}, 1000)
    assert direct["particle_volume"] == pytest.approx(1.25e-4)
    assert total["particle_volume"] == pytest.approx(1.25e-4)
    assert shape["particle_volume"] == pytest.approx(1.25e-4)
    # their cube: prod(size) / N with resolution 10 per axis on a 0.5 m cube
    assert shape["particle_volume_source"] == "prod(shape.size) / N"


def test_config_lookup_finds_their_resolved_config(tmp_path):
    """Their eval.py saves the resolved config one level above the state folder.

    OmegaConf.save(cfg, exp_root / 'hydra.yaml') with state_root =
    exp_root / 'state', so state_root/../hydra.yaml is where it lands. Hydra's
    own .hydra folder is disabled in their configs/default.yaml.
    """
    state = tmp_path / "state"
    state.mkdir()
    assert ing._config_path(state) is None
    (tmp_path / "hydra.yaml").write_text("sim:\n  dt: 5.0e-4\n")
    assert ing._config_path(state) == tmp_path / "hydra.yaml"


def test_manifest_rejects_unknown_frame_convention(exported):
    state_dir, man = exported
    with pytest.raises(ing.ManifestError):
        ing.resolve_manifest(state_dir, {**man, "frame_convention": "xup"}, 10)


# ---------------------------------------------------------------------------
# Round trip: arrays
# ---------------------------------------------------------------------------

def test_ingested_dump_is_schema_valid(ingested: Path):
    meta = validate_dump_schema(ingested)
    assert meta.law == "corotated"
    assert meta.has_pressure
    assert meta.extra["n_grid"] == N_GRID
    assert meta.extra["grid_lim"] == pytest.approx(GRID_LIM)
    assert meta.extra["gravity"] == pytest.approx([0.0, 0.0, -9.8])
    assert meta.extra["channel_provenance"] == {
        "x": "measured", "v": "measured", "F": "measured", "L": "measured",
        "stress": "measured", "volume": "derived", "mass": "derived",
        "volume0": "manifest"}


def test_round_trip_positions_are_exact_and_tensors_hold_to_float32(dump, ingested):
    a, b = np.load(dump), np.load(ingested)
    # x is rotated about the box centre (an affine map, not a bare signed
    # permutation), so the centring add/subtract costs one float32 ULP; v is a
    # free vector under a plain linear rotation and stays bitwise exact.
    assert np.abs(a["x"].astype(np.float64) - b["x"].astype(np.float64)).max() < 1e-7
    assert np.array_equal(a["v"], b["v"])
    assert np.array_equal(a["mass"], b["mass"])
    assert np.array_equal(a["volume0"], b["volume0"])
    assert np.array_equal(a["volume"], b["volume"])
    for key in ("L", "F"):
        u, w = a[key].astype(np.float64), b[key].astype(np.float64)
        assert np.abs(u - w).max() / np.abs(u).max() < 1e-6, key
    # stress goes out as Kirchhoff J sigma and comes back divided by J, so it
    # carries one float32 store of the product; everything else is exact.
    u, w = a["stress"].astype(np.float64), b["stress"].astype(np.float64)
    rel = np.abs(u - w).max() / np.abs(u).max()
    assert rel < 1e-6, f"stress relative round-trip error {rel:.2e}"


def test_kirchhoff_conversion_is_the_only_stress_loss(dump, tmp_path):
    """With the manifest declaring Cauchy, the stress round trip is exact too."""
    state = tmp_path / "state_cauchy"
    ing.export_to_nclaw(dump, state, stress_kind="cauchy", log=lambda *_: None)
    man = ing.manifest_for_export(dump, "jelly", stress_kind="cauchy")
    back = tmp_path / "cauchy.npz"
    ing.read_nclaw_dir(state, man, back, log=lambda *_: None)
    assert np.array_equal(np.load(dump)["stress"], np.load(back)["stress"])


def test_probe_verdict_agrees_with_the_dump_probe(dump, ingested):
    from experiments.nclaw.probe_l_convention import probe
    direct = probe(dump)
    meta = validate_dump_schema(ingested)
    rt = meta.extra["l_convention_probe"]
    assert direct["verdict"] == rt["verdict"]
    assert direct["verdict"].startswith("L == dv_i/dx_j")
    assert rt["matches_expected"], rt
    assert rt["C_transposed_on_ingest"] is False
    assert rt["median_err_vs_L"] < rt["median_err_vs_LT"]


def test_transposed_C_is_measured_and_undone(dump, tmp_path):
    """A folder whose C is the transpose must still ingest to OUR convention.

    The convention is measured on the data, so writing the transposed channel
    flips the verdict and the ingest transposes it back; a hard-coded reading
    would silently store dv_j/dx_i.
    """
    state = tmp_path / "state_T"
    ing.export_to_nclaw(dump, state, log=lambda *_: None)
    import torch
    for f in sorted(state.glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=True)
        d["C"] = d["C"].transpose(1, 2).contiguous()
        torch.save(d, f)
    back = tmp_path / "transposed.npz"
    res = ing.read_nclaw_dir(state, ing.manifest_for_export(dump, "jelly"), back,
                             log=lambda *_: None)
    probe = res.meta["l_convention_probe"]
    assert probe["C_transposed_on_ingest"] is True
    assert probe["median_err_vs_LT"] < probe["median_err_vs_L"]
    a, b = np.load(dump), np.load(back)
    u, w = a["L"].astype(np.float64), b["L"].astype(np.float64)
    assert np.abs(u - w).max() / np.abs(u).max() < 1e-6


def test_stress_lag_alignment_shifts_and_drops_one_frame(dump, tmp_path):
    """Their eval.py saves elasticity(F_{k-1}) beside state k; lag 1 undoes it.

    The folder here is written WITH their lag, so the ingest must undo it. The
    same folder read with lag 0 is the negative control: it pairs each stress
    with the state one frame ahead of the one that produced it and misses by
    order one.
    """
    state = tmp_path / "state_lag"
    ing.export_to_nclaw(dump, state, stress_lag_steps=1, log=lambda *_: None)
    base = ing.manifest_for_export(dump, "jelly")
    a = np.load(dump)

    back = tmp_path / "lag1.npz"
    ing.read_nclaw_dir(state, {**base, "stress_lag_steps": 1}, back, log=lambda *_: None)
    b = np.load(back)
    assert b["x"].shape[0] == a["x"].shape[0] - 1
    # the state keeps its frames; one float32 ULP from the box-centre affine
    # rotation (see test_round_trip_positions_are_exact_and_tensors_hold_to_float32)
    assert np.abs(a["x"][:-1].astype(np.float64) - b["x"].astype(np.float64)).max() < 1e-7
    u = a["stress"][:-1].astype(np.float64)
    w = b["stress"].astype(np.float64)
    assert np.abs(u - w).max() / np.abs(u).max() < 1e-6
    assert validate_dump_schema(back).extra["frames_dropped_for_stress_lag"] == 1

    wrong = tmp_path / "lag0.npz"
    ing.read_nclaw_dir(state, {**base, "stress_lag_steps": 0}, wrong, log=lambda *_: None)
    c = np.load(wrong)["stress"][:-1].astype(np.float64)
    assert np.abs(u - c).max() / np.abs(u).max() > 1e-2, "the lag must matter"


# ---------------------------------------------------------------------------
# Round trip: identification
# ---------------------------------------------------------------------------

def _identify_elastic(path: Path) -> dict:
    arr = suite._load_arrays(path)
    return suite.identify_elastic(arr, window_frames=8, frame_stride=2,
                                  log=lambda *_: None)


def test_round_trip_identify_reproduces_theta(dump, ingested):
    direct = _identify_elastic(dump)
    rt = _identify_elastic(ingested)
    assert not direct["refused"] and not rt["refused"]
    assert direct["n_rows"] == rt["n_rows"] and direct["n_rows"] > 8
    # 1e-6: the box-centre affine position rotation costs one float32 ULP
    # (see test_round_trip_positions_are_exact_and_tensors_hold_to_float32),
    # which the weak-form solve amplifies less than the other round-trip
    # channels already toleranced at this level.
    for key in ("mu", "lam", "E", "nu"):
        assert abs(rt[key] / direct[key] - 1.0) < 1e-6, (
            f"{key}: direct {direct[key]!r} vs round trip {rt[key]!r}")


# ---------------------------------------------------------------------------
# Degradation tier: positions only
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def x_only(exported, dump: Path) -> Path:
    state_dir, man = exported
    stripped = ing.strip_to_positions(state_dir, dump.parent / "state_x_only",
                                      log=lambda *_: None)
    out = dump.parent / "x_only.npz"
    ing.read_nclaw_dir(stripped, man, out, log=lambda *_: None)
    return out


def test_x_only_records_every_derived_channel(x_only: Path):
    meta = validate_dump_schema(x_only)
    prov = meta.extra["channel_provenance"]
    assert prov["x"] == "measured"
    assert prov["v"] == "derived" and prov["L"] == "derived" and prov["F"] == "derived"
    assert prov["stress"] == "absent"
    assert meta.pressure_source == "absent"
    assert meta.has_pressure is False
    assert meta.extra["has_oracle_pressure"] is False
    notes = " ".join(meta.extra["degradation_notes"])
    assert "finite differences" in notes and "least squares" in notes
    assert "oracle pressure" in notes


def test_x_only_elastic_identification_runs_and_the_error_is_recorded(x_only, dump, capsys):
    """The kinematics-only tier: no accuracy bar, the measurement is the point.

    The comparison is against the measured-channel path on the SAME trajectory,
    not against the manufactured moduli. This motion is kinematically smooth but
    not a momentum-balance solution, so neither path recovers E here (the direct
    path reads about 9.8e5 against 1e5). What the tier costs on a real
    trajectory is measured by the round-trip run on the suite dumps, where the
    measured-channel path recovers E to 0.09 percent.
    """
    direct = _identify_elastic(dump)
    out = _identify_elastic(x_only)
    assert not out["refused"], out
    with capsys.disabled():
        print(f"\n[x-only tier] mu {out['mu']:.4e} vs measured-channel "
              f"{direct['mu']:.4e} ({100 * (out['mu'] / direct['mu'] - 1.0):+.2f} percent), "
              f"lam {out['lam']:.4e} vs {direct['lam']:.4e}, rows {out['n_rows']} vs "
              f"{direct['n_rows']}, cond {out['cond_AtA']:.2e}")
    assert np.isfinite(out["mu"]) and np.isfinite(out["lam"])
    assert out["n_rows"] == direct["n_rows"]


def test_x_only_friction_leg_refuses_for_want_of_pressure(x_only: Path):
    arr = suite._load_arrays(x_only)
    out = suite.identify_friction(arr, window_frames=8, frame_stride=2,
                                  log=lambda *_: None)
    assert out["refused"] is True
    assert "oracle pressure" in out["reason"]
    assert out["pressure_source"] == "absent"


# ---------------------------------------------------------------------------
# Seeding a rollout from their cloud (no simulator: the seed itself)
# ---------------------------------------------------------------------------

def test_cloud_from_dump_matches_frame_zero(ingested: Path):
    cloud = suite.cloud_from_dump(ingested)
    d = np.load(ingested)
    assert np.array_equal(cloud["pts"], d["x"][0])
    assert np.array_equal(cloud["v0"], d["v"][0])
    assert np.array_equal(cloud["vol0"], d["volume0"])
    assert cloud["n_frames"] == d["x"].shape[0] - 1
    assert cloud["t_end"] == pytest.approx(float(d["times"][-1]))
    assert cloud["n_grid"] == N_GRID


@pytest.mark.slow
def test_cross_stage_is_one_command(exported, tmp_path, monkeypatch):
    """The whole chain: ingest, identify, roll out from THEIR cloud, score.

    This is the only test here that runs the simulator, on a 2197-particle
    20-frame rollout, so it is marked slow. What it checks is the plumbing: the
    rollout is seeded from the ingested frame-0 cloud, so the score is computed
    between two trajectories in 1:1 particle correspondence.

    The identified pair is replaced by the manufactured one before the rollout.
    The manufactured motion is not a momentum-balance solution, so the solve on
    it returns nu of about 5.3, and a rollout at an unphysical law is refused by
    _wave_speed (measured: it used to reach the engine and bus-error on a NaN
    time step). Identification accuracy is measured on the suite dumps, not here.
    """
    state_dir, man = exported
    monkeypatch.setattr(suite, "OUT", tmp_path)
    monkeypatch.setattr(suite, "DUMPS", tmp_path / "dumps")
    monkeypatch.setattr(suite, "theta_for_engine",
                        lambda material, ident, **_: ({"E": E_TRUE, "nu": NU_TRUE}, [], {}))
    out = suite.stage_cross("jelly", state_dir, man, name="rt", window_frames=8,
                            log=lambda *_: None)
    assert out["n_particles"] == out["mse"]["n_particles"]
    assert out["mse"]["mse"] >= 0.0
    assert out["channel_provenance"]["x"] == "measured"
    assert out["n_grid"] == N_GRID
    rec = np.load(tmp_path / "dumps" / out["rollout_dump"])
    truth = np.load(tmp_path / "dumps" / out["ingested_dump"])
    # 1:1 correspondence, and the rollout starts from their state
    assert rec["x"].shape[1] == truth["x"].shape[1]
    assert np.abs(rec["x"][0] - truth["x"][0]).max() == 0.0
    assert np.abs(rec["v"][0] - truth["v"][0]).max() == 0.0


# ---------------------------------------------------------------------------
# The same chain on a real suite dump, when one is present
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("material", ["jelly", "sand"])
def test_roundtrip_on_a_real_dump(material, tmp_path):
    """Opt-in: NCLAW_ROUNDTRIP_DUMPS=<dir> with <material>_cube_truth.npz inside.

    The synthetic tests above cover the path; this one measures it on the real
    trajectories the comparison reports, which is a several-minute, several-GB
    run and therefore not part of the default suite.
    """
    root = os.environ.get("NCLAW_ROUNDTRIP_DUMPS")
    if not root:
        pytest.skip("set NCLAW_ROUNDTRIP_DUMPS to the folder of suite dumps")
    src = Path(root) / f"{material}_cube_truth.npz"
    if not src.exists():
        pytest.skip(f"{src} not present")
    res = ing.roundtrip_check(src, material, tmp_path, log=lambda *_: None)
    assert res["x_bitwise_identical"]
    assert res["probe_verdicts_agree"]
    assert res["rel_max_diff"]["L"] == 0.0 and res["rel_max_diff"]["F"] == 0.0
    assert res["rel_max_diff"]["stress"] < 1e-6
    # measured on both suite dumps: jelly exactly 0, sand 2.6e-11, the residue
    # of the one float32 store of the Kirchhoff product
    for k, v in res["theta_rel_diff"].items():
        assert v is None or v < 1e-9, f"{k}: {v}"

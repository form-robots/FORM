"""NCLaw-matched comparison: identify from one cube throw, roll out on their shapes.

NCLaw (ICML 2023) learns a neural constitutive law by differentiating an MPM
rollout, from ONE thrown-blob trajectory per material, and reports the
box-normalized per-particle position MSE on the training scene and on a
held-out geometry. This runs their protocol with our engine and our weak-form
identification: the law comes from a convex, linear-in-theta solve on the cube
trajectory alone, the simulator is never differentiated, and the recovered law
is then re-simulated on the cube and on held-out shapes.

Protocol, from NCLaw/experiments/configs (verified):
  0.5 m cube of particles centred at (0.5, 0.5, 0.5) in a unit box, thrown with
  linear velocity [1.0, -1.5, -2.0] and angular velocity [4, 4, 4] in their
  y-up frame, freeslip walls with a 3-cell bound, 1000 steps of dt 5e-4 = 0.5 s.
  Our frame is z-up, so the throw and gravity are rotated by the same proper
  rotation (their +y becomes our +z) and our engine picks its own stable dt.

Stages:
  gen       truth trajectories for the cube and the evaluation shapes
  identify  recover the law from the CUBE trajectory only
  rollout   re-simulate every scene at the recovered law, score the MSE
  report    results.json plus report.md plus the comparison figure
  cross     the cross-engine chain on a folder of THEIR trajectories: ingest,
            identify, roll out from their frame-0 cloud, score

Run:
  .venv/bin/python -m experiments.nclaw.suite gen      --material jelly --shapes cube,bunny
  .venv/bin/python -m experiments.nclaw.suite identify --material jelly
  .venv/bin/python -m experiments.nclaw.suite rollout  --material jelly --shapes cube,bunny
  .venv/bin/python -m experiments.nclaw.suite report   --material jelly
  .venv/bin/python -m experiments.nclaw.suite cross    --material jelly \\
      --nclaw-dir /path/to/their/state_root --manifest manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out" / "nclaw_suite"
DUMPS = OUT / "dumps"
import os


def _find_nclaw() -> Path:
    """The NCLaw clone (meshes + configs). NCLAW_DIR wins; then common layouts."""
    cands = [Path(os.environ["NCLAW_DIR"])] if "NCLAW_DIR" in os.environ else []
    cands += [ROOT.parent / "NCLaw", ROOT / "NCLaw", ROOT.parent.parent / "NCLaw"]
    for c in cands:
        if (c / "nclaw" / "assets").is_dir():
            return c
    raise SystemExit("NCLaw clone not found; clone github NCLaw next to this repo "
                     "or set NCLAW_DIR to it (needed for their meshes and configs)")


def assets() -> Path:
    """Their asset folder, resolved on first use.

    Looked up lazily so that importing this module needs no clone: the ingest
    path (experiments/nclaw/ingest.py) reads their trajectories and their
    rotation from here and touches no mesh.
    """
    return _find_nclaw() / "nclaw" / "assets"

# ---------------------------------------------------------------------------
# Their scene, in their frame, then rotated once into ours
# ---------------------------------------------------------------------------

# proper rotation taking their y-up frame to our z-up frame: (x, y, z) -> (x, -z, y)
R_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])
GRAVITY_YUP = np.array([0.0, -9.8, 0.0])
LIN_VEL_YUP = np.array([1.0, -1.5, -2.0])       # configs/env/blob/vel/preset.yaml
ANG_VEL_YUP = np.array([4.0, 4.0, 4.0])
GRAVITY = R_YUP_TO_ZUP @ GRAVITY_YUP            # (0, 0, -9.8)
LIN_VEL = R_YUP_TO_ZUP @ LIN_VEL_YUP            # (1.0, 2.0, -1.5)
ANG_VEL = R_YUP_TO_ZUP @ ANG_VEL_YUP            # (4.0, -4.0, 4.0); a pseudovector
                                                # under a PROPER rotation
# NCLaw's geometry evals do NOT use the preset throw: eval/shape.py switches to
# vel/mild.yaml, linear [1.0, -1.5, -1.5], angular [1, 1, 1] (y-up). The
# preset throw drives the sand blub cloud through the freeslip walls and the
# rollout goes NaN. The mild throw is the primary config for the their-mesh
# cells; the preset-throw cells stay as the geometry-isolated secondary set.
VELS = {
    "preset": (LIN_VEL, ANG_VEL),
    "mild": (R_YUP_TO_ZUP @ np.array([1.0, -1.5, -1.5]),
             R_YUP_TO_ZUP @ np.array([1.0, 1.0, 1.0])),
}
GRID_LIM = 1.0
N_GRID = 32                                     # sim/high.yaml; their training used 20
BOUND_CELLS = 3

# The engine's grid semantics at NCLaw's dataset settings, for
# MPM_Simulator_WARP.set_grid_semantics. freeslip_bound and particle_clip_cells
# come from their configs (sim/*.yaml bound: 3; every env/blob/*.yaml
# clip_bound: 0.5). mass_eps is 1e-7 in sim/low.yaml (num_grids 20, the grid
# their released dataset and the ingested dumps use) and 0.0 in sim/high.yaml
# and sim/super.yaml, so a grid-32 or grid-64 comparison should pass 0.0.
NCLAW_GRID_SEMANTICS = {"freeslip_bound": BOUND_CELLS, "mass_eps": 1.0e-7,
                        "empty_node_gravity": True, "mls_transfer": True,
                        "particle_clip_cells": 0.5}

# particle_clip_cells above is right for every env/blob/*.yaml EXCEPT the
# mesh-shape scenes, which override clip_bound per NCLaw's own configs:
# bunny.yaml, blub.yaml and spot.yaml set it to ${sim.bound} (BOUND_CELLS at
# both the low and high quality presets we use), armadillo.yaml fixes it at 1.
# Missing this made the shape scenes rest against a clip 6x looser than
# NCLaw's, which is why they were the one axis where truth-theta rollouts
# diverged hard from NCLaw's own trajectory even with the correct law.
NCLAW_SHAPE_CLIP_BOUND = {"bunny": BOUND_CELLS, "blub": BOUND_CELLS,
                          "spot": BOUND_CELLS, "armadillo": 1.0}
T_END = 0.5                                     # 1000 steps of dt 5e-4
N_FRAMES = 125                                  # our dump cadence, 4 ms
CENTER = np.array([0.5, 0.5, 0.5])

# NCLaw shape configs: mesh name and the bounding-box size they scale it to.
SHAPES: dict[str, dict] = {
    "cube": {"kind": "cube", "size": 0.5},
    "bunny": {"kind": "mesh", "mesh": "bunny", "size": 0.7},
    "spot": {"kind": "mesh", "mesh": "spot", "size": 0.8},
    "dragon": {"kind": "mesh", "mesh": "dragon_full", "size": 0.8},
    "armadillo": {"kind": "mesh", "mesh": "armadillo", "size": 0.7},
    # blub's fins are 1 to 2 particles thick at the default pitch and the sand
    # run goes NaN. pitch_div 3 gives about 27k particles, near NCLaw's own
    # count.
    "blub": {"kind": "mesh", "mesh": "blub", "size": 0.8, "pitch_div": 3},
}

# The held-out shape NCLaw's own geometry-generalization column uses, per
# material. Their figure pairs a different mesh with each material, so a cell
# of our table is only strictly matched when the shape agrees.
NCLAW_HELD_OUT_SHAPE = {"jelly": "armadillo", "plasticine": "bunny",
                        "sand": "blub", "water": "spot"}

MATERIALS: dict[str, dict] = {
    # engine: warp-mpm material name; law: dump schema law tag
    "jelly": {
        "engine": "jelly", "law": "corotated", "rho": 1000.0,
        "truth": {"E": 1.0e5, "nu": 0.2},
        "theta_names": ["mu", "lam"],
        # their material/jelly.yaml: corotated_jelly + identity
        "nclaw_law": {"elasticity": "corotated", "plasticity": "identity"},
    },
    "plasticine": {
        # their dataset config is SigmaElasticity + VonMisesPlasticity, i.e.
        # HENCKY elasticity with the von Mises return: our "metal" (mat 1).
        # The fork's "plasticine" (mat 5) pairs the SAME return map with
        # fixed-corotated elasticity, which is their corotated_plasticine
        # variant, not the one their dataset uses.
        "engine": "metal", "law": "vonmises", "rho": 1000.0,
        "truth": {"E": 3.0e5, "nu": 0.25, "yield_stress": 5.0e3},
        "theta_names": ["mu", "lam", "yield_stress"],
        # their material/plasticine.yaml: sigma_plasticine + von_mises_plasticine
        "nclaw_law": {"elasticity": "hencky", "plasticity": "von_mises"},
    },
    "sand": {
        "engine": "sand", "law": "drucker_prager", "rho": 1000.0,
        "truth": {"E": 1.0e6, "nu": 0.2, "friction_angle": 25.0},
        "theta_names": ["friction_angle"],
        # their material/sand.yaml: sigma_sand + drucker_prager_sand (cohesion 0)
        "nclaw_law": {"elasticity": "hencky", "plasticity": "drucker_prager",
                      "cohesion": 0.0},
    },
    "water": {
        "engine": "fluid", "law": "eos_fluid", "rho": 1000.0,
        # their volumetric elasticity E, nu; the engine takes a bulk modulus
        "truth": {"E": 1.0e5, "nu": 0.3},
        "theta_names": ["bulk_modulus"],
        # their material/water.yaml: volume_water (mode taichi) + sigma. Their
        # pressure is linear in J - 1; ours is a gamma = 1.1 power law. The two
        # are not reparameterizations of each other.
        "nclaw_law": {"elasticity": "volume_taichi", "plasticity": "sigma"},
    },
}


# Rollout-only entries, for re-simulating a recovered CURVE instead of a named
# scalar. The engine's tabulated materials read the curve off the model as a
# table, so a function-encoder recovery (sim/fe_ls_baseline.py in the parent
# tree) rolls out through these. They carry no NCLaw published column and no
# identify leg of their own: the identify and report stages take the four
# physical materials above, and these entries only ever reach run_scene.
MATERIALS["sand_table"] = {
    "engine": "tabulated_mu_i", "law": "tabulated_mu_i", "rho": 1000.0,
    # E, nu and the grain scales of the sand truth run, so the tabulated rollout
    # differs from it in the friction law alone; d and rho_s fix the inertial
    # number the table is read at and must match the identification's.
    "truth": {"E": 1.0e6, "nu": 0.2,
              "grain_diameter": 1.0e-3, "grain_density": 1000.0},
    "theta_names": ["mu_table"],
}
MATERIALS["visc_table"] = {
    "engine": "tabulated_viscous", "law": "tabulated_viscous", "rho": 1000.0,
    "truth": {"E": 1.0e5, "nu": 0.3},
    "theta_names": ["eta_table"],
}
MATERIALS["yield_table"] = {
    # learned perfect plasticity: Hencky elasticity plus a tabulated yield
    # surface sqrt(J2(dev tau)) <= h(p) on a linear Kirchhoff-pressure grid.
    # A flat table is von Mises, a line through the origin is the cohesionless
    # Drucker-Prager cone; the elastic pair comes in through theta per run.
    "engine": "tabulated_yield", "law": "tabulated_yield", "rho": 1000.0,
    "truth": {"E": 3.0e5, "nu": 0.25},
    "theta_names": ["eta_table"],
}

# Keys the engine consumes verbatim. A recovered curve is a table plus its
# grid. It passes through engine_params unchanged; the grain scales set the
# inertial number the table is read at.
ENGINE_PASSTHROUGH = ("eta_table", "eta_table_smin", "eta_table_smax",
                      "grain_diameter", "grain_density", "bulk_modulus")


def bulk_from_E_nu(E: float, nu: float) -> float:
    """K = E / (3 (1 - 2 nu)), the standard mapping; recorded because NCLaw's
    water config states (E, nu) while our fluid material takes a bulk modulus."""
    return float(E / (3.0 * (1.0 - 2.0 * nu)))


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Particle seeding: deterministic, so truth and rollout are 1:1
# ---------------------------------------------------------------------------

def seed_cloud(shape: str, n_grid: int = N_GRID, grid_lim: float = GRID_LIM
               ) -> tuple[np.ndarray, np.ndarray]:
    """Particle positions and reference volumes for one NCLaw shape.

    Pitch is dx/2 (eight particles per cell), our engine's working density.
    NCLaw seeds the cube at resolution 10 per axis, about 1.1 of their cells,
    so our particle counts are larger. The metric compares our truth to our
    rollout on identical clouds, so the count only scales the number. The
    deviation is recorded in the report.
    """
    cfg = SHAPES[shape]
    dx = grid_lim / n_grid
    h = dx / float(cfg.get("pitch_div", 2))
    if cfg["kind"] == "cube":
        half = cfg["size"] / 2.0
        ax = np.arange(-half, half + 0.5 * h, h)
        pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
        pts = pts + CENTER
    else:
        import trimesh
        m = trimesh.load(assets() / f"{cfg['mesh']}.obj", force="mesh")
        m.apply_scale(cfg["size"] / m.extents.max())
        pts = np.asarray(m.voxelized(pitch=h).fill().points, dtype=np.float64)
        pts += np.random.default_rng(0).uniform(-0.2 * h, 0.2 * h, pts.shape)
        pts += CENTER - pts.mean(axis=0)
    vol0 = np.full(len(pts), h ** 3, dtype=np.float32)
    return pts.astype(np.float32), vol0


def throw_velocity(pts: np.ndarray, vel: str = "preset") -> np.ndarray:
    """v = lin + ang x (x - centre), exactly NCLaw's MPMStateInitializer."""
    lin, ang = VELS[vel]
    return (lin[None, :]
            + np.cross(ang[None, :], pts.astype(np.float64) - CENTER[None, :])
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# Forward run
# ---------------------------------------------------------------------------

def engine_params(material: str, theta: dict | None = None,
                  nclaw_law: bool = False) -> dict:
    """warp-mpm parameter dict for a material, at truth or at recovered theta.

    ``nclaw_law`` selects the engine's composed material (14) at the elasticity
    and plasticity kinds NCLaw's own config for this material uses, from the
    entry's ``nclaw_law`` spec. The four physical materials keep their own
    engine materials unless ``nclaw_law`` is set.
    """
    spec = MATERIALS[material]
    p = dict(spec["truth"])
    if theta:
        p.update(theta)
    if nclaw_law and "nclaw_law" in spec:
        law = spec["nclaw_law"]
        kw = {"material": "composed", "density": spec["rho"],
              "g": list(GRAVITY), "E": p["E"], "nu": p["nu"],
              "elasticity": law["elasticity"], "plasticity": law["plasticity"]}
        if "yield_stress" in p:
            kw["yield_stress"] = p["yield_stress"]
        if "friction_angle" in p:
            kw["friction_angle"] = p["friction_angle"]
        if "cohesion" in law:
            kw["cohesion"] = law["cohesion"]
        if "eos_gamma" in law:
            kw["eos_gamma"] = law["eos_gamma"]
        return kw
    kw = {"material": spec["engine"], "density": spec["rho"],
          "g": list(GRAVITY)}
    if spec["engine"] == "fluid":
        kw["bulk_modulus"] = p.get("bulk_modulus",
                                   bulk_from_E_nu(p["E"], p["nu"]))
        kw["E"], kw["nu"] = p["E"], p["nu"]
    else:
        kw["E"], kw["nu"] = p["E"], p["nu"]
    if "yield_stress" in p:
        kw["yield_stress"] = p["yield_stress"]
        kw["softening"] = 0.0          # NCLaw's von Mises has no damage term
    if "friction_angle" in p:
        kw["friction_angle"] = p["friction_angle"]
    for key in ENGINE_PASSTHROUGH:
        if key in p:
            kw[key] = p[key]
    return kw


def _wave_speed(material: str, kw: dict) -> float:
    """The p-wave speed the time step is sized from.

    Refuses non-physical laws. nu outside (-1, 0.5) makes the wave speed NaN
    and the first p2g2p writes out of the grid (a bus error on this machine).
    """
    rho = MATERIALS[material]["rho"]
    if "bulk_modulus" in kw:
        bulk = float(kw["bulk_modulus"])
        if not np.isfinite(bulk) or bulk <= 0.0:
            raise ValueError(f"non-physical bulk modulus for a rollout: {bulk!r}")
        return float(np.sqrt(bulk / rho))
    E, nu = float(kw["E"]), float(kw["nu"])
    if not (np.isfinite(E) and E > 0.0) or not (np.isfinite(nu) and -1.0 < nu < 0.5):
        raise ValueError(
            f"non-physical elastic pair for a rollout: E={E!r}, nu={nu!r}. "
            "nu must lie in (-1, 0.5) and E must be positive, or the wave speed "
            "and the time step come out NaN.")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(np.sqrt((lam + 2.0 * mu) / rho))     # p-wave speed


def cloud_from_dump(path: str | Path) -> dict:
    """Frame-0 state and run geometry of a dump, for a rollout in 1:1 particle
    correspondence with that trajectory.

    Seeding from their frame-0 cloud puts our rollout in 1:1 particle
    correspondence with their trajectory.
    """
    from ident.io.schema import validate_dump_schema
    meta = validate_dump_schema(path)
    d = np.load(path)
    times = np.asarray(d["times"], dtype=float)
    return {
        "pts": np.ascontiguousarray(d["x"][0].astype(np.float32)),
        "vol0": np.ascontiguousarray(
            (d["volume0"] if "volume0" in d.files else d["volume"][0]).astype(np.float32)),
        "v0": np.ascontiguousarray(d["v"][0].astype(np.float32)),
        "n_frames": int(times.shape[0]) - 1,
        "t_end": float(times[-1] - times[0]),
        "n_grid": int(meta.extra["n_grid"]),
        "grid_lim": float(meta.extra["grid_lim"]),
        "source": str(Path(path).name),
    }


def run_scene(material: str, shape: str, out_path: Path, theta: dict | None = None,
              n_grid: int = N_GRID, grid_lim: float = GRID_LIM,
              t_end: float = T_END, n_frames: int = N_FRAMES,
              cfl: float = 0.35, vel: str = "preset", cloud: dict | None = None,
              nclaw_bc: bool | dict = False, nclaw_law: bool = False,
              substeps: int | None = None, device: str = "cpu",
              xonly: bool = False,
              log=print, timing: dict | None = None) -> Path:
    """One truth or rollout trajectory, dumped schema-valid with F and V0.

    ``timing``, if given, is updated in place with a sub-breakdown of this
    call's wall time: ``setup_s`` (warp/simulator init, particle seeding,
    everything before the frame loop), ``step_s`` (the physics substeps
    alone), ``snapshot_s`` (buffering each frame's state into the writer, not
    counting the substeps themselves), ``finalize_s`` (the single
    ``np.savez``/``np.savez_compressed`` call at the end -- uncompressed by
    default here, see ``DumpWriter.compress``), and ``total_s``.

    ``xonly`` writes positions and times only, dropping v/L/stress/F/volume/
    active/mass and the mu-table/flowing-I diagnostics. Scoring
    (``nclaw_position_mse``) and rendering (``blender_scene.py``) only ever
    read ``x`` back from a rollout's own dump, so a comparison prediction that
    will never be re-seeded from can use this for a large disk and write-time
    saving; a dump meant as truth or identification input must keep the
    default full state.

    ``cloud`` (from ``cloud_from_dump``) replaces the analytic seeding with a
    provided particle cloud: frame-0 positions, reference volumes and
    velocities, plus that trajectory's grid and horizon. Everything downstream
    is unchanged, so a rollout on their cloud is scored by the same metric.

    ``nclaw_bc`` asks the engine for NCLaw's grid semantics
    (``MPM_Simulator_WARP.set_grid_semantics``): ``True`` takes all of
    ``NCLAW_GRID_SEMANTICS``, and a dict overrides individual options for
    one-behavior measurements. The walls come from exactly one
    source: the engine's freeslip grid operator when ``freeslip_bound`` is set,
    otherwise the six collider slip planes.

    ``nclaw_law`` rolls the material out on NCLaw's own constitutive pair for it
    (engine material 14, see ``engine_params``), which for water is a different
    equation of state and not a reparameterization of ours.

    ``substeps`` fixes the substep count per dumped frame instead of taking it
    from the CFL. The truth-theta control needs it: their trajectory is their
    discrete solution at their dt, so the comparison must use that dt.
    """
    t_start = time.time()
    import warp as wp
    wp.config.quiet = True
    wp.init()
    import torch

    from experiments.nclaw.dump_writer import DumpWriter
    from experiments.nclaw.probe_l_convention import L_CONVENTION_STRING
    from warpmpm.kernels import MPM_Simulator_WARP

    if cloud is None:
        pts, vol0 = seed_cloud(shape, n_grid, grid_lim)
        v0 = throw_velocity(pts, vel)
    else:
        pts, vol0, v0 = cloud["pts"], cloud["vol0"], cloud["v0"]
        n_grid, grid_lim = cloud["n_grid"], cloud["grid_lim"]
        t_end, n_frames = cloud["t_end"], cloud["n_frames"]
    kw = engine_params(material, theta, nclaw_law=nclaw_law)
    dx = grid_lim / n_grid
    frame_dt = t_end / n_frames
    c = _wave_speed(material, kw)
    v_max = float(np.abs(v0).max())
    dt_cfl = cfl * dx / max(c + v_max, 1e-9)
    cfl_substeps = max(int(np.ceil(frame_dt / dt_cfl)), 1)
    if substeps is None:
        substeps = cfl_substeps
    elif substeps < cfl_substeps:
        # A fixed substep count matches the other engine's dt. Subdividing
        # below their dt moves the rollout away from their discrete solution,
        # so log when the request is coarser than our CFL.
        log(f"[gen] substeps={substeps} requested below the CFL's {cfl_substeps} "
            f"(dt {frame_dt / substeps:.2e} vs {dt_cfl:.2e})")
    dt = frame_dt / substeps
    log(f"[gen] {material}/{shape} N={len(pts)} grid={n_grid}^3 c={c:.0f}m/s "
        f"dt={dt:.2e} sub={substeps} frames={n_frames} theta={theta or 'truth'}")

    s = MPM_Simulator_WARP(len(pts), device=device)
    s.load_initial_data_from_torch(
        torch.from_numpy(np.ascontiguousarray(pts)),
        torch.from_numpy(np.ascontiguousarray(vol0)),
        n_grid=n_grid, grid_lim=grid_lim, device=device)
    s.import_particle_v_from_torch(
        torch.from_numpy(np.ascontiguousarray(v0)), device=device)
    s.set_parameters_dict(kw, device=device)
    s.finalize_mu_lam(device=device)
    kw_bc = None
    if nclaw_bc:
        kw_bc = dict(NCLAW_GRID_SEMANTICS)
        # their sim/low.yaml (grid 20) sets 1e-7; high and super set 0.0, so
        # the epsilon follows the trajectory's grid rather than a constant
        kw_bc["mass_eps"] = 1.0e-7 if n_grid == 20 else 0.0
        # a compare.py scene label like "shape_bunny" has that prefix stripped
        # to match NCLAW_SHAPE_CLIP_BOUND's bare shape-name keys
        kw_bc["particle_clip_cells"] = NCLAW_SHAPE_CLIP_BOUND.get(
            shape.removeprefix("shape_"), 0.5)
        if isinstance(nclaw_bc, dict):
            kw_bc.update(nclaw_bc)
        s.set_grid_semantics(**kw_bc)
        log(f"[gen] grid semantics {kw_bc}")
    if kw_bc is None or not kw_bc["freeslip_bound"]:
        # Collider slip planes on all six faces when the grid freeslip operator
        # is off. A bisection leg that enables one other behavior keeps these
        # planes.
        pad = BOUND_CELLS * dx
        for pt, nrm in (((pad, 0, 0), (1, 0, 0)), ((grid_lim - pad, 0, 0), (-1, 0, 0)),
                        ((0, pad, 0), (0, 1, 0)), ((0, grid_lim - pad, 0), (0, -1, 0)),
                        ((0, 0, pad), (0, 0, 1)), ((0, 0, grid_lim - pad), (0, 0, -1))):
            s.add_surface_collider(tuple(map(float, pt)), tuple(map(float, nrm)), "slip")

    writer = DumpWriter(
        s, grain_diameter=1.0e-3, rho_s=MATERIALS[material]["rho"],
        rho_bulk=MATERIALS[material]["rho"], packing_fraction=1.0,
        gravity_inplane=(float(GRAVITY[0]), float(GRAVITY[2])),
        law=MATERIALS[material]["law"],
        law_params=engine_params(material, theta, nclaw_law=nclaw_law),
        theta_true=None, l_convention=L_CONVENTION_STRING, store_F=True,
        xonly=xonly,
        extra={"material": material, "shape": shape, "n_grid": n_grid,
               "grid_lim": grid_lim, "dt": dt, "substeps_per_frame": substeps,
               "gravity": list(GRAVITY),
               "vel_name": vel if cloud is None else "seeded_from_cloud",
               "lin_vel": [float(x) for x in VELS[vel][0]],
               "ang_vel": [float(x) for x in VELS[vel][1]],
               "bound_cells": BOUND_CELLS,
               "seeded_from": None if cloud is None else cloud["source"],
               "collider_bc": "freeslip_grid_op" if kw_bc else "slip",
               "grid_semantics": kw_bc, "cfl_substeps": cfl_substeps,
               "nclaw_law": MATERIALS[material].get("nclaw_law") if nclaw_law else None,
               "recovered": theta is not None})
    t0 = time.time()
    setup_s = t0 - t_start
    step = 0
    t_snap = t_step = 0.0
    for frame in range(n_frames + 1):
        t1 = time.time()
        ok = writer.snapshot(frame * frame_dt)
        t_snap += time.time() - t1
        if not ok:
            log(f"[gen] NaN at frame {frame}; truncating")
            break
        if frame == n_frames:
            break
        t1 = time.time()
        for _ in range(substeps):
            s.p2g2p(step, dt, device=device)
            step += 1
        t_step += time.time() - t1
    t1 = time.time()
    writer.finalize(out_path, frame_dt=frame_dt)
    finalize_s = time.time() - t1
    log(f"[gen] wrote {out_path.name} ({time.time() - t0:.2f}s: "
        f"substeps {t_step:.2e}s, snapshots {t_snap:.2e}s, "
        f"file write {finalize_s:.2e}s)")
    if timing is not None:
        timing.update({"setup_s": setup_s, "step_s": t_step,
                       "snapshot_s": t_snap, "finalize_s": finalize_s,
                       "total_s": time.time() - t_start})
    return out_path


def dump_path(material: str, shape: str, kind: str, vel: str = "preset",
              n_grid: int = N_GRID) -> Path:
    """Grid 32 dumps carry no grid tag; other grids append _g<n> so a grid-20
    truth never collides with a grid-32 dump."""
    tag = "" if vel == "preset" else f"_{vel}"
    gtag = "" if n_grid == N_GRID else f"_g{n_grid}"
    return DUMPS / f"{material}_{shape}{tag}{gtag}_{kind}.npz"


def stage_gen(material: str, shapes: list[str], force: bool = False,
              n_grid: int = N_GRID, vel: str = "preset", log=print) -> dict:
    DUMPS.mkdir(parents=True, exist_ok=True)
    made = {}
    for shape in shapes:
        p = dump_path(material, shape, "truth", vel, n_grid)
        if p.exists() and not force:
            log(f"[gen] skip {p.name} (exists)")
        else:
            run_scene(material, shape, p, theta=None, n_grid=n_grid, vel=vel, log=log)
        made[shape] = str(p)
    return made


# ---------------------------------------------------------------------------
# Scoring: NCLaw's metric
# ---------------------------------------------------------------------------

def nclaw_position_mse(truth: Path, pred: Path, grid_lim: float = GRID_LIM,
                       frame_step: int = 5, strict: bool = True) -> dict:
    """NCLaw's metric: mean over particles AND coordinates of the squared
    position difference, in their unit box, averaged over sampled frames.

    Their eval averages every frame_step-th frame, and their box is 1.0 m, so
    the box normalization is a division by grid_lim^2 that is unity here. Note
    the mean over the three coordinates: summing them instead would report a
    number three times larger.
    """
    t = np.load(truth)
    r = np.load(pred)
    if strict and t["x"].shape != r["x"].shape:
        # a rollout the dump writer truncated at a NaN would otherwise be
        # scored over its successful prefix, which flatters a divergent law;
        # callers that handle divergence themselves pass strict=False
        raise ValueError(
            f"shape mismatch: truth {t['x'].shape} vs prediction "
            f"{r['x'].shape} ({Path(pred).name}); the rollout was truncated "
            "at a NaN. Pass strict=False to score the surviving prefix.")
    nf = min(t["x"].shape[0], r["x"].shape[0])
    n = min(t["x"].shape[1], r["x"].shape[1])
    diff = (t["x"][:nf, :n] - r["x"][:nf, :n]) / grid_lim
    per_frame = (diff ** 2).mean(axis=(1, 2))
    sel = per_frame[::frame_step]
    return {
        "mse": float(sel.mean()),
        "mse_final_frame": float(per_frame[-1]),
        "rmse_mm": float(np.sqrt((diff ** 2).sum(-1).mean()) * 1e3),
        "n_frames": int(nf),
        "n_particles": int(n),
        "per_frame": per_frame.tolist(),
    }


# ---------------------------------------------------------------------------
# Identification, from the CUBE trajectory alone
# ---------------------------------------------------------------------------

SQRT3 = float(np.sqrt(3.0))


def friction_to_mu(phi_deg: float) -> float:
    """Drucker-Prager friction angle -> the mu = sqrt(J2) / p friction coefficient.

    Derivation follows the fork's ``set_parameters_dict`` and
    ``sand_return_mapping``. ``set_parameters_dict`` stores
    alpha = sqrt(2/3) * 2 sin phi / (3 - sin phi), and ``sand_return_mapping``
    yields when

        ||dev eps|| + (3 lam + 2 mu) / (2 mu) * tr eps * alpha > 0.

    Hencky elasticity gives ||dev tau|| = 2 mu ||dev eps|| and
    tr tau = (3 lam + 2 mu) tr eps = -3 p, so the yield surface is
    ||dev tau|| = 3 alpha p. This repository measures friction as
    mu = sqrt(J2) / p with sqrt(J2) = ||dev tau|| / sqrt(2), hence

        mu = 3 alpha / sqrt(2) = 2 sqrt(3) sin phi / (3 - sin phi),

    which is the textbook compression-cone Drucker-Prager constant. At 25
    degrees this is 0.5679.
    """
    s = float(np.sin(np.deg2rad(phi_deg)))
    return 2.0 * SQRT3 * s / (3.0 - s)


def mu_to_friction(mu_c: float) -> float:
    """Inverse of friction_to_mu: sin phi = 3 mu / (2 sqrt(3) + mu)."""
    s = 3.0 * mu_c / (2.0 * SQRT3 + mu_c)
    return float(np.rad2deg(np.arcsin(np.clip(s, -1.0, 1.0))))


def _load_arrays(path: Path) -> dict:
    """Positions, velocities, F, volumes and the run's grid config from a dump."""
    from ident.io.schema import validate_dump_schema
    # schema rule: read keys through validate_dump_schema only
    meta = validate_dump_schema(path)
    d = np.load(path)
    T, P = d["x"].shape[0], d["x"].shape[1]
    out = {
        "meta": meta,
        "x": d["x"].astype(np.float64),
        "v": d["v"].astype(np.float64),
        "L": d["L"].astype(np.float64).reshape(T, P, 3, 3),
        "stress": d["stress"].astype(np.float64).reshape(T, P, 3, 3),
        "volume": d["volume"].astype(np.float64),
        "mass": d["mass"].astype(np.float64),
        "frame_dt": float(d["frame_dt"]),
        "n_grid": int(meta.extra["n_grid"]),
        "grid_lim": float(meta.extra["grid_lim"]),
        "g": np.asarray(meta.extra["gravity"], dtype=float),
    }
    if "F" in d.files:
        out["F"] = d["F"].astype(np.float64).reshape(T, P, 3, 3)
        out["vol0"] = d["volume0"].astype(np.float64)
    return out


def wall_planes(n_grid: int, grid_lim: float) -> list:
    """The six freeslip planes of the NCLaw box, as (point, inward normal)."""
    pad = BOUND_CELLS * (grid_lim / n_grid)
    return [((pad, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((grid_lim - pad, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            ((0.0, pad, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, grid_lim - pad, 0.0), (0.0, -1.0, 0.0)),
            ((0.0, 0.0, pad), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, grid_lim - pad), (0.0, 0.0, -1.0))]


def _longest_run(frames: list[int], counts: list[int], min_count: int) -> list[int]:
    """Longest contiguous stretch of the frame list whose count clears min_count."""
    best: list[int] = []
    cur: list[int] = []
    for f, c in zip(frames, counts, strict=True):
        if c >= min_count:
            cur.append(f)
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    return best


def _hencky_dev_norm(F: np.ndarray) -> np.ndarray:
    """||dev log sigma(F)||, the rotation-invariant deviatoric strain measure."""
    sig = np.linalg.svd(F, compute_uv=False)
    eps = np.log(np.clip(np.abs(sig), 1e-12, None))
    return np.linalg.norm(eps - eps.mean(axis=-1, keepdims=True), axis=-1)


def identify_elastic(arr: dict, window_frames: int = 26, frame_stride: int = 2,
                     margin_cells: float = 3.0,
                     frames: list[int] | None = None,
                     columns: str = "corotated", log=print) -> dict:
    """(mu, lambda) of the fixed-corotated pair by the Step 0 time-weak assembly.

    ``frames`` restricts the fit to an explicit uniformly spaced frame list.
    The positions-only plasticine path uses it for the pre-yield window, where
    total and elastic deformation still coincide and the tier's MLS F is
    therefore the right state to read.
    """
    from ident.weakform.elastic_grid import (
        assemble_elastic_timeweak,
        moduli_to_E_nu,
        solve_elastic_grid,
    )
    if frames is None:
        frames = list(range(0, arr["x"].shape[0], frame_stride))
    else:
        frames = [int(f) for f in frames]
        spacing = np.diff(frames)
        if spacing.size and not np.all(spacing == spacing[0]):
            raise ValueError("frames must be uniformly spaced; the temporal "
                             "weight assumes a constant frame spacing")
        frame_stride = int(spacing[0]) if spacing.size else 1
    t_asm = time.time()
    sysm = assemble_elastic_timeweak(
        arr["x"], arr["F"], arr["v"], arr["vol0"], arr["mass"], arr["g"],
        arr["frame_dt"] * frame_stride, arr["n_grid"], arr["grid_lim"],
        frames=frames, window_frames=window_frames,
        collider_planes=wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells, columns=columns)
    wall_assemble_s = time.time() - t_asm
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": 0.0}
    t_fit = time.time()
    out = solve_elastic_grid(sysm)
    wall_fit_s = time.time() - t_fit
    E, nu = moduli_to_E_nu(out["mu"], out["lam"])
    out.update({"E": E, "nu": nu, "refused": False, "estimator": "weak_form",
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "row_survival": sysm.row_survival,
                "strain_coverage": list(sysm.strain_coverage),
                "n_frames_used": len(sysm.frames_used),
                "window_frames": window_frames, "frame_stride": frame_stride,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": wall_fit_s})
    log(f"[ident] elastic mu={out['mu']:.4e} lam={out['lam']:.4e} "
        f"E={E:.4e} nu={nu:.4f} rows={sysm.n_rows} cond={out['cond_AtA']:.2e}")
    return out


def identify_yield(arr: dict, mu_hat: float, plateau_pct: float = 99.9,
                   plateau_frac_min: float = 0.05, log=print) -> dict:
    """von Mises yield from the saturation of the deviatoric Hencky strain.

    The return map caps ||dev eps|| at yield / (2 mu), so the cap IS the yield
    stress divided by twice the shear modulus, and the yield stress follows from
    the mu the elastic solve already recovered. Identifiable only if particles
    reached the cap. Sub-yield loading gives a lower bound, so the fit refuses.
    """
    t_fit = time.time()
    T = arr["F"].shape[0]
    vals = np.concatenate([_hencky_dev_norm(arr["F"][f]) for f in range(0, T, 4)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"refused": True, "reason": "no finite deformation gradient",
                "wall_assemble_s": 0.0, "wall_fit_s": time.time() - t_fit}
    cap = float(np.percentile(vals, plateau_pct))
    at_cap = float((vals >= 0.98 * cap).mean())
    # The return map clips every yielded particle's strain norm to
    # tau_y / (2 mu), so yielded samples pile up at one value and the
    # histogram shows a spike there. Elastic data falls off smoothly
    # and has no spike. The refusal condition looks for the spike.
    below = float(((vals >= 0.94 * cap) & (vals < 0.96 * cap)).mean())
    concentration = at_cap / max(below, 1e-6)
    res = {"eps_y": cap, "plateau_fraction": at_cap,
           "plateau_concentration": concentration,
           "yield_stress": float(2.0 * mu_hat * cap),
           "plateau_pct": plateau_pct, "estimator": "closed_form_strain_cap",
           # no matrix assembly here: a closed-form statistic of the stored F
           "wall_assemble_s": 0.0, "wall_fit_s": time.time() - t_fit}
    if at_cap < plateau_frac_min or concentration < 3.0:
        res.update({"refused": True,
                    "reason": ("strain cap not reached: "
                               f"{100 * at_cap:.2f} percent of particle-frames "
                               f"in the top band, concentration {concentration:.2f} "
                               f"(needs >= {100 * plateau_frac_min:g} percent "
                               "at >= 3x); the estimate is a lower bound")})
    else:
        res["refused"] = False
    log(f"[ident] yield eps_y={cap:.4e} plateau_frac={at_cap:.2e} "
        f"yield={res['yield_stress']:.4e} refused={res['refused']}")
    return res


def identify_friction(arr: dict, window_frames: int = 26, frame_stride: int = 2,
                      margin_cells: float = 3.0, eps_gamma: float = 0.02,
                      gd_min: float = 1.0, yield_frac_min: float = 0.97,
                      yield_band: float = 0.05,
                      pressure: np.ndarray | None = None,
                      dev_stress: np.ndarray | None = None,
                      pressure_label: str = "stress_trace", log=print) -> dict:
    """Constant-friction Mode C in 3D: one column, mu = sqrt(J2) / p.

    sigma = -p I + mu * p * (2 D / |gamma_dot|_eps) is linear in mu with the
    pressure as DATA (the 3D stress trace, the oracle pressure), so the
    pressure term goes into the known part of the load and the single column is
    V p (2 D / |gamma_dot|_eps). Only particles that are shearing under positive
    pressure enter, since the yield relation holds at yield and not below it;
    yield_frac_min then sets how much of a node's support mass must come from
    them, the same node check ``grid_assembly.py`` applies (flow_frac_min
    there). Requiring all of it drops every node, because roughly
    a quarter of the particles sit at or below zero pressure on the free surface
    at any instant.

    The flow direction comes from D. The return map scales the deviatoric
    elastic strain, so this rate-form reading is exact only where the two are
    coaxial. It is the same
    reading the mu(I) legs of this tree use, and the recovered coefficient is
    therefore an effective friction.

    ``pressure`` supplies the pressure field (frames by particles, Cauchy
    pressure in Pa) from a stated model instead of the stress trace, which is
    what a run without the stress channel needs; ``pressure_label`` names the
    model in the result. ``dev_stress`` supplies the matching deviatoric Cauchy
    stress, which the yield-set check and the cone-level estimator need. With
    ``pressure`` given and ``dev_stress`` left out, the yield set is selected on
    kinematics alone (shearing under positive pressure), the cone-level
    estimator is unavailable, and a solve whose residual exceeds the bar refuses
    rather than falling back.
    """
    from common.conventions import (
        equivalent_shear_rate,
        pressure_from_cauchy_3d_trace,
        sym,
    )
    from ident.weakform.elastic_grid import assemble_columns_timeweak, solve_elastic_grid

    # the pressure is DATA to this leg, read from the 3D stress trace unless the
    # caller states a model. With no stress channel and no stated model, refuse
    # and name the missing channel.
    if pressure is None and not arr["meta"].has_pressure:
        return {"refused": True, "n_rows": 0, "n_rows_before_gating": 0,
                "reason": ("no oracle pressure: the dump's pressure_source is "
                           f"{arr['meta'].pressure_source!r}, so the stress trace this "
                           "leg needs is absent. Supply a stress channel or state a "
                           "pressure closure."),
                "pressure_source": arr["meta"].pressure_source}

    D = sym(arr["L"])
    gd = equivalent_shear_rate(D, eps_gamma)
    eye = np.eye(3)[None]
    if pressure is None:
        p = pressure_from_cauchy_3d_trace(arr["stress"])
        dev_stress = arr["stress"] - (np.trace(arr["stress"], axis1=2, axis2=3) / 3.0
                                      )[:, :, None, None] * eye
        pressure_label = "stress_trace"
    else:
        p = np.asarray(pressure, dtype=float)
        if p.shape != arr["x"].shape[:2]:
            raise ValueError(f"pressure has shape {p.shape}, expected "
                             f"{arr['x'].shape[:2]} (frames by particles)")

    # Yield-set check. The cone relation holds at yield only; sub-yield
    # shearing particles bias the fit high. Stage 1 reads the cone level as the
    # mode of r over shearing high-pressure particles; stage 2 keeps particles
    # within yield_band of it.
    if dev_stress is None:
        # no deviatoric stress at all: the cone level is not observable, so the
        # yield set is selected on kinematics alone and there is no cone level
        # to fall back on when the solve's residual is high.
        r_cone = None
        mu_plateau = None
    else:
        dev = np.asarray(dev_stress, dtype=float)
        j2 = 0.5 * np.sum(dev * dev, axis=(2, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cone = np.sqrt(j2) / p
        cand = np.isfinite(r_cone) & (p > np.nanpercentile(p[p > 0], 25)) & (gd > gd_min)
        r_pool = r_cone[cand]
        hist, edges = np.histogram(r_pool, bins=256,
                                   range=(0.0, float(np.nanpercentile(r_pool, 99.5))))
        mu_plateau = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))

    def columns_fn(f: int):
        finite = np.isfinite(p[f]) & np.isfinite(D[f]).all(axis=(1, 2))
        on_cone = (True if r_cone is None else
                   np.abs(r_cone[f] / max(mu_plateau, 1e-9) - 1.0) < yield_band)
        at_yield = finite & (p[f] > 0.0) & (gd[f] > gd_min) & on_cone
        # A cohesionless particle at or below zero pressure is stress free
        # (sand_return_mapping sets F_elastic = U V^T), so it stays valid with
        # zero contribution. Only a positive-pressure sub-yield particle
        # invalidates its nodes.
        free = finite & (p[f] <= 0.0)
        ok = at_yield | free
        if at_yield.sum() < 50:
            return None
        Vp = arr["volume"][f]
        gsafe = np.where(gd[f] > 0.0, gd[f], 1.0)
        flow = 2.0 * D[f] / gsafe[:, None, None]
        w = np.where(at_yield, Vp * p[f], 0.0)
        Vsig = w[:, None, None, None] * flow[:, None, :, :]
        Vsig_known = -w[:, None, None] * eye
        return Vsig, Vsig_known, ok, gd[f]

    # a time-weak row sums a whole window, so one unusable frame kills it. The
    # frame list is the longest contiguous run of shearing frames; the early
    # rigid-rotation frames drop out. The run must stay uniformly spaced: the
    # temporal weight assumes a constant frame spacing.
    #
    # Keep frames after the kinetic energy peaks and decays below a fraction of
    # the peak; the impact frames are collisional and off-model. Try fractions
    # 0.1, 0.2, 0.5 until the window fits.
    ke = 0.5 * np.einsum("p,fpi->f", arr["mass"], arr["v"] ** 2)
    k_peak = int(np.argmax(ke))
    frames, ke_frac_used = [], None
    for frac in (0.1, 0.2, 0.5):
        decayed = np.flatnonzero(ke <= frac * ke[k_peak])
        k_start = int(decayed[decayed > k_peak][0]) if np.any(decayed > k_peak) else k_peak
        all_frames = [f for f in range(0, arr["x"].shape[0], frame_stride) if f >= k_start]
        counts = [int(((gd[f] > gd_min) & (p[f] > 0.0)).sum()) for f in all_frames]
        frames = _longest_run(all_frames, counts, 50)
        if len(frames) >= window_frames:
            ke_frac_used = frac
            break

    if len(frames) < window_frames:
        return {"refused": True,
                "reason": (f"only {len(frames)} contiguous shearing frames, "
                           f"fewer than the {window_frames}-frame window"),
                "n_rows": 0, "n_rows_before_gating": 0,
                "wall_assemble_s": 0.0, "wall_fit_s": 0.0}
    t_asm = time.time()
    sysm = assemble_columns_timeweak(
        arr["x"], arr["v"], arr["mass"], arr["g"],
        arr["frame_dt"] * frame_stride, arr["n_grid"], arr["grid_lim"],
        columns_fn, n_columns=1, frames=frames, window_frames=window_frames,
        collider_planes=wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells, valid_frac_min=yield_frac_min)
    wall_assemble_s = time.time() - t_asm
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows, "yield_frac_min": yield_frac_min,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": 0.0}
    t_fit = time.time()
    out = solve_elastic_grid(sysm)
    wall_fit_s = time.time() - t_fit
    out["wall_assemble_s"], out["wall_fit_s"] = wall_assemble_s, wall_fit_s
    mu_c = out["theta"][0]
    # When the solve's relative residual exceeds solve_residual_bar and a cone
    # level exists, the cone level is the estimate. Both numbers are recorded.
    solve_residual_bar = 0.15
    mu_c_solve = float(mu_c)
    used = "solve"
    high_residual = float(out.get("residual_rel", 0.0)) > solve_residual_bar
    if high_residual and mu_plateau is not None:
        mu_c = mu_plateau
        used = "plateau"
    out.update({"ke_frac_used": ke_frac_used,
                "mu_c_solve": mu_c_solve,
                "friction_angle_solve": mu_to_friction(mu_c_solve),
                "mu_estimator_used": used,
                "solve_residual_bar": solve_residual_bar,
                "pressure_source": pressure_label,
                "mu_plateau_stage1": mu_plateau,
                "friction_angle_stage1": (None if mu_plateau is None
                                          else mu_to_friction(mu_plateau)),
                "yield_band": float(yield_band),
                "mu_c": mu_c, "friction_angle": mu_to_friction(mu_c),
                "estimator": "weak_form" if used == "solve" else "cone_level_reading",
                "refused": False, "row_survival": sysm.row_survival,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "yield_frac_min": yield_frac_min, "gd_min": gd_min,
                "n_shearing_frames": len(frames),
                "shear_rate_coverage": list(sysm.strain_coverage)})
    if high_residual and mu_plateau is None:
        # the kinematic check did not remove the bias and there is no cone
        # level to read.
        out.update({"refused": True,
                    "reason": (f"solve residual {out['residual_rel']:.3f} exceeds "
                               f"{solve_residual_bar} and no deviatoric stress is "
                               f"available to read the cone level from, with pressure "
                               f"from {pressure_label}")})
    log(f"[ident] friction mu_c={mu_c:.4f} -> phi={out['friction_angle']:.2f} deg "
        f"rows={sysm.n_rows} cond={out['cond_AtA']:.2e} "
        f"pressure={pressure_label} refused={out['refused']}")
    return out


def identify_eos(arr: dict, gamma: float = 1.1, window_frames: int = 26,
                 frame_stride: int = 2, margin_cells: float = 3.0,
                 cond_max: float = 1.0e12, form: str = "power_law",
                 log=print) -> dict:
    """Stiffness of a volumetric equation of state from the volumetric weak form.

    Two forms, one linear unknown each, so the solve stays the same convex
    least-squares problem:

    power_law
        our fluid material, ``kirchoff_stress_water``: Cauchy =
        -bulk (J^-gamma - 1) I, column -(J^-gamma - 1) I, unknown the bulk
        modulus.
    linear
        NCLaw's VolumeElasticity in mode taichi: Cauchy = lam (J - 1) I, column
        (J - 1) I, unknown lam. Their water was generated with this one, and the
        two forms are not reparameterizations of each other at their splash's
        compression (J down to 0.039).

    Water is a hard case for identification either way: the compression the
    throw produces is small, so the column can be starved. The conditioning and
    the realized volumetric strain are reported and a starved solve refuses.
    """
    if form not in ("power_law", "linear"):
        raise ValueError(f"unknown EOS form {form!r}")
    from ident.weakform.elastic_grid import assemble_columns_timeweak, solve_elastic_grid

    eye = np.eye(3)[None]
    Jall = []

    def columns_fn(f: int):
        Vp = arr["volume"][f]
        J = Vp / np.maximum(arr["vol0"], 1e-30)
        ok = np.isfinite(J) & (J > 1e-6)
        if ok.sum() < 50:
            return None
        Jall.append(J[ok])
        col = (J - 1.0 if form == "linear"
               else -(np.power(np.clip(J, 1e-6, None), -gamma) - 1.0))
        Vsig = (Vp * col)[:, None, None, None] * eye[:, None, :, :]
        return Vsig, None, ok, np.abs(J - 1.0)

    frames = list(range(0, arr["x"].shape[0], frame_stride))
    t_asm = time.time()
    sysm = assemble_columns_timeweak(
        arr["x"], arr["v"], arr["mass"], arr["g"],
        arr["frame_dt"] * frame_stride, arr["n_grid"], arr["grid_lim"],
        columns_fn, n_columns=1, frames=frames, window_frames=window_frames,
        collider_planes=wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells)
    wall_assemble_s = time.time() - t_asm
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": 0.0}
    t_fit = time.time()
    out = solve_elastic_grid(sysm)
    out["wall_assemble_s"], out["wall_fit_s"] = wall_assemble_s, time.time() - t_fit
    stiffness = out["theta"][0]
    Jc = np.concatenate(Jall) if Jall else np.ones(1)
    key = "lam" if form == "linear" else "bulk_modulus"
    out.update({key: stiffness, "form": form, "gamma": gamma, "refused": False,
                "estimator": "weak_form",
                "row_survival": sysm.row_survival,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "volumetric_strain_p99": float(np.percentile(np.abs(Jc - 1.0), 99))})
    if not np.isfinite(stiffness) or stiffness <= 0.0 or out["cond_AtA"] > cond_max:
        out.update({"refused": True,
                    "reason": ("too little volumetric strain to identify: "
                               f"{key}={stiffness:.3e}, cond={out['cond_AtA']:.3e}")})
    log(f"[ident] eos {form} {key}={stiffness:.4e} rows={sysm.n_rows} "
        f"cond={out['cond_AtA']:.2e} refused={out['refused']}")
    return out


def lam_to_E(lam: float, nu: float) -> float:
    """E from lam at fixed nu, the inverse of lam = E nu / ((1+nu)(1-2nu)).

    NCLaw's volumetric water elasticity reads lam and nothing else, so nu is
    bookkeeping there: the rollout depends on the recovered lam alone, and nu is
    carried at their config value so the engine's (E, nu) arguments reproduce it.
    """
    return float(lam * (1.0 + nu) * (1.0 - 2.0 * nu) / nu)


def theta_for_engine(material: str, ident: dict, nclaw_law: bool = False,
                     ) -> tuple[dict, list[str], dict[str, str]]:
    """Recovered parameters in the engine's own arguments, plus what was refused.

    A refused parameter falls back to its known-class prior value, which is the
    truth entry here, and the returned list names the fallback for the report.

    The third return value labels, per parameter key, how it was produced:
    ``"assumed"`` (never attempted, or attempted and refused -- either way the
    value is the truth prior, not an estimate), ``"weak_form"`` (the convex
    momentum-balance solve), or whatever the estimator names itself
    (``"closed_form_strain_cap"`` for plasticine's yield plateau read,
    ``"rollout_scan"`` for the derivative-free scan, ``"replay_yield"`` /
    ``"replay_friction"`` for the positions-only replay attempt -- the last
    two are only reachable if a caller substitutes a variant's theta for the
    primary one; the primary path here only ever assigns "assumed" or
    "weak_form"/"closed_form_strain_cap"/"rollout_scan").
    """
    from ident.weakform.elastic_grid import moduli_to_E_nu
    truth = MATERIALS[material]["truth"]
    theta: dict = {}
    refused: list[str] = []
    estimator: dict[str, str] = {}
    if material in ("jelly", "plasticine"):
        el = ident.get("elastic", {})
        if el.get("refused", True):
            refused += ["E", "nu"]
            theta.update({"E": truth["E"], "nu": truth["nu"]})
            estimator["E"] = estimator["nu"] = "assumed"
        else:
            E, nu = moduli_to_E_nu(el["mu"], el["lam"])
            theta.update({"E": E, "nu": nu})
            estimator["E"] = estimator["nu"] = el.get("estimator", "weak_form")
        if material == "plasticine":
            y = ident.get("yield", {})
            if y.get("refused", True):
                refused.append("yield_stress")
                theta["yield_stress"] = truth["yield_stress"]
                estimator["yield_stress"] = "assumed"
            else:
                theta["yield_stress"] = y["yield_stress"]
                estimator["yield_stress"] = y.get("estimator", "closed_form_strain_cap")
    elif material == "sand":
        fr = ident.get("friction", {})
        if fr.get("refused", True):
            refused.append("friction_angle")
            theta["friction_angle"] = truth["friction_angle"]
            estimator["friction_angle"] = "assumed"
        else:
            theta["friction_angle"] = fr["friction_angle"]
            estimator["friction_angle"] = fr.get("estimator", "weak_form")
        # the elastic pair below yield is not excited by this throw; prior-fixed
        refused += ["E", "nu"]
        theta.update({"E": truth["E"], "nu": truth["nu"]})
        estimator["E"] = estimator["nu"] = "assumed"
    elif material == "water":
        eos = ident.get("eos", {})
        if nclaw_law:
            # their linear volumetric EOS: one unknown, lam, carried into the
            # engine as (E, nu) at their nu
            nu = truth["nu"]
            if eos.get("refused", True):
                refused.append("lam")
                theta.update({"E": truth["E"], "nu": nu})
                estimator["E"] = "assumed"
            else:
                theta.update({"E": lam_to_E(eos["lam"], nu), "nu": nu})
                estimator["E"] = eos.get("estimator", "weak_form")
        else:
            if eos.get("refused", True):
                refused.append("bulk_modulus")
                theta["bulk_modulus"] = bulk_from_E_nu(truth["E"], truth["nu"])
                estimator["bulk_modulus"] = "assumed"
            else:
                theta["bulk_modulus"] = eos["bulk_modulus"]
                estimator["bulk_modulus"] = eos.get("estimator", "weak_form")
            theta.update({"E": truth["E"], "nu": truth["nu"]})
            estimator.setdefault("E", "assumed")
        estimator["nu"] = "assumed"
    return theta, refused, estimator


# Time-weak window length, in sampled frames, per material. Longer is better
# for the elastic pair, whose remaining error is the temporal quadrature; sand
# wants a short window because only the frames after the throw starts shearing
# are usable and a row must fit entirely inside them.
WINDOW_FRAMES: dict[str, int] = {"jelly": 26, "plasticine": 26,
                                 "sand": 10, "water": 16}


def stage_identify(material: str, n_grid: int = N_GRID,
                   window_frames: int | None = None, dump: str | Path | None = None,
                   tag: str | None = None, nclaw_law: bool = False,
                   log=print) -> dict:
    """Recover the law from ONE trajectory: the cube throw by default.

    ``dump`` points the same legs at another schema-valid trajectory, which is
    how an ingested NCLaw folder is identified from; ``tag`` names the results
    file so an ingested run does not overwrite the suite's own.
    """
    if window_frames is None:
        window_frames = WINDOW_FRAMES[material]
    cube = Path(dump) if dump is not None else dump_path(material, "cube", "truth")
    if not cube.exists():
        raise SystemExit(f"missing {cube}; run the gen stage first")
    t0 = time.time()
    arr = _load_arrays(cube)
    ident: dict = {"source_dump": cube.name, "n_grid": arr["n_grid"]}
    if material in ("jelly", "plasticine"):
        ident["elastic"] = identify_elastic(
            arr, window_frames=window_frames,
            columns="hencky" if material == "plasticine" else "corotated", log=log)
        if material == "plasticine" and not ident["elastic"].get("refused", True):
            ident["yield"] = identify_yield(arr, ident["elastic"]["mu"], log=log)
    elif material == "sand":
        ident["friction"] = identify_friction(arr, window_frames=window_frames, log=log)
    elif material == "water":
        # their water is the linear volumetric EOS; ours is the power law
        ident["eos"] = identify_eos(
            arr, window_frames=window_frames,
            form="linear" if nclaw_law else "power_law", log=log)
    identify_total_s = time.time() - t0
    # Full tier: every channel (x, v, F, stress) is measured directly, so
    # there is no field reconstruction step; the assemble/fit split comes
    # straight from whichever identify_* function(s) ran above.
    sub_dicts = [d for d in (ident.get("elastic"), ident.get("yield"),
                             ident.get("friction"), ident.get("eos")) if d]
    ident["timing_breakdown_s"] = {
        "reconstruct_fields_s": 0.0,
        "assemble_equations_s": sum(d.get("wall_assemble_s", 0.0) for d in sub_dicts),
        "fit_parameters_s": sum(d.get("wall_fit_s", 0.0) for d in sub_dicts),
        "simulate_s": 0.0,
        "identify_total_s": identify_total_s,
    }
    theta, refused, param_estimator = theta_for_engine(
        material, ident, nclaw_law=nclaw_law)
    ident["theta_engine"] = theta
    ident["refused_parameters"] = refused
    ident["parameter_estimator"] = param_estimator
    ident["truth"] = MATERIALS[material]["truth"]
    ident["nclaw_law"] = MATERIALS[material].get("nclaw_law") if nclaw_law else None
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"identify_{material}.json" if tag is None else f"identify_{material}_{tag}.json"
    (OUT / name).write_text(json.dumps(ident, indent=2, default=float))
    log(f"[ident] theta={theta} refused={refused} ({identify_total_s:.2f}s)")
    return ident


def stage_rollout(material: str, shapes: list[str], force: bool = False,
                  n_grid: int = N_GRID, vel: str = "preset", log=print) -> dict:
    """Re-simulate every scene at the recovered law and score NCLaw's metric."""
    ipath = OUT / f"identify_{material}.json"
    if not ipath.exists():
        raise SystemExit(f"missing {ipath}; run the identify stage first")
    ident = json.loads(ipath.read_text())
    theta = ident["theta_engine"]
    rpath_prev = OUT / f"rollout_{material}.json"
    scores: dict = json.loads(rpath_prev.read_text()) if rpath_prev.exists() else {}
    for shape in shapes:
        truth = dump_path(material, shape, "truth", vel, n_grid)
        if not truth.exists():
            log(f"[rollout] skip {shape}: no truth dump")
            continue
        pred = dump_path(material, shape, "rec", vel, n_grid)
        if not pred.exists() or force:
            run_scene(material, shape, pred, theta=theta, n_grid=n_grid, vel=vel, log=log)
        s = nclaw_position_mse(truth, pred)
        s["role"] = "reconstruction" if shape == "cube" else "generalization"
        s["vel"] = vel
        scores[shape if vel == "preset" else f"{shape}@{vel}"] = s
        log(f"[rollout] {material}/{shape} ({s['role']}) N={s['n_particles']} "
            f"MSE={s['mse']:.3e}  final-frame {s['mse_final_frame']:.3e}  "
            f"RMS {s['rmse_mm']:.2f} mm")
    (OUT / f"rollout_{material}.json").write_text(
        json.dumps(scores, indent=2, default=float))
    return scores


# ---------------------------------------------------------------------------
# Cross-engine chain: their trajectory in, our rollout out
# ---------------------------------------------------------------------------

CROSS_CAVEAT = (
    "Cross-engine cell. The truth trajectory is NCLaw's own simulator output, "
    "ingested through experiments/nclaw/ingest.py; the law is identified from "
    "that trajectory by our convex weak form; the rollout is our engine seeded "
    "from their frame-0 cloud and frame-0 velocities, so the two trajectories "
    "are in 1:1 particle correspondence. The MSE therefore carries both the "
    "identification error and the difference between the two integrators, and "
    "it is not comparable to the same-engine cells of the suite."
)


def stage_cross(material: str, nclaw_dir: str | Path, manifest: str | Path | dict | None,
                name: str | None = None, force: bool = False,
                window_frames: int | None = None, log=print) -> dict:
    """One command per material once their data exists.

    ingest their folder -> identify from the ingested trajectory -> roll out our
    engine from their frame-0 cloud -> score their trajectory against our
    rollout in their own metric.
    """
    from experiments.nclaw.ingest import read_nclaw_dir
    DUMPS.mkdir(parents=True, exist_ok=True)
    tag = name or Path(nclaw_dir).name
    truth = DUMPS / f"{material}_{tag}_nclaw_truth.npz"
    if not truth.exists() or force:
        res = read_nclaw_dir(nclaw_dir, manifest, truth, log=log)
        provenance, probe = res.provenance, res.meta["l_convention_probe"]
    else:
        log(f"[cross] reuse {truth.name} (exists)")
        meta = _load_arrays(truth)["meta"]
        provenance = meta.extra.get("channel_provenance", {})
        probe = meta.extra.get("l_convention_probe", {})

    ident = stage_identify(material, dump=truth, tag=f"nclaw_{tag}",
                           window_frames=window_frames, log=log)
    pred = DUMPS / f"{material}_{tag}_nclaw_rec.npz"
    cloud = cloud_from_dump(truth)
    if not pred.exists() or force:
        run_scene(material, tag, pred, theta=ident["theta_engine"], cloud=cloud, log=log)
    score = nclaw_position_mse(truth, pred)
    score["role"] = "cross_engine"

    # the engine-gap control: the TRUTH parameters through our engine against
    # their trajectory. This row is pure engine difference; the recovered-theta
    # row above is engine difference plus identification error, so their ratio
    # says how much of the cross-engine cell any identifier could ever remove.
    pred_t = DUMPS / f"{material}_{tag}_nclaw_truththeta.npz"
    if not pred_t.exists() or force:
        run_scene(material, tag, pred_t, theta=dict(MATERIALS[material]["truth"]),
                  cloud=cloud, log=log)
    score_t = nclaw_position_mse(truth, pred_t)
    score_t["role"] = "engine_gap_baseline"
    out = {
        "schema_version": "nclaw-cross-1.0",
        "material": material, "scene": tag,
        "nclaw_dir": str(nclaw_dir),
        "ingested_dump": truth.name, "rollout_dump": pred.name,
        "git_rev_mpm_engine": _git_rev(ROOT),
        "channel_provenance": provenance,
        "l_convention_probe": probe,
        "theta_engine": ident["theta_engine"],
        "refused_parameters": ident.get("refused_parameters", []),
        "truth_parameters_for_reference": MATERIALS[material]["truth"],
        "n_grid": cloud["n_grid"], "grid_lim": cloud["grid_lim"],
        "n_particles": cloud["pts"].shape[0], "t_end": cloud["t_end"],
        "mse": {k: score[k] for k in
                ("mse", "mse_final_frame", "rmse_mm", "n_frames", "n_particles")},
        "engine_gap_baseline": {k: score_t[k] for k in
                             ("mse", "mse_final_frame", "rmse_mm")},
        "identification_excess_over_baseline": (
            float(score["mse"] / score_t["mse"]) if score_t["mse"] > 0 else None),
        "nclaw_published": NCLAW_PUBLISHED[material],
        "caveat": CROSS_CAVEAT,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cross_{material}_{tag}.json").write_text(
        json.dumps(out, indent=2, default=float))
    log(f"[cross] {material}/{tag} theta={ident['theta_engine']} "
        f"MSE={score['mse']:.3e} (RMS {score['rmse_mm']:.2f} mm over "
        f"{score['n_frames']} frames, {score['n_particles']} particles); "
        f"engine-gap baseline at truth theta {score_t['mse']:.3e} "
        f"(identification excess {score['mse'] / max(score_t['mse'], 1e-300):.2f}x)")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# NCLaw's published position MSE, their own method's row ("ours"), in squared
# metres on a 1.0 m box. Reconstruction is Table 1 (page 5); the three
# generalization axes are Table 2 (page 7), tasks (a) longer horizon,
# (b) initial velocity, (c) geometry. Their per-table Overall columns do not
# recompute from the cells, so only the per-cell values are used here. A cell
# left as None is not published for the matched case and prints blank.
# ENGINE_CAVEAT accompanies the table in every report.
NCLAW_PUBLISHED: dict[str, dict[str, float | None]] = {
    "jelly": {"reconstruction": 2.4e-4, "generalization": 4.1e-4,
              "time": 9.8e-4, "velocity": 2.4e-4},
    "plasticine": {"reconstruction": 6.5e-5, "generalization": 2.3e-4,
                   "time": 1.4e-4, "velocity": 4.6e-5},
    "sand": {"reconstruction": 2.6e-5, "generalization": 3.6e-4,
             "time": 4.2e-5, "velocity": 6.5e-5},
    "water": {"reconstruction": 2.0e-5, "generalization": 2.4e-4,
              "time": 3.5e-4, "velocity": 1.9e-5},
}
# Strongest published baseline, for context on the reconstruction column: their
# "neural" row beats them on jelly reconstruction at 1.2e-5. Their labelled and
# system-identification oracles sit below the horizontal divider in their
# tables and are excluded from their own shading, so they are not comparison
# targets here.
NCLAW_GEOMETRY_CONFOUND = (
    "NCLaw's geometry column, task (c), changes four things at once against "
    "their reconstruction column: the mesh, the particle count (about 30k), the "
    "grid (20 to 32) and the horizon (1000 to 2000 steps), and it also switches "
    "the throw from vel/preset to vel/mild, linear [1.0, -1.5, -1.5] and angular "
    "[1, 1, 1]. Our generalization column changes the mesh alone, at one grid, "
    "one horizon and the preset throw, so it isolates geometry where theirs does "
    "not."
)
ENGINE_CAVEAT = (
    "NCLaw's column is their published number from their own MPM engine, grid, "
    "particle count and time step. Ours is our engine's truth against our "
    "engine's rollout on identical particle clouds. The columns are two "
    "self-consistent measurements of the same protocol on two engines."
)


def stage_report(material: str, log=print) -> dict:
    """results.json, report.md and the comparison figure for one material."""
    ipath = OUT / f"identify_{material}.json"
    rpath = OUT / f"rollout_{material}.json"
    if not ipath.exists():
        raise SystemExit(f"missing {ipath}; run the identify stage first")
    ident = json.loads(ipath.read_text())
    scores = json.loads(rpath.read_text()) if rpath.exists() else {}
    truth = MATERIALS[material]["truth"]

    recovered_vs_truth = {}
    for k, v in ident["theta_engine"].items():
        t = truth.get(k)
        if k == "bulk_modulus" and t is None:
            t = bulk_from_E_nu(truth["E"], truth["nu"])
        recovered_vs_truth[k] = {
            "recovered": v, "truth": t,
            "rel_err": (abs(v / t - 1.0) if t not in (None, 0.0) else None),
            "prior_fixed": k in ident.get("refused_parameters", []),
        }

    diag = {}
    for key in ("elastic", "yield", "friction", "eos"):
        if key in ident:
            d = ident[key]
            diag[key] = {kk: d.get(kk) for kk in
                         ("n_rows", "n_rows_before_gating", "row_survival",
                          "cond_AtA", "cond_AtA_scaled", "residual_rel",
                          "effective_rank", "strain_coverage",
                          "shear_rate_coverage", "volumetric_strain_p99",
                          "plateau_fraction", "refused", "reason")
                         if kk in d}

    results = {
        "schema_version": "nclaw-suite-1.0",
        "material": material,
        "git_rev_mpm_engine": _git_rev(ROOT),
        "git_rev_staging_tree": _git_rev(ROOT.parent),
        "protocol": {
            "grid_lim": GRID_LIM, "n_grid": ident.get("n_grid", N_GRID),
            "bound_cells": BOUND_CELLS, "collider_bc": "slip (freeslip)",
            "t_end": T_END, "n_frames": N_FRAMES,
            "gravity": list(GRAVITY), "lin_vel": list(LIN_VEL),
            "ang_vel": list(ANG_VEL),
            "rotation_yup_to_zup": R_YUP_TO_ZUP.tolist(),
            "nclaw_held_out_shape": NCLAW_HELD_OUT_SHAPE[material],
        },
        "identified_from": ident.get("source_dump"),
        "recovered_vs_truth": recovered_vs_truth,
        "refused_parameters": ident.get("refused_parameters", []),
        "diagnostics": diag,
        "mse": {s: {"role": v["role"], "mse": v["mse"],
                    "mse_final_frame": v["mse_final_frame"],
                    "rmse_mm": v["rmse_mm"], "n_particles": v["n_particles"],
                    "n_frames": v["n_frames"]}
                for s, v in scores.items()},
        "nclaw_published": NCLAW_PUBLISHED[material],
        "nclaw_published_source": ("their own method's row; reconstruction from "
                                   "Table 1 page 5, generalization axes from "
                                   "Table 2 page 7 tasks (a) time, "
                                   "(b) velocity, (c) geometry"),
        "engine_caveat": ENGINE_CAVEAT,
        "nclaw_geometry_confound": NCLAW_GEOMETRY_CONFOUND,
        "metric": ("mean over particles and coordinates of the squared position "
                   "difference in the unit box, averaged over every fifth frame"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{material}.json").write_text(
        json.dumps(results, indent=2, default=float))

    lines = [f"# NCLaw-matched comparison: {material}", "",
             "Law identified from the cube throw alone, by a convex weak-form",
             "solve on the grid-consistent momentum residual. The simulator is",
             "never differentiated.", "",
             "## Recovered parameters", "",
             "| parameter | recovered | truth | rel err | source |",
             "| --- | --- | --- | --- | --- |"]
    for k, v in recovered_vs_truth.items():
        rel = "" if v["rel_err"] is None else f"{100 * v['rel_err']:.2f} %"
        src = "prior (refused)" if v["prior_fixed"] else "identified"
        t = "" if v["truth"] is None else f"{v['truth']:.4e}"
        lines += [f"| {k} | {v['recovered']:.4e} | {t} | {rel} | {src} |"]
    if results["refused_parameters"]:
        lines += ["", "Refused and held at the known-class prior: "
                  + ", ".join(results["refused_parameters"]) + "."]

    lines += ["", "## Position MSE (NCLaw's metric)", "",
              "| scene | role | ours | NCLaw published | particles |",
              "| --- | --- | --- | --- | --- |"]
    for s, v in results["mse"].items():
        pub = NCLAW_PUBLISHED[material].get(v["role"])
        pubs = "" if pub is None else f"{pub:.2e}"
        lines += [f"| {s} | {v['role']} | {v['mse']:.3e} | {pubs} | "
                  f"{v['n_particles']} |"]
    lines += ["", f"NCLaw's own held-out shape for {material} is "
              f"{NCLAW_HELD_OUT_SHAPE[material]}.", "", ENGINE_CAVEAT, "",
              NCLAW_GEOMETRY_CONFOUND, ""]
    (OUT / f"report_{material}.md").write_text("\n".join(lines) + "\n")

    if results["mse"]:
        _figure(material, results)
    log("\n".join(lines))
    log(f"[report] wrote {OUT / f'results_{material}.json'} and "
        f"{OUT / f'report_{material}.md'}")
    return results


def _figure(material: str, results: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results["mse"].keys())
    ours = [results["mse"][n]["mse"] for n in names]
    pub = [NCLAW_PUBLISHED[material].get(results["mse"][n]["role"]) for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    xs = np.arange(len(names))
    ax[0].bar(xs - 0.19, ours, width=0.36, color="#1c7ed6", label="ours")
    have = [i for i, p in enumerate(pub) if p is not None]
    if have:
        ax[0].bar(xs[have] + 0.19, [pub[i] for i in have], width=0.36,
                  color="#e8590c", label="NCLaw published")
    ax[0].set_yscale("log")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"{n}\n({results['mse'][n]['role'][:5]})" for n in names])
    ax[0].set_ylabel("position MSE (unit box)")
    ax[0].set_title(f"(a) {material}: identify from the cube, roll out on shapes")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y", which="both")

    for n in names:
        s = json.loads((OUT / f"rollout_{material}.json").read_text())[n]
        ax[1].semilogy(np.maximum(s["per_frame"], 1e-14), lw=1.8, label=n)
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("position MSE")
    ax[1].set_title("(b) error growth over the throw")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / f"nclaw_{material}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["gen", "identify", "rollout", "report", "all", "cross"])
    # the four physical materials only: MATERIALS also holds the rollout-only
    # table entries, which have no identify leg and no published column
    ap.add_argument("--material", default="jelly", choices=sorted(NCLAW_PUBLISHED))
    ap.add_argument("--shapes", default="cube,bunny,spot,dragon")
    ap.add_argument("--n-grid", type=int, default=N_GRID)
    ap.add_argument("--window-frames", type=int, default=0,
                    help="0 uses the per-material default in WINDOW_FRAMES")
    ap.add_argument("--vel", default="preset", choices=sorted(VELS),
                    help="throw preset: NCLaw vel/preset or their geometry-eval vel/mild")
    ap.add_argument("--force", action="store_true",
                    help="re-run scenes whose dumps already exist")
    ap.add_argument("--nclaw-dir", default=None,
                    help="cross stage: their state_root of 0000.pt, 0001.pt, ...")
    ap.add_argument("--manifest", default=None,
                    help="cross stage: the ingest manifest json (see ingest.MANIFEST_SCHEMA)")
    ap.add_argument("--name", default=None,
                    help="cross stage: scene name for the artefacts; default the folder name")
    ap.add_argument("--dump", default=None,
                    help="identify stage: identify from THIS schema-valid trajectory "
                         "instead of the default cube dump (a grid-tagged truth, an "
                         "ingested folder)")
    ap.add_argument("--tag", default=None,
                    help="identify stage: suffix for the results file, so a run on "
                         "another dump does not overwrite identify_<material>.json")
    a = ap.parse_args(argv)

    if a.stage == "cross":
        if not a.nclaw_dir:
            raise SystemExit("the cross stage needs --nclaw-dir (and normally --manifest)")
        stage_cross(a.material, a.nclaw_dir, a.manifest, name=a.name, force=a.force,
                    window_frames=a.window_frames or None)
        return

    shapes = [s.strip() for s in a.shapes.split(",") if s.strip()]
    bad = [s for s in shapes if s not in SHAPES]
    if bad:
        raise SystemExit(f"unknown shapes: {bad}; known: {sorted(SHAPES)}")

    if a.stage in ("gen", "all"):
        stage_gen(a.material, shapes, force=a.force, n_grid=a.n_grid, vel=a.vel)
    if a.stage in ("identify", "all"):
        stage_identify(a.material, n_grid=a.n_grid,
                       window_frames=a.window_frames or None,
                       dump=a.dump, tag=a.tag)
    if a.stage in ("rollout", "all"):
        stage_rollout(a.material, shapes, force=a.force, n_grid=a.n_grid, vel=a.vel)
    if a.stage in ("report", "all"):
        stage_report(a.material)


if __name__ == "__main__":
    main()

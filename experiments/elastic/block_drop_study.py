"""Two elastic block drops: identify E from particle states, then change height.

This is an isolated precursor to flexible-object manipulation. It reuses the
current MPM solver and contact-excluding weak form, not the historical radial
drop estimator. No controller or manuscript files are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "material_law": "fixed-corotated isotropic elasticity; no plasticity or viscosity",
    "E_pa": {"A": 80000.0, "B": 240000.0},
    "known_nu": 0.45, "known_density_kg_m3": 1000.0,
    "size_m": [0.060, 0.040, 0.025], "domain_m": 0.20, "floor_z_m": 0.015,
    "probe_height_m": 0.040, "validation_height_m": 0.060,
    "floor_contact": "separable", "floor_friction": 0.0,
    "grid": 80, "refinement_grid": 96, "dt_s": 0.000020,
    "frame_dt_s": 0.0005, "duration_s": 0.30,
    "observation_interval_s": [0.060, 0.180],
    "temporal_window_frames": 26, "contact_margin_cells": 3.0,
    "seed": 0, "particle_jitter": False,
    "estimated": ["Young modulus E"],
    "known": ["Poisson ratio", "density", "geometry", "gravity", "contact", "law family"],
    "fit_channels": ["particle x", "particle v", "particle F", "reference volumes", "mass"],
    "fit_excludes": ["true E", "stress", "contact force", "validation trajectories"],
    "selection": "Fixed two materials and heights before simulation; keep all runs.",
}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def initialize(root):
    root.mkdir(parents=True, exist_ok=False)
    save(root / "protocol.json", PROTOCOL)
    sources = [Path(__file__), ROOT / "src/ident/weakform/elastic_grid.py",
               ROOT / "src/warpmpm/core/solver.py",
               ROOT / "src/warpmpm/kernels/mpm_solver_warp.py",
               ROOT / "src/warpmpm/kernels/mpm_utils.py",
               ROOT / "src/warpmpm/kernels/warp_utils.py"]
    snapshot = root / "source"
    for path in sources:
        dst = snapshot / path.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
    save(root / "provenance.json", {
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                               text=True),
        "python": platform.python_version(), "numpy": np.__version__,
    })


def specimen(p, grid, height):
    size = np.asarray(p["size_m"])
    counts = np.rint(size / (p["domain_m"] / grid / 2)).astype(int)
    spacing = size / counts
    axes = [(np.arange(n) + .5) * h - length / 2
            for n, h, length in zip(counts, spacing, size, strict=True)]
    x = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    x += [p["domain_m"] / 2, p["domain_m"] / 2,
          p["floor_z_m"] + height + size[2] / 2]
    vol = np.full(len(x), np.prod(spacing), dtype=np.float32)
    return x.astype(np.float32), vol, spacing


def record(root, name, E, height, grid, device, states):
    from warpmpm.core.solver import GridConfig, Solver
    from warpmpm.materials import elastic

    p = json.loads((root / "protocol.json").read_text())
    case = root / name
    case.mkdir(exist_ok=False)
    x0, vol, spacing = specimen(p, grid, height)
    save(case / "simulation_config.json", dict(
        E_pa=E, height_m=height, grid=grid, device=device, particle_count=len(x0),
        particle_spacing_m=spacing.tolist(), states_recorded=states,
        protocol_sha256=sha(root / "protocol.json")))
    solver = Solver(GridConfig(grid, p["domain_m"]), device=device,
                    inversion_policy="raise", sort_interval=0).load_particles(x0, vol)
    solver.set_material(elastic(E=E, nu=p["known_nu"], density=p["known_density_kg_m3"]),
                        rpic_damping=0.0, grid_v_damping_scale=1.0)
    solver.add_plane((0, 0, p["floor_z_m"]), (0, 0, 1),
                     surface=p["floor_contact"], friction=p["floor_friction"])
    step_count = round(p["frame_dt_s"] / p["dt_s"])
    assert np.isclose(step_count * p["dt_s"], p["frame_dt_s"], atol=1e-12)
    times = np.arange(round(p["duration_s"] / p["frame_dt_s"]) + 1) * p["frame_dt_s"]
    obs = np.flatnonzero((times >= p["observation_interval_s"][0] - 1e-12) &
                         (times <= p["observation_interval_s"][1] + 1e-12))
    np.save(case / "time.npy", times)
    np.save(case / "vol0.npy", vol)
    np.save(case / "initial.npy", x0)
    X = np.lib.format.open_memmap(case / "x.npy", mode="w+", dtype="float32",
                                  shape=(len(times), len(x0), 3))
    if states:
        V = np.lib.format.open_memmap(case / "observed_v.npy", mode="w+", dtype="float32",
                                      shape=(len(obs), len(x0), 3))
        F = np.lib.format.open_memmap(case / "observed_F.npy", mode="w+", dtype="float32",
                                      shape=(len(obs), len(x0), 3, 3))
        np.save(case / "observation_frames.npy", obs)
    diagnostics = []
    start = time.monotonic()
    oi = 0
    for i, t in enumerate(times):
        x, v, f = solver.x(), solver.v(), solver.F()
        if not all(np.isfinite(a).all() for a in (x, v, f)):
            raise RuntimeError(f"Nonfinite state at {t}")
        J = np.linalg.det(f)
        if J.min() <= 0:
            raise RuntimeError(f"Inversion at {t}")
        X[i] = x
        if states and oi < len(obs) and i == obs[oi]:
            V[oi], F[oi] = v, f
            oi += 1
        diagnostics.append([
            t, *np.mean(x, axis=0, dtype=np.float64), *np.ptp(x, axis=0),
            float(x[:, 2].min() - p["floor_z_m"]), float(J.min()), float(J.max()),
            float(np.sum(J * vol)), float(np.sqrt(np.mean(np.sum(v*v, axis=1)))),
        ])
        if i % 100 == 0:
            print(f"{name}: t={t:.3f} s, elapsed={time.monotonic()-start:.1f} s", flush=True)
        if i + 1 < len(times):
            solver.step(p["dt_s"], substeps=step_count)
    X.flush()
    if states:
        assert oi == len(obs)
        V.flush()
        F.flush()
    np.savetxt(case / "diagnostics.csv", diagnostics, delimiter=",", comments="",
               header="t,com_x,com_y,com_z,extent_x,extent_y,extent_z,min_z_above_floor,J_min,J_max,volume_m3,rms_speed")
    # Ballistic check uses only the first 40 ms, before contact for either height.
    d = np.asarray(diagnostics)
    free = times <= .040
    exact_z = d[0, 3] - .5 * 9.81 * times**2
    result = dict(wall_s=time.monotonic()-start, all_frames_completed=True,
                  inverted_count=int(solver.inverted_count()),
                  ballistic_com_max_error_mm=float(np.max(abs(d[free, 3]-exact_z[free]))*1000),
                  min_particle_z_above_floor_mm=float(d[:, 7].min()*1000),
                  min_J=float(d[:, 8].min()), max_J=float(d[:, 9].max()),
                  particle_mass_kg=float(vol.sum(dtype=np.float64)*p["known_density_kg_m3"]))
    result["outputs_sha256"] = {f.name: sha(f) for f in case.glob("*.npy")}
    save(case / "completion.json", result)
    print(json.dumps(dict(case=name, **result)), flush=True)


def identify(root, name):
    from ident.weakform.elastic_grid import assemble_elastic_timeweak

    p = json.loads((root / "protocol.json").read_text())
    case = root / name
    assert (case / "completion.json").exists()
    # Grid metadata are known measurement discretization. True E is not read.
    grid = p["refinement_grid"] if name.startswith("refinement") else p["grid"]
    frames = np.load(case / "observation_frames.npy")
    x = np.load(case / "x.npy", mmap_mode="r")[frames]
    v = np.load(case / "observed_v.npy", mmap_mode="r")
    f = np.load(case / "observed_F.npy", mmap_mode="r")
    vol = np.load(case / "vol0.npy")
    start = time.monotonic()
    system = assemble_elastic_timeweak(
        x, f, v, vol, vol*p["known_density_kg_m3"], np.array([0., 0., -9.81]),
        p["frame_dt_s"], grid, p["domain_m"],
        window_frames=p["temporal_window_frames"],
        collider_planes=[((0, 0, p["floor_z_m"]), (0, 0, 1))],
        collider_margin_cells=p["contact_margin_cells"], columns="corotated")
    nu = p["known_nu"]
    per_E = np.array([1/(2*(1+nu)), nu/((1+nu)*(1-2*nu))])
    a = system.A @ per_E
    if len(a) < 2 or float(a @ a) <= 1e-30:
        raise RuntimeError("Insufficient informative rows for stiffness identification")
    E = float(a @ system.b / (a @ a))
    residual = a*E-system.b
    if not np.isfinite(E) or E <= 0:
        raise RuntimeError("Nonphysical stiffness estimate")
    np.savez(case / "weak_system.npz", A=system.A, a_E=a, b=system.b,
             row_frame=system.node_frame, observation_frames=frames)
    # Splits are sensitivity diagnostics, not independent trials or error bars.
    mid = np.median(system.node_frame)
    split = {}
    for tag, mask in [("early", system.node_frame <= mid), ("late", system.node_frame > mid)]:
        if np.count_nonzero(mask) > 1 and a[mask] @ a[mask] > 1e-30:
            split[tag] = float(a[mask] @ system.b[mask] / (a[mask] @ a[mask]))
    fit = dict(E_pa=E, known_nu=nu, n_rows=system.n_rows,
               residual_relative_l2=float(np.linalg.norm(residual)/np.linalg.norm(system.b)),
               strain_norm_p05_p99=list(system.strain_coverage),
               temporal_split_E_pa=split, wall_s=time.monotonic()-start,
               inputs=p["fit_channels"], excluded=p["fit_excludes"],
               observation_interval_s=p["observation_interval_s"],
               force_used=False, true_E_used=False, measured_noise_added=False)
    save(case / "identification.json", fit)
    print(json.dumps(dict(case=name, **fit)), flush=True)
    return fit


def validate(root, label, device):
    p = json.loads((root / "protocol.json").read_text())
    fit = json.loads((root / f"probe_{label}/identification.json").read_text())
    for tag, E in [("true", p["E_pa"][label]), ("identified", fit["E_pa"])]:
        record(root, f"validation_{tag}_{label}", E, p["validation_height_m"],
               p["grid"], device, states=False)
    truth = np.load(root / f"validation_true_{label}/x.npy", mmap_mode="r")
    pred = np.load(root / f"validation_identified_{label}/x.npy", mmap_mode="r")
    t = np.load(root / f"validation_true_{label}/time.npy")
    rms = np.array([np.sqrt(np.mean(np.sum((a.astype(float)-b)**2, axis=1)))*1000
                    for a, b in zip(truth, pred, strict=True)])
    # Fixed interval begins before contact and covers impact/rebound at 60 mm.
    use = t >= .090
    save(root / f"validation_{label}.json", dict(
        height_m=p["validation_height_m"], fit_height_m=p["probe_height_m"],
        all_time_particle_rmse_mm=float(np.sqrt(np.mean(rms*rms))),
        post_090s_particle_rmse_mm=float(np.sqrt(np.mean(rms[use]**2))),
        max_frame_particle_rmse_mm=float(rms.max()),
        metric="3D particle correspondence RMSE, no alignment or rescaling",
        prediction_frozen_before_validation=True))
    np.savetxt(root / f"validation_error_{label}.csv", np.column_stack([t, rms]),
               delimiter=",", header="time_s,particle_rmse_mm", comments="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["init", "probe", "validate", "refine"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--material", choices=["A", "B"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "init":
        initialize(args.out)
        return
    if args.material is None:
        parser.error("--material is required")
    p = json.loads((args.out / "protocol.json").read_text())
    if args.stage == "validate":
        validate(args.out, args.material, args.device)
    else:
        refinement = args.stage == "refine"
        name = f"{'refinement' if refinement else 'probe'}_{args.material}"
        record(args.out, name, p["E_pa"][args.material], p["probe_height_m"],
               p["refinement_grid"] if refinement else p["grid"], args.device, states=True)
        identify(args.out, name)


if __name__ == "__main__":
    main()

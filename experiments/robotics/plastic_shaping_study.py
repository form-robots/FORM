"""Reproducible simulation study for the identification/shaping Results figure.

Run with the repository environment, for example:
    .venv/bin/python -m experiments.robotics.plastic_shaping_study prepare --device cuda:1
    .venv/bin/python -m experiments.robotics.plastic_shaping_study run --device cuda:1

Unlike the historical gripper demo, fitted parameters are computed from a saved
probe, and every scored/rendered execution uses the specimen's generating law.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "plastic_shaping_20260908"
TRUTH = {"A": dict(E=80000.0, nu=0.30, yield_stress=1000.0),
         "B": dict(E=80000.0, nu=0.30, yield_stress=10000.0)}
NOMINAL = dict(E=80000.0, nu=0.30, yield_stress=5500.0)
# Fixed before evaluating shaping outcomes. Targets are off the action lattice.
TARGET_ACTIONS = [[0.067, 0.058], [0.083, 0.074], [0.093, 0.066]]
ACTION_GRID = np.array([(gx, gy) for gx in np.linspace(0.06, 0.10, 11)
                        for gy in np.linspace(0.05, 0.09, 11)])


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, data):
    """Atomic checkpoints: an interrupted rollout never becomes a cache hit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


def cached(path):
    path = Path(path)
    if not path.exists():
        return False
    # Includes compatibility with checkpoints from the initial, non-atomic run.
    try:
        with np.load(path) as data:
            return bool(np.all(np.isfinite(data["x"])))
    except (OSError, ValueError, EOFError, BadZipFile):
        return False


def press(law, *, device, n_grid=48, speed=0.08, depth=0.014,
          release_time=0.12, record_state=True):
    """One constant-speed plate press, followed by contact removal and free motion.

    State exports are x, v, and the simulator's elastic F (not reconstructed
    total deformation). Force is the signed plate grid reaction. No stress or
    pressure labels enter the recorded identification data.
    """
    grid = GridConfig(n_grid=n_grid, grid_lim=0.30)
    size = (0.12, 0.08, 0.06)
    pos, vol0, floor = block(grid, size=size, ppc=2, seed=0)
    dt, sub = 1e-4, 8
    tick = dt * sub
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(grid, device=device, inversion_policy="raise").load_particles(pos, vol0)
        solver.set_material(vonmises(**law, density=1000.0))
    solver.add_plane(point=(0, 0, floor), normal=(0, 0, 1), surface="sticky")
    half = (0.07, 0.05, 2 * grid.dx)
    z_initial = floor + size[2] + half[2]
    plate = solver.add_box(center=(0.15, 0.15, z_initial), half_size=half)
    npress = int(np.ceil(depth / (speed * tick)))
    velocity = depth / (npress * tick)
    data = {k: [] for k in ("x", "v", "F", "time", "disp", "force")}

    def record(t, displacement, force):
        data["x"].append(solver.x().copy())
        if record_state:
            data["v"].append(solver.v().copy())
            data["F"].append(solver.F().copy())
        data["time"].append(t)
        data["disp"].append(displacement)
        data["force"].append(force)

    record(0.0, 0.0, 0.0)
    for f in range(npress):
        solver.set_box(plate, center=(0.15, 0.15, z_initial - f * velocity * tick),
                       velocity=(0, 0, -velocity))
        solver.reset_tool_force(plate)
        solver.step(dt, substeps=sub)
        record((f + 1) * tick, (f + 1) * velocity * tick,
               float(solver.tool_force(plate, tick)[2]))
    pressed = solver.x().copy()
    solver.set_box(plate, center=(0.15, 0.15, 0.27), velocity=(0, 0, 0))
    for _ in range(round(release_time / tick)):
        solver.step(dt, substeps=sub)
    data = {k: np.asarray(v) for k, v in data.items()}
    data.update(initial=pos, pressed=pressed, released=solver.x().copy(),
                released_v=solver.v().copy(), vol0=vol0,
                mass=1000.0 * vol0, floor=np.array(floor), n_grid=np.array(n_grid),
                grid_lim=np.array(grid.grid_lim), frame_dt=np.array(tick),
                depth=np.array(depth), speed=np.array(velocity),
                release_time=np.array(release_time))
    if not all(np.all(np.isfinite(a)) for a in data.values()):
        raise RuntimeError("Nonfinite press data")
    return data


def identify(data, *, stride=2, window_frames=26, margin=2.0):
    """Hencky weak-balance moduli and elastic-strain plateau yield estimate.

    The top exclusion plane is the deepest plate position, so every retained
    spatial test stays clear of the moving contact throughout the probe.
    """
    from experiments.nclaw.suite import identify_yield
    from ident.weakform.elastic_grid import assemble_elastic_timeweak, solve_elastic_grid

    floor = float(data["floor"])
    planes = [((0, 0, floor), (0, 0, 1)),
              ((0, 0, floor + 0.06 - float(data["depth"])), (0, 0, -1))]
    system = assemble_elastic_timeweak(
        data["x"], data["F"], data["v"], data["vol0"], data["mass"],
        np.array([0, 0, -9.81]), float(data["frame_dt"]) * stride,
        int(data["n_grid"]), float(data["grid_lim"]),
        frames=list(range(0, len(data["x"]), stride)), window_frames=window_frames,
        collider_planes=planes, collider_margin_cells=margin, columns="hencky")
    fit = solve_elastic_grid(system)
    yfit = identify_yield({"F": data["F"]}, fit["mu"], log=lambda _: None)
    if yfit["refused"] or fit["E"] <= 0 or not 0 < fit["nu"] < 0.49:
        raise RuntimeError(f"Identification refused: {fit}, {yfit}")
    return dict(law={"E": fit["E"], "nu": fit["nu"],
                     "yield_stress": yfit["yield_stress"]},
                elastic=fit, plastic=yfit, frame_stride=stride,
                window_frames=window_frames, contact_margin_cells=margin)


def grip_boxes(axis, gap, floor):
    """Two full-height fingers, with the long face perpendicular to closing."""
    ax = 0 if axis == "x" else 1
    half = [0.07, 0.07, 0.065]
    half[ax] = 0.0125
    centers = []
    for sign in (-1, 1):
        center = [0.15, 0.15, floor + 0.065]
        center[ax] += sign * (gap / 2 + half[ax])
        centers.append(center)
    return np.array(centers), np.array(half)


def shape(law, action, *, device, n_grid=48, release_time=0.30, record=False,
          dt=1e-4):
    """Two centered orthogonal squeezes; action gives their final face gaps.

    Fresh, correctly oriented collider pairs are created for each axis. Tools
    are removed by parking them above the material, without tangential drag.
    Outcomes are evaluated at a fixed delay after release, not called static.
    """
    grid = GridConfig(n_grid=n_grid, grid_lim=0.30)
    pos, vol0, floor = block(grid, size=(0.12, 0.08, 0.06), ppc=2, seed=0)
    tick = 0.0008
    sub = round(tick / dt)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(grid, device=device, inversion_policy="raise").load_particles(pos, vol0)
        solver.set_material(vonmises(**law, density=1000.0))
    solver.add_plane(point=(0, 0, floor), normal=(0, 0, 1), surface="sticky")
    snapshots = {"initial": pos}
    opening, speed = 0.16, 0.10
    elapsed = 0.0
    for axis, gap in zip(("x", "y"), action, strict=True):
        if not 0.04 <= gap <= opening:
            raise ValueError("Invalid gripper gap")
        start, half = grip_boxes(axis, opening, floor)
        handles = [solver.add_box(center=c, half_size=half) for c in start]
        ax = 0 if axis == "x" else 1
        nclose = int(np.ceil((opening - gap) / (2 * speed * tick)))
        vel = np.zeros((2, 3))
        vel[:, ax] = [1, -1]
        vel *= (opening - gap) / (2 * nclose * tick)
        for f in range(nclose):
            for h, c, v in zip(handles, start + f * tick * vel, vel, strict=True):
                solver.set_box(h, center=c, velocity=v)
            solver.step(dt, substeps=sub)
        elapsed += nclose * tick
        if record:
            snapshots[f"{axis}_pressed"] = solver.x().copy()
            snapshots[f"{axis}_boxes"] = grip_boxes(axis, gap, floor)[0]
            snapshots[f"{axis}_half"] = half
        for h in handles:
            solver.set_box(h, center=(0.15, 0.15, 0.27), velocity=(0, 0, 0))
        for _ in range(round(release_time / tick)):
            solver.step(dt, substeps=sub)
        elapsed += release_time
        if record:
            snapshots[f"{axis}_released"] = solver.x().copy()
    x = solver.x().copy()
    v = solver.v().copy()
    if not np.all(np.isfinite(x)) or solver.inverted_count():
        raise RuntimeError("Invalid shaping rollout")
    return dict(x=x, v=v, floor=np.array(floor), elapsed=np.array(elapsed), **snapshots)


def chamfer_mm(x, target):
    """Mean of the two mean Euclidean nearest-neighbor distances, in mm."""
    from scipy.spatial import cKDTree
    return float(500 * (cKDTree(target).query(x)[0].mean() +
                        cKDTree(x).query(target)[0].mean()))


def pilot(args):
    folder = OUT / "pilot"
    folder.mkdir(parents=True, exist_ok=True)
    for name, yld in (("A", 1000.0), ("B", 6000.0)):
        law = dict(E=80000.0, nu=0.30, yield_stress=yld)
        path = folder / f"press_{name}.npz"
        started = time.monotonic()
        if cached(path):
            data = dict(np.load(path))
        else:
            data = press(law, device=args.device)
            save_npz(path, data)
        fit = identify(data)
        save_json(folder / f"fit_{name}.json", dict(truth=law, **fit))
        print(json.dumps(dict(name=name, fit=fit, seconds=time.monotonic()-started)), flush=True)


def prepare(args):
    """Freeze the protocol, fit probes, and evaluate the independent press."""
    folder = OUT / "study"
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(truth=TRUTH, nominal=NOMINAL, target_actions=TARGET_ACTIONS,
                  candidate_actions=ACTION_GRID.tolist(), n_grid=48, grid_lim=0.30,
                  size=[0.12, 0.08, 0.06], ppc=2, seed=0, density=1000.0,
                  dt=1e-4, control_dt=0.0008, gripper_release_time=0.30,
                  probe_depth=0.014, probe_speed=0.08,
                  validation_depth=0.024, validation_speed=0.12,
                  fit_observations=["x", "v", "elastic F", "mass", "reference volume"],
                  metric="0.5*(mean NN distance X->T + mean NN distance T->X), mm",
                  evaluation="All plans executed in the specimen's true material",
                  target_protocol="Three reachable shapes per material, generated from off-grid actions; actions hidden from planner")
    if (folder / "protocol.json").exists():
        if json.loads((folder / "protocol.json").read_text()) != config:
            raise RuntimeError("Existing protocol differs; use a new output directory")
    else:
        save_json(folder / "protocol.json", config)
    fits = {}
    for name in TRUTH:
        probe_path = folder / f"probe_{name}.npz"
        if not cached(probe_path):
            data = press(TRUTH[name], device=args.device)
            save_npz(probe_path, data)
        data = dict(np.load(probe_path))
        fits[name] = identify(data)
        save_json(folder / f"identification_{name}.json", fits[name])
        print("Identified", name, fits[name]["law"], flush=True)
    for key, law in [("true_A", TRUTH["A"]), ("true_B", TRUTH["B"]),
                     ("identified_A", fits["A"]["law"]),
                     ("identified_B", fits["B"]["law"]), ("nominal", NOMINAL)]:
        path = folder / f"validation_{key}.npz"
        if not cached(path):
            data = press(law, device=args.device, speed=0.12, depth=0.024, record_state=False)
            save_npz(path, data)
        print("Validation", key, flush=True)
    provenance = dict(git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        device=args.device, source_sha256={})
    for rel in ["experiments/robotics/plastic_shaping_study.py", "experiments/nclaw/suite.py",
                "src/ident/weakform/elastic_grid.py", "src/warpmpm/core/solver.py",
                "src/warpmpm/kernels/mpm_utils.py", "src/warpmpm/kernels/mpm_solver_warp.py"]:
        provenance["source_sha256"][rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    save_json(folder / "provenance.json", provenance)


def verify(args):
    """Recompute reported metrics, verify replay provenance, and archive inputs."""
    del args
    import importlib.metadata

    folder = OUT / "study"
    protocol = json.loads((folder / "protocol.json").read_text())
    results = json.loads((folder / "shaping_results.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    for rel, digest in provenance["source_sha256"].items():
        if rel != "experiments/robotics/plastic_shaping_study.py":
            if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Scientific dependency changed during the run: {rel}")
    if len(results) != 18:
        raise AssertionError("Expected all 18 material/target/model executions")
    max_replay_chamfer = 0.0
    for r in results:
        name, target, method = r["material"], r["target"], r["method"]
        idx = r["action_index"]
        np.testing.assert_allclose(r["action"], protocol["candidate_actions"][idx], atol=0, rtol=0)
        assert idx == int(np.argmin(r["candidate_errors_mm"]))
        truth = np.load(folder / "library" / f"true_{name}" / f"{idx:03d}.npz")["x"]
        executed = np.load(folder / f"execution_{name}_{target}_{method}.npz")["x"]
        goal = np.load(folder / f"target_{name}_{target}.npz")["x"]
        np.testing.assert_allclose(chamfer_mm(executed, goal), r["executed_error_mm"], rtol=1e-10)
        replay_error = chamfer_mm(truth, executed)
        max_replay_chamfer = max(max_replay_chamfer, replay_error)
        # Parallel float32 accumulation is not bitwise deterministic on CUDA.
        if replay_error > 0.10:
            raise AssertionError(f"Independent true-material replay differs by {replay_error} mm")
    for method, mean in summary["mean_shaping_error_mm"].items():
        measured = np.mean([r["executed_error_mm"] for r in results if r["method"] == method])
        np.testing.assert_allclose(measured, mean, rtol=1e-12)
    for name in TRUTH:
        actual = np.load(folder / f"validation_true_{name}.npz")["force"]
        for method, key in (("identified", f"identified_{name}"), ("nominal", "nominal")):
            prediction = np.load(folder / f"validation_{key}.npz")["force"]
            value = np.linalg.norm(actual-prediction)/np.linalg.norm(actual)
            np.testing.assert_allclose(value, summary["force_relative_l2"][name][method], rtol=1e-12)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    save_json(folder / "environment.json", dict(python=sys.version, packages=packages))
    (folder / "environment.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(packages.items()))+"\n")
    snapshot_dir = folder / "source_snapshot" / "final"
    for rel in ["experiments/robotics/plastic_shaping_study.py", "experiments/robotics/plastic_shaping_figure.py",
                "experiments/nclaw/suite.py", "src/ident/weakform/elastic_grid.py",
                "src/warpmpm/core/solver.py", "src/warpmpm/kernels/mpm_utils.py",
                "src/warpmpm/kernels/mpm_solver_warp.py"]:
        dest = snapshot_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.rglob("*")) if p.is_file()
              and p.name not in {"checksums.json", "verification.json"}}
    save_json(folder / "checksums.json", hashes)
    result = dict(passed=True, executions_checked=len(results), files_hashed=len(hashes),
                  max_independent_true_replay_chamfer_mm=max_replay_chamfer,
                  replay_tolerance_mm=0.10)
    launch_file = folder / "source_snapshot" / "plastic_shaping_study-at-launch.py"
    if launch_file.exists():
        import ast
        def physics_functions(path):
            return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name in
                    {"press", "identify", "grip_boxes", "shape", "chamfer_mm"}}
        unchanged = physics_functions(launch_file) == physics_functions(Path(__file__))
        result["physics_and_identification_unchanged_since_launch"] = unchanged
        if not unchanged:
            raise AssertionError("Scientific computation changed since the library was launched")
    save_json(folder / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


def run_library(args):
    """Evaluate identical candidate sets once per model, reused across targets."""
    folder = OUT / "study"
    protocol = json.loads((folder / "protocol.json").read_text())
    if (protocol["truth"] != TRUTH or protocol["nominal"] != NOMINAL
            or protocol["candidate_actions"] != ACTION_GRID.tolist()
            or protocol["target_actions"] != TARGET_ACTIONS):
        raise RuntimeError("Frozen protocol differs from this code; use a new output directory")
    fits = {n: json.loads((folder / f"identification_{n}.json").read_text()) for n in TRUTH}
    laws = {"nominal": NOMINAL, **{f"identified_{n}": fits[n]["law"] for n in TRUTH},
            **{f"true_{n}": TRUTH[n] for n in TRUTH}}
    for n, law in TRUTH.items():
        for j, action in enumerate(TARGET_ACTIONS):
            path = folder / f"target_{n}_{j}.npz"
            if not cached(path):
                data = shape(law, action, device=args.device, record=True)
                save_npz(path, data)
    for name, law in laws.items():
        target_folder = folder / "library" / name
        target_folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        for i, action in enumerate(ACTION_GRID):
            path = target_folder / f"{i:03d}.npz"
            if not cached(path):
                data = shape(law, action, device=args.device)
                save_npz(path, data)
            if (i + 1) % 10 == 0 or i == len(ACTION_GRID)-1:
                print(name, i+1, "/", len(ACTION_GRID),
                      "seconds", round(time.monotonic()-started, 1), flush=True)
    evaluate(args)


def evaluate(args):
    folder = OUT / "study"
    results = []
    for name, true_law in TRUTH.items():
        for j in range(len(TARGET_ACTIONS)):
            target = np.load(folder / f"target_{name}_{j}.npz")["x"]
            for method, model in (("nominal", "nominal"),
                                  ("identified", f"identified_{name}"),
                                  ("oracle", f"true_{name}")):
                losses = [chamfer_mm(np.load(folder / "library" / model / f"{i:03d}.npz")["x"], target)
                          for i in range(len(ACTION_GRID))]
                best = int(np.argmin(losses))
                action = ACTION_GRID[best]
                # Independent replay in the true simulator, including the oracle.
                executed = shape(true_law, action, device=args.device, record=True)
                output = folder / f"execution_{name}_{j}_{method}.npz"
                save_npz(output, executed)
                error = chamfer_mm(executed["x"], target)
                result = dict(material=name, target=j, method=method, model=model,
                              action_index=best, action=action.tolist(),
                              predicted_error_mm=losses[best], executed_error_mm=error,
                              candidate_errors_mm=losses,
                              final_rms_speed=float(np.sqrt(np.mean(executed["v"]**2))))
                results.append(result)
                print(name, j, method, "predicted", round(losses[best], 4),
                      "executed", round(error, 4), "action", action.tolist(), flush=True)
                save_json(folder / "shaping_results.json", results)
    force = {}
    for name in TRUTH:
        true = np.load(folder / f"validation_true_{name}.npz")["force"]
        force[name] = {}
        for method, key in (("identified", f"identified_{name}"), ("nominal", "nominal")):
            pred = np.load(folder / f"validation_{key}.npz")["force"]
            force[name][method] = float(np.linalg.norm(pred-true)/np.linalg.norm(true))
    means = {m: float(np.mean([r["executed_error_mm"] for r in results if r["method"] == m]))
             for m in ("nominal", "identified", "oracle")}
    summary = dict(force_relative_l2=force, mean_shaping_error_mm=means,
                   nominal_to_identified_ratio=means["nominal"] / means["identified"],
                   n_material_target_pairs=6, trials="deterministic; one execution per plan")
    save_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def numerics(args):
    """Replay the first task's frozen plans with smaller dt and a finer grid.

    Targets are regenerated from the same reference actions at each resolution.
    Parameters and chosen actions stay fixed. This is a sensitivity check, not
    a claim of spatial convergence or cross-simulator validation.
    """
    folder = OUT / "study"
    destination = folder / "numerics"
    destination.mkdir(exist_ok=True)
    results = []
    for name, law in TRUTH.items():
        original_target = np.load(folder / f"target_{name}_0.npz")["x"]
        actions = {}
        for method, model in (("nominal", "nominal"), ("identified", f"identified_{name}")):
            losses = [chamfer_mm(np.load(folder / "library" / model / f"{i:03d}.npz")["x"],
                                 original_target) for i in range(len(ACTION_GRID))]
            actions[method] = ACTION_GRID[int(np.argmin(losses))]
        for tag, grid, dt in (("half_dt", 48, 5e-5), ("grid64", 64, 1e-4)):
            goal = shape(law, TARGET_ACTIONS[0], device=args.device, n_grid=grid, dt=dt)
            save_npz(destination / f"{name}_{tag}_target.npz", goal)
            for method, action in actions.items():
                executed = shape(law, action, device=args.device, n_grid=grid, dt=dt)
                save_npz(destination / f"{name}_{tag}_{method}.npz", executed)
                result = dict(material=name, target=0, refinement=tag, n_grid=grid, dt=dt,
                              method=method, action=action.tolist(),
                              error_mm=chamfer_mm(executed["x"], goal["x"]))
                results.append(result)
                save_json(folder / "numerical_sensitivity.json", results)
                print(json.dumps(result), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "grip-pilot", "prepare", "run", "evaluate", "verify", "numerics"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out", type=Path, default=OUT, help="Study root; use a fresh path for independent reproduction")
    args = parser.parse_args()
    OUT = args.out.resolve()
    if args.stage == "pilot":
        pilot(args)
    elif args.stage == "grip-pilot":
        for name, law in TRUTH.items():
            started = time.monotonic()
            result = shape(law, [0.08, 0.07], device=args.device, record=True)
            save_npz(OUT / "pilot" / f"correct_grip_{name}.npz", result)
            print(name, "seconds", time.monotonic()-started,
                  "extent mm", 1000*np.ptp(result["x"], axis=0),
                  "rms speed", np.sqrt(np.mean(result["v"]**2)), flush=True)
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage == "run":
        run_library(args)
    elif args.stage == "evaluate":
        evaluate(args)
    elif args.stage == "numerics":
        numerics(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()

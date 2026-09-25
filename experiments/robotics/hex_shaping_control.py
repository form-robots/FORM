"""Optimize three squeeze gaps against a fixed, volume-matched hexagonal prism.

This is a true-parameter feasibility experiment, separate from the paper's
identified-versus-nominal benchmark. Use a fresh output directory for each setup.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize

from experiments.robotics.hex_shaping_pilot import ROOT, TRUTH, shape
from experiments.robotics.plastic_shaping_study import chamfer_mm, save_json, save_npz
from warpmpm import GridConfig
from warpmpm.scenes import block

OUT = ROOT/"out/hex_shaping_control_20260908"


def target_prism(n_grid=48, diameter=.090):
    """Geometric target fixed independently of simulated outcomes and material law."""
    grid = GridConfig(n_grid=n_grid, grid_lim=.30)
    _, volume, floor = block(grid, size=(.12, .08, .06), ppc=2, seed=0)
    # Fix the geometric volume across grid checks to the baseline particle volume.
    _, baseline_volume, _ = block(GridConfig(n_grid=48, grid_lim=.30),
                                  size=(.12, .08, .06), ppc=2, seed=0)
    total_volume = float(baseline_volume.astype(float).sum())
    height = total_volume/(np.sqrt(3)/2*diameter**2)
    spacing = grid.dx/2
    nx = int(np.ceil(diameter/spacing))
    ny = int(np.ceil(2*diameter/np.sqrt(3)/spacing))
    nz = int(np.ceil(height/spacing))
    xs = (np.arange(nx)+.5-nx/2)*spacing
    ys = (np.arange(ny)+.5-ny/2)*spacing
    zs = (np.arange(nz)+.5)*height/nz
    xyz = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    normals = np.column_stack((np.cos(np.deg2rad([0, 60, 120])),
                               np.sin(np.deg2rad([0, 60, 120]))))
    xyz = xyz[np.all(np.abs(xyz[:, :2]@normals.T) <= diameter/2, axis=1)]
    xyz += np.array([.15, .15, floor])
    vol = np.full(len(xyz), total_volume/len(xyz), dtype=np.float32)
    return dict(x=xyz.astype(np.float32), vol0=vol, floor=np.array(floor),
                diameter=np.array(diameter), height=np.array(height),
                volume=np.array(total_volume), initial_volume=np.array(float(volume.sum())))


def snapshot_sources(folder, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Existing protocol differs; use a fresh output directory")
    save_json(path, protocol)
    files = [Path(__file__), ROOT/"experiments/robotics/hex_shaping_pilot.py",
             ROOT/"experiments/robotics/plastic_shaping_study.py"]
    files += list((ROOT/"src/warpmpm").rglob("*.py"))
    hashes = {}
    for source in files:
        rel = source.relative_to(ROOT)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        dest = folder/"source_snapshot"/rel
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed: {rel}; use a fresh output directory")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(rel)] = digest
    save_json(folder/"source_sha256.json", hashes)
    save_json(folder/"environment.json", dict(python=sys.version,
        packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    (folder/"gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))


def optimize(args):
    law = dict(TRUTH[args.material])
    if args.young_kpa is not None:
        law["E"] = args.young_kpa*1000
    if args.yield_kpa is not None:
        law["yield_stress"] = args.yield_kpa*1000
    initial = np.array(args.initial_mm or ([75., 85., 90.] if args.material == "A"
                                          else [65., 70., 75.]))
    simplex = np.tile(initial, (4, 1))
    simplex[1:] -= 10*np.eye(3)
    protocol = dict(material=args.material, law=law, cycles=args.cycles,
        n_grid=args.n_grid, dt=args.dt, diameter_m=args.diameter_mm/1000,
        initial_gaps_mm=initial.tolist(), initial_simplex_mm=simplex.tolist(),
        bounds_mm=[[40., 115.]]*3, maximum_objective_evaluations=args.max_evals,
        objective="Symmetric mean unsquared nearest-neighbor distance of full particle clouds, mm",
        target="Analytic regular hexagonal prism; 90 mm baseline width; initial baseline volume",
        method="Bounded Nelder-Mead; true material parameters; no claim of global optimum",
        scope="Feasibility only, not a nominal-versus-identified benchmark",
        device=args.device)
    folder = args.out.resolve()
    snapshot_sources(folder, protocol)
    target = target_prism(args.n_grid, args.diameter_mm/1000)
    save_npz(folder/"target.npz", target)
    evaluations = []
    cache = folder/"evaluations"
    cache.mkdir(exist_ok=True)

    def objective(gaps_mm):
        gaps_mm = np.round(gaps_mm, 6)
        key = hashlib.sha256(gaps_mm.tobytes()).hexdigest()[:16]
        path = cache/f"{key}.npz"
        started = time.monotonic()
        if not path.exists():
            data = shape(law, gaps_mm/1000, device=args.device, n_grid=args.n_grid,
                         dt=args.dt, cycles=args.cycles)
            save_npz(path, data)
        else:
            data = dict(np.load(path))
        error = chamfer_mm(data["x"], target["x"])
        row = dict(index=len(evaluations), gaps_mm=gaps_mm.tolist(), error_mm=error,
                   file=str(path.relative_to(folder)), seconds=time.monotonic()-started)
        evaluations.append(row)
        save_json(folder/"evaluations.json", evaluations)
        best = min(evaluations, key=lambda r: r["error_mm"])
        save_json(folder/"best.json", best)
        print(json.dumps(dict(material=args.material, evaluation=len(evaluations),
                              error_mm=error, best_mm=best["error_mm"],
                              gaps_mm=row["gaps_mm"], seconds=row["seconds"])), flush=True)
        return error

    result = minimize(objective, initial, method="Nelder-Mead", bounds=[(40, 115)]*3,
                      options=dict(maxfev=args.max_evals, xatol=.25, fatol=.005,
                                   initial_simplex=simplex))
    best = min(evaluations, key=lambda r: r["error_mm"])
    shutil.copy2(folder/best["file"], folder/"selected.npz")
    # Independently execute the selected motion to check numerical replay.
    replay = shape(law, np.array(best["gaps_mm"])/1000, device=args.device,
                   n_grid=args.n_grid, dt=args.dt, cycles=args.cycles)
    save_npz(folder/"execution.npz", replay)
    save_json(folder/"result.json", dict(best=best, optimizer_success=bool(result.success),
        optimizer_message=str(result.message), evaluations=len(evaluations),
        executed_error_mm=chamfer_mm(replay["x"], target["x"]),
        independent_replay_error_mm=chamfer_mm(replay["x"], np.load(folder/"selected.npz")["x"])))
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json"}
    save_json(folder/"checksums.json", hashes)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from experiments.robotics.hex_shaping_report import image_of

    folder = args.out.resolve()
    data = np.load(folder/("execution.npz" if (folder/"execution.npz").exists() else
                          json.loads((folder/"best.json").read_text())["file"]))
    target = np.load(folder/"target.npz")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), constrained_layout=True)
    for ax, d, key, title in zip(axes[:2], [target, data], ["x", "x"],
                                ["Geometric target", "Released execution"], strict=True):
        ax.imshow(image_of(d, key)); ax.axis("off"); ax.set_title(title)
    ax = axes[2]
    z = data["x"][:, 2]-data["floor"]
    for q in [.15, .5, .85]:
        points = data["x"][abs(z-np.quantile(z, q)) < .003, :2]*1000-150
        hull = ConvexHull(points)
        points = points[np.r_[hull.vertices, hull.vertices[0]]]
        ax.plot(points[:, 0], points[:, 1], label=f"{q:.0%} height")
    angles = np.deg2rad(np.arange(30, 391, 60))
    radius = float(target["diameter"])*1000/np.sqrt(3)
    ax.plot(radius*np.cos(angles), radius*np.sin(angles), "k--", label="Target")
    ax.set(aspect="equal", xlim=(-65, 65), ylim=(-65, 65), title="Cross-sections", xlabel="x (mm)")
    ax.legend(fontsize=7)
    fig.savefig(folder/"preview.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("optimize", "render"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--material", choices=("A", "B"), default="A")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--n-grid", type=int, default=48)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--diameter-mm", type=float, default=90.)
    parser.add_argument("--initial-mm", nargs=3, type=float)
    parser.add_argument("--max-evals", type=int, default=44)
    parser.add_argument("--young-kpa", type=float)
    parser.add_argument("--yield-kpa", type=float)
    args = parser.parse_args()
    if args.stage == "optimize":
        optimize(args)
    else:
        render(args)


if __name__ == "__main__":
    main()

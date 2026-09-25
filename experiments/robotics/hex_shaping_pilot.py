"""Separate reproducible feasibility pilot for a released hexagonal shaping target.

Run: .venv/bin/python -m experiments.robotics.hex_shaping_pilot run --device cuda:0 --out out/hex_shaping_reproduction
View: .venv/bin/python -m experiments.robotics.hex_shaping_pilot render
The existing manuscript study and its fitted parameters are never overwritten.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from experiments.robotics.plastic_shaping_study import (
    ROOT, TRUTH, NOMINAL, save_json, save_npz,
)
from warpmpm import GridConfig, Solver
from warpmpm.geometry import SDFData
from warpmpm.materials import vonmises
from warpmpm.scenes import block

OUT = ROOT / "out" / "hex_shaping_20260908"
SOURCE_STUDY = ROOT / "out" / "plastic_shaping_20260908" / "study"
ANGLES = (0.0, 60.0, 120.0)
GAPS = (0.080, 0.090, 0.100)
HALF = np.array([0.0125, 0.095, 0.080])
BAND = 0.0002


def box_sdf(cell=0.002, extent=0.110):
    """Sample the analytic box distance without a mesh-to-SDF conversion."""
    coords = np.linspace(-extent, extent, round(2*extent/cell)+1)
    xyz = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    q = np.abs(xyz) - HALF
    values = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    grads = np.stack(np.gradient(values, cell, edge_order=2), axis=-1)
    return SDFData(values.astype(np.float32), grads.astype(np.float32),
                   np.full(3, -extent), cell, float(values.max()))


def jaw_pose(angle, gap, floor):
    rad = np.deg2rad(angle)
    normal = np.array([np.cos(rad), np.sin(rad), 0.0])
    # Account for the SDF contact band: its inner boundary has the requested gap.
    center = np.array([0.15, 0.15, floor+HALF[2]])
    centers = center + np.array([-1, 1])[:, None]*(gap/2+HALF[0]+BAND)*normal
    return centers, normal, np.array([0., 0., np.sin(rad/2), np.cos(rad/2)])


def shape(law, gaps, *, device, n_grid=48, dt=1e-4, cycles=1):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.30)
    pos, vol0, floor = block(grid, size=(0.12, 0.08, 0.06), ppc=2, seed=0)
    tick, release, opening, speed = 0.0008, 0.30, 0.160, 0.10
    sdf = box_sdf()
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(grid, device=device, inversion_policy="raise").load_particles(pos, vol0)
        solver.set_material(vonmises(**law, density=1000.0))
    solver.add_plane(point=(0, 0, floor), normal=(0, 0, 1), surface="sticky")
    data = dict(initial=pos, vol0=vol0, floor=np.array(floor),
                angles=np.tile(ANGLES, cycles), gaps=np.tile(gaps, cycles),
                half=HALF, contact_band=np.array(BAND), dt=np.array(dt),
                n_grid=np.array(n_grid), release_time=np.array(release))
    elapsed = 0.0
    for stage, (angle, gap) in enumerate(zip(data["angles"], data["gaps"], strict=True)):
        start, normal, quat = jaw_pose(angle, opening, floor)
        nclose = int(np.ceil((opening-gap)/(2*speed*tick)))
        end_time = elapsed+nclose*tick
        velocity = np.array([1, -1])[:, None]*normal*(opening-gap)/(2*nclose*tick)
        handles = [solver.add_sdf_collider(sdf, center=c, quat=quat, band=BAND,
                   surface="sticky", friction=0., start_time=elapsed-0.5*dt,
                   end_time=end_time-0.5*dt) for c in start]
        for f in range(nclose):
            for h, c, v in zip(handles, start+f*tick*velocity, velocity, strict=True):
                solver.set_sdf_pose(h, center=c, velocity=v)
            solver.step(dt, substeps=round(tick/dt))
        data[f"stage_{stage}_pressed"] = solver.x().copy()
        data[f"stage_{stage}_centers"] = jaw_pose(angle, gap, floor)[0]
        data[f"stage_{stage}_quat"] = quat
        for _ in range(round(release/tick)):
            solver.step(dt, substeps=round(tick/dt))
        data[f"stage_{stage}_released"] = solver.x().copy()
        elapsed = end_time+release
    data.update(x=solver.x().copy(), v=solver.v().copy(), elapsed=np.array(elapsed),
                inverted_count=np.array(solver.inverted_count()))
    if data["inverted_count"] or not all(np.all(np.isfinite(v)) for v in data.values()):
        raise RuntimeError("Nonfinite or inverted shaping rollout")
    return data


def freeze(folder, gaps, cycles, ramp):
    folder.mkdir(parents=True, exist_ok=True)
    before = folder/"manuscript_before.json"
    if not before.exists():
        paths = ["paper/icra2027/paper.tex", "paper/icra2027/paper.pdf",
                 "paper/icra2027/figs/identification_plastic_shaping.pdf",
                 "paper/icra2027/figs/identification_plastic_shaping.png"]
        save_json(before, {rel: hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
                           for rel in paths})
    protocol = dict(angles_deg=ANGLES, gaps_m=gaps, truth=TRUTH, nominal=NOMINAL,
                    cycles=cycles, gap_increment_per_direction_m=ramp,
                    initial_size_m=[0.12, 0.08, 0.06], n_grid=48, grid_lim_m=0.30,
                    ppc=2, seed=0, density_kg_m3=1000.0,
                    dt_s=1e-4, control_dt_s=0.0008, release_s=0.30,
                    jaw_half_m=HALF.tolist(), sdf_cell_m=0.002, contact_band_m=BAND,
                    contact="Sticky SDF jaws, sticky floor; each pair removed before the next",
                    decision="Inspect both materials and all three sizes for six recognizable released faces before model comparison",
                    status="Exploratory geometry pilot; not a manuscript replacement")
    protocol = json.loads(json.dumps(protocol))
    path = folder/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Protocol changed; use a fresh output directory")
    save_json(path, protocol)
    sources = ["experiments/robotics/hex_shaping_pilot.py",
               "experiments/robotics/plastic_shaping_study.py",
               "src/warpmpm/core/solver.py", "src/warpmpm/kernels/mpm_solver_warp.py",
               "src/warpmpm/kernels/mpm_utils.py", "src/warpmpm/scenes.py"]
    hashes = {}
    for rel in sources:
        digest = hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        hashes[rel] = digest
        dest = folder/"source_snapshot"/rel
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed since launch: {rel}; use a fresh output directory")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    save_json(folder/"source_sha256.json", hashes)
    for name in TRUTH:
        src = SOURCE_STUDY/f"identification_{name}.json"
        dest = folder/src.name
        if dest.exists() and dest.read_bytes() != src.read_bytes():
            raise RuntimeError("Original identification changed")
        shutil.copy2(src, dest)


def run(folder, device, gaps, cycles, ramp):
    freeze(folder, gaps, cycles, ramp)
    (folder/"gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    packages = sorted(f"{d.metadata['Name']}=={d.version}"
                      for d in importlib.metadata.distributions())
    (folder/"environment.txt").write_text("\n".join(packages)+"\n")
    for name, law in TRUTH.items():
        for gap in gaps:
            path = folder/f"true_{name}_{round(gap*1000)}.npz"
            if path.exists():
                continue
            started = time.monotonic()
            data = shape(law, [gap+i*ramp for i in range(3)], device=device, cycles=cycles)
            save_npz(path, data)
            print(json.dumps(dict(material=name, gap_mm=gap*1000,
                                  seconds=time.monotonic()-started,
                                  height_mm=float((data["x"][:, 2].max()-data["floor"])*1000),
                                  max_speed_m_s=float(np.linalg.norm(data["v"], axis=1).max()))), flush=True)
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json"}
    save_json(folder/"checksums.json", hashes)


def render(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    protocol = json.loads((folder/"protocol.json").read_text())
    gaps = protocol["gaps_m"]
    cycles = protocol.get("cycles", 1)
    ramp = protocol.get("gap_increment_per_direction_m", 0.)
    fig, axes = plt.subplots(2, len(gaps), figsize=(10, 6.5), constrained_layout=True, squeeze=False)
    for row, name in enumerate(TRUTH):
        for col, gap in enumerate(gaps):
            data = np.load(folder/f"true_{name}_{round(gap*1000)}.npz")
            ax = axes[row, col]
            # Display actual particle slices; no smoothing or silhouette adjustment.
            height = data["x"][:, 2]-data["floor"]
            for frac, color in [(0.15, "#9db8cc"), (0.50, "#276994"), (0.85, "#c17139")]:
                center = np.quantile(height, frac)
                points = data["x"][np.abs(height-center)<0.003, :2]*1000-150
                hull = ConvexHull(points)
                p = points[np.r_[hull.vertices, hull.vertices[0]]]
                ax.plot(p[:, 0], p[:, 1], color=color, lw=1.5, label=f"{frac:.0%} height quantile")
            # Ideal regular-hexagon footprint with the requested flat-to-flat size.
            angles = np.deg2rad(np.arange(30, 391, 60))
            radius = (gap+2*ramp)*1000/np.sqrt(3)
            ax.plot(radius*np.cos(angles), radius*np.sin(angles), "k--", lw=1,
                    label="Hexagon at final gap (guide)")
            gap_label = "/".join(f"{(gap+i*ramp)*1000:.0f}" for i in range(3))
            ax.set(title=f"{name}: {gap_label} mm", aspect="equal",
                   xlim=(-70, 70), ylim=(-70, 70), xlabel="x (mm)", ylabel="y (mm)")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Released cross-sections: {cycles} cycle(s) at 0°, 60°, 120°", fontsize=13)
    fig.savefig(folder/"feasibility.png", dpi=160)
    plt.close(fig)


def numerics(folder, device):
    """Check the near-hexagonal two-cycle, 90 mm sequence at finer dt and grid."""
    freeze(folder, (.090,), 2, 0.)
    checks = dict(half_dt=dict(n_grid=48, dt=5e-5),
                  grid64=dict(n_grid=64, dt=1e-4))
    save_json(folder/"numerical_protocol.json", checks)
    for name, law in TRUTH.items():
        for tag, settings in checks.items():
            path = folder/f"true_{name}_{tag}.npz"
            if path.exists():
                continue
            started = time.monotonic()
            data = shape(law, [.090]*3, device=device, cycles=2, **settings)
            save_npz(path, data)
            print(json.dumps(dict(material=name, check=tag,
                                  seconds=time.monotonic()-started)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "render", "numerics"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gaps-mm", type=float, nargs="+", default=[g*1000 for g in GAPS])
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--ramp-mm", type=float, default=0.)
    args = parser.parse_args()
    if args.stage == "run":
        run(args.out.resolve(), args.device, tuple(g/1000 for g in args.gaps_mm),
            args.cycles, args.ramp_mm/1000)
    elif args.stage == "render":
        render(args.out.resolve())
    else:
        numerics(args.out.resolve(), args.device)


if __name__ == "__main__":
    main()

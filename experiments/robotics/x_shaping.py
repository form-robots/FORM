"""Cylindrical-finger X shaping with explicit position-controlled robot motions.

Standalone experimental protocol; historical paper experiments are untouched.
All dimensions are SI except command-line gaps, which are millimetres.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pyvista as pv

from experiments.robotics.plastic_shaping_study import ROOT, save_json, save_npz
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_figure import surface
from warpmpm import GridConfig, Solver
from warpmpm.geometry import SDFData
from warpmpm.materials import vonmises

SOURCE = ROOT / "out/hex_shaping_study_20260908"
OUT = ROOT / "out/x_shaping_pilot_20260909"
CONFIG = dict(domain=.16, size=[.060, .060, .025], floor=.010,
              radius=.014, finger_height=.045, finger_bottom=.0005,
              opening=.072, finger_speed=.05, vertical_speed=.10,
              hover_bottom=.060, rotation_speed=np.pi,
              band=.0002, sdf_cell=.001, tool_friction=.3, floor_friction=.5,
              tick=.004, release_s=1., target_width=.070, target_notch_radius=.014,
              voxel=.00125, seed=0, density=1000.)


def cylinder_distance(xyz, radius, half_height):
    q = np.stack([np.linalg.norm(xyz[..., :2], axis=-1)-radius,
                  np.abs(xyz[..., 2])-half_height], axis=-1)
    return np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(q.max(-1), 0)


def cylinder_sdf(radius, height):
    cell = CONFIG["sdf_cell"]
    extent = np.ceil((max(radius, height/2)+.004)/cell)*cell
    axis = np.linspace(-extent, extent, round(2*extent/cell)+1)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    values = cylinder_distance(xyz, radius, height/2)
    grads = np.stack(np.gradient(values, cell, edge_order=2), axis=-1)
    return SDFData(values.astype(np.float32), grads.astype(np.float32),
                   np.full(3, -extent), cell, float(values.max()))


def specimen(n_grid=64):
    # Identical physical bounds and exact volume at every numerical resolution.
    size = np.array(CONFIG["size"]); center = CONFIG["domain"]/2
    counts = np.round(size/(CONFIG["domain"]/n_grid/2)).astype(int)
    spacing = size/counts
    axes = [(np.arange(n)+.5)*h-s/2+center for n, h, s in zip(counts[:2], spacing[:2], size[:2], strict=True)]
    axes.append(CONFIG["floor"]+(np.arange(counts[2])+.5)*spacing[2])
    x = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    x += np.random.default_rng(CONFIG["seed"]).uniform(-.2, .2, x.shape)*spacing
    return x.astype(np.float32), np.full(len(x), size.prod()/len(x), dtype=np.float32)


def target_polygon():
    # A square with an inward semicircle at the midpoint of each side.
    # CCW polygon, fixed independently of simulation outcomes.
    half = CONFIG["target_width"]/2; r = CONFIG["target_notch_radius"]
    bottom = [[-half, -half], [-r, -half]]
    theta = np.linspace(np.pi, 0, 65)
    bottom.extend(np.column_stack([r*np.cos(theta), -half+r*np.sin(theta)])[1:].tolist())
    bottom.append([half, -half])
    points = []
    for i in range(4):
        a = i*np.pi/2; rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        points.extend((np.array(bottom[:-1])@rotation.T).tolist())
    return np.array(points)+CONFIG["domain"]/2


def target_mesh():
    xy = target_polygon()
    area = .5*abs(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1)))
    height = np.prod(CONFIG["size"])/area
    points = np.column_stack([xy, np.full(len(xy), CONFIG["floor"])])
    face = pv.PolyData(points, np.r_[len(points), np.arange(len(points))])
    mesh = face.extrude((0, 0, height), capping=True).triangulate().clean()
    # Subdivision gives comparable quadrature density on large caps and walls.
    return mesh.compute_normals(auto_orient_normals=True, consistent_normals=True).subdivide(3, subfilter="linear")


def score(data, voxel=None):
    actual = surface(data["x_after_1s"], data["vol0"], h=CONFIG["voxel"] if voxel is None else voxel)
    return mesh_distance_mm(actual, target_mesh())


def centers(angle, gap, bottom):
    normal = np.array([np.cos(angle), np.sin(angle), 0.])
    middle = np.array([CONFIG["domain"]/2]*2+[CONFIG["floor"]+bottom+CONFIG["finger_height"]/2])
    return middle+np.array([-1., 1.])[:, None]*(gap/2+CONFIG["radius"])*normal


def execute(law, gaps_mm, *, device="cuda:0", n_grid=64, dt=.00005):
    gaps = np.array(gaps_mm)/1000
    if len(gaps) not in (2, 4) or np.any((gaps < .016) | (gaps > .060)):
        raise ValueError("Use two or four physical surface gaps in [16,60] mm")
    tick = CONFIG["tick"]; substeps = round(tick/dt)
    assert abs(substeps*dt-tick) < 1e-12
    x, vol = specimen(n_grid)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(GridConfig(n_grid=n_grid, grid_lim=CONFIG["domain"]), device=device,
                        inversion_policy="raise").load_particles(x, vol)
        solver.set_material(vonmises(**law, density=CONFIG["density"]))
    solver.add_plane(point=(0, 0, CONFIG["floor"]), normal=(0, 0, 1),
                     surface="separable", friction=CONFIG["floor_friction"])
    sdf = cylinder_sdf(CONFIG["radius"], CONFIG["finger_height"])
    pose = centers(0., CONFIG["opening"], CONFIG["hover_bottom"])
    handles = [solver.add_sdf_collider(sdf, center=c, band=CONFIG["band"],
               surface="separable", friction=CONFIG["tool_friction"]) for c in pose]
    data = dict(initial=x, vol0=vol, floor=np.array(CONFIG["floor"]),
                n_grid=np.array(n_grid), dt=np.array(dt), gaps_mm=np.array(gaps_mm),
                angles_deg=np.tile([0., 90.], len(gaps)//2))
    elapsed = 0.; frames = []; phase_ids = []; forces = []; times = []; phase_records = []

    def move(end, duration, phase, angle_start=None, angle_end=None):
        nonlocal pose, elapsed
        start = pose.copy(); count = max(1, int(np.ceil(duration/tick)))
        phase_records.append(dict(name=phase, start_s=elapsed, duration_s=count*tick,
                                  start_centers=start.tolist(), end_centers=end.tolist()))
        for j in range(count):
            if angle_start is None:
                next_pose = start+(end-start)*(j+1)/count
            else:
                next_pose = centers(angle_start+(angle_end-angle_start)*(j+1)/count,
                                    CONFIG["opening"], CONFIG["hover_bottom"])
            velocity = (next_pose-pose)/tick
            for h, c, v in zip(handles, pose, velocity, strict=True):
                solver.set_sdf_pose(h, center=c, velocity=v)
                solver.reset_sdf_force(h)
            solver.step(dt, substeps=substeps)
            force = np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles])
            # Full control-rate poses and reaction forces, without particle trajectories.
            frames.append(next_pose.copy()); forces.append(force); times.append(elapsed+tick)
            phase_ids.append(len(phase_records)-1)
            pose = next_pose; elapsed += tick

    for i, gap in enumerate(gaps):
        angle = (i % 2)*np.pi/2
        if i:
            previous = ((i-1) % 2)*np.pi/2
            move(centers(angle, CONFIG["opening"], CONFIG["hover_bottom"]),
                 abs(angle-previous)/CONFIG["rotation_speed"], f"{i}:rotate", previous, angle)
        move(centers(angle, CONFIG["opening"], CONFIG["finger_bottom"]),
             (CONFIG["hover_bottom"]-CONFIG["finger_bottom"])/CONFIG["vertical_speed"], f"{i}:approach")
        move(centers(angle, gap, CONFIG["finger_bottom"]),
             (CONFIG["opening"]-gap)/(2*CONFIG["finger_speed"]), f"{i}:close")
        data[f"stage_{i}_pressed"] = solver.x().copy()
        data[f"stage_{i}_centers"] = pose.copy()
        move(centers(angle, CONFIG["opening"], CONFIG["finger_bottom"]),
             (CONFIG["opening"]-gap)/(2*CONFIG["finger_speed"]), f"{i}:open")
        move(centers(angle, CONFIG["opening"], CONFIG["hover_bottom"]),
             (CONFIG["hover_bottom"]-CONFIG["finger_bottom"])/CONFIG["vertical_speed"], f"{i}:withdraw")
        data[f"stage_{i}_released"] = solver.x().copy()
    data["x_after_withdrawal"] = solver.x().copy()
    move(pose.copy(), CONFIG["release_s"], "final:wait")
    data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
                inverted_count=np.array(solver.inverted_count()),
                time=np.array(times), tool_centers=np.array(frames), reaction_force=np.array(forces),
                phase_id=np.array(phase_ids))
    if int(data["inverted_count"]) or not all(np.all(np.isfinite(v)) for v in data.values()):
        raise RuntimeError("Invalid X-shaping rollout")
    return data, phase_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT); p.add_argument("--model", default="true_A")
    p.add_argument("--gaps", type=float, nargs="+", default=[36., 36.])
    p.add_argument("--device", default="cuda:0"); p.add_argument("--grid", type=int, default=64)
    p.add_argument("--dt", type=float, default=.00005)
    p.add_argument("--E", type=float, help="Exploratory material modulus in Pa")
    p.add_argument("--yield-stress", type=float, help="Exploratory yield threshold in Pa")
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    model = json.loads((SOURCE/"models.json").read_text())[args.model]
    if args.E is not None:
        model["E"] = args.E
    if args.yield_stress is not None:
        model["yield_stress"] = args.yield_stress
    config = dict(config=CONFIG, model=args.model, law=model, gaps_mm=args.gaps,
                  n_grid=args.grid, dt=args.dt, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    stem = args.out/(args.model+"_"+"_".join(f"{g:g}" for g in args.gaps))
    if stem.with_suffix(".npz").exists():
        raise RuntimeError("Existing result; use another output folder")
    save_json(stem.with_suffix(".config.json"), config)
    start = time.monotonic()
    data, phases = execute(model, args.gaps, device=args.device, n_grid=args.grid, dt=args.dt)
    save_npz(stem.with_suffix(".npz"), data); save_json(stem.with_suffix(".phases.json"), phases)
    record = dict(**config, surface_mm=score(data), seconds=time.monotonic()-start,
                  max_force_per_finger_n=np.linalg.norm(data["reaction_force"], axis=-1).max(axis=0).tolist())
    save_json(stem.with_suffix(".json"), record); print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

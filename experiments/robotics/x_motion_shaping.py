"""Prescribed-motion X shaping: generate finger poses, simulate, and render.

Only saved finger poses drive execution. Forces are recorded, never commanded.
Planning lives in x_motion_search.py. This module contains one execution
method and has no force-control mode.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from experiments.robotics import x_shaping as geometry
from experiments.robotics.plastic_shaping_study import save_json, save_npz
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "out/x_consistent_force_study_20260911"
OUT = ROOT / "out/x_motion_shaping_20260911"
COLORS = {"A": "#7aacc7", "B": "#d39b76"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def motion(actions):
    """Build pose trajectories from openings and planar gripper poses.

    Closing has 40 ms ramps and <=20 mm/s per finger. Opening, withdrawal,
    and raised repositioning retain the original timing convention.
    """
    cfg = geometry.CONFIG
    tick = cfg["tick"]
    assert actions and all(12 <= a["gap_mm"] <= 54 for a in actions)

    def centers(angle, gap, bottom, offset):
        return geometry.centers(angle, gap, bottom) + np.r_[offset, 0.]

    first = actions[0]
    angle = np.deg2rad(first["angle_deg"])
    offset = np.array(first["offset_xy_mm"], dtype=float) / 1000
    pose = centers(angle, cfg["opening"], cfg["hover_bottom"], offset)
    start_pose = pose.copy()
    times, poses, phase_ids, pinch_ids, phases = [], [], [], [], []
    elapsed = 0.

    def phase(name):
        phases.append(dict(name=name, start_s=elapsed, start_centers=pose.tolist()))

    def advance(nxt, pinch=-1):
        nonlocal pose, elapsed
        pose = nxt
        elapsed += tick
        times.append(elapsed); poses.append(nxt.copy())
        phase_ids.append(len(phases) - 1); pinch_ids.append(pinch)
        phases[-1]["end_centers"] = pose.tolist()

    def move(end, duration, name):
        phase(name)
        start = pose.copy()
        count = max(1, int(np.ceil(duration / tick)))
        for j in range(count):
            advance(start + (end - start) * (j + 1) / count)

    for i, action in enumerate(actions):
        new_angle = np.deg2rad(action["angle_deg"])
        new_offset = np.array(action["offset_xy_mm"], dtype=float) / 1000
        if i:
            phase(f"{i}:reposition")
            count = round(1. / tick)
            for j in range(count):
                alpha = (j + 1) / count
                advance(centers(angle + (new_angle - angle) * alpha, cfg["opening"],
                                cfg["hover_bottom"], offset + (new_offset - offset) * alpha))
        angle, offset = new_angle, new_offset
        move(centers(angle, cfg["opening"], cfg["finger_bottom"], offset),
             (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], f"{i}:lower")
        phase(f"{i}:close")
        distance = (cfg["opening"] - action["gap_mm"] / 1000) / 2
        ramps = round(.04 / tick)
        count = int(np.ceil(distance / (.020 * tick))) + ramps
        j = np.arange(count) + .5
        weights = np.minimum(np.minimum(j / ramps, (count - j) / ramps), 1.)
        speed = weights * distance / (tick * weights.sum())
        assert speed.max() <= .020 + 1e-12
        for v in speed:
            gap = np.linalg.norm(pose[1] - pose[0]) - 2 * cfg["radius"]
            advance(centers(angle, gap - 2 * v * tick, cfg["finger_bottom"], offset), pinch=i)
        gap = np.linalg.norm(pose[1] - pose[0]) - 2 * cfg["radius"]
        np.testing.assert_allclose(gap * 1000, action["gap_mm"], atol=1e-7)
        move(centers(angle, cfg["opening"], cfg["finger_bottom"], offset),
             abs(cfg["opening"] - gap) / (2 * cfg["finger_speed"]), f"{i}:open")
        move(centers(angle, cfg["opening"], cfg["hover_bottom"], offset),
             (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], f"{i}:withdraw")
    move(pose.copy(), cfg["release_s"], "final:wait")
    for i, p in enumerate(phases):
        p["duration_s"] = (phases[i + 1]["start_s"] if i + 1 < len(phases) else elapsed) - p["start_s"]
    command = dict(time=np.array(times), tool_centers=np.array(poses), phase_id=np.array(phase_ids),
                   pinch_id=np.array(pinch_ids), start_pose=start_pose,
                   gaps_mm=np.array([a["gap_mm"] for a in actions]))
    return command, phases


def simulate(law, command, phases, initial, device="cuda:0", grid=64, dt=.00005):
    cfg = geometry.CONFIG
    tick = cfg["tick"]
    substeps = round(tick / dt)
    assert abs(substeps * dt - tick) < 1e-12
    with contextlib.redirect_stdout(io.StringIO()):
        solver = geometry.Solver(geometry.GridConfig(n_grid=grid, grid_lim=cfg["domain"]),
            device=device, inversion_policy="raise").load_particles(initial["initial"], initial["vol0"])
        solver.set_material(geometry.vonmises(**law, density=cfg["density"]))
    solver.add_plane(point=(0, 0, cfg["floor"]), normal=(0, 0, 1),
                     surface="separable", friction=cfg["floor_friction"])
    sdf = geometry.cylinder_sdf(cfg["radius"], cfg["finger_height"])
    pose = command["start_pose"].copy()
    handles = [solver.add_sdf_collider(sdf, center=cp, band=cfg["band"],
               surface="separable", friction=cfg["tool_friction"]) for cp in pose]
    forces, velocities, extents = [], [], []
    data = dict(**initial, **command)

    for j, nxt in enumerate(command["tool_centers"]):
        velocity = (nxt - pose) / tick
        for handle, cp, v in zip(handles, pose, velocity, strict=True):
            solver.set_sdf_pose(handle, center=cp, velocity=v)
            solver.reset_sdf_force(handle)
        solver.step(dt, substeps=substeps)
        forces.append(np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles]))
        velocities.append(velocity)
        pose = nxt
        if (j + 1) % 25 == 0:
            xyz = solver.x()
            assert np.all(np.isfinite(xyz))
            assert xyz[:, :2].min() > .003 and xyz[:, :2].max() < cfg["domain"] - .003
            extents.append(np.r_[command["time"][j], xyz.min(0), xyz.max(0)])
        if j + 1 == len(command["time"]) or command["phase_id"][j + 1] != command["phase_id"][j]:
            name = phases[int(command["phase_id"][j])]["name"]
            if name.endswith(":close") or name.endswith(":withdraw"):
                data[name.replace(":", "_")] = solver.x().copy()
    data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
        reaction_force=np.array(forces), tool_velocity=np.array(velocities),
        extent_samples=np.array(extents), inverted_count=np.array(solver.inverted_count()))
    assert data["inverted_count"] == 0 and all(np.all(np.isfinite(v)) for v in data.values())
    return data


def render(mesh, scale, color, mask=False):
    plotter = pv.Plotter(off_screen=True, window_size=(650, 700))
    plotter.set_background("white")
    plotter.add_mesh(mesh, color="black" if mask else color, lighting=not mask,
        smooth_shading=True, split_sharp_edges=True, ambient=.4, diffuse=.7, specular=.1)
    if not mask:
        plotter.add_mesh(pv.Plane(center=(.08, .08, .0099), i_size=.13, j_size=.13),
                         color="#f1f2f2", lighting=False)
    focal = np.array([.08, .08, .027])
    plotter.camera_position = [focal + [0, -.13, .24], focal, (0, 1, 0)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = scale
    if not mask:
        plotter.enable_anti_aliasing("ssaa")
    picture = plotter.screenshot(return_img=True)
    plotter.close()
    return picture

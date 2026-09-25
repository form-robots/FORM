"""Independent replay, rounded controls, and numerical checks for a selected plan."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.robotics.hex_shaping_control import snapshot_sources, target_prism
from experiments.robotics.hex_shaping_pilot import ANGLES, BAND, HALF, box_sdf, jaw_pose
from experiments.robotics.plastic_shaping_study import chamfer_mm, save_json, save_npz
from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block


def execute(law, gaps_mm, cycles, *, device, n_grid=48, dt=1e-4):
    """Replay the jaw commands, retaining the simulator for a longer release check."""
    grid = GridConfig(n_grid=n_grid, grid_lim=.30)
    pos, vol0, floor = block(grid, size=(.12, .08, .06), ppc=2, seed=0)
    tick, release, opening, speed = .0008, .30, .160, .10
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(grid, device=device, inversion_policy="raise").load_particles(pos, vol0)
        solver.set_material(vonmises(**law, density=1000.0))
    solver.add_plane(point=(0, 0, floor), normal=(0, 0, 1), surface="sticky")
    sdf = box_sdf()
    data = dict(initial=pos, vol0=vol0, floor=np.array(floor), half=HALF,
                angles=np.tile(ANGLES, cycles), gaps=np.tile(np.asarray(gaps_mm)/1000, cycles),
                dt=np.array(dt), n_grid=np.array(n_grid))
    elapsed = 0.0
    for stage, (angle, gap) in enumerate(zip(data["angles"], data["gaps"], strict=True)):
        centers, normal, quat = jaw_pose(angle, opening, floor)
        count = int(np.ceil((opening-gap)/(2*speed*tick)))
        stop = elapsed+count*tick
        velocities = np.array([1, -1])[:, None]*normal*(opening-gap)/(2*count*tick)
        handles = [solver.add_sdf_collider(sdf, center=c, quat=quat, band=BAND,
                   surface="sticky", friction=0., start_time=elapsed-.5*dt,
                   end_time=stop-.5*dt) for c in centers]
        for step in range(count):
            for h, center, velocity in zip(handles, centers+step*tick*velocities, velocities, strict=True):
                solver.set_sdf_pose(h, center=center, velocity=velocity)
            solver.step(dt, substeps=round(tick/dt))
        data[f"stage_{stage}_pressed"] = solver.x().copy()
        data[f"stage_{stage}_centers"] = jaw_pose(angle, gap, floor)[0]
        data[f"stage_{stage}_quat"] = quat
        for _ in range(round(release/tick)):
            solver.step(dt, substeps=round(tick/dt))
        data[f"stage_{stage}_released"] = solver.x().copy()
        elapsed = stop+release
    data.update(x=solver.x().copy(), v=solver.v().copy(), elapsed=np.array(elapsed))
    for _ in range(round(.70/tick)):
        solver.step(dt, substeps=round(tick/dt))
    data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
                inverted_count=np.array(solver.inverted_count()))
    if data["inverted_count"] or not all(np.all(np.isfinite(x)) for x in data.values()):
        raise RuntimeError("Invalid independent execution")
    return data


def run(args):
    source = args.plan.resolve()
    protocol = json.loads((source/"protocol.json").read_text())
    best = json.loads((source/"best.json").read_text())
    rounded = np.round(best["gaps_mm"]).tolist()
    checks = dict(exact=dict(gaps_mm=best["gaps_mm"], n_grid=48, dt=1e-4),
                  rounded=dict(gaps_mm=rounded, n_grid=48, dt=1e-4),
                  half_dt=dict(gaps_mm=rounded, n_grid=48, dt=5e-5),
                  grid64=dict(gaps_mm=rounded, n_grid=64, dt=1e-4))
    folder = args.out.resolve()
    snapshot_sources(folder, dict(source_plan=str(source), material=protocol["material"],
                                 law=protocol["law"], cycles=protocol["cycles"],
                                 checks=checks, total_final_release_s=1.0))
    rel = "experiments/robotics/hex_shaping_check.py"
    dest = folder/"source_snapshot"/rel
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Verification source changed; use a fresh output directory")
    shutil.copy2(Path(__file__), dest)
    hashes = json.loads((folder/"source_sha256.json").read_text())
    hashes[rel] = digest
    save_json(folder/"source_sha256.json", hashes)
    summary = {}
    for tag, settings in checks.items():
        started = time.monotonic()
        path = folder/f"{tag}.npz"
        if path.exists():
            d = dict(np.load(path))
        else:
            d = execute(protocol["law"], cycles=protocol["cycles"], device=args.device, **settings)
            save_npz(path, d)
        target = target_prism(settings["n_grid"], protocol["diameter_m"])
        save_npz(folder/f"target_{settings['n_grid']}.npz", target)
        result = dict(error_mm=chamfer_mm(d["x"], target["x"]),
                      error_after_1s_mm=chamfer_mm(d["x_after_1s"], target["x"]),
                      shape_change_during_extra_wait_mm=chamfer_mm(d["x"], d["x_after_1s"]),
                      rms_speed_after_03s_m_s=float(np.sqrt(np.mean(np.sum(d["v"]**2, axis=1)))),
                      rms_speed_after_1s_m_s=float(np.sqrt(np.mean(np.sum(d["v_after_1s"]**2, axis=1)))),
                      initial_volume_m3=float(d["vol0"].astype(float).sum()),
                      seconds=time.monotonic()-started)
        if tag == "exact":
            result["independent_replay_difference_mm"] = chamfer_mm(d["x"], np.load(source/"execution.npz")["x"])
            assert result["independent_replay_difference_mm"] < .05, result
        summary[tag] = result
        save_json(folder/"checks.json", summary)
        print(json.dumps(dict(check=tag, **result)), flush=True)
    save_json(folder/"selected_controls.json", dict(gaps_mm=rounded, angles_deg=ANGLES,
              cycles=protocol["cycles"], law=protocol["law"], speed_mm_s=100,
              opening_mm=160, inter_squeeze_release_s=.30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

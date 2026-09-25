"""Run a fixed-viscosity angle sweep, then forward-check five new target pours.

All generated motions and scene parameters come from the 60-degree recording.
The measured endpoints of the 45/50-degree recordings are not planner inputs.
Run as a module with MUJOCO_GL=egl; see out/pour_angle_handoff/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from examples import pour_recorded_twin as twin
from scipy.spatial.transform import Rotation

from experiments.pour.pour_angle_sweep import (
    build_motion,
    interp_columns,
    invert_angles,
    write_planned_episode,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "pouring_real_data/09-04-60-2s"
GEOMETRY = ROOT / "out/pour_wf/readout_workaround/candidate_geometry_60.json"
MESH = ROOT / "out/pour_wf/09-04-60-2s/cup_render.obj"
OUTPUT = ROOT / "out/pour_angle_plan"
ETA = 4.0
GRID = 192
TARGETS = [50.0, 80.0, 110.0, 140.0, 170.0]
# An inspected 1.2-1.4 mL table spill occurs near the 110 mL target. The planning
# quantity is receiver volume, which excludes that spill. Keep it explicit rather
# than requiring the complete-catch assumption used for the 60-degree calibration.
MAX_FINAL_OUTSIDE_ML = 2.0


def angle_name(angle):
    return f"angle_{angle:06.2f}"


def check_motion():
    """Verify the reference identity, phase joins, joint range and target FK."""
    motion, reference = build_motion(REFERENCE, MESH)
    geometry = json.loads(GEOMETRY.read_text())
    checks = []
    for angle in [45.0, 50.0, 55.0, 60.0, 63.0, 65.0]:
        t = motion.sample_times(angle)
        ack, ret, done = motion.timing(angle)
        episode = dict(reference, ts=t + twin.PRE_ROLL, qs=motion.joints(t, angle),
                       widths=np.interp(motion.reference_clock(t, angle), motion.reference_t,
                                        motion.reference_width),
                       t_hold=ack + twin.PRE_ROLL, t_return_start=ret + twin.PRE_ROLL,
                       t_return_done=done + twin.PRE_ROLL,
                       duration=done + motion.end_s - motion.return_ack_s + twin.PRE_ROLL)
        arm = twin.RecordedPanda(episode, MESH, height=64, width=64, max_geom=4000,
                                 cup_reference_pos=geometry["cup_reference_pos"],
                                 cup_reference_quat=geometry["cup_reference_quat"])
        cup_angle = arm.tilt_degrees(arm.cup_pose_at(ack + twin.PRE_ROLL)[1])
        r_hand = twin.quat_to_mat(arm.data.xquat[arm.ee])
        tcp = arm.data.xpos[arm.ee] + r_hand @ arm.TCP_LOCAL
        desired_pos = [0.5 - angle / 1200, 0, 0.1 + angle / 1200]
        c, s = np.cos(np.radians(angle / 2)), np.sin(np.radians(angle / 2))
        desired_rot = Rotation.from_quat([0.5*(c+s), 0.5*(c-s), -0.5*(c-s), 0.5*(c+s)])
        q = motion.joints(np.linspace(-1, done + 0.8, 1500), angle)
        limits = arm.model.jnt_range[:7]
        margin = float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min())
        join = max(float(np.max(np.abs(np.diff(
            motion.joints([b - 1e-8, b + 1e-8], angle), axis=0)))) for b in [0, ack, ret, done])
        row = dict(angle_deg=angle, tcp_mm_error=float(np.linalg.norm(tcp - desired_pos)*1000),
                   orientation_deg_error=float(np.degrees(
                       (Rotation.from_matrix(r_hand) * desired_rot.inv()).magnitude())),
                   joint_limit_margin_rad=margin, join_max_rad=join, cup_angle_deg=cup_angle)
        if angle == 60:
            times = np.linspace(-0.4, reference["duration"], 3000)
            error = interp_columns(times, reference["ts"], reference["qs"]) - interp_columns(
                times, episode["ts"], episode["qs"])
            row["identity_max_joint_error_rad"] = float(np.max(np.abs(error)))
            if row["identity_max_joint_error_rad"] > 1e-8:
                raise ValueError("Reference trajectory changed")
        arm.close()
        if (margin <= 0 or join > 1e-6 or row["tcp_mm_error"] > 1.5
                or row["orientation_deg_error"] > 0.2):
            raise ValueError(f"Motion preflight failed: {row}")
        checks.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "motion_checks.json").write_text(json.dumps(checks, indent=2))
    return checks


def fingerprint(angle):
    paths = [REFERENCE / "states.jsonl", REFERENCE / "actions.jsonl",
             REFERENCE / "meta.json", GEOMETRY, Path(__file__).with_name("pour_angle_sweep.py"),
             Path(twin.__file__), ROOT / "src/warpmpm/geometry/measuring_cup.py"]
    return dict(angle_deg=float(angle), eta_pa_s=ETA, n_grid=GRID, initial_volume_ml=300.0,
                files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths})


def simulate(angle, device):
    if abs(angle - round(angle, 2)) > 1e-8:
        raise ValueError("Command angles must be rounded to 0.01 degrees before simulation")
    result_path = OUTPUT / "simulations" / angle_name(angle) / "result.json"
    provenance = fingerprint(angle)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["provenance"] == provenance:
            print(f"Cached {angle:.2f} degrees: {result['receiver_ml']:.3f} mL", flush=True)
            return result
        raise RuntimeError(f"Stale result: move {result_path.parent} before rerunning")
    motion, ep = build_motion(REFERENCE, MESH)
    episode = OUTPUT / "episodes" / angle_name(angle)
    planning = write_planned_episode(REFERENCE, episode, motion, ep, angle)
    geometry = json.loads(GEOMETRY.read_text())
    scene = {k: geometry[k] for k in
             ["cup_reference_pos", "cup_reference_quat", "receiver_xy", "table_z"]}
    twin.OUT_ROOT = OUTPUT / "simulations"
    simulation = twin.run(episode, device=device, n_grid=GRID, video=False,
                          side_by_side=False, rebake=True, eta=ETA, volume_ml=300.0, **scene)
    rows = simulation["rows"]
    row = rows[-1]
    count = sum(rows[0][key] for key in ["n_src", "n_rcv", "n_air_spill"])
    count_drift = max(abs(sum(r[k] for k in ["n_src", "n_rcv", "n_air_spill"]) - count)
                      for r in rows)
    if count_drift:
        raise RuntimeError(f"Particle ledger changed by {count_drift}")
    receiver = 300.0 * row["n_rcv"] / count
    depletion = 300.0 * (1.0 - row["n_src"] / count)
    spill = 300.0 * row["n_air_spill"] / count
    result = dict(provenance=provenance, planning=planning, receiver_ml=receiver,
                  source_depletion_ml=depletion, outside_both_cups_ml=spill,
                  simulated_particle_count=int(count), particle_count_drift=int(count_drift),
                  final_time_s=float(row["t"]),
                  status="prospective prediction; not experimentally validated")
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def run_batch(angles):
    """Two independent GPU workers; the parent never initializes a GPU solver."""
    results = []
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(angles), 2):
        workers = []
        for index, angle in enumerate(angles[offset:offset + 2]):
            log_path = log_dir / f"{angle_name(angle)}.log"
            log = log_path.open("w")
            command = [sys.executable, "-u", "-m", "experiments.pour.pour_angle_plan",
                       "--single-angle", str(angle), "--device", f"cuda:{index}"]
            env = dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=str(ROOT))
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT)
            workers.append((angle, worker, log, log_path))
            print(f"Simulating {angle:.2f} degrees on cuda:{index}", flush=True)
        failures = []
        for angle, worker, log, log_path in workers:
            rc = worker.wait()
            log.close()
            if rc:
                failures.append(str(log_path))
            else:
                path = OUTPUT / "simulations" / angle_name(angle) / "result.json"
                result = json.loads(path.read_text())
                results.append(result)
                print(f"{angle:.2f} degrees: receiver {result['receiver_ml']:.3f} mL; "
                      f"outside {result['outside_both_cups_ml']:.3f} mL", flush=True)
        if failures:
            raise RuntimeError("Simulation failed; see " + ", ".join(failures))
    return results


def bracket_candidate(angles, volumes, target):
    """Refine an actual crossing without imposing global monotonicity on noisy MPM."""
    angles, volumes = np.asarray(angles), np.asarray(volumes)
    if len(angles) != len(volumes) or np.any(np.diff(angles) <= 0):
        raise ValueError("Samples must be ordered by distinct angles")
    crossings = np.flatnonzero((volumes[:-1] < target) & (volumes[1:] > target))
    if not len(crossings):
        raise ValueError(f"No increasing local bracket for {target} mL")
    i = min(crossings, key=lambda k: angles[k + 1] - angles[k])
    lo, hi = angles[i:i + 2]
    candidate = lo + (hi - lo) * (target - volumes[i]) / (volumes[i + 1] - volumes[i])
    candidate = round(float(np.clip(candidate, lo + 0.01, hi - 0.01)), 2)
    if candidate in angles or not lo < candidate < hi:
        raise ValueError(f"Angle resolution exhausted near target {target} mL")
    return candidate


def collect_existing():
    results = []
    for path in sorted((OUTPUT / "simulations").glob("*/result.json")):
        result = json.loads(path.read_text())
        if result["provenance"] != fingerprint(result["provenance"]["angle_deg"]):
            raise ValueError(f"Stale simulation inputs at {path}")
        results.append(result)
    return sorted(results, key=lambda r: r["provenance"]["angle_deg"])


def refine_existing():
    for iteration in range(9):
        results = collect_existing()
        captured = [r for r in results if r["outside_both_cups_ml"] <= MAX_FINAL_OUTSIDE_ML]
        closest = [min(captured, key=lambda r, target=target: abs(r["receiver_ml"] - target))
                   for target in TARGETS]
        unresolved = [t for t, r in zip(TARGETS, closest, strict=True)
                      if abs(r["receiver_ml"] - t) > 1.5]
        (OUTPUT / "all_forward_results.json").write_text(json.dumps(results, indent=2))
        if not unresolved:
            break
        if iteration == 8:
            raise ValueError(f"Targets remain unresolved: {unresolved}")
        angles = [r["provenance"]["angle_deg"] for r in results]
        volumes = [r["receiver_ml"] for r in results]
        requested = sorted({bracket_candidate(angles, volumes, target) for target in unresolved})
        print(f"Refining targets {unresolved} at angles {requested}", flush=True)
        run_batch(requested)
    table = []
    for target, result in zip(TARGETS, closest, strict=True):
        angle = result["provenance"]["angle_deg"]
        metrics = np.genfromtxt(OUTPUT / "simulations" / angle_name(angle) / "metrics.csv",
                                delimiter=",", names=True)
        tail = metrics["t"] >= metrics["t"][-1] - 0.5
        endpoint_variation = float(np.ptp(metrics["n_rcv"][tail])
                                   / result["simulated_particle_count"] * 300)
        if endpoint_variation > 0.5:
            raise ValueError(f"Receiver endpoint is still changing at {angle} degrees")
        table.append(dict(target_ml=target, command_angle_deg=angle,
                          pour_command_duration_s=angle / 10, dwell_s=2.0,
                          return_command_duration_s=2.0,
                          predicted_receiver_ml=result["receiver_ml"],
                          target_error_ml=result["receiver_ml"] - target,
                          predicted_source_depletion_ml=result["source_depletion_ml"],
                          outside_both_cups_ml=result["outside_both_cups_ml"],
                          endpoint_variation_last_half_second_ml=endpoint_variation,
                          predicted_peak_cup_tilt_deg=float(metrics["tilt_deg"].max()),
                          extrapolates_recorded_angle=angle > 60.0))
    (OUTPUT / "table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2), flush=True)
    if any(abs(row["target_error_ml"]) > 1.5 for row in table):
        raise RuntimeError("A forward prediction misses by >1.5 mL: refine before export")
    if any(row["outside_both_cups_ml"] > MAX_FINAL_OUTSIDE_ML for row in table):
        raise RuntimeError("Predicted catch loss exceeds the inspected range")
    reversals = []
    for left, right in pairwise(results):
        if right["receiver_ml"] < left["receiver_ml"]:
            reversals.append(dict(from_angle_deg=left["provenance"]["angle_deg"],
                                  to_angle_deg=right["provenance"]["angle_deg"],
                                  volume_change_ml=right["receiver_ml"] - left["receiver_ml"]))
    diagnostic = dict(local_decreases=reversals, number_of_forward_simulations=len(results),
                      method="local bracket refinement; all evaluations retained",
                      note="Finite-grid response need not be strictly monotone. "
                           "No volume sorting, isotonic correction or "
                           "experimental endpoint fitting.")
    (OUTPUT / "refinement_diagnostics.json").write_text(json.dumps(diagnostic, indent=2))


def plan():
    check_motion()
    sweep = run_batch([45.0, 50.0, 55.0, 60.0, 63.0])
    (OUTPUT / "sweep.json").write_text(json.dumps(sweep, indent=2))
    angles = [r["provenance"]["angle_deg"] for r in sweep]
    volumes = [r["receiver_ml"] for r in sweep]
    requested = np.round(invert_angles(angles, volumes, TARGETS), 2).tolist()
    print(f"Initial target angles: {requested}", flush=True)
    run_batch(requested)
    refine_existing()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-angle", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refine-existing", action="store_true")
    arguments = parser.parse_args()
    if arguments.single_angle is not None:
        simulate(arguments.single_angle, arguments.device)
    elif arguments.refine_existing:
        refine_existing()
    else:
        plan()

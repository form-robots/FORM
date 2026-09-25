"""Isolated full MPM replay with fixed weak viscosity and explicit source contact.

This forward runner does not read any measured receiver endpoint. A separate
calibration driver may use ONLY the 60-degree measurement to choose one boundary
parameter. Overnight target planning uses the previously frozen contact setting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from examples import pour_recorded_twin as twin
from experiments.pour.pour_navier_quadratic_wall import install_navier_quadratic_wall
from experiments.pour.pour_navier_traction_wall import install_navier_traction_wall
from experiments.pour.pour_angle_sweep import build_motion, write_planned_episode
from warpmpm.geometry.measuring_cup import make_cup_mesh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_navier_calibration"
REFERENCE = ROOT / "pouring_real_data/09-04-60-2s"
GEOMETRY = ROOT / "out/pour_weakform_recovery/identified/geometry.json"
OBSERVATIONS = ROOT / "out/pour_three_video_identification/09-04-60-2s/observations.npz"
MESH = ROOT / "out/pour_wf/09-04-60-2s/cup_render.obj"
ETA = 3.4392377844275503


def planned_motion(max_angle=60.):
    """Retain the recorded-path mapping; explicitly extend its timing to 70°.

    The existing motion planner permits 65°. Above that limit its unchanged
    angle/10 inclination timing continues linearly. Neither the spatial-path
    extrapolation nor any motion below 65° is changed.
    """
    if max_angle not in (60., 62., 70.):
        raise ValueError("Unsupported explicit planning bound")
    motion, ep = build_motion(REFERENCE, MESH)
    original_timing = motion.timing

    def timing(angle):
        if not np.isfinite(angle) or not 35. <= angle <= max_angle:
            raise ValueError("Angle exceeds the explicitly selected planning bound")
        if angle <= 65.:
            return original_timing(angle)
        delta = motion.command_duration_s * (angle - 65.) / motion.reference_angle_deg
        return tuple(t + delta for t in original_timing(65.))

    motion.timing = timing
    return motion, ep


def fixed_scene(geometry):
    ep = twin.load_episode(REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ep, MESH, height=64, width=64, max_geom=4000,
                             cup_reference_pos=geometry["cup_reference_pos"],
                             cup_reference_quat=geometry["cup_reference_quat"])
    hand_cup, grasp = arm._hand_cup.copy(), arm._grasp.copy()
    receiver = np.r_[geometry["receiver_xy"], geometry["table_z"]]
    # Freeze the crop using the largest collision shell in the intended grid pair.
    vertices, _ = make_cup_mesh(twin.SPEC, *twin.collision_extras(.35/160), n_theta=96, n_z=24)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for t in np.arange(0, arm.duration + 1e-9, 1/60):
        p, q = arm.cup_pose_at(float(t))
        points = vertices @ twin.quat_to_mat(q).T + p
        lower, upper = np.minimum(lower, points.min(0)), np.maximum(upper, points.max(0))
    points = vertices @ twin.quat_to_mat(twin.Q_RCV).T + receiver
    lower, upper = np.minimum(lower, points.min(0)), np.maximum(upper, points.max(0))
    ideal = .5 * (.35 - lower - upper)
    original = twin.world_to_mpm_offset(arm, receiver, .7/384)
    offset = original - np.rint((original - ideal)/(.7/384)) * (.7/384)
    arm.close()
    return offset, lower, upper, hand_cup, grasp


def run(args):
    source_friction = float(getattr(args, "source_friction", .05))
    if not np.isfinite(source_friction) or not 0. <= source_friction <= .2:
        raise ValueError("Original-contact exploration is bounded to coefficients [0, 0.2]")
    if args.grid not in (160, 192):
        raise ValueError("This bounded trial authorizes only the 160/192 grid pair")
    if args.wall != "original-separable" and (args.slip_mm is None or not np.isfinite(args.slip_mm) or not .25 <= args.slip_mm <= 4.):
        raise ValueError("Slip outside the benchmarked trial range [0.25, 4] mm")
    geometry = json.loads(GEOMETRY.read_text())
    # Recompute the documented weak fit from the 60-degree observations only.
    from experiments.pour.pour_weakform_recovery import fit
    pour, ret, _ = twin.recorded_pour_actions(REFERENCE)
    window = [pour["t_ack"] - pour["t_send"] + .15, ret["t_send"] - pour["t_send"] - .15]
    identified_eta = fit(dict(np.load(OBSERVATIONS)), window)[0]
    if abs(identified_eta - ETA) > 1e-10:
        raise ValueError("The fixed one-video viscosity no longer reproduces")
    angle_label = "recorded60" if args.angle is None else f"planned{args.angle:g}"
    slip_label = "NA" if args.wall == "original-separable" else f"{args.slip_mm:.6f}"
    name = f"{angle_label}_b{slip_label}_n{args.grid}_phase{args.phase:g}_dt{args.dt_scale:g}"
    if args.wall == "original-separable" and source_friction != .05:
        name += f"_mu{source_friction:.6f}"
    destination = OUT / "replays" / args.wall / name
    destination.mkdir(parents=True, exist_ok=False)
    episode = REFERENCE
    if args.angle is not None:
        max_angle = float(getattr(args, "max_angle", 60.))
        if max_angle not in (60., 62., 70.) or not 35. <= args.angle <= max_angle:
            raise ValueError("Planned angle outside the explicitly selected trial bounds")
        motion, ep = planned_motion(max_angle)
        episode = destination / "planned_episode"
        write_planned_episode(REFERENCE, episode, motion, ep, args.angle)
    offset, lower, upper, hand_cup, grasp = fixed_scene(geometry)
    dx = .35/args.grid
    offset += args.phase * dx * np.array([1., 0., 1.])
    margin = np.minimum(lower + offset, .35 - upper - offset)
    if margin.min() < .03:
        raise ValueError("Insufficient crop margin")
    paths = [Path(__file__), ROOT / "experiments/pour/pour_navier_quadratic_wall.py",
             ROOT / "experiments/pour/pour_navier_traction_wall.py",
             ROOT / "experiments/pour/pour_weakform_identify.py",
             ROOT / "experiments/pour/pour_weakform_recovery.py",
             ROOT / "experiments/pour/pour_angle_sweep.py", Path(twin.__file__), GEOMETRY, OBSERVATIONS,
             *sorted((ROOT / "src/warpmpm").rglob("*.py")),
             *[REFERENCE / k for k in ["states.jsonl", "actions.jsonl", "meta.json"]]]
    protocol = dict(name=name, identification_episode=REFERENCE.name, simulation_episode=episode.name,
                    planned_angle_deg=args.angle, eta_pa_s=ETA, initial_volume_ml=300.,
                    planned_angle_limit_deg=float(getattr(args, "max_angle", 60.)),
                    slip_length_mm=None if args.wall == "original-separable" else args.slip_mm,
                    source_wall=args.wall, source_coulomb_friction=source_friction if args.wall == "original-separable" else 0.,
                    receiver_wall="separable, friction=0.05",
                    weak_fit_window_s=window, viscosity_fitted_by_simulator=False,
                    measured_endpoints_read_by_forward_runner=[], validation_outcomes_used=[],
                    extent_m=.35, n_grid=args.grid, dx_m=dx, sdf_resolution=256,
                    phase_xz_cells=args.phase, dt_scale=args.dt_scale,
                    world_to_grid_offset=offset.tolist(), geometry_margin_m=margin.tolist(),
                    purpose="Exploratory single-parameter forward replay; numerical acceptance required",
                    input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    old = (twin.Solver, twin.RecordedPanda, twin.OUT_ROOT, twin.GRID_LIM, twin.SDF_RES,
           twin.world_to_mpm_offset, twin.substeps_per_tick)
    clearances = []

    class TrialSolver(old[0]):
        def add_sdf_collider(self, *positional, **keywords):
            index = getattr(self, "_navier_source_count", 0)
            if index == 0:
                keywords = dict(keywords, surface="separable", friction=source_friction if args.wall == "original-separable" else 0.)
            handle = super().add_sdf_collider(*positional, **keywords)
            if index == 0 and args.wall != "original-separable":
                installer = install_navier_traction_wall if args.wall == "wet-traction" else install_navier_quadratic_wall
                installer(self, handle, slip_length_m=args.slip_mm*.001, eta_pa_s=ETA)
            self._navier_source_count = index + 1
            return handle

        def step(self, *positional, **keywords):
            result = super().step(*positional, **keywords)
            x = self.x()
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite liquid positions")
            clearance = float(min(x.min(), .35-x.max()))
            if clearance <= 3*dx:
                np.savez_compressed(destination / "failure_state.npz", x=x, v=self.v())
                (destination / "failure.json").write_text(json.dumps(dict(
                    reason="Liquid reached artificial crop boundary", simulation_time_s=float(self._sim.time),
                    minimum_xyz=x.min(0).tolist(), maximum_xyz=x.max(0).tolist(),
                    maximum_speed_m_s=float(np.linalg.norm(self.v(), axis=1).max()),
                    calibration_result=False), indent=2) + "\n")
                raise ValueError("Liquid reached the artificial crop boundary")
            clearances.append(clearance)
            return result

    class TrialPanda(old[1]):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self._hand_cup, self._grasp = hand_cup.copy(), grasp.copy()

    twin.Solver, twin.RecordedPanda, twin.OUT_ROOT = TrialSolver, TrialPanda, destination
    twin.GRID_LIM, twin.SDF_RES = .35, 256
    twin.world_to_mpm_offset = lambda arm, receiver, dx: offset.copy()
    twin.substeps_per_tick = lambda liquid, dx, dt: int(np.ceil(old[-1](liquid, dx, dt) / args.dt_scale))
    start = time.monotonic()
    try:
        simulation = twin.run(episode, device=args.device, n_grid=args.grid, video=False,
                               side_by_side=False, rebake=True, eta=ETA, volume_ml=300., **geometry)
    finally:
        (twin.Solver, twin.RecordedPanda, twin.OUT_ROOT, twin.GRID_LIM, twin.SDF_RES,
         twin.world_to_mpm_offset, twin.substeps_per_tick) = old
    rows = simulation["rows"]
    counts = np.array([[r["n_src"], r["n_rcv"], r["n_air_spill"]] for r in rows])
    if np.any(counts < 0) or np.ptp(counts.sum(1)):
        raise ValueError("Invalid particle ledger")
    n = int(counts[0].sum())
    tail = [r["n_rcv"] for r in rows if r["t"] >= rows[-1]["t"] - .5]
    result = dict(**protocol, receiver_ml=300.*counts[-1, 1]/n,
                  source_depletion_ml=300.*(1-counts[-1, 0]/n), outside_ml=300.*counts[-1, 2]/n,
                  tail_variation_ml=300.*float(np.ptp(tail))/n, particle_count=n,
                  minimum_boundary_clearance_m=min(clearances), elapsed_s=time.monotonic()-start,
                  status="Forward result only; calibration and numerical checks are separate")
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "input_sha256"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, default=160)
    p.add_argument("--phase", type=float, choices=[0., .5], default=0.)
    p.add_argument("--slip-mm", type=float)
    p.add_argument("--source-friction", type=float, default=.05)
    p.add_argument("--angle", type=float)
    p.add_argument("--max-angle", type=float, choices=[60., 62., 70.], default=60.,
                   help="Explicit planning bound; 70 supports the requested 180/200 mL pilot extension")
    p.add_argument("--dt-scale", type=float, choices=[1., .5, .25], default=1.)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--wall", choices=["wet-traction", "quadratic", "original-separable"], default="wet-traction")
    run(p.parse_args())

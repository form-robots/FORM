"""Plan final-angle pours from one recorded joint trajectory.

The reference's spatial path and inclination/return progress profiles are retained.
Inclination duration follows the recorded command rule (angle / 10 degrees/s),
including the reference acknowledgement overhead. Dwell and return timing stay
fixed. Generated trajectories are predictions, not additional measured recordings.

The 60-degree reference is reproduced exactly by the joint mapping. For another
angle the recorded forward and return joint paths are evaluated at scaled angular
progress. A smooth correction joins the paths continuously at the dwell. Limited
extrapolation uses the terminal spatial slope; FK checks report the resulting pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from examples.pour_recorded_twin import (
    HOLD_SECONDS,
    PRE_ROLL,
    RecordedPanda,
    load_episode,
    recorded_pour_actions,
)
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def interp_columns(x, xp, values):
    x = np.atleast_1d(x)
    return np.stack([np.interp(x, xp, column) for column in values.T], axis=-1)


def monotone_path(progress, joints):
    """Coalesce tiny progress jitter before using it as a spatial coordinate."""
    order = np.argsort(progress, kind="stable")
    p, q = np.asarray(progress)[order], np.asarray(joints)[order]
    keep = np.r_[True, np.diff(p) > 1e-7]
    p, q = p[keep], q[keep]
    if len(p) < 8:
        raise ValueError("Too few distinct poses in the recorded motion")
    return p, q


def spatial_joints(progress, path):
    p, q = path
    progress = np.atleast_1d(progress)
    result = interp_columns(progress, p, q)
    # Estimate a slope over several moving frames, avoiding stopped-pose jitter.
    tail = (p >= 0.82) & (p <= 0.985)
    if tail.sum() < 3:
        tail = np.arange(len(p)) >= max(0, len(p) - 6)
    slope = np.polyfit(p[tail], q[tail], 1)[0]
    above = progress > p[-1]
    result[above] = q[-1] + (progress[above, None] - p[-1]) * slope
    return result


@dataclass
class AngleMotion:
    reference_t: np.ndarray
    reference_q: np.ndarray
    reference_width: np.ndarray
    forward_t: np.ndarray
    forward_progress: np.ndarray
    return_t: np.ndarray
    return_progress: np.ndarray
    forward_path: tuple
    return_path: tuple
    reference_angle_deg: float
    command_duration_s: float
    pour_ack_s: float
    return_send_s: float
    return_ack_s: float
    end_s: float

    def timing(self, angle_deg):
        if not np.isfinite(angle_deg) or not 35 <= angle_deg <= 65:
            raise ValueError("Angle must be between 35 and 65 degrees")
        scale = angle_deg / self.reference_angle_deg
        ack = self.pour_ack_s + self.command_duration_s * (scale - 1)
        ret = ack + self.return_send_s - self.pour_ack_s
        done = ret + self.return_ack_s - self.return_send_s
        return ack, ret, done

    def reference_clock(self, times, angle_deg):
        ack, ret, done = self.timing(angle_deg)
        t = np.atleast_1d(np.asarray(times, float))
        return np.select(
            [t <= 0, t <= ack, t <= ret, t <= done],
            [t, t * self.pour_ack_s / ack, self.pour_ack_s + t - ack,
             self.return_send_s + t - ret],
            default=self.return_ack_s + t - done,
        )

    def joints(self, times, angle_deg):
        ack, ret, done = self.timing(angle_deg)
        times = np.atleast_1d(np.asarray(times, float))
        clock = self.reference_clock(times, angle_deg)
        q = interp_columns(clock, self.reference_t, self.reference_q)
        scale = angle_deg / self.reference_angle_deg
        if scale == 1:
            return q
        delta_forward = (spatial_joints([scale], self.forward_path)
                         - spatial_joints([1.0], self.forward_path))[0]
        delta_return = (spatial_joints([scale], self.return_path)
                        - spatial_joints([1.0], self.return_path))[0]
        incline = (times > 0) & (times <= ack)
        p = np.interp(clock[incline], self.forward_t, self.forward_progress)
        q[incline] += (spatial_joints(p * scale, self.forward_path)
                       - spatial_joints(p, self.forward_path))
        dwell = (times > ack) & (times <= ret)
        q[dwell] += delta_forward
        returning = (times > ret) & (times <= done)
        p = np.interp(clock[returning], self.return_t, self.return_progress)
        blend = p * p * (3 - 2 * p)
        q[returning] += (spatial_joints(p * scale, self.return_path)
                         - spatial_joints(p, self.return_path)
                         + blend[:, None] * (delta_forward - delta_return))
        return q

    def sample_times(self, angle_deg, dt=0.02):
        ack, ret, done = self.timing(angle_deg)
        # Retain all mapped original breakpoints as well as a dense export grid.
        t = self.reference_t
        mapped = np.select(
            [t <= 0, t <= self.pour_ack_s, t <= self.return_send_s,
             t <= self.return_ack_s],
            [t, t * ack / self.pour_ack_s, ack + t - self.pour_ack_s,
             ret + t - self.return_send_s],
            default=done + t - self.return_ack_s,
        )
        end = done + self.end_s - self.return_ack_s
        samples = np.r_[mapped, np.arange(-PRE_ROLL - 0.5, end + 0.5, dt),
                        0.0, ack, ret, done, end]
        # Near-coincident floating-point knots cause unstable gradient diagnostics.
        return np.unique(np.round(samples, 9))


def build_motion(episode_dir, mesh_path):
    ep = load_episode(episode_dir, PRE_ROLL, HOLD_SECONDS)
    pour, _, _ = recorded_pour_actions(episode_dir)
    if float(pour.get("pour_angle_deg", 0)) <= 0:
        raise ValueError("Reference must contain its commanded pour angle")
    arm = RecordedPanda(ep, mesh_path, height=64, width=64, max_geom=4000)
    ts = ep["ts"] - ep["t_pour"]
    ack = ep["t_hold"] - ep["t_pour"]
    rstart = ep["t_return_start"] - ep["t_pour"]
    rend = ep["t_return_done"] - ep["t_pour"]
    arm.set_time(ep["t_pour"])
    r0 = Rotation.from_quat(np.roll(arm.data.xquat[arm.ee], -1))
    arm.set_time(ep["t_hold"])
    rh = Rotation.from_quat(np.roll(arm.data.xquat[arm.ee], -1))
    axis = (rh * r0.inv()).as_rotvec()
    magnitude = np.linalg.norm(axis)
    axis /= magnitude

    def phase(begin, end, increasing):
        times = np.unique(np.r_[begin, ts[(ts > begin) & (ts < end)], end])
        qs = interp_columns(times, ts, ep["qs"])
        raw = []
        for time in times:
            arm.set_time(time + ep["t_pour"])
            rotation = Rotation.from_quat(np.roll(arm.data.xquat[arm.ee], -1))
            raw.append(np.dot((rotation * r0.inv()).as_rotvec(), axis) / magnitude)
        progress = np.clip(np.asarray(raw), 0, 1)
        if increasing:
            progress[0], progress[-1] = 0, 1
            progress = np.maximum.accumulate(progress)
        else:
            progress[0], progress[-1] = 1, 0
            progress = np.minimum.accumulate(progress)
        return times, progress, monotone_path(progress, qs)

    ft, fp, forward = phase(0, ack, True)
    rt, rp, returning = phase(rstart, rend, False)
    arm.close()
    return AngleMotion(ts, ep["qs"], ep["widths"], ft, fp, rt, rp,
                       forward, returning, float(pour["pour_angle_deg"]),
                       float(pour["duration_s"]), ack, rstart, rend,
                       ep["duration"] - ep["t_pour"]), ep


def write_planned_episode(reference_dir, output_dir, motion, ep, angle_deg):
    """Write a clearly marked synthetic episode accepted by the existing twin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t = motion.sample_times(angle_deg)
    q = motion.joints(t, angle_deg)
    clock = motion.reference_clock(t, angle_deg)
    width = np.interp(clock, motion.reference_t, motion.reference_width)
    ack, ret, done = motion.timing(angle_deg)
    end = done + motion.end_s - motion.return_ack_s
    source_hash = hashlib.sha256((reference_dir / "states.jsonl").read_bytes()).hexdigest()
    meta = dict(ep["meta"])
    meta["synthetic_planned_episode"] = True
    meta["planning"] = dict(reference_episode=reference_dir.name,
                            reference_states_sha256=source_hash,
                            angle_deg=angle_deg, duration_s=angle_deg / 10,
                            dwell_s=2.0, return_command_duration_s=2.0,
                            reference_clock_ack_s=ack, reference_clock_return_send_s=ret,
                            method="scaled angular progress on the 60-degree recorded joint paths",
                            extrapolates_spatial_path=angle_deg > motion.reference_angle_deg)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    upright = [0.5, 0.5, -0.5, 0.5]
    theta = np.radians(angle_deg / 2)
    c, s = np.cos(theta), np.sin(theta)
    target_quat = [0.5 * (c + s), 0.5 * (c - s), -0.5 * (c - s), 0.5 * (c + s)]
    target_pos = [0.5 - angle_deg / 1200, 0.0, 0.1 + angle_deg / 1200]
    actions = [dict(type="execute_trajectory", synthetic=True, t_send=0.0, t_ack=ack,
                    pour_angle_deg=angle_deg, target_pos=target_pos,
                    target_quat_xyzw=target_quat, duration_s=angle_deg / 10, dwell_s=2.0),
               dict(type="go_to_pose", synthetic=True, t_send=ret, t_ack=done,
                    pour_angle_deg=0.0, target_pos=[0.5, 0.0, 0.1],
                    target_quat_xyzw=upright, duration_s=2.0),
               dict(type="go_to_pose", synthetic=True, t_send=end + 0.05, t_ack=end + 2,
                    target_quat_xyzw=upright)]
    (output_dir / "actions.jsonl").write_text("\n".join(json.dumps(a) for a in actions))
    with (output_dir / "states.jsonl").open("w") as f:
        for time, joints, w in zip(t, q, width, strict=True):
            f.write(json.dumps(dict(t=float(time), synthetic=True,
                                   state=dict(joint_position=joints.tolist(),
                                              gripper_width=float(w)))) + "\n")
    np.savetxt(output_dir / "joint_trajectory.csv", np.c_[t, q, width], delimiter=",",
               header="time_from_pour_command_s," + ",".join(f"joint_{j}" for j in range(1, 8))
                      + ",gripper_width_m", comments="", fmt="%.10f")
    return meta["planning"]


def invert_angles(angles, volumes, targets):
    """Invert an increasing simulation curve; never sort volumes to hide reversals."""
    angles, volumes = np.asarray(angles, float), np.asarray(volumes, float)
    if (len(angles) < 2 or len(angles) != len(volumes)
            or not np.isfinite(angles).all() or not np.isfinite(volumes).all()
            or np.any(np.diff(angles) <= 0) or np.any(np.diff(volumes) <= 0)):
        raise ValueError("Angle and simulated volume must both increase strictly")
    curve = PchipInterpolator(angles, volumes, extrapolate=False)
    result = []
    for target in targets:
        if not volumes[0] <= target <= volumes[-1]:
            raise ValueError(f"Target {target} mL is outside the simulated volume range")
        result.append(float(brentq(lambda a, v=target: float(curve(a)) - v,
                                   angles[0], angles[-1])))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--angle", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    motion, ep = build_motion(args.reference, args.mesh)
    info = write_planned_episode(args.reference, args.output, motion, ep, args.angle)
    print(json.dumps(info, indent=2))

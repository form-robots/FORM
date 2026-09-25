"""Transfer robot timing residuals without using measured liquid outcomes.

Keep the original 60-degree spatial joint path. Average time-at-progress
differences from the two supplied joint/action logs, in seconds, separately for
inclination and return. This is additional robot motion characterization, not
additional constitutive identification. No outcome-dependent correction is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from experiments.pour.pour_angle_sweep import build_motion
from experiments.pour import pour_navier_reference as reference

LOG_ROOT = reference.ROOT / "out/pour_physics_audit/validation_motion_47p06_52p92"
TIMING_ANGLES = (47.06, 52.92)


def progress_times(motion, progress, returning=False):
    t = motion.return_t if returning else motion.forward_t
    p = motion.return_progress if returning else motion.forward_progress
    if returning:
        t, p = t[::-1], p[::-1]
    unique, indices = np.unique(p, return_index=True)
    return np.interp(progress, unique, t[indices])


def characterize(base):
    progress = np.linspace(.02, .98, 49)
    cases = []
    for angle in TIMING_ANGLES:
        measured, _ = build_motion(LOG_ROOT / f"recorded_{angle:g}", reference.MESH)
        ack, ret, done = base.timing(angle)
        forward = progress_times(base, progress) * ack / base.pour_ack_s
        backward = progress_times(base, progress, True) - base.return_send_s
        cases.append(dict(angle_deg=angle,
            forward_residual_s=(progress_times(measured, progress)-forward).tolist(),
            return_residual_s=(progress_times(measured, progress, True)
                               - measured.return_send_s-backward).tolist(),
            ack_residual_s=measured.pour_ack_s-ack,
            hold_residual_s=(measured.return_send_s-measured.pour_ack_s)-(ret-ack),
            return_duration_residual_s=(measured.return_ack_s-measured.return_send_s)-(done-ret)))
    fields = ["forward_residual_s", "return_residual_s", "ack_residual_s",
              "hold_residual_s", "return_duration_residual_s"]
    means = {k: np.mean([r[k] for r in cases], axis=0).tolist() for k in fields}
    disagreement = {}
    for phase in ("forward", "return"):
        delta = np.array(cases[0][phase+"_residual_s"])-cases[1][phase+"_residual_s"]
        disagreement[phase] = dict(rms_seconds=float(np.sqrt(np.mean(delta**2))),
                                   max_abs_seconds=float(np.max(abs(delta))))
    return dict(method="Equal-weight mean time-at-normalized-angular-progress residual, in seconds",
        progress=progress.tolist(), cases=cases, mean=means,
        between_log_disagreement=disagreement, liquid_outcomes_used=[],
        interpretation="Two motion samples; variation is retained, not a statistical confidence interval")


@dataclass
class MeasuredTimingMotion:
    base: object
    characterization: dict

    def __getattr__(self, name):
        return getattr(self.base, name)

    def timing(self, angle):
        ack, ret, done = self.base.timing(angle)
        m = self.characterization["mean"]
        new_ack = ack+m["ack_residual_s"]
        new_ret = ret+m["ack_residual_s"]+m["hold_residual_s"]
        new_done = done+new_ret-ret+m["return_duration_residual_s"]
        return new_ack, new_ret, new_done

    def clock_knots(self, angle):
        ack, ret, done = self.base.timing(angle)
        new_ack, new_ret, new_done = self.timing(angle)
        m = self.characterization["mean"]
        p = np.asarray(self.characterization["progress"])
        ft = progress_times(self.base, p)*ack/self.base.pour_ack_s
        rt = progress_times(self.base, p, True)[::-1]-self.base.return_send_s
        old = np.r_[0., ft, ack, ret, ret+rt, done]
        new = np.r_[0., ft+m["forward_residual_s"], new_ack, new_ret,
                    new_ret+rt+np.array(m["return_residual_s"])[::-1], new_done]
        if np.any(np.diff(old) <= 0) or np.any(np.diff(new) <= 0):
            raise ValueError("Measured timing produces a non-monotone clock; review rather than clip")
        return old, new

    @staticmethod
    def map_clock(times, source, target):
        t = np.atleast_1d(np.asarray(times, float))
        mapped = np.interp(t, source, target)
        before, after = t < source[0], t > source[-1]
        mapped[before] = t[before]+target[0]-source[0]
        mapped[after] = t[after]+target[-1]-source[-1]
        return mapped

    def base_clock(self, times, angle):
        old, new = self.clock_knots(angle)
        return self.map_clock(times, new, old)

    def joints(self, times, angle):
        return self.base.joints(self.base_clock(times, angle), angle)

    def reference_clock(self, times, angle):
        return self.base.reference_clock(self.base_clock(times, angle), angle)

    def sample_times(self, angle, dt=.02):
        old, new = self.clock_knots(angle)
        mapped = self.map_clock(self.base.sample_times(angle, dt), old, new)
        return np.unique(np.round(np.r_[mapped, new, np.arange(mapped.min(), mapped.max(), dt)], 9))


def load_motion(path: Path, max_angle=70.):
    base, ep = reference.planned_motion(max_angle)
    data = json.loads(path.read_text())
    return MeasuredTimingMotion(base, data), ep

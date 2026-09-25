"""Force-feedback controllers for robot-to-material coupling.

The controllers map a measured reaction force to the next tool command. They can stop a
press at a target force or let a compliant position target move in response to contact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForceAdmittance:
    """First-order vertical admittance for a target contact force.

        v_down = clip((f_target - f_react) / damping, -v_max, +v_max)

    ``f_react`` is the upward force exerted by the material and is nonnegative in contact.
    In free space, the tool descends at ``v_max``. Its speed falls as the reaction rises and
    reaches zero at ``f_target``. If the reaction exceeds the target and retraction is
    enabled, the commanded velocity reverses. The calling loop should still impose a hard
    z limit as a safety constraint.

    damping has units N/(m/s): the contact force error that produces full descent speed is
    f_target at v_max when damping = f_target / v_max.
    """

    f_target: float = 40.0
    v_max: float = 0.4
    damping: float | None = None    # N/(m/s); default f_target/v_max -> full v_max at zero force
    allow_retract: bool = True

    def _damp(self) -> float:
        # default: full descent speed at zero reaction, linearly to zero at f_target
        return self.damping if self.damping is not None else self.f_target / self.v_max

    def velocity_down(self, f_react: float) -> float:
        """Commanded downward speed (m/s, positive = descending) given the reaction force."""
        v = (self.f_target - float(f_react)) / self._damp()
        lo = -self.v_max if self.allow_retract else 0.0
        return float(np.clip(v, lo, self.v_max))


@dataclass
class Impedance1D:
    """Second-order vertical impedance for a contact tool (height z, up positive):

        m*z'' + b*z' + k*(z - z_ref) = f_react

    With finite stiffness and ``z_ref`` below the surface, the tool settles where
    ``k * (z_eq - z_ref) = f_react(z_eq)`` and stops short of ``z_ref``. Integration is
    semi-implicit with damping treated implicitly. Use this controller for a compliant
    position target rather than a regulated force.
    """

    m: float = 1.0
    b: float = 120.0
    k: float = 4000.0
    z: float = 0.0
    zd: float = 0.0
    z_floor: float | None = None

    def step(self, dt: float, f_react: float, z_ref: float) -> tuple[float, float]:
        a = (float(f_react) - self.k * (self.z - z_ref)) / self.m
        self.zd = (self.zd + dt * a) / (1.0 + dt * self.b / self.m)  # implicit damping
        self.z += dt * self.zd
        if self.z_floor is not None and self.z < self.z_floor:
            self.z = self.z_floor
            self.zd = max(self.zd, 0.0)
        return self.z, self.zd

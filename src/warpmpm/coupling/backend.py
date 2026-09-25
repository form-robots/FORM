"""Coupling adapter between a robot simulator and the MPM material solver.

A tool is represented by a kinematic box collider. Each control tick supplies its
start-of-tick center and velocity; set_box documents how the collider is integrated over
the substeps. A stationary tool acts as a no-slip boundary.

Two force readouts are available. get_tool_wrench integrates stress over a contact band
for quasi-static controller feedback. get_tool_reaction returns the accumulated collider
grid impulse for moving-contact measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from warpmpm.core.solver import Solver
from warpmpm.coupling.wrench import box_contact_wrench


@dataclass
class WarpMPMBackend:
    """Connect one Solver to kinematic tool commands and reaction-wrench readouts."""

    solver: Solver
    _tools: dict[int, dict[str, Any]] = field(default_factory=dict)

    # --- tool lifecycle --------------------------------------------------------------
    def attach_tool(self, center, half, velocity=(0.0, 0.0, 0.0)) -> int:
        """Add a kinematic box end-effector; returns a tool id."""
        tid = self.solver.add_box(center, half, velocity)
        self._tools[tid] = {"half": tuple(map(float, half)),
                            "center": tuple(map(float, center)),
                            "velocity": tuple(map(float, velocity))}
        return tid

    def set_tool_kinematics(self, tool_id: int, center, velocity) -> None:
        """Set a tool's start-of-tick center and per-tick velocity.

        This follows the contract documented by Solver.set_box. step() advances the cached
        center to its post-step position, which get_tool_wrench uses by default.
        """
        self.solver.set_box(tool_id, center=center, velocity=velocity)
        self._tools[tool_id]["center"] = tuple(map(float, center))
        self._tools[tool_id]["velocity"] = tuple(map(float, velocity))

    # --- exchange --------------------------------------------------------------------
    def step(self, dt: float, substeps: int = 1) -> None:
        self.solver.step(dt, substeps)
        dt_ctrl = dt * substeps                  # advance each cached centre to the end-of-tick
        for t in self._tools.values():           # position the fork integrated the collider to
            v, c = t.get("velocity", (0.0, 0.0, 0.0)), t["center"]
            t["center"] = (c[0] + v[0] * dt_ctrl, c[1] + v[1] * dt_ctrl, c[2] + v[2] * dt_ctrl)

    def get_tool_wrench(self, tool_id: int, at_center=None, **kw) -> dict:
        """Estimate a quasi-static contact wrench by integrating stress over the box.

        This readout is intended for admittance or impedance feedback. It differs from
        get_tool_reaction, which measures ``sum m * (v_free - v_imposed) / dt`` and is used
        for moving-contact identification. At a static halt, that grid-impulse signal tends
        to zero as the two velocities match, while the contact stress remains nonzero.

        The stress-integral estimate has a known bias. ``at_center`` sets the post-step
        center at which torque is evaluated; the cached post-step center is the default.
        """
        t = self._tools[tool_id]
        c = tuple(map(float, at_center)) if at_center is not None else t["center"]
        return box_contact_wrench(
            self.solver.x(), self.solver.cauchy(), self.solver.vol(), c, t["half"], **kw
        )

    def reset_tool_force(self, tool_id: int) -> None:
        """Zero the tool's grid-impulse reaction accumulator (call before step())."""
        self.solver.reset_tool_force(tool_id)

    def get_tool_reaction(self, tool_id: int, dt: float):
        """Return the collider grid impulse since reset_tool_force, divided by ``dt``.

        The result is ``force[3]`` with positive z in compression. It uses no contact band,
        layer thickness, or stress gate.
        """
        return self.solver.tool_force(tool_id, dt)

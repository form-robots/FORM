"""Warp MLS/APIC simulation for robot interaction with deformable and granular materials.

The public package exports the typed solver, material and scene builders, contact-wrench
readouts, force-feedback controllers, and MuJoCo coupling adapter.
"""
from __future__ import annotations

from warpmpm.colliders.glass import GlassProfile, cup_fill
from warpmpm.core.solver import GridConfig, Solver
from warpmpm.coupling.admittance import ForceAdmittance, Impedance1D
from warpmpm.coupling.backend import WarpMPMBackend
from warpmpm.coupling.wrench import box_contact_wrench
from warpmpm.materials import (
    Material,
    elastic,
    granular,
    newtonian,
    tabulated_viscous,
    vonmises,
)
from warpmpm.scenes import block, dough

__version__ = "0.0.1"
__all__ = [
    "ForceAdmittance",
    "GlassProfile",
    "GridConfig",
    "Impedance1D",
    "Material",
    "Solver",
    "WarpMPMBackend",
    "__version__",
    "block",
    "box_contact_wrench",
    "cup_fill",
    "dough",
    "elastic",
    "granular",
    "newtonian",
    "tabulated_viscous",
    "vonmises",
]

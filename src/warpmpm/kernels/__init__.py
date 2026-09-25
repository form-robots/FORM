"""Warp MLS-MPM kernels used by the engine.

``mpm_solver_warp`` contains the explicit solver, ``mpm_utils`` contains constitutive
stress and return-mapping kernels, and ``warp_utils`` defines the model and state structs.
The base solver, quadratic B-spline transfer, and materials 0-8 derive from the UCLA
warp-mpm project by Zeshun Zong and collaborators. UT Austin extensions include moving
robot colliders, grid-impulse force measurement, a six-DoF revolved-SDF glass collider,
multi-material scenes, a point-cloud loader, and the mu(I) and viscoplastic models in
material slots 9-13. See AUTHORS.md for attribution and citations.

The typed ``core.Solver`` wrapper imports ``MPM_Simulator_WARP`` and the low-level structs
from this module.
"""
from __future__ import annotations

from warpmpm.kernels.mpm_solver_warp import MATERIAL_NAME_TO_ID, MPM_Simulator_WARP
from warpmpm.kernels.warp_utils import MPMModelStruct, MPMStateStruct  # low-level structs (sim/ probes)

__all__ = [
    "MATERIAL_NAME_TO_ID",
    "MPMModelStruct",
    "MPMStateStruct",
    "MPM_Simulator_WARP",
]

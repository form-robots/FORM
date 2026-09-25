"""Collider geometry: analytic tool/container shapes shared by the kernel BCs, the
scene fill samplers, and the containment/leak masks. The Warp kernels that impose these
shapes on the grid live in kernels/mpm_solver_warp.py; this package holds the host-side
profiles and reference SDFs."""
from __future__ import annotations

from warpmpm.colliders.glass import (
    GlassProfile,
    angular_velocity_between,
    cavity_mask,
    cup_fill,
    glass_sdf,
    glass_sdf_local,
    project_out_of_solid,
    quat_from_axis_angle,
    quat_mul,
    quat_to_mat,
    solid_mask,
    world_to_local,
)

__all__ = [
    "GlassProfile",
    "angular_velocity_between",
    "cavity_mask",
    "cup_fill",
    "glass_sdf",
    "glass_sdf_local",
    "project_out_of_solid",
    "quat_from_axis_angle",
    "quat_mul",
    "quat_to_mat",
    "solid_mask",
    "world_to_local",
]

"""Geometry helpers for the engine: procedural meshes and mesh->signed-distance fields.

The SDF is the collision/coupling representation for an arbitrary watertight mesh driven
by the robot (the cup in the pour demo, an imported tool later). It mirrors the design used
by Genesis (a stored value grid + gradient grid in a normalized mesh frame, trilinearly
interpolated at query time) but the builder is self-contained numpy: generalized winding
number for the inside/outside sign and exact point-triangle distance for the magnitude, so
the engine keeps its numpy + warp footprint and the field is fully testable.
"""
from __future__ import annotations

from warpmpm.geometry.measuring_cup import (
    MeasuringCupSpec,
    build_cup_sdf,
    write_cup_obj,
)
from warpmpm.geometry.mesh_sdf import (
    CDFData,
    SDFData,
    build_sdf,
    build_sdf_cached,
    build_surface_cdf,
    build_surface_cdf_cached,
    load_cdf,
    load_sdf,
    make_cup_mesh,
    revolve_profile,
    revolve_profile_open,
    save_cdf,
    save_sdf,
)

__all__ = [
    "CDFData",
    "MeasuringCupSpec",
    "SDFData",
    "build_cup_sdf",
    "build_sdf",
    "build_sdf_cached",
    "build_surface_cdf",
    "build_surface_cdf_cached",
    "load_cdf",
    "load_sdf",
    "make_cup_mesh",
    "revolve_profile",
    "revolve_profile_open",
    "save_cdf",
    "save_sdf",
    "write_cup_obj",
]

"""Experimental Navier traction with a wetted-area correction.

Physical boundary law: tangential traction = -(eta / b) * relative velocity.
The surface integral is regularized over a quadratic B-spline of width 3 dx;
the nodal surface area is dx**2 * B2(signed_distance / dx). Nodal friction
is mass lumped and integrated implicitly. This is an approximation to the
boundary law, to be verified on film flow before any pouring calibration.

This opt-in experiment replaces only its own collider callback. Existing
production contact modes and the shared solver source are unchanged.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from warpmpm.kernels.mpm_solver_warp import sdf_trilerp, sdf_trilerp_vec
from warpmpm.kernels.warp_utils import MPMModelStruct, MPMStateStruct, SDFCollider


def install_navier_traction_wall(solver, handle: int, *, slip_length_m: float, eta_pa_s: float, density: float = 1260.):
    """Install immediately after adding an SDF collider; b is finite and positive."""
    if not np.isfinite(slip_length_m) or slip_length_m <= 0:
        raise ValueError("Navier slip length must be finite and positive")
    if not np.isfinite(eta_pa_s) or eta_pa_s <= 0:
        raise ValueError("Viscosity must be finite and positive")
    sim = solver._sim
    if handle != len(sim.collider_params) - 1:
        raise ValueError("Install Navier contact immediately after creating the collider")
    if len(sim.grid_postprocess) != len(sim.collider_params):
        raise ValueError("Unexpected collider callback indexing")
    beta = wp.constant(float(eta_pa_s / slip_length_m))
    rho = wp.constant(float(density))

    @wp.kernel
    def contact(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct,
                param: SDFCollider, lo: wp.vec3i):
        i, j, k = wp.tid()
        i += lo[0]
        j += lo[1]
        k += lo[2]
        mass = state.grid_m[i, j, k]
        if mass <= 0.0 or time < param.start_time or time >= param.end_time:
            return
        x = wp.vec3(float(i), float(j), float(k)) * model.dx
        rel = x - param.center
        body = wp.quat_rotate_inv(param.quat, rel)
        fidx = (body - param.origin) / param.cell
        last = float(param.res - 1)
        if (fidx[0] < 0.0 or fidx[1] < 0.0 or fidx[2] < 0.0
                or fidx[0] > last or fidx[1] > last or fidx[2] > last):
            return
        sd = sdf_trilerp(param.sdf_val, param.res, fidx)
        if sd > wp.max(param.band, 1.5 * model.dx):
            return
        g = sdf_trilerp_vec(param.sdf_grad, param.res, fidx)
        if wp.length(g) < 1.0e-12:
            return
        normal = wp.quat_rotate(param.quat, wp.normalize(g))
        wall = param.velocity + wp.cross(param.omega, rel)
        before = state.grid_v_out[i, j, k]
        velocity = before - wall
        vn = wp.dot(velocity, normal)
        tangent = velocity - vn * normal
        if sd <= param.band:
            vn = wp.max(vn, 0.0)  # unilateral nonpenetration, release allowed
        distance = wp.abs(sd / model.dx)
        weight = float(0.0)
        if distance < 0.5:
            weight = 0.75 - distance * distance
        elif distance < 1.5:
            weight = 0.5 * (1.5 - distance) * (1.5 - distance)
        # A narrow/free-edge fluid patch covers only part of a wall cell. Divide
        # the measured nodal mass fraction by the normal half-space occupancy
        # to isolate the tangential wetted fraction. Without this correction,
        # support nodes outside a narrow film apply spurious extra wall area.
        signed = sd / model.dx
        normal_fraction = float(0.0)
        if signed >= 1.5:
            normal_fraction = 1.0
        elif signed >= 0.5:
            normal_fraction = 1.0 - wp.pow(1.5 - signed, 3.0) / 6.0
        elif signed >= -0.5:
            normal_fraction = 0.5 + 0.75 * signed - signed * signed * signed / 3.0
        elif signed > -1.5:
            normal_fraction = wp.pow(signed + 1.5, 3.0) / 6.0
        full_mass = rho * model.dx * model.dx * model.dx * normal_fraction
        coverage = wp.min(1.0, mass / wp.max(full_mass, 1.0e-20))
        area = model.dx * model.dx * weight * coverage
        scale = 1.0 / (1.0 + dt * beta * area / mass)
        after = wall + tangent * scale + vn * normal
        impulse = mass * (before - after)
        wp.atomic_add(param.force, 0, impulse)
        wp.atomic_add(param.torque, 0, wp.cross(rel, impulse))
        state.grid_v_out[i, j, k] = after

    sim.grid_postprocess[-1] = contact
    return dict(method="Regularized weak Navier traction with wetted-area correction",
                slip_length_m=float(slip_length_m), eta_pa_s=float(eta_pa_s),
                boundary_friction_pa_s_per_m=float(eta_pa_s / slip_length_m),
                surface_kernel="quadratic B-spline, support +/-1.5 dx",
                density_kg_m3=float(density),
                wetted_area="nodal mass / normal half-space occupancy, capped at a full wall cell",
                integration="implicit tangential damping; unilateral normal contact")

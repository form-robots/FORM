"""Second-order physical-wall Robin extrapolation for the pouring experiment.

Use two fluid-side samples to reconstruct u(s)=a*(b+s)+c*s**2. This exactly
satisfies u(0)=b*u'(0), retaining the curvature of pressure/gravity-driven films.
The sample's first AND second distance moments account for grid interpolation.
All velocity reads are strictly outside the write band. Thin films with just
one readable exterior sample fall back to linear Robin extrapolation.
"""
import numpy as np
import warp as wp

from warpmpm.kernels.mpm_solver_warp import sdf_trilerp, sdf_trilerp_vec
from warpmpm.kernels.warp_utils import MPMModelStruct, MPMStateStruct, SDFCollider


@wp.func
def exterior_sample(target: wp.vec3, state: MPMStateStruct, model: MPMModelStruct,
                    param: SDFCollider):
    grid_target = target / model.dx
    base = wp.vec3i(int(wp.floor(grid_target[0])), int(wp.floor(grid_target[1])), int(wp.floor(grid_target[2])))
    fraction = grid_target - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    total, moment1, moment2 = float(0.0), float(0.0), float(0.0)
    velocity = wp.vec3(0.0)
    last = float(param.res - 1)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i, j, k = base[0] + di, base[1] + dj, base[2] + dk
                if (i < 0 or j < 0 or k < 0 or i >= state.grid_m.shape[0]
                        or j >= state.grid_m.shape[1] or k >= state.grid_m.shape[2]):
                    continue
                rel = wp.vec3(float(i), float(j), float(k)) * model.dx - param.center
                body = wp.quat_rotate_inv(param.quat, rel)
                fi = (body - param.origin) / param.cell
                if (fi[0] < 0.0 or fi[1] < 0.0 or fi[2] < 0.0
                        or fi[0] > last or fi[1] > last or fi[2] > last):
                    continue
                distance = sdf_trilerp(param.sdf_val, param.res, fi)
                if distance <= param.band or state.grid_m[i, j, k] <= 0.0:
                    continue
                wi = fraction[0] if di == 1 else 1.0 - fraction[0]
                wj = fraction[1] if dj == 1 else 1.0 - fraction[1]
                wk = fraction[2] if dk == 1 else 1.0 - fraction[2]
                weight = wi * wj * wk
                wall = param.velocity + wp.cross(param.omega, rel)
                velocity += weight * (state.grid_v_out[i, j, k] - wall)
                moment1 += weight * distance
                moment2 += weight * distance * distance
                total += weight
    if total > 1.0e-6:
        velocity /= total
        moment1 /= total
        moment2 /= total
    return velocity, moment1, moment2, total


def install_navier_quadratic_wall(solver, handle, *, slip_length_m, eta_pa_s):
    if not np.isfinite(slip_length_m) or slip_length_m < 0:
        raise ValueError("Slip length must be finite and nonnegative")
    sim = solver._sim
    if handle != len(sim.collider_params) - 1 or len(sim.grid_postprocess) != len(sim.collider_params):
        raise ValueError("Install immediately after the SDF collider")
    b = wp.constant(float(slip_length_m))

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
        if sd > param.band:
            return
        g = sdf_trilerp_vec(param.sdf_grad, param.res, fidx)
        if wp.length(g) < 1.0e-12:
            return
        normal = wp.quat_rotate(param.quat, wp.normalize(g))
        wall = param.velocity + wp.cross(param.omega, rel)
        before = state.grid_v_out[i, j, k]
        vn = wp.max(wp.dot(before - wall, normal), 0.0)
        target1 = x + (param.band + model.dx - sd) * normal
        v1, d1, m1, total1 = exterior_sample(target1, state, model, param)
        v2, d2, m2, total2 = exterior_sample(target1 + model.dx * normal, state, model, param)
        tangent = before - wall - wp.dot(before - wall, normal) * normal
        if total1 > 1.0e-6:
            candidate = v1 * wp.clamp((b + sd) / (b + d1), -1.0, 1.0)
            det = (b + d1) * m2 - (b + d2) * m1
            det_scale = wp.abs((b + d1) * m2) + wp.abs((b + d2) * m1)
            if total2 > 1.0e-6 and wp.abs(det) > 1.0e-5 * det_scale:
                weight1 = ((b + sd) * m2 - sd * sd * (b + d2)) / det
                weight2 = (sd * sd * (b + d1) - (b + sd) * m1) / det
                candidate = weight1 * v1 + weight2 * v2
            tangent = candidate - wp.dot(candidate, normal) * normal
        after = wall + tangent + vn * normal
        impulse = mass * (before - after)
        wp.atomic_add(param.force, 0, impulse)
        wp.atomic_add(param.torque, 0, wp.cross(rel, impulse))
        state.grid_v_out[i, j, k] = after

    sim.grid_postprocess[-1] = contact
    return dict(method="Second-order SDF Robin extrapolation", slip_length_m=float(slip_length_m),
                eta_pa_s=float(eta_pa_s), boundary_relation="u_t = b * partial_n u_t",
                reconstruction="a*(b+s)+c*s**2 using two exterior distance moments",
                readable_exterior_rule="Mass-bearing nodes strictly outside write band",
                single_sample_rule="Linear Robin extrapolation",
                no_exterior_rule="Preserve tangential velocity; enforce normal contact")

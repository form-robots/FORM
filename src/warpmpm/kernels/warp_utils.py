import warp
import os

import warp as wp

# The simulator is never differentiated (project invariant), so adjoint
# codegen is off by default; it cuts module compile time 2.5x. Set
# WARPMPM_ENABLE_BACKWARD=1 to compile adjoints for a project that
# consciously retires the invariant.
wp.set_module_options({"enable_backward":
                       os.environ.get("WARPMPM_ENABLE_BACKWARD", "0") == "1"})

try:  # warp.torch was a separate submodule import in warp<=1.10; gone/needless in 1.14+
    import warp.torch  # noqa: F401
except Exception:
    pass
import torch

# CPIC thin-boundary lanes: 2 tag bits per lane plus a 2-bit owner id fit an int32
# with room to raise later; 4 keeps the per-lane vote accumulators in one wp.vec4
# inside the (register-heavy) fused transfer kernel. Node tag bit layout: bit 2l =
# lane l valid, bit 2l+1 = lane l side, bits [CDF_OWNER_SHIFT, +2) = owner lane id.
MAX_CDF = 4
CDF_OWNER_SHIFT = 2 * MAX_CDF
CDF_OWNER_SHIFT_C = wp.constant(CDF_OWNER_SHIFT)
CDF_OWNER_MASK_C = wp.constant(3 << CDF_OWNER_SHIFT)
# coarse tag-occupancy block edge (cells per block per axis, power of two): the
# transfer kernels test at most 2^3 blocks per particle before the 27-node vote
CDF_BLOCK = 4
CDF_BLOCK_SHIFT_C = wp.constant(2)


@wp.struct
class MPMModelStruct:
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    bulk: wp.array(dtype=float)
    material: int

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)
    friction_angle: float
    alpha: wp.array(dtype=float)
    gravitational_accelaration: wp.vec3
    hardening: wp.array(dtype=float)
    xi: wp.array(dtype=float)
    plastic_viscosity: wp.array(dtype=float)
    softening: wp.array(dtype=float)

    ######## local mu(I) rheology (material 9, TrackEUCLID), per-type ####
    muI_mu_s: wp.array(dtype=float)        # static friction coefficient
    muI_delta_mu: wp.array(dtype=float)    # mu_2 - mu_s (0 for constant mu)
    muI_I0: wp.array(dtype=float)          # Pouliquen reference inertial number
    muI_d: wp.array(dtype=float)           # grain diameter
    muI_rho_s: wp.array(dtype=float)       # grain density
    # compressible mu(I)-Phi(I) dilatancy (material 11), per-type
    muI_phi_init: wp.array(dtype=float)    # rest (densest) solid fraction Phi_init
    muI_phi_chi: wp.array(dtype=float)     # max dilatancy fraction: Phi_c(I)=Phi_init*(1-chi*I/(I+I0))

    ######## tabulated apparent viscosity (material 12, TrackEUCLID FE rollout) ####
    # eta_app(gd) read from a uniform log10(gd) table with clamped linear interp, so an
    # FE-recovered eta_app curve can be re-simulated directly (no parametric fit). One
    # global table (a single tabulated material per sim); s = log10(gd) in [smin, smax].
    eta_table: wp.array(dtype=float)       # (n,) eta_app samples on the log10(gd) grid
    eta_table_smin: float
    eta_table_smax: float
    eta_table_n: int

    ####### for damping
    rpic_damping: float
    grid_v_damping_scale: float

    ####### for PhysGaussian: covariance
    update_cov_with_F: int

    ######## composed material (14): one elasticity kind and one plasticity kind
    # per material type, chosen independently. Inert for every other material.
    # Reuses mu, lam, yield_stress and alpha; carries only what has no home yet.
    compose_elastic: wp.array(dtype=int)    # ELASTICITY_KIND value
    compose_plastic: wp.array(dtype=int)    # PLASTICITY_KIND value
    compose_gamma: wp.array(dtype=float)    # ziran EOS exponent (their value: 2)
    compose_cohesion: wp.array(dtype=float) # Drucker-Prager cohesion (their sand: 0)

    ####### grid semantics and transfer form (0/-1 = the historical default) ####
    # Set through MPM_Simulator_WARP.set_grid_semantics. Each option is
    # independent, and all of them are off unless asked for.
    freeslip_bound: int    # > 0: approach-only wall clamp on the outer `bound` node layers
    grid_mass_eps: float   # grid velocity is mv / (m + grid_mass_eps)
    empty_node_gravity: int    # 1: zero-mass nodes carry v = gravity * dt into g2p
    mls_transfer: int      # 1: MLS-MPM stress transfer and (I + dt C) F update
    particle_clip_cells: float  # advected position clamped this many cells inside
                                # the box; < 0 = no clamp

    ####### CPIC thin-boundary (CDF) colliders: 0 = feature off, transfers untouched
    n_cdf: int
    # periodic boundary along x: transfers wrap the x node index and advection
    # wraps particle x (0 = off, transfers untouched). Used by chute flows.
    periodic_x: int
    # test hook: 1 forces the full tag vote even where the block mask is empty
    # (the mask early-out is exact, and the bitwise gate proves it against this)
    cdf_mask_off: int


@wp.struct
class MPMStateStruct:
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_x_roundoff: wp.array(dtype=wp.vec3)  # compensated advection residual
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_F_roundoff: wp.array(dtype=wp.mat33)  # compensated elastic increment residual
    particle_init_cov: wp.array(dtype=float)  # initial covariance matrix
    particle_cov: wp.array(dtype=float)  # current covariance matrix
    particle_F_trial: wp.array(
        dtype=wp.mat33
    )  # apply return mapping on this to obtain elastic def grad
    particle_R: wp.array(dtype=wp.mat33)  # rotation matrix
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density
    particle_Jp: wp.array(dtype=float)

    particle_L: wp.array(dtype=wp.mat33)  # velocity gradient from g2p, L_ij = dv_i/dx_j (TrackEUCLID dump)

    # CPIC color persistence (paper 5.3.3): a particle keeps its side of each thin
    # boundary while any node in its kernel still has that lane's affinity, so a
    # slight penetration by advection error cannot flip its allegiance (the fresh
    # vote would, and the particle would fall through). Written by g2p, read by the
    # next p2g; in the fused kernel both halves run in one thread in that order.
    particle_cdf_tag: wp.array(dtype=int)

    particle_selection: wp.array(dtype=int) # only particle_selection[p] = 0 will be simulated
    particle_material: wp.array(dtype=int)  # material type per particle (0=jelly,1=metal,2=sand,5=plasticine,6=fluid,7=stationary,8=rigid; 3,4 retired)
    particle_rigid_id: wp.array(dtype=int)  # rigid body id for mat==8 particles; -1 for non-rigid
    particle_x_ref: wp.array(dtype=wp.vec3) # body-frame reference position for rigid particles

    # grid
    grid_m: wp.array(dtype=float, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity
    grid_v_out: wp.array(
        dtype=wp.vec3, ndim=3
    )  # grid node momentum/velocity, after grid update

    # CPIC colored distance field (thin-boundary colliders; docs in mpm_solver_warp).
    # Node tag bit layout: bits 2l (valid) and 2l+1 (side) per lane l < MAX_CDF, then
    # 2 owner-lane bits at OWNER_SHIFT. grid_cdf_d holds the owner lane's unsigned
    # distance. The *_prev twins hold the previous substep's stamp so the fused
    # kernel's g2p half reads colors(s-1) while its p2g half reads colors(s); the
    # split pipeline copies cur -> prev after stamping so both reads see pose(s).
    # (1,1,1) placeholders until the first add_cdf_collider (lazy allocation).
    grid_cdf_tag: wp.array(dtype=int, ndim=3)
    grid_cdf_tag_prev: wp.array(dtype=int, ndim=3)
    grid_cdf_d: wp.array(dtype=float, ndim=3)
    grid_cdf_d_prev: wp.array(dtype=float, ndim=3)
    # coarse occupancy: one int per CDF_BLOCK^3 cell block, nonzero when any cell
    # of the block carries a tag in either the cur or prev grid. The transfer
    # kernels test the blocks a particle's stencil touches before the 27-node tag
    # vote; an all-empty stencil yields pc == 0 through the full path too, so the
    # early-out is exact. Without it every particle paid the vote's ~108 extra
    # global loads per substep whether or not it was near any surface.
    grid_cdf_block: wp.array(dtype=int, ndim=3)
    # per-lane pose/material (length MAX_CDF; g2p_particle cannot take collider
    # structs, so the ghost projection reads these) and reaction accumulators
    cdf_lane_center: wp.array(dtype=wp.vec3)
    cdf_lane_velocity: wp.array(dtype=wp.vec3)
    cdf_lane_omega: wp.array(dtype=wp.vec3)
    cdf_lane_center_prev: wp.array(dtype=wp.vec3)
    cdf_lane_velocity_prev: wp.array(dtype=wp.vec3)
    cdf_lane_omega_prev: wp.array(dtype=wp.vec3)
    cdf_lane_friction: wp.array(dtype=float)
    cdf_lane_type: wp.array(dtype=int)
    cdf_reaction_force: wp.array(dtype=wp.vec3)
    cdf_reaction_torque: wp.array(dtype=wp.vec3)


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2

    # Optional reaction-impulse accumulator. It is set only by colliders that measure
    # force, such as set_velocity_on_cuboid, and is unused by other colliders.
    force: wp.array(dtype=wp.vec3)

@wp.struct
class PointCloudCollider:
    occupancy_grid: wp.array(dtype=int, ndim=3)
    start_time: float
    end_time: float


# Kinematic open-top glass driven by the robot end-effector. Stores a six-DoF pose,
# rigid velocity field, analytic profile parameters, and reaction-impulse accumulators.
@wp.struct
class RevolvedCollider:
    point: wp.vec3          # glass centre (mid-height), world frame
    rot: wp.mat33           # local -> world rotation
    velocity: wp.vec3       # linear velocity of the centre
    omega: wp.vec3          # angular velocity, world frame

    start_time: float
    end_time: float
    friction: float         # Coulomb friction of the near-surface separable contact

    outer_radius: float
    inner_radius: float
    half_height: float
    inner_floor_z: float    # local z of the cavity floor
    fillet_radius: float    # cavity floor-edge fillet
    sticky_depth: float     # deeper than this inside the solid: full velocity grab
    contact_band: float     # BC also acts this far OUTSIDE the surface (approach-only)

    force: wp.array(dtype=wp.vec3)   # sum_substeps sum_nodes m*(v_free - v_imposed)
    torque: wp.array(dtype=wp.vec3)  # sum of (x_node - point) x impulse


@wp.struct
class SDFCollider:
    # a watertight mesh represented as a stored signed-distance field in its body frame:
    # negative inside the solid, positive outside. The collider may translate and rotate; the
    # field is queried by mapping each grid node into the body frame and trilinearly
    # interpolating sdf_val (distance) and sdf_grad (outward gradient).
    sdf_val: wp.array(dtype=float, ndim=3)
    sdf_grad: wp.array(dtype=wp.vec3, ndim=3)
    res: int
    origin: wp.vec3          # body-frame coordinate of voxel index (0,0,0)
    cell: float              # body-frame metres per voxel (isotropic)

    center: wp.vec3          # world position of the body-frame origin (pivot)
    quat: wp.quat            # orientation body -> world
    velocity: wp.vec3        # linear velocity of the pivot (world)
    omega: wp.vec3           # angular velocity (world)

    band: float              # contact band thickness (world metres); constrain nodes with sd < band
    surface_type: int        # 0 sticky, 1 slip (frictionless), 2 separable + Coulomb friction
    friction: float

    start_time: float
    end_time: float

    # Reaction accumulators in the world frame. force = sum_nodes m*(v_free - v_new), and
    # torque = sum_nodes (x - center) x impulse.
    # Reaction wrench = (force, torque) / elapsed dt.
    force: wp.array(dtype=wp.vec3)
    torque: wp.array(dtype=wp.vec3)



@wp.struct
class CDFCollider:
    # CPIC thin boundary: an OPEN oriented mid-surface stored as a side-signed
    # distance field plus validity mask in its body frame (geometry.CDFData). Unlike
    # SDFCollider it has no grid-BC kernel; a stamp kernel writes side/validity bits
    # into state.grid_cdf_tag each substep and the transfer kernels enforce the
    # discontinuity. Reaction accumulators live on MPMStateStruct per lane.
    cdf_val: wp.array(dtype=float, ndim=3)
    cdf_valid: wp.array(dtype=float, ndim=3)
    res: int
    origin: wp.vec3          # body-frame coordinate of voxel index (0,0,0)
    cell: float              # body-frame metres per voxel (isotropic)

    center: wp.vec3          # world position of the body-frame origin (pivot)
    quat: wp.quat            # orientation body -> world
    velocity: wp.vec3        # linear velocity of the pivot (world)
    omega: wp.vec3           # angular velocity (world)

    band: float              # world metres; nodes with |d| <= band get tagged
    surface_type: int        # 0 sticky, 1 slip, 2 separable + Coulomb friction
    friction: float
    lane: int                # bit lane in grid_cdf_tag, < MAX_CDF

    start_time: float
    end_time: float


@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int)


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float

@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float

@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    
    start_time: float

    end_time: float

    velocity: wp.vec3

    mask: wp.array(dtype=int)




@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def set_value_to_int_array(target_array: wp.array(dtype=int), value: int):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]


def _torch_on_device(t, dvc):
    dev = torch.device(str(dvc))
    if t.device != dev:
        t = t.to(dev)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def torch2warp_quat(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 4
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_int(t, copy=False, dtype=wp.int32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.int32,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_float(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.float32,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_vec3(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_mat33(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

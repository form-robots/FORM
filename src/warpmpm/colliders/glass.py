"""Analytic open-top glass: profile, SDF, containment masks, and particle fill.

One `GlassProfile` describes a pouring/receiving glass as a solid of revolution: an
outer capped cylinder minus an inner cavity whose floor edge is filleted. The cavity is
exactly the fillet-radius dilation of a smaller capped cylinder, so the closed-form SDF
here (numpy, the host reference) and its Warp twin in kernels/mpm_solver_warp.py stay in
lockstep; tests assert they agree.

Local frame: z up, origin at the glass mid-height. The cavity floor sits at
`inner_floor_z`, the rim at `half_height`; above the rim the glass is open.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

# ---- wxyz quaternion helpers (host-side pose bookkeeping) ----------------------------


def quat_to_mat(q) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion (columns = local axes in world frame)."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    half = 0.5 * float(angle)
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def quat_from_mat(m) -> np.ndarray:
    """wxyz quaternion of a rotation matrix (Shepperd's branch selection)."""
    m = np.asarray(m, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def angular_velocity_between(q0, q1, dt: float) -> np.ndarray:
    """World-frame angular velocity that rotates q0 into q1 over dt (wxyz quaternions)."""
    delta = quat_mul(np.asarray(q1, dtype=np.float64), quat_conj(q0))
    delta = delta / np.linalg.norm(delta)
    if delta[0] < 0.0:
        delta = -delta
    vec_norm = float(np.linalg.norm(delta[1:]))
    if vec_norm < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vec_norm, float(delta[0]))
    return delta[1:] / vec_norm * (angle / dt)


def world_to_local(points, pos, quat) -> np.ndarray:
    """Map world points into the glass local frame (row-stacked: (x - p) @ R)."""
    r = quat_to_mat(quat)
    return (np.asarray(points, dtype=np.float64) - np.asarray(pos, dtype=np.float64)) @ r


# ---- the glass ------------------------------------------------------------------------


@dataclass(frozen=True)
class GlassProfile:
    """Open-top glass as a solid of revolution (metres). Defaults = the Genesis study's glass:
    outer 0.089 m, wall 0.020 m, height 0.24 m, 0.050 m base, 0.012 m floor fillet."""

    outer_radius: float = 0.089
    inner_radius: float = 0.069
    height: float = 0.24
    base_thickness: float = 0.050
    fillet_radius: float = 0.012

    def __post_init__(self):
        if not (0.0 < self.inner_radius < self.outer_radius):
            raise ValueError("need 0 < inner_radius < outer_radius")
        if not (0.0 < self.base_thickness < self.height):
            raise ValueError("need 0 < base_thickness < height")
        if not (0.0 <= self.fillet_radius < self.inner_radius):
            raise ValueError("need 0 <= fillet_radius < inner_radius")

    @property
    def half_height(self) -> float:
        return 0.5 * self.height

    @property
    def inner_floor_z(self) -> float:
        return -0.5 * self.height + self.base_thickness

    @property
    def rim_z(self) -> float:
        return 0.5 * self.height

    @property
    def wall_thickness(self) -> float:
        return self.outer_radius - self.inner_radius

    def inner_radius_at_z(self, z) -> np.ndarray:
        """Cavity radius vs local z: constant above the fillet, quarter-circle inside it
        (the dilated-cylinder boundary), collapsing to inner_radius - fillet at the floor."""
        z = np.asarray(z, dtype=np.float64)
        fr = self.fillet_radius
        r = np.full_like(z, self.inner_radius)
        if fr > 0.0:
            corner = (z >= self.inner_floor_z) & (z < self.inner_floor_z + fr)
            dz = z[corner] - (self.inner_floor_z + fr)
            r[corner] = self.inner_radius - fr + np.sqrt(np.maximum(fr * fr - dz * dz, 0.0))
            r = np.where(z < self.inner_floor_z, self.inner_radius - fr, r)
        return r

    def cavity_volume(self, depth: float) -> float:
        """Liquid volume held by `depth` of fill above the cavity floor (fillet-aware)."""
        zs = np.linspace(self.inner_floor_z, self.inner_floor_z + depth, 2048)
        return float(np.trapezoid(np.pi * self.inner_radius_at_z(zs) ** 2, zs))


def glass_cavity_profile(profile: GlassProfile | None = None) -> np.ndarray:
    """Open (r, z) polyline of the glass CAVITY surface: the inner floor disk and the
    inner wall up to the rim. Revolve with revolve_profile_open and feed
    build_surface_cdf; this is the CPIC contact sheet for the glass.

    The sheet traces the cavity, not the solid's mid-surface. For a shell much
    thinner than a grid cell the two coincide, but the default glass has a 20 mm
    wall and a 50 mm base, so a mid-surface sheet would sit 10 mm (wall) and 25 mm
    (floor) inside the solid, and fluid would sink that far into the glass before
    touching anything. Tracing the cavity keeps CDF and SDF contact geometry
    identical. Two cavity details are not represented: the floor fillet (thinner
    than the CDF band) and contact on the OUTSIDE of the glass (the sheet's band
    ends a couple of cells past the cavity surface, so runoff down the outer wall
    crosses solid regions unopposed)."""
    p = GlassProfile() if profile is None else profile
    return np.array([(0.0, p.inner_floor_z), (p.inner_radius, p.inner_floor_z),
                     (p.inner_radius, p.rim_z)])


def _capped_cylinder_sdf(r_xy, z, radius: float, z0: float, z1: float):
    """Exact SDF of the solid cylinder r <= radius, z in [z0, z1] (vectorized)."""
    zc, hh = 0.5 * (z0 + z1), 0.5 * (z1 - z0)
    dr = r_xy - radius
    dz = np.abs(z - zc) - hh
    outside = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    inside = np.minimum(np.maximum(dr, dz), 0.0)
    return outside + inside


def glass_sdf_local(points_local, profile: GlassProfile):
    """Signed distance to the glass solid in the local frame (negative inside the
    material). solid = capped outer cylinder minus the fillet-dilated cavity cylinder;
    the subtraction max(a, -b) is exact at the surface, which is all the BC needs."""
    p = np.asarray(points_local, dtype=np.float64)
    r_xy = np.linalg.norm(p[..., :2], axis=-1)
    z = p[..., 2]
    fr = profile.fillet_radius
    d_outer = _capped_cylinder_sdf(r_xy, z, profile.outer_radius,
                                   -profile.half_height, profile.half_height)
    # cavity = dilate(smaller capped cylinder, fillet): floor at inner_floor_z with the
    # quarter-circle fillet, wall at inner_radius, extended above the rim so the top is open
    d_cavity = _capped_cylinder_sdf(
        r_xy, z, profile.inner_radius - fr,
        profile.inner_floor_z + fr, profile.half_height + profile.outer_radius,
    ) - fr
    return np.maximum(d_outer, -d_cavity)


def glass_sdf(points_world, pos, quat, profile: GlassProfile):
    """Signed distance to the glass solid for world points at pose (pos, wxyz quat)."""
    return glass_sdf_local(world_to_local(points_world, pos, quat), profile)


# ---- masks (liquid accounting + leak audit) -------------------------------------------


def cavity_mask(points_world, pos, quat, profile: GlassProfile,
                pad: float = 0.0, brim_clearance: float = 0.0) -> np.ndarray:
    """Points inside the open cavity below the rim, the 'liquid held by this glass'
    account. `pad` loosens the radius (e.g. 0.75x the particle spacing);
    `brim_clearance` trims the count band below the rim."""
    local = world_to_local(points_world, pos, quat)
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    z = local[:, 2]
    return (
        (r_xy < profile.inner_radius_at_z(z) + pad)
        & (z >= profile.inner_floor_z - 1e-3)
        & (z <= profile.rim_z - brim_clearance)
    )


def solid_mask(points_world, pos, quat, profile: GlassProfile, tol: float = 0.0) -> np.ndarray:
    """Leak audit: points embedded inside the glass material by more than `tol`."""
    return glass_sdf(points_world, pos, quat, profile) < -tol


def project_out_of_solid(x, v, pos, quat, profile: GlassProfile, clearance: float = 0.0,
                         solid_velocity=None):
    """Rescue net for boundary creep (wall correction on the MPM side):
    any particle embedded in the glass solid is moved along the SDF gradient back to
    `clearance` outside the surface, and the inward normal component of its velocity
    relative to the local wall velocity is removed. The grid BC (contact band) makes
    this rare; call once per control tick and log the count: a growing count means
    the BC needs attention, not the net.

    x, v: (N,3) world positions/velocities (modified copies are returned).
    solid_velocity: optional (v_lin[3], omega[3]) of the glass; default static.
    Returns (x_new, v_new, n_projected)."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sdf = glass_sdf(x, pos, quat, profile)
    inside = sdf < 0.0
    n_bad = int(inside.sum())
    if n_bad == 0:
        return x, v, 0
    r = quat_to_mat(quat)
    local = world_to_local(x[inside], pos, quat)
    h = 1e-5
    grad = np.stack(
        [
            glass_sdf_local(local + np.array([h, 0, 0]), profile)
            - glass_sdf_local(local - np.array([h, 0, 0]), profile),
            glass_sdf_local(local + np.array([0, h, 0]), profile)
            - glass_sdf_local(local - np.array([0, h, 0]), profile),
            glass_sdf_local(local + np.array([0, 0, h]), profile)
            - glass_sdf_local(local - np.array([0, 0, h]), profile),
        ],
        axis=1,
    )
    n_local = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    push = (clearance - sdf[inside])[:, None] * n_local
    x_new = x.copy()
    x_new[inside] = (local + push) @ r.T + np.asarray(pos, dtype=np.float64)
    # cancel the inward normal component of the velocity relative to the wall
    n_world = n_local @ r.T
    if solid_velocity is None:
        v_wall = np.zeros_like(v[inside])
    else:
        v_lin, omega = (np.asarray(a, dtype=np.float64) for a in solid_velocity)
        v_wall = v_lin + np.cross(omega, x_new[inside] - np.asarray(pos, dtype=np.float64))
    v_rel = v[inside] - v_wall
    vn = np.sum(v_rel * n_world, axis=1)
    v_rel -= n_world * np.minimum(vn, 0.0)[:, None]
    v_new = v.copy()
    v_new[inside] = v_wall + v_rel
    return x_new, v_new, n_bad


# ---- render mesh -----------------------------------------------------------------------


def write_glass_obj(profile: GlassProfile, path, segments: int = 48,
                    fillet_segments: int = 8):
    """Write the watertight open-top glass render mesh (OBJ, triangles, outward
    winding): outer wall, rim annulus, inner wall, filleted cavity floor, bottom cap.
    Pure numpy (no trimesh). Local frame = the SDF/collider frame."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = segments
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    def ring(radius: float, z: float) -> np.ndarray:
        return np.stack([radius * ca, radius * sa, np.full(n, z)], axis=1)

    r_o, r_i = profile.outer_radius, profile.inner_radius
    base_z, rim_z, floor_z = -profile.half_height, profile.rim_z, profile.inner_floor_z
    fr = min(max(profile.fillet_radius, 0.0), r_i - 1e-6)

    # fillet rings from the cavity floor edge (r_i - fr, floor_z) up to the inner wall
    # start (r_i, floor_z + fr); with fr == 0 this collapses to one ring
    if fr > 0.0:
        theta = np.linspace(0.0, 0.5 * np.pi, fillet_segments + 1)
        fillet = [ring(r_i - fr + fr * np.sin(t), floor_z + fr * (1.0 - np.cos(t)))
                  for t in theta]
    else:
        fillet = [ring(r_i, floor_z)]

    rings = [ring(r_o, base_z), ring(r_o, rim_z), ring(r_i, rim_z), *fillet]
    off = np.cumsum([0] + [n] * len(rings))
    ob, ot, it = off[0], off[1], off[2]
    fil = off[3:3 + len(fillet)]
    verts = np.concatenate(rings, axis=0)
    bot_c = len(verts)
    floor_c = len(verts) + 1
    verts = np.concatenate([verts, [[0.0, 0.0, base_z], [0.0, 0.0, floor_z]]], axis=0)

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[ob + i, ob + j, ot + j], [ob + i, ot + j, ot + i]]     # outer wall (out)
        faces += [[ot + i, ot + j, it + j], [ot + i, it + j, it + i]]     # rim annulus (+z)
        top_f = fil[-1]                                                    # inner wall (in)
        faces += [[top_f + i, it + i, it + j], [top_f + i, it + j, top_f + j]]
        for a, b in itertools.pairwise(fil):                               # fillet strips
            faces += [[a + i, b + i, b + j], [a + i, b + j, a + j]]
        faces += [[bot_c, ob + j, ob + i]]                                 # bottom cap (-z)
        faces += [[floor_c, fil[0] + i, fil[0] + j]]                       # cavity floor (+z)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# warpmpm glass: R_o={r_o} R_i={r_i} h={profile.height} "
                f"base={profile.base_thickness} fillet={fr}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return path


# ---- particle fill ---------------------------------------------------------------------


def cup_fill(profile: GlassProfile, h: float, fill_fraction: float = 0.80,
             clearance: float | None = None, floor_clearance: float | None = None,
             brim_clearance: float = 0.006, seed: int = 0):
    """Jittered particle lattice filling the cavity to `fill_fraction` of its usable
    height (usable = rim - floor - brim/floor clearances). An MPM
    lattice at rest density needs no settle-overfill calibration. Returns
    (pos_local[N,3] float32, vol[N] float32); place with x_world = pos + R(quat) @ x_local.
    h is the lattice spacing (grid.dx / ppc)."""
    clearance = h if clearance is None else clearance
    floor_clearance = h if floor_clearance is None else floor_clearance
    usable = profile.rim_z - profile.inner_floor_z - brim_clearance - floor_clearance
    fill_h = fill_fraction * usable
    if fill_h <= 0:
        raise ValueError("fill height is non-positive; check clearances")
    z0 = profile.inner_floor_z + floor_clearance
    r_max = profile.inner_radius - clearance
    xs = np.arange(-r_max + 0.5 * h, r_max, h)
    zs = np.arange(z0 + 0.5 * h, z0 + fill_h, h)
    g = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    g = g + rng.uniform(-0.25 * h, 0.25 * h, size=g.shape)
    keep = (
        (np.linalg.norm(g[:, :2], axis=1) < profile.inner_radius_at_z(g[:, 2]) - clearance)
        & (g[:, 2] >= z0)
        & (g[:, 2] <= z0 + fill_h)
    )
    pos = g[keep].astype(np.float32)
    vol = np.full(len(pos), h**3, dtype=np.float32)
    return pos, vol


__all__ = [
    "GlassProfile",
    "angular_velocity_between",
    "cavity_mask",
    "cup_fill",
    "glass_sdf",
    "glass_sdf_local",
    "project_out_of_solid",
    "quat_conj",
    "quat_from_axis_angle",
    "quat_from_mat",
    "quat_mul",
    "quat_to_mat",
    "solid_mask",
    "world_to_local",
    "write_glass_obj",
]

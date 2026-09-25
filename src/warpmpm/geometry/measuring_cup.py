"""The measured 500 mL plastic measuring cup, as a parametric solid for the pour twin.

Replaces the revolved Genesis glass with the real cup used in the hardware pouring
experiment. The cup is NOT a solid of revolution: a tapered ELLIPTICAL frustum (top
outer 99.60 x 97.82 mm, bottom outer 84.01 x 78.58 mm over H = 99.06 mm), a spout on
the +x side (a smooth x-displacement of both wall sheets, 22.17 mm at the rim apex,
fading over the top 24 mm and across the 37.66 mm spout width), a planar floor at
z = 4.39 mm, and a J-handle on the -x side (render-only: the liquid never reaches it,
and leaving it out keeps the collision SDF box tight). All numbers are the user's
calipers, in the spec frame: origin on the ellipse axis, z = 0 at the external base,
+x toward the spout, +z up.

Three consumers, one set of closed forms:
  collision  build_cup_sdf evaluates the signed field directly on the voxel lattice
             (no mesh, no winding numbers) for Solver.add_sdf_collider. The cavity is
             always the MEASURED surface; the wall/base may be thickened OUTWARD /
             DOWNWARD via extra_wall / extra_base, because the real 2.25 mm wall is
             thinner than the MPM grid spacing at production resolutions and a
             sub-cell wall leaks between grid nodes. Fill volume, levels, and the
             wrench all read the exact inner geometry.
  audits     cavity_mask / solid_mask / project_out_of_solid / cavity_volume mirror
             colliders.glass so pour_franka's per-frame ledger carries over.
  render     make_cup_mesh / write_cup_obj emit the true-dimension watertight body
             (plus the J-handle) for MuJoCo.

Ellipse distances are the scaled radial approximation (sign exact, magnitude within
the ~7% axis ratio of these near-circular sections), which is more than enough for a
contact band and for rescue-net pushes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuringCupSpec:
    """Measured geometry (metres). Defaults are the 500 mL cup's calipers.

    Field <-> measurement sheet: height=H, a/b_top_outer=D_{x,t}^o/2, D_{y,t}^o/2,
    a/b_bot_outer=D_{x,b}^o/2, D_{y,b}^o/2, wall=t_w, base=t_b (= the internal floor
    height z_f), b_top_inner=D_{y,t}^i/2, rim_blend=h_rim, spout_width=W_s,
    spout_drop=h_s, back_to_tip=L_back-tip, total_length=L_total,
    handle_z_min=z_{h,min}, handle_height=H_h, handle_z_ref=z_{h,ref}."""

    height: float = 0.09906
    a_top_outer: float = 0.04980
    b_top_outer: float = 0.04891
    a_bot_outer: float = 0.042005
    b_bot_outer: float = 0.03929
    wall: float = 0.00225
    base: float = 0.00439
    b_top_inner: float = 0.04390
    rim_blend: float = 0.003
    spout_width: float = 0.03766
    spout_drop: float = 0.02400
    back_to_tip: float = 0.12177
    total_length: float = 0.163576
    handle_z_min: float = 0.02571
    handle_height: float = 0.072898
    handle_z_ref: float = 0.08848

    # ---- derived, all straight from the sheet's constraint algebra -------------------
    @property
    def rim_width(self) -> float:
        """Radial rim width, measured on the minor axis and assumed on the major."""
        return self.b_top_outer - self.b_top_inner

    @property
    def a_top_inner(self) -> float:
        return self.a_top_outer - self.rim_width

    @property
    def floor_z(self) -> float:
        return self.base

    @property
    def rim_z(self) -> float:
        return self.height

    @property
    def spout_z0(self) -> float:
        return self.height - self.spout_drop

    @property
    def spout_len(self) -> float:
        """Max +x displacement of the spout: tip minus the undeformed front edge."""
        return self.back_to_tip - 2.0 * self.a_top_outer

    @property
    def tip_x(self) -> float:
        return self.a_top_outer + self.spout_len

    @property
    def handle_x_min(self) -> float:
        return self.tip_x - self.total_length

    @property
    def handle_z_max(self) -> float:
        return self.handle_z_min + self.handle_height

    @property
    def handle_thickness(self) -> float:
        return self.handle_z_max - self.handle_z_ref

    @property
    def brim_volume(self) -> float:
        return self.cavity_volume(self.rim_z - self.floor_z)

    # ---- section semi-axes ------------------------------------------------------------
    def outer_semi_axes(self, z):
        """(a_o(z), b_o(z)): linear taper bottom -> top."""
        s = np.clip(np.asarray(z, dtype=np.float64) / self.height, 0.0, 1.0)
        a = self.a_bot_outer + (self.a_top_outer - self.a_bot_outer) * s
        b = self.b_bot_outer + (self.b_top_outer - self.b_bot_outer) * s
        return a, b

    def inner_semi_axes(self, z):
        """(a_i(z), b_i(z)): the nominal wall offset, smoothstep-blended over the top
        rim_blend band into the measured top opening (the rim is thicker than the
        wall: 5.01 mm vs 2.25 mm)."""
        z = np.asarray(z, dtype=np.float64)
        a_o, b_o = self.outer_semi_axes(z)
        a, b = a_o - self.wall, b_o - self.wall
        u = np.clip((z - (self.height - self.rim_blend)) / self.rim_blend, 0.0, 1.0)
        s = u * u * (3.0 - 2.0 * u)
        return (1.0 - s) * a + s * self.a_top_inner, (1.0 - s) * b + s * self.b_top_inner

    def spout_dx(self, y, z):
        """+x displacement of both wall sheets: Delta_x(y,z) = L_s * B_y(y) * B_z(v)."""
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        v = np.clip((z - self.spout_z0) / self.spout_drop, 0.0, 1.0)
        bz = v * v * (3.0 - 2.0 * v)
        q = 2.0 * y / self.spout_width
        by = np.where(np.abs(q) <= 1.0, (1.0 - q * q) ** 2, 0.0)
        return self.spout_len * by * bz

    # ---- cavity volume (the metering instrument) --------------------------------------
    def cavity_area(self, z):
        """Horizontal cavity cross-section area at height z: the inner ellipse plus the
        exact spout strip (displacing the +x boundary by Delta_x adds integral
        Delta_x dy = L_s * B_z * W_s * 8/15)."""
        z = np.asarray(z, dtype=np.float64)
        a_i, b_i = self.inner_semi_axes(z)
        v = np.clip((z - self.spout_z0) / self.spout_drop, 0.0, 1.0)
        bz = v * v * (3.0 - 2.0 * v)
        return np.pi * a_i * b_i + self.spout_len * bz * self.spout_width * (8.0 / 15.0)

    def cavity_volume(self, depth: float) -> float:
        """Liquid volume (m^3) at fill depth above the internal floor."""
        depth = float(np.clip(depth, 0.0, self.rim_z - self.floor_z))
        if depth <= 0.0:
            return 0.0
        z = np.linspace(self.floor_z, self.floor_z + depth, 2048)
        return float(np.trapezoid(self.cavity_area(z), z))


# --------------------------------------------------------------------------------------
# closed-form signed fields (local/spec frame)
# --------------------------------------------------------------------------------------
def _ell_sdf(x, y, a, b):
    """Scaled radial signed distance to the ellipse x^2/a^2 + y^2/b^2 = 1 (sign exact,
    magnitude ~axis-ratio accurate; these sections are within 7% of circular)."""
    rho = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    return (rho - 1.0) * np.minimum(a, b)


def _spout_warp(spec: MeasuringCupSpec, x, y, z):
    """Pull the +x sheet back by the spout displacement so the warped point can be
    tested against the plain ellipse. The blend-in over 0 < x < 0.3*a_bot keeps the
    field smooth through the axis (Delta_x is only nonzero at |y| < W_s/2, far inside
    the section there, so the warp never misclassifies a boundary point)."""
    w = np.clip(x / (0.3 * spec.a_bot_outer), 0.0, 1.0)
    return x - w * spec.spout_dx(y, z)


def cavity_sdf_local(points, spec: MeasuringCupSpec):
    """Approximate signed distance to the cavity boundary (negative INSIDE the liquid
    space): inner wall sheets (spout-warped ellipse) and the planar floor. Open top."""
    p = np.asarray(points, dtype=np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    a_i, b_i = spec.inner_semi_axes(z)
    lateral = _ell_sdf(_spout_warp(spec, x, y, z), y, a_i, b_i)
    return np.maximum(lateral, spec.floor_z - z)


def solid_sdf_local(points, spec: MeasuringCupSpec, extra_wall: float = 0.0,
                    extra_base: float = 0.0):
    """Approximate signed distance to the cup SOLID (negative inside the plastic):
    inside the outer sheets, below the rim plane, above the (extended) base plane,
    and not in the cavity. extra_wall/extra_base thicken the collision solid outward
    and downward only; the cavity stays the measured surface."""
    p = np.asarray(points, dtype=np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    a_o, b_o = spec.outer_semi_axes(z)
    xw = _spout_warp(spec, x, y, z)
    outer = _ell_sdf(xw, y, a_o + extra_wall, b_o + extra_wall)
    return np.maximum.reduce([
        outer,
        z - spec.rim_z,
        -(z + extra_base),
        -cavity_sdf_local(p, spec),
    ])


# --------------------------------------------------------------------------------------
# masks / rescue net / fill (pour_franka's per-frame ledger, colliders.glass semantics)
# --------------------------------------------------------------------------------------
def cavity_mask(points_world, pos, quat, spec: MeasuringCupSpec, pad: float = 0.0):
    """Particles inside the cavity (pad loosens the wall test; z within floor..rim)."""
    from warpmpm.colliders.glass import world_to_local

    local = world_to_local(np.asarray(points_world, dtype=np.float64), pos, quat)
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    a_i, b_i = spec.inner_semi_axes(z)
    lateral = _ell_sdf(_spout_warp(spec, x, y, z), y, a_i, b_i)
    return (lateral < pad) & (z >= spec.floor_z - 1e-3) & (z <= spec.rim_z)


def solid_mask(points_world, pos, quat, spec: MeasuringCupSpec, tol: float = 0.0,
               extra_wall: float = 0.0, extra_base: float = 0.0):
    """Particles embedded in the cup solid (the leak audit). Audit the same thickened
    solid the collider enforces, or legitimate near-wall liquid reads as embedded."""
    from warpmpm.colliders.glass import world_to_local

    local = world_to_local(np.asarray(points_world, dtype=np.float64), pos, quat)
    return solid_sdf_local(local, spec, extra_wall, extra_base) < -tol


def project_out_of_solid(x, v, pos, quat, spec: MeasuringCupSpec, clearance: float = 0.0,
                         solid_velocity=None, extra_wall: float = 0.0,
                         extra_base: float = 0.0):
    """Rescue net for boundary creep, the colliders.glass one on the parametric field:
    embedded particles move along the finite-difference field gradient to `clearance`
    outside the surface and lose the inward normal velocity relative to the wall.
    Returns (x_new, v_new, n_projected)."""
    from warpmpm.colliders.glass import quat_to_mat, world_to_local

    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    local_all = world_to_local(x, pos, quat)
    sdf = solid_sdf_local(local_all, spec, extra_wall, extra_base)
    inside = sdf < 0.0
    n_bad = int(inside.sum())
    if n_bad == 0:
        return x, v, 0
    r = quat_to_mat(quat)
    local = local_all[inside]
    h = 1e-5
    grad = np.stack(
        [solid_sdf_local(local + off, spec, extra_wall, extra_base)
         - solid_sdf_local(local - off, spec, extra_wall, extra_base)
         for off in (np.array([h, 0, 0]), np.array([0, h, 0]), np.array([0, 0, h]))],
        axis=1,
    )
    n_local = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    push = (clearance - sdf[inside])[:, None] * n_local
    x_new = x.copy()
    x_new[inside] = (local + push) @ r.T + np.asarray(pos, dtype=np.float64)
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


def cup_fill(spec: MeasuringCupSpec, h: float, fill_fraction: float = 0.80,
             clearance: float | None = None, floor_clearance: float | None = None,
             brim_clearance: float = 0.006, seed: int = 0):
    """Jittered particle lattice filling the cavity to `fill_fraction` of its usable
    height (same semantics as colliders.glass.cup_fill). The lattice stays inside the
    plain inner ellipse (the spout sliver above spout_z0 holds under a millimetre of
    displacement at any realistic fill and is not worth seeding). Returns
    (pos_local[N,3] f32, vol[N] f32)."""
    clearance = h if clearance is None else clearance
    floor_clearance = h if floor_clearance is None else floor_clearance
    usable = spec.rim_z - spec.floor_z - brim_clearance - floor_clearance
    fill_h = fill_fraction * usable
    if fill_h <= 0:
        raise ValueError("fill height is non-positive; check clearances")
    z0 = spec.floor_z + floor_clearance
    r_max = max(spec.a_top_inner, spec.b_top_inner)
    xs = np.arange(-r_max + 0.5 * h, r_max, h)
    zs = np.arange(z0 + 0.5 * h, z0 + fill_h, h)
    g = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    g = g + rng.uniform(-0.25 * h, 0.25 * h, size=g.shape)
    a_i, b_i = spec.inner_semi_axes(g[:, 2])
    keep = (
        (_ell_sdf(g[:, 0], g[:, 1], a_i, b_i) < -clearance)
        & (g[:, 2] >= z0)
        & (g[:, 2] <= z0 + fill_h)
    )
    pos = g[keep].astype(np.float32)
    vol = np.full(len(pos), h**3, dtype=np.float32)
    return pos, vol


# --------------------------------------------------------------------------------------
# collision SDF (analytic voxelization; no mesh, no winding numbers)
# --------------------------------------------------------------------------------------
def build_cup_sdf(spec: MeasuringCupSpec, res: int = 160, margin: float = 0.008,
                  extra_wall: float = 0.0, extra_base: float = 0.0):
    """Voxelize solid_sdf_local straight onto a cubic lattice as an SDFData for
    Solver.add_sdf_collider. `margin` metres of stored field beyond the solid on every
    side must exceed the collider's contact band (add_sdf_collider enforces this).
    The handle is excluded: render-only, and skipping it keeps the box tight."""
    from warpmpm.geometry.mesh_sdf import SDFData

    lo = np.array([-(spec.a_top_outer + extra_wall) - margin,
                   -(spec.b_top_outer + extra_wall) - margin,
                   -extra_base - margin])
    hi = np.array([spec.tip_x + extra_wall + margin,
                   (spec.b_top_outer + extra_wall) + margin,
                   spec.height + margin])
    center = 0.5 * (lo + hi)
    span = float((hi - lo).max())
    cell = span / (res - 1)
    origin = center - 0.5 * span
    axes = [origin[d] + cell * np.arange(res) for d in range(3)]
    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    signed = solid_sdf_local(pts, spec, extra_wall, extra_base).reshape(res, res, res)
    grads = np.stack(np.gradient(signed, cell, edge_order=2), axis=-1)
    return SDFData(values=signed.astype(np.float64), grads=grads.astype(np.float64),
                   origin=origin.astype(np.float64), cell=float(cell),
                   sdf_max=float(np.abs(signed).max()))


# --------------------------------------------------------------------------------------
# render mesh (true dimensions; watertight body + J-handle)
# --------------------------------------------------------------------------------------
def _ring(spec: MeasuringCupSpec, z: float, a: float, b: float, n: int):
    """Closed section curve at height z: the ellipse with the +x sheet displaced by the
    spout. Smooth because Delta_x -> 0 before |y| reaches the section's flanks."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = a * np.cos(th)
    y = b * np.sin(th)
    x = x + np.where(np.cos(th) > 0.0, spec.spout_dx(y, np.full(n, z)), 0.0)
    return np.stack([x, y, np.full(n, z)], axis=1)


def _loft(rings: list[np.ndarray]) -> tuple[np.ndarray, list[list[int]]]:
    n = len(rings[0])
    verts = np.concatenate(rings, axis=0)
    faces = []
    for k in range(len(rings) - 1):
        o0, o1 = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces += [[o0 + i, o0 + j, o1 + j], [o0 + i, o1 + j, o1 + i]]
    return verts, faces


def make_cup_mesh(spec: MeasuringCupSpec, extra_wall: float = 0.0, extra_base: float = 0.0,
                  n_theta: int = 96, n_z: int = 40):
    """Watertight cup body (verts, faces): bottom cap, outer wall, rim annulus, inner
    wall, floor cap. True dimensions by default; extra_wall/extra_base mirror the
    collision thickening for debug views of what the solver actually enforces."""
    z_bot = -extra_base
    z_wall = np.concatenate([[z_bot], np.linspace(0.0, spec.height, n_z)])
    outer = []
    for z in z_wall:
        a, b = spec.outer_semi_axes(z)
        outer.append(_ring(spec, float(z), float(a) + extra_wall, float(b) + extra_wall,
                           n_theta))
    inner = []
    for z in np.linspace(spec.height, spec.floor_z, n_z):
        a, b = spec.inner_semi_axes(z)
        inner.append(_ring(spec, float(z), float(a), float(b), n_theta))
    verts, faces = _loft(outer + inner)  # outer up, rim strip (last outer->first inner
    # rings share z=H so the loft segment between them IS the rim annulus), inner down
    n = n_theta
    bot_c = len(verts)
    floor_c = len(verts) + 1
    verts = np.concatenate([verts, [[0.0, 0.0, z_bot], [0.0, 0.0, spec.floor_z]]], axis=0)
    faces = list(faces)
    last_inner = (len(outer) + len(inner) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        faces.append([bot_c, j, i])                                # bottom cap (-z out)
        faces.append([floor_c, last_inner + i, last_inner + j])    # cavity floor (+z up)
    return verts, np.asarray(faces, dtype=np.int64)


def make_handle_mesh(spec: MeasuringCupSpec, width: float = 0.020, n_path: int = 24):
    """Render-only J-handle: a rectangular tube swept in the y=0 plane from the upper
    rear wall, out to the measured outermost point, down the column, curling to the
    free lower tip. Ends capped; the root end is embedded in the wall so the junction
    never shows. Path centreline honours z_h,max (top surface), x_h,min (outer face)
    and z_h,min (lowest tip surface)."""
    t2 = 0.5 * spec.handle_thickness
    w2 = 0.5 * width
    a_top, _ = spec.outer_semi_axes(spec.height - 0.5 * spec.handle_thickness)
    x_col = spec.handle_x_min + t2                       # column centreline
    z_top = spec.handle_z_max - t2                       # top arm centreline
    z_tip = spec.handle_z_min + t2                       # tip centreline
    ctrl = np.array([
        [-(a_top - 0.004), z_top],
        [x_col + 0.55 * (-(a_top - 0.004) - x_col), z_top],
        [x_col, z_top - 0.006],
        [x_col, 0.55 * (z_top + z_tip)],
        [x_col + 0.004, z_tip + 0.014],
        [x_col + 0.016, z_tip + 0.002],
        [x_col + 0.026, z_tip],
    ])
    # Catmull-Rom-ish resample for a smooth sweep
    t = np.linspace(0.0, 1.0, n_path)
    seg = np.linspace(0.0, 1.0, len(ctrl))
    path = np.stack([np.interp(t, seg, ctrl[:, 0]), np.interp(t, seg, ctrl[:, 1])], axis=1)
    for _ in range(2):  # midpoint smoothing keeps the corners from rendering as kinks
        path[1:-1] = 0.5 * path[1:-1] + 0.25 * (path[:-2] + path[2:])

    tang = np.gradient(path, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)    # in-plane normal
    rings = []
    for (px, pz), (nx, nz) in zip(path, nrm, strict=True):
        ring = [
            [px + t2 * nx, -w2, pz + t2 * nz],
            [px + t2 * nx, +w2, pz + t2 * nz],
            [px - t2 * nx, +w2, pz - t2 * nz],
            [px - t2 * nx, -w2, pz - t2 * nz],
        ]
        rings.append(np.asarray(ring))
    verts = np.concatenate(rings, axis=0)
    faces = []
    for k in range(len(rings) - 1):
        o0, o1 = 4 * k, 4 * (k + 1)
        for i in range(4):
            j = (i + 1) % 4
            faces += [[o0 + i, o0 + j, o1 + j], [o0 + i, o1 + j, o1 + i]]
    last = 4 * (len(rings) - 1)
    faces += [[0, 2, 1], [0, 3, 2], [last, last + 1, last + 2], [last, last + 2, last + 3]]
    return verts, np.asarray(faces, dtype=np.int64)


def write_cup_obj(spec: MeasuringCupSpec, path, with_handle: bool = True,
                  n_theta: int = 96, n_z: int = 40):
    """True-dimension render OBJ (body + J-handle) for the MuJoCo glass asset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts, faces = make_cup_mesh(spec, n_theta=n_theta, n_z=n_z)
    if with_handle:
        hv, hf = make_handle_mesh(spec)
        faces = np.concatenate([faces, hf + len(verts)], axis=0)
        verts = np.concatenate([verts, hv], axis=0)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# warpmpm measuring cup: H={spec.height} top_o={2*spec.a_top_outer}x"
                f"{2*spec.b_top_outer} bot_o={2*spec.a_bot_outer}x{2*spec.b_bot_outer} "
                f"wall={spec.wall} base={spec.base} tip_x={spec.tip_x:.6f}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return path


__all__ = [
    "MeasuringCupSpec",
    "build_cup_sdf",
    "cavity_mask",
    "cavity_sdf_local",
    "cup_fill",
    "make_cup_mesh",
    "make_handle_mesh",
    "project_out_of_solid",
    "solid_mask",
    "solid_sdf_local",
    "write_cup_obj",
]

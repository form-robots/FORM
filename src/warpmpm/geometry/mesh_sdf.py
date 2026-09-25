"""Mesh -> signed distance field, self-contained in numpy.

A watertight triangle mesh (verts (V,3), faces (F,3)) is voxelized on a cubic grid covering
its bounding box plus a margin. Each voxel stores the signed distance to the surface (negative
inside the solid, positive outside) and the field gradient (an outward-pointing normal field).
The collider queries this with a body-frame transform and trilinear interpolation.

Sign comes from the generalized winding number (Jacobson et al. 2013): the sum of signed solid
angles subtended by the faces, which is ~1 inside a consistently oriented closed mesh and ~0
outside, and is robust on concave shapes (a cup). Magnitude is the unsigned distance to the
nearest triangle: an exact vectorized point-triangle distance, optionally accelerated by a
scipy cKDTree over a dense surface sampling when scipy is installed. The global orientation
sign is auto-detected from a known-interior probe, so the revolution mesh does not need its
faces hand-oriented.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SDFData:
    """A cubic signed-distance field in the mesh (body) frame.

    values[res,res,res]  signed distance in metres (negative inside the solid)
    grads[res,res,res,3] outward field gradient (normalized later by the kernel)
    origin (3,)          mesh-frame coordinate of voxel index (0,0,0)
    cell                 mesh-frame metres per voxel (isotropic, cubic grid)
    sdf_max              max stored distance (the outside-grid proxy bound)
    """

    values: np.ndarray
    grads: np.ndarray
    origin: np.ndarray
    cell: float
    sdf_max: float

    @property
    def res(self) -> int:
        return int(self.values.shape[0])


@dataclass
class CDFData:
    """A cubic colored-distance field around an OPEN oriented mid-surface, for CPIC
    thin-boundary colliders (Hu et al., SIGGRAPH 2018).

    Unlike SDFData, the surface has no interior: values is the signed distance to the
    surface sheet itself, with the sign taken from the nearest face's normal (which
    side of the sheet the point is on). The field is only meaningful within a narrow
    band of the sheet AND away from the sheet's open rim, where no consistent side
    exists; valid marks that region.

    values[res,res,res]  signed distance to the mid-surface (side-signed) in metres
    valid[res,res,res]   1.0 where the distance and side are trustworthy, else 0.0
    origin (3,)          mesh-frame coordinate of voxel index (0,0,0)
    cell                 mesh-frame metres per voxel (isotropic, cubic grid)
    band                 metres from the sheet within which valid can be 1
    """

    values: np.ndarray
    valid: np.ndarray
    origin: np.ndarray
    cell: float
    band: float

    @property
    def res(self) -> int:
        return int(self.values.shape[0])


# --------------------------------------------------------------------------------------
# procedural meshes
# --------------------------------------------------------------------------------------
def revolve_profile(profile_rz, n_theta: int = 64):
    """Revolve a closed 2D profile (r,z), r>=0, around the z-axis into a watertight solid.

    profile_rz is the closed polygon bounding the solid cross-section (the last point connects
    back to the first). Vertices on the axis (r ~ 0) collapse to a single shared point so the
    caps are triangle fans rather than degenerate quads. Face winding is uniform around the
    revolution, giving a consistently oriented closed mesh (the global in/out sign is resolved
    later by build_sdf, so the orientation need not be outward here).
    """
    profile = np.asarray(profile_rz, dtype=float)
    P = len(profile)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(thetas), np.sin(thetas)
    axis_eps = 1e-9

    verts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for r, z in profile:
        if r <= axis_eps:
            idx = len(verts)
            verts.append((0.0, 0.0, float(z)))
            rings.append([idx] * n_theta)
        else:
            base = len(verts)
            for j in range(n_theta):
                verts.append((float(r) * ct[j], float(r) * st[j], float(z)))
            rings.append([base + j for j in range(n_theta)])

    faces: list[tuple[int, int, int]] = []
    for i in range(P):
        i2 = (i + 1) % P
        ri, ri2 = rings[i], rings[i2]
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a, b, c, d = ri[j], ri[j2], ri2[j2], ri2[j]
            on_axis_i = a == b
            on_axis_i2 = c == d
            if on_axis_i and on_axis_i2:
                continue
            if on_axis_i:
                faces.append((a, c, d))
            elif on_axis_i2:
                faces.append((a, b, c))
            else:
                faces.append((a, b, c))
                faces.append((a, c, d))
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def revolve_profile_open(profile_rz, n_theta: int = 64):
    """Revolve an OPEN 2D polyline (r,z), r>=0, around the z-axis into an open surface
    (no closure back to the first point, no caps). Shares revolve_profile's ring
    machinery; the result is the mid-surface input for build_surface_cdf. Face winding
    is uniform, so face normals consistently point to one side of the sheet."""
    profile = np.asarray(profile_rz, dtype=float)
    P = len(profile)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(thetas), np.sin(thetas)
    axis_eps = 1e-9

    verts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for r, z in profile:
        if r <= axis_eps:
            idx = len(verts)
            verts.append((0.0, 0.0, float(z)))
            rings.append([idx] * n_theta)
        else:
            base = len(verts)
            for j in range(n_theta):
                verts.append((float(r) * ct[j], float(r) * st[j], float(z)))
            rings.append([base + j for j in range(n_theta)])

    faces: list[tuple[int, int, int]] = []
    for i in range(P - 1):
        ri, ri2 = rings[i], rings[i + 1]
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a, b, c, d = ri[j], ri[j2], ri2[j2], ri2[j]
            on_axis_i = a == b
            on_axis_i2 = c == d
            if on_axis_i and on_axis_i2:
                continue
            if on_axis_i:
                faces.append((a, c, d))
            elif on_axis_i2:
                faces.append((a, b, c))
            else:
                faces.append((a, b, c))
                faces.append((a, c, d))
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def make_cup_mesh(
    inner_radius: float = 0.035,
    wall_thickness: float = 0.006,
    height: float = 0.085,
    base_thickness: float = 0.008,
    n_theta: int = 64,
):
    """A watertight open-top cup (surface of revolution): a base disk plus an annular wall,
    open at the top so a fluid filled inside pours over the rim when the cup is tilted.

    Returns (verts (V,3), faces (F,3)) centred on the z-axis with the base sitting at z=0.
    """
    r_in = inner_radius
    r_out = inner_radius + wall_thickness
    h = height
    t = base_thickness
    # closed cross-section polygon of the cup material, counter-clockwise in (r, z):
    #   axis-bottom -> base outer corner -> wall outer top -> rim -> wall inner top of cavity
    #   -> cavity floor -> back to the axis
    profile = [
        (0.0, 0.0),      # base centre, bottom
        (r_out, 0.0),    # base outer, bottom
        (r_out, h),      # outer wall, top
        (r_in, h),       # rim (flat top)
        (r_in, t),       # inner wall, down to the cavity floor
        (0.0, t),        # cavity floor centre
    ]
    return revolve_profile(profile, n_theta=n_theta)


# --------------------------------------------------------------------------------------
# winding number (sign) and point-triangle distance (magnitude)
# --------------------------------------------------------------------------------------
def _winding_number(points: np.ndarray, verts: np.ndarray, faces: np.ndarray,
                    chunk: int = 4096) -> np.ndarray:
    """Generalized winding number at each point, summed signed solid angles / (4 pi).

    ~1 (or ~-1) inside a consistently oriented closed mesh, ~0 outside. Chunked over points
    and vectorized over faces (Van Oosterom-Strackee solid-angle formula).
    """
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    out = np.empty(len(points), dtype=float)
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk][:, None, :]            # (m,1,3)
        A = a[None] - p                                # (m,F,3)
        B = b[None] - p
        C = c[None] - p
        la = np.linalg.norm(A, axis=2)
        lb = np.linalg.norm(B, axis=2)
        lc = np.linalg.norm(C, axis=2)
        num = np.einsum("mfi,mfi->mf", A, np.cross(B, C))
        den = (la * lb * lc
               + np.einsum("mfi,mfi->mf", A, B) * lc
               + np.einsum("mfi,mfi->mf", B, C) * la
               + np.einsum("mfi,mfi->mf", C, A) * lb)
        omega = 2.0 * np.arctan2(num, den)
        out[s:s + chunk] = omega.sum(axis=1) / (4.0 * np.pi)
    return out


def _point_triangle_distance_exact(points: np.ndarray, a: np.ndarray, b: np.ndarray,
                                   c: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Exact min distance from each point to the triangle set (Ericson region test),
    vectorized over triangles, chunked over points. The numpy fallback when scipy is absent."""
    ab = b - a
    ac = c - a
    out = np.empty(len(points), dtype=float)
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk][:, None, :]            # (m,1,3)
        ap = p - a[None]
        d1 = np.einsum("mfi,fi->mf", ap, ab)
        d2 = np.einsum("mfi,fi->mf", ap, ac)
        bp = p - b[None]
        d3 = np.einsum("mfi,fi->mf", bp, ab)
        d4 = np.einsum("mfi,fi->mf", bp, ac)
        cp = p - c[None]
        d5 = np.einsum("mfi,fi->mf", cp, ab)
        d6 = np.einsum("mfi,fi->mf", cp, ac)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = va + vb + vc
        # barycentric clamp into the triangle (handle the 7 Voronoi regions)
        v = np.zeros_like(d1)
        w = np.zeros_like(d1)
        # default: interior face
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(denom) > 1e-30, 1.0 / denom, 0.0)
            v_face = vb * inv
            w_face = vc * inv
        v = v_face
        w = w_face
        # vertex / edge regions override the interior solution
        regA = (d1 <= 0) & (d2 <= 0)
        regB = (d3 >= 0) & (d4 <= d3)
        regC = (d6 >= 0) & (d5 <= d6)
        regAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        regAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        regBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        v = np.where(regA, 0.0, v); w = np.where(regA, 0.0, w)
        v = np.where(regB, 1.0, v); w = np.where(regB, 0.0, w)
        v = np.where(regC, 0.0, v); w = np.where(regC, 1.0, w)
        tAB = np.where(np.abs(d1 - d3) > 1e-30, d1 / (d1 - d3), 0.0)
        v = np.where(regAB, tAB, v); w = np.where(regAB, 0.0, w)
        tAC = np.where(np.abs(d2 - d6) > 1e-30, d2 / (d2 - d6), 0.0)
        v = np.where(regAC, 0.0, v); w = np.where(regAC, tAC, w)
        tBC = np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-30,
                       (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0)
        v = np.where(regBC, 1.0 - tBC, v); w = np.where(regBC, tBC, w)
        closest = a[None] + v[..., None] * ab[None] + w[..., None] * ac[None]
        dist = np.linalg.norm(p - closest, axis=2)
        out[s:s + chunk] = dist.min(axis=1)
    return out


def _sample_surface(verts: np.ndarray, faces: np.ndarray, spacing: float) -> np.ndarray:
    """Dense barycentric samples on the mesh surface, ~spacing apart, for a KD-tree query."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pts = [verts]  # vertices always included
    for i in range(len(faces)):
        n = int(max(1, np.ceil(np.sqrt(areas[i]) / max(spacing, 1e-9))))
        if n <= 1:
            pts.append(((a[i] + b[i] + c[i]) / 3.0)[None])
            continue
        u, vv = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1), indexing="ij")
        u = u.ravel(); vv = vv.ravel()
        m = (u + vv) <= 1.0
        u, vv = u[m], vv[m]
        pts.append(a[i] + u[:, None] * (b[i] - a[i]) + vv[:, None] * (c[i] - a[i]))
    return np.concatenate(pts, axis=0)


def _unsigned_distance(grid_pts: np.ndarray, verts: np.ndarray, faces: np.ndarray,
                      cell: float) -> np.ndarray:
    """Unsigned distance from each grid point to the mesh surface. Uses a scipy cKDTree over a
    dense surface sampling when available (fast, accurate to the sample spacing); otherwise the
    exact vectorized point-triangle distance."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        a = verts[faces[:, 0]]; b = verts[faces[:, 1]]; c = verts[faces[:, 2]]
        return _point_triangle_distance_exact(grid_pts, a, b, c)
    samples = _sample_surface(verts, faces, spacing=0.4 * cell)
    tree = cKDTree(samples)
    dist, _ = tree.query(grid_pts, workers=-1)
    return dist


# --------------------------------------------------------------------------------------
# build / cache
# --------------------------------------------------------------------------------------
def build_sdf(verts: np.ndarray, faces: np.ndarray, res: int = 64, margin_cells: float = 4.0,
             interior_probe: np.ndarray | None = None) -> SDFData:
    """Voxelize a watertight mesh into a cubic SDFData covering its bbox plus a margin.

    res            voxels per axis (cubic grid)
    margin_cells   empty voxels of padding around the mesh bbox on every side
    interior_probe a point known to be inside the solid, used to fix the global sign when the
                   mesh face orientation is unknown. Defaults to the mesh centroid, which is
                   inside for a convex solid; pass an explicit interior point for thin/concave
                   shells (e.g. a point in the middle of a cup wall).
    """
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = 0.5 * (lo + hi)
    span = float((hi - lo).max())
    # cubic grid of side L with `margin_cells` of padding; res points across it
    cell = (span) / (res - 1 - 2 * margin_cells)
    L = cell * (res - 1)
    origin = center - 0.5 * L
    axes = [origin[d] + cell * np.arange(res) for d in range(3)]
    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    unsigned = _unsigned_distance(grid_pts, verts, faces, cell)
    wn = _winding_number(grid_pts, verts, faces)
    # auto-fix global orientation: winding at a known-interior probe should be the inside value
    probe = center if interior_probe is None else np.asarray(interior_probe, dtype=float)
    w_probe = _winding_number(probe[None], verts, faces)[0]
    inside = (wn * np.sign(w_probe)) > 0.5 if abs(w_probe) > 0.1 else (np.abs(wn) > 0.5)
    signed = np.where(inside, -unsigned, unsigned).reshape(res, res, res)

    grads = np.stack(np.gradient(signed, cell, edge_order=2), axis=-1)
    return SDFData(values=signed.astype(np.float64), grads=grads.astype(np.float64),
                   origin=origin.astype(np.float64), cell=float(cell),
                   sdf_max=float(np.abs(signed).max()))


def _sample_surface_with_faces(verts: np.ndarray, faces: np.ndarray, spacing: float
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Like _sample_surface but each sample carries its source face index, so the
    nearest sample also yields the nearest face's normal for side classification."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pts: list[np.ndarray] = []
    fid: list[np.ndarray] = []
    for i in range(len(faces)):
        n = int(max(1, np.ceil(np.sqrt(areas[i]) / max(spacing, 1e-9))))
        if n <= 1:
            pts.append(((a[i] + b[i] + c[i]) / 3.0)[None])
            fid.append(np.array([i]))
            continue
        u, vv = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1),
                            indexing="ij")
        u = u.ravel(); vv = vv.ravel()
        m = (u + vv) <= 1.0
        u, vv = u[m], vv[m]
        pts.append(a[i] + u[:, None] * (b[i] - a[i]) + vv[:, None] * (c[i] - a[i]))
        fid.append(np.full(m.sum(), i))
    return np.concatenate(pts, axis=0), np.concatenate(fid, axis=0)


def _boundary_loop_samples(verts: np.ndarray, faces: np.ndarray, spacing: float
                           ) -> np.ndarray:
    """Dense samples on the mesh's open-boundary edges (edges with exactly one
    incident face); empty (0,3) for a closed mesh."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    open_edges = uniq[counts == 1]
    if len(open_edges) == 0:
        return np.zeros((0, 3))
    pts = []
    for i0, i1 in open_edges:
        a, b = verts[i0], verts[i1]
        n = int(max(2, np.ceil(np.linalg.norm(b - a) / max(spacing, 1e-9)) + 1))
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(a[None] * (1 - t) + b[None] * t)
    return np.concatenate(pts, axis=0)


def build_surface_cdf(verts: np.ndarray, faces: np.ndarray, res: int = 64,
                      band_cells: float = 3.0, margin_cells: float = 4.0) -> CDFData:
    """Voxelize an OPEN oriented mid-surface into a CDFData (side-signed distance plus
    a validity mask) for CPIC thin-boundary colliders.

    Distance and side come from the nearest surface sample (spacing 0.4 cells, the
    accuracy build_sdf already accepts): magnitude = distance to the sample, sign =
    which side of that sample's face normal the point is on. Validity requires BOTH
    |d| <= band AND distance to the surface's open rim > band: past the rim no
    consistent side exists, and without the rim exclusion trilinear interpolation
    would fabricate a phantom surface there. The excluded rim tube (radius ~band) is
    inherent to open surfaces; material flows through it untagged, which is the
    behavior a pour needs at a glass rim.
    """
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = 0.5 * (lo + hi)
    span = float((hi - lo).max())
    cell = span / (res - 1 - 2 * margin_cells)
    band = band_cells * cell
    L = cell * (res - 1)
    origin = center - 0.5 * L
    axes = [origin[d] + cell * np.arange(res) for d in range(3)]
    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    fa = verts[faces[:, 0]]
    fb = verts[faces[:, 1]]
    fc = verts[faces[:, 2]]
    normals = np.cross(fb - fa, fc - fa)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)

    samples, sample_face = _sample_surface_with_faces(verts, faces, spacing=0.4 * cell)
    rim = _boundary_loop_samples(verts, faces, spacing=0.4 * cell)
    try:
        from scipy.spatial import cKDTree
        dist, idx = cKDTree(samples).query(grid_pts, workers=-1)
        rim_dist = (cKDTree(rim).query(grid_pts, workers=-1)[0]
                    if len(rim) else np.full(len(grid_pts), np.inf))
    except Exception:
        # numpy fallback: chunked argmin over samples (procedural meshes are small)
        dist = np.empty(len(grid_pts))
        idx = np.empty(len(grid_pts), dtype=np.int64)
        for s in range(0, len(grid_pts), 2048):
            d2 = np.linalg.norm(grid_pts[s:s + 2048, None, :] - samples[None], axis=2)
            idx[s:s + 2048] = d2.argmin(axis=1)
            dist[s:s + 2048] = d2.min(axis=1)
        if len(rim):
            rim_dist = np.empty(len(grid_pts))
            for s in range(0, len(grid_pts), 2048):
                rim_dist[s:s + 2048] = np.linalg.norm(
                    grid_pts[s:s + 2048, None, :] - rim[None], axis=2).min(axis=1)
        else:
            rim_dist = np.full(len(grid_pts), np.inf)

    n_near = normals[sample_face[idx]]
    side = np.sign(np.einsum("ij,ij->i", grid_pts - samples[idx], n_near))
    side[side == 0.0] = 1.0
    signed = (side * dist).reshape(res, res, res)
    valid = ((dist <= band) & (rim_dist > band)).astype(np.float64
                                                        ).reshape(res, res, res)
    return CDFData(values=signed.astype(np.float64), valid=valid,
                   origin=origin.astype(np.float64), cell=float(cell),
                   band=float(band))


_CDF_CACHE_VERSION = 1


def save_cdf(cdf: CDFData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=cdf.values, valid=cdf.valid, origin=cdf.origin,
                        cell=cdf.cell, band=cdf.band)


def load_cdf(path: str | Path) -> CDFData:
    d = np.load(path)
    return CDFData(values=d["values"], valid=d["valid"], origin=d["origin"],
                   cell=float(d["cell"]), band=float(d["band"]))


def build_surface_cdf_cached(verts, faces, res: int = 64, band_cells: float = 3.0,
                             margin_cells: float = 4.0,
                             cache_dir: str | Path | None = None) -> CDFData:
    """build_surface_cdf with a versioned cache keyed by geometry and grid."""
    if cache_dir is None:
        return build_surface_cdf(verts, faces, res, band_cells, margin_cells)
    cache_dir = Path(cache_dir)
    h = hashlib.sha1()
    for arr in (np.array([_CDF_CACHE_VERSION], dtype=np.int64),
                np.ascontiguousarray(verts, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int64),
                np.array([res, band_cells, margin_cells], dtype=np.float64)):
        h.update(arr.tobytes())
    path = cache_dir / f"cdf_{h.hexdigest()[:16]}.npz"
    if path.exists():
        return load_cdf(path)
    cdf = build_surface_cdf(verts, faces, res, band_cells, margin_cells)
    save_cdf(cdf, path)
    return cdf


_SDF_CACHE_VERSION = 2


def _hashkey(verts, faces, res, margin_cells, interior_probe=None) -> str:
    verts64 = np.ascontiguousarray(verts, dtype=np.float64)
    if interior_probe is None:
        probe = 0.5 * (verts64.min(axis=0) + verts64.max(axis=0))
    else:
        probe = np.asarray(interior_probe, dtype=np.float64)
    if probe.shape != (3,) or not np.isfinite(probe).all():
        raise ValueError(f"interior_probe must be a finite 3-vector, got {probe}")
    h = hashlib.sha1()
    for arr in (np.array([_SDF_CACHE_VERSION], dtype=np.int64),
                verts64,
                np.ascontiguousarray(faces, dtype=np.int64),
                np.array([res, margin_cells], dtype=np.float64),
                np.ascontiguousarray(probe, dtype=np.float64)):
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def save_sdf(sdf: SDFData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=sdf.values, grads=sdf.grads, origin=sdf.origin,
                        cell=sdf.cell, sdf_max=sdf.sdf_max)


def load_sdf(path: str | Path) -> SDFData:
    d = np.load(path)
    return SDFData(values=d["values"], grads=d["grads"], origin=d["origin"],
                   cell=float(d["cell"]), sdf_max=float(d["sdf_max"]))


def build_sdf_cached(verts, faces, res: int = 64, margin_cells: float = 4.0,
                    cache_dir: str | Path | None = None, interior_probe=None) -> SDFData:
    """build_sdf with a versioned cache keyed by geometry, grid, and sign probe."""
    if cache_dir is None:
        return build_sdf(verts, faces, res, margin_cells, interior_probe)
    cache_dir = Path(cache_dir)
    key = _hashkey(verts, faces, res, margin_cells, interior_probe)
    path = cache_dir / f"sdf_{key}.npz"
    if path.exists():
        return load_sdf(path)
    sdf = build_sdf(verts, faces, res, margin_cells, interior_probe)
    save_sdf(sdf, path)
    return sdf

"""Grid-consistent (Bubnov-Galerkin) weak form for the fixed-corotated elastic pair.

Tests on the discrete space the forward solver uses, the quadratic B-spline
grid basis, node by node (the granular counterpart is
``ident/weakform/grid_assembly.py``). An analytic window that does not vanish
on a cube's faces leaks the floor-contact traction and biases the recovered
stiffness by order ten percent; the grid basis has no such leak.

The discrete MPM momentum balance is exact at every grid node the collider does
not touch. warp-mpm deposits the internal force as

    f_i = - sum_p V_p^0 tau_p grad N_i(x_p),        tau = J sigma (Kirchhoff)

and then ``grid_normalization_and_gravity`` sets
v_out = v_in / m_i + dt g. So for a node whose velocity the collider never
overwrites,

    sum_p V_p^0 tau_p[d,:] . grad N_i(x_p) = m_i (g - a_i)_d,
    m_i = sum_p m_p N_i(x_p).

A surface node carries the balance because MPM applies zero traction there, so
no shape-fitted window is needed.

Linearity in theta. For the fixed corotated model with energy
psi = mu ||F - R||_F^2 + (lam/2) (J - 1)^2 the Kirchhoff stress is

    tau = mu * 2 (F - R) F^T  +  lam * J (J - 1) I,

which is exactly ``kirchoff_stress_FCR`` in the fork, so the two stress columns
are read off with no approximation. In Cauchy form (the form the plan states)

    sigma_mu  = (2 / J) (F - R) F^T,     sigma_lam = (J - 1) I.

Updated-Lagrangian pairing. Volumes and stresses
are paired as CURRENT volume with CAUCHY stress, V_p = J_p V_p^0 with sigma_p.
That product is identically V_p^0 tau_p, the pairing the engine uses, so the two
readings agree to the last bit and the assembler carries the reference volume
plus J rather than a separately dumped current volume. Mass is frame invariant,
m_p = rho_0 V_p^0 = (rho_0 / J_p) (J_p V_p^0), so the load term needs no J at
all. Both stress columns are symmetrized before use, matching the engine's own
``(tau + tau^T) / 2``.

The load, and why there are two routes. The exact statement carries the GRID
acceleration a_i, which a dump of particle trajectories does not hold. The
instantaneous route substitutes the mass-weighted particle-acceleration
interpolation

    b_(i,d) = sum_p m_p (g - a_p)_d N_i(x_p),

consistent to the temporal discretization in the same sense as
``grid_assembly.py`` documents for the granular case (G2P makes v_p a nodal
interpolation, so the particle sum is a mass-weighted smoothing of the nodal
accelerations). On the granular collapse that is
enough. On an elastic bounce it is not enough. The gap between the two
accelerations is the P2G-then-G2P projection residual. The load divides that
residual by dt, and the residual grows with the gradient of the test
function, so a single-node basis suffers the most.
On the sphere drop this route errs 0.6 percent on E.

The time-weak route removes the acceleration from the data instead of
approximating it. Integrating the balance against a test function that vanishes
at both ends of a time window gives

    INT dt sum_p m_p a_p . w chi
        = - INT dt sum_p m_p [ v_p (v_p . grad w) chi + v_p w chi' ],

the M2 load already used for the granular space-time form in
``galerkin_spacetime.py``. Only x, v and F are read; no derivative of a measured
quantity appears. ``assemble_elastic_grid`` keeps the instantaneous route for the
manufactured tests, where the load is exact by construction.

Scope. ``assemble_columns_timeweak`` is the law-independent engine. A caller
supplies the volume-weighted Cauchy stress columns of any law linear in
theta, and may supply a known stress part that enters as data. Node
filtering, collider clearance, spatial windows, and the temporal weight are
shared. The fixed-corotated pair is the first client; constant-friction and
EOS comparisons reuse it.

Node filtering, three reasons a node is dropped:
  1. Collider reach. ``collide`` in the fork overwrites grid_v_out at nodes with
     dot(x_i - point, normal) < 0, so those nodes carry an unmodelled contact
     impulse. The contamination also travels: a kept node reads particles within
     1.5 dx, and those particles read nodes within a further 1.5 dx, so a margin
     of about 3 cells is needed for the interpolated load to be clean. The
     margin is a parameter and the validation run reports its sensitivity.
  2. Support mass. A node with little mass in its support has a small, noisy
     balance and the largest particle-to-node interpolation error.
  3. Non-finite or inverted F anywhere in the support. Such a particle's stress
     column is meaningless even though the engine consumed it, so the node's row
     is dropped rather than trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ident.weakform.grid_assembly import _bspline_weights_1d

__all__ = [
    "ElasticGridSystem",
    "bspline_stencil",
    "polar_rotation",
    "corotated_cauchy_columns",
    "hencky_cauchy_columns",
    "ELASTIC_COLUMNS",
    "assemble_elastic_grid",
    "assemble_elastic_timeweak",
    "assemble_columns_timeweak",
    "aggregate_elastic_rows",
    "solve_elastic_grid",
    "moduli_to_E_nu",
    "E_nu_to_moduli",
]


@dataclass
class ElasticGridSystem:
    """Assembled rows of the elastic grid-consistent weak form.

    Columns of A are ordered [mu, lambda]; b is the acceleration-form load.
    """

    A: np.ndarray                    # (n_rows, 2)
    b: np.ndarray                    # (n_rows,)
    node_id: np.ndarray              # (n_rows,) flat node index
    node_dir: np.ndarray             # (n_rows,) 0/1/2 component
    node_frame: np.ndarray           # (n_rows,) source frame index
    node_mass_frac: np.ndarray       # (n_rows,) support mass / max support mass
    n_rows: int
    n_rows_before_gating: int
    row_survival: float
    strain_coverage: tuple[float, float]   # 5th/99th pct of ||log sigma(F)||
    frames_used: list[int] = field(default_factory=list)


def bspline_stencil(
    x: np.ndarray, n_grid: int, grid_lim: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quadratic B-spline stencil of a 3D particle cloud on the MPM grid.

    Reconstructs exactly what warp-mpm's ``p2g_particle`` / ``g2p_particle``
    evaluate: base = floor(x / dx - 0.5), the three 1D weights per axis from
    ``grid_assembly._bspline_weights_1d``, and dweight = (dw_a w_b w_c) / dx.

    Returns (node_id, N, gradN, valid) with shapes (P, 27), (P, 27),
    (P, 27, 3), (P, 27). node_id is the flat index
    ix * n_grid^2 + iy * n_grid + iz, clamped to 0 where valid is False.
    """
    x = np.asarray(x, dtype=float)
    inv_dx = n_grid / grid_lim
    gp = x * inv_dx
    base = np.floor(gp - 0.5).astype(np.int64)          # (P, 3)
    fx = gp - base
    w0, dw0 = _bspline_weights_1d(fx[:, 0])
    w1, dw1 = _bspline_weights_1d(fx[:, 1])
    w2, dw2 = _bspline_weights_1d(fx[:, 2])

    P = x.shape[0]
    node_id = np.zeros((P, 27), dtype=np.int64)
    N = np.zeros((P, 27))
    gradN = np.zeros((P, 27, 3))
    valid = np.zeros((P, 27), dtype=bool)

    for i in range(3):
        ix = base[:, 0] + i
        for j in range(3):
            iy = base[:, 1] + j
            for k in range(3):
                iz = base[:, 2] + k
                s = 9 * i + 3 * j + k
                ok = ((ix >= 0) & (ix < n_grid) & (iy >= 0) & (iy < n_grid)
                      & (iz >= 0) & (iz < n_grid))
                valid[:, s] = ok
                node_id[:, s] = np.where(
                    ok, ix * n_grid * n_grid + iy * n_grid + iz, 0)
                N[:, s] = w0[:, i] * w1[:, j] * w2[:, k]
                gradN[:, s, 0] = dw0[:, i] * w1[:, j] * w2[:, k] * inv_dx
                gradN[:, s, 1] = w0[:, i] * dw1[:, j] * w2[:, k] * inv_dx
                gradN[:, s, 2] = w0[:, i] * w1[:, j] * dw2[:, k] * inv_dx
    return node_id, N, gradN, valid


def polar_rotation(F: np.ndarray) -> np.ndarray:
    """Rotation R from the polar decomposition F = R S, det R = +1.

    Same construction as the radial-window recovery: R = U V^T from the SVD,
    with the last column of U flipped when the SVD hands back a reflection.
    """
    F = np.asarray(F, dtype=float)
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    bad = np.linalg.det(R) < 0.0
    if np.any(bad):
        U = U.copy()
        U[bad, :, -1] *= -1.0
        R = U @ Vt
    return R


def corotated_cauchy_columns(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cauchy stress columns of the fixed-corotated pair, sigma = mu s_mu + lam s_lam.

        s_mu  = (2 / J) (F - R) F^T          s_lam = (J - 1) I

    Both are symmetrized, as the engine symmetrizes its own tau.
    """
    F = np.asarray(F, dtype=float)
    R = polar_rotation(F)
    J = np.linalg.det(F)
    A = (F - R) @ np.transpose(F, (0, 2, 1))
    s_mu = (2.0 / J)[:, None, None] * 0.5 * (A + np.transpose(A, (0, 2, 1)))
    s_lam = (J - 1.0)[:, None, None] * np.eye(3)[None, :, :]
    return s_mu, s_lam


def hencky_cauchy_columns(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cauchy stress columns of the Hencky pair, sigma = mu s_mu + lam s_lam.

        tau_mu = 2 U diag(log sig) U^T      tau_lam = tr(log sig) I
        s_*    = tau_* / J

    The columns for a material whose generating elasticity is SigmaElasticity.
    Identifying such data with the corotated columns is exact only at small strain;
    at the von Mises cap of about 2 percent deviatoric strain the difference is
    second order but measurable.
    """
    F = np.asarray(F, dtype=float)
    U, sig, _ = np.linalg.svd(F)
    eps = np.log(np.clip(sig, 1e-12, None))
    J = np.exp(eps.sum(-1))
    tau_mu = 2.0 * np.einsum("pij,pj,pkj->pik", U, eps, U)
    s_mu = tau_mu / J[:, None, None]
    s_lam = (eps.sum(-1) / J)[:, None, None] * np.eye(3)[None, :, :]
    return s_mu, s_lam


ELASTIC_COLUMNS = {"corotated": corotated_cauchy_columns,
                   "hencky": hencky_cauchy_columns}


def _collider_node_mask(
    n_grid: int,
    grid_lim: float,
    planes: Sequence[tuple[Sequence[float], Sequence[float]]],
    margin_cells: float,
) -> np.ndarray:
    """True where a node is far enough from every collider plane to be trusted.

    A plane is (point, inward normal); the fork's ``collide`` kernel rewrites
    nodes with dot(x_i - point, normal) < 0, and the interpolated load stays
    contaminated for a few more cells, hence the margin.
    """
    dx = grid_lim / n_grid
    idx = np.arange(n_grid) * dx
    XX, YY, ZZ = np.meshgrid(idx, idx, idx, indexing="ij")
    pos = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)
    keep = np.ones(pos.shape[0], dtype=bool)
    for point, normal in planes:
        pt = np.asarray(point, dtype=float)
        nrm = np.asarray(normal, dtype=float)
        nrm = nrm / np.linalg.norm(nrm)
        keep &= ((pos - pt) @ nrm) >= margin_cells * dx
    return keep


def _window_coefficients(
    keep: np.ndarray,
    n_grid: int,
    grid_lim: float,
    taper_cells: float,
    n_modes: int,
) -> np.ndarray | None:
    """Smooth nodal coefficient sets that vanish outside the kept-node region.

    A single node row is exact but narrow, and the one inexact term in the load
    (particle-interpolated instead of nodal acceleration) scales with the
    gradient of the test function, so the narrowest basis pays the most for it.
    Any linear combination of kept rows is equally exact, so combining them with
    a SMOOTH coefficient field keeps exactness and cuts that term by the ratio
    of dx to the window width. The coefficients are built to vanish on every
    dropped node; this keeps the collider traction out. The free
    surface needs no taper, and gets one only because the kept region ends
    there too.

    Returns (n_modes, n_nodes) coefficients, or None if the kept region is too
    thin for the requested taper.
    """
    from scipy import ndimage

    K = keep.reshape(n_grid, n_grid, n_grid)
    dist = ndimage.distance_transform_edt(K)      # cells to the nearest dropped node
    u = np.clip(dist / max(taper_cells, 1e-9), 0.0, 1.0)
    W = u * u * (3.0 - 2.0 * u)                   # smoothstep, C1 at both ends
    if W.max() <= 0.0:
        return None
    Wf = W.reshape(-1)
    if not np.any(Wf > 0.5):
        return None

    dx = grid_lim / n_grid
    idx = np.arange(n_grid) * dx
    XX, YY, ZZ = np.meshgrid(idx, idx, idx, indexing="ij")
    pos = np.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], axis=1)
    sel = Wf > 0.0
    centre = pos[sel].mean(axis=0)
    span = max(float(np.abs(pos[sel] - centre).max()), 1e-9)
    q = (pos - centre) / span

    modes = [np.ones(pos.shape[0])]
    for d in range(3):
        modes.append(q[:, d])
    modes += [q[:, 0] * q[:, 1], q[:, 1] * q[:, 2], q[:, 0] * q[:, 2],
              q[:, 0] ** 2 - q[:, 2] ** 2, q[:, 1] ** 2 - q[:, 2] ** 2]
    modes = modes[:max(1, n_modes)]
    return np.stack([Wf * m for m in modes], axis=0)


def _frame_nodal_terms(
    x_f: np.ndarray,
    Vsig: np.ndarray,
    mass: np.ndarray,
    ok_p: np.ndarray,
    n_grid: int,
    grid_lim: float,
    body: np.ndarray | None = None,
    v_f: np.ndarray | None = None,
    Vsig_known: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Scatter one frame's particle quantities onto the grid nodes.

    Vsig is (P, K, 3, 3): the CURRENT volume times each Cauchy stress column,
    so the assembled A is linear in theta with K columns. Vsig_known, if given,
    is the same product for the part of the stress that is data rather than
    unknown (the pressure term of a granular closure), and its nodal force lands
    in b_known.

    Always returns A (3, n_nodes, K) and the support masses. With ``body`` it
    also returns the instantaneous load b_inst; with ``v_f`` it returns the two
    velocity pieces of the time-weak load, the convective momentum flux
    (b_conv) and the momentum (b_mom).

    The 27-slot stencil is flattened so each accumulated quantity costs one
    bincount per frame instead of one per slot.
    """
    n_nodes = n_grid ** 3
    K = Vsig.shape[1]
    nid, N, gradN, valid = bspline_stencil(x_f, n_grid, grid_lim)
    sel = valid.reshape(-1)
    nid_f = nid.reshape(-1)[sel]
    N_f = N.reshape(-1)[sel]
    gN_f = gradN.reshape(-1, 3)[sel]
    pidx = np.repeat(np.arange(x_f.shape[0]), 27)[sel]

    mN = mass[pidx] * N_f
    out: dict[str, np.ndarray] = {
        "m_tot": np.bincount(nid_f, weights=mN, minlength=n_nodes),
        "m_ok": np.bincount(nid_f, weights=mN * ok_p[pidx], minlength=n_nodes),
    }
    A_acc = np.zeros((3, n_nodes, K))
    for d in range(3):
        for k in range(K):
            A_acc[d, :, k] = np.bincount(
                nid_f, weights=np.einsum("nj,nj->n", Vsig[pidx, k, d, :], gN_f),
                minlength=n_nodes)
    out["A"] = A_acc
    if Vsig_known is not None:
        bk = np.zeros((3, n_nodes))
        for d in range(3):
            bk[d] = np.bincount(
                nid_f,
                weights=np.einsum("nj,nj->n", Vsig_known[pidx, d, :], gN_f),
                minlength=n_nodes)
        out["b_known"] = bk

    if body is not None:
        b_inst = np.zeros((3, n_nodes))
        for d in range(3):
            b_inst[d] = np.bincount(
                nid_f, weights=body[pidx, d] * N_f, minlength=n_nodes)
        out["b_inst"] = b_inst

    if v_f is not None:
        vg = np.einsum("nj,nj->n", v_f[pidx], gN_f)        # v . grad N
        b_conv = np.zeros((3, n_nodes))
        b_mom = np.zeros((3, n_nodes))
        for d in range(3):
            b_conv[d] = np.bincount(
                nid_f, weights=mass[pidx] * v_f[pidx, d] * vg, minlength=n_nodes)
            b_mom[d] = np.bincount(
                nid_f, weights=mass[pidx] * v_f[pidx, d] * N_f, minlength=n_nodes)
        out["b_conv"] = b_conv
        out["b_mom"] = b_mom
    return out


def _particle_validity(
    F_f: np.ndarray, x_f: np.ndarray, extra: np.ndarray | None,
    max_hencky: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite, non-inverted particles plus the rotation-invariant strain measure.

    The SVD must not see a non-finite F (LAPACK raises), so finiteness is
    checked first and the decomposition runs on the survivors only.
    """
    finite = np.isfinite(F_f).all(axis=(1, 2)) & np.isfinite(x_f).all(axis=1)
    if extra is not None:
        finite &= np.isfinite(extra).all(axis=1)
    hencky = np.full(F_f.shape[0], np.inf)
    good = np.zeros(F_f.shape[0], dtype=bool)
    if finite.any():
        sig = np.linalg.svd(F_f[finite], compute_uv=False)
        sig_pos = sig.min(axis=1) > 1.0e-9
        idx_f = np.where(finite)[0]
        good[idx_f[sig_pos]] = True
        hencky[idx_f[sig_pos]] = np.linalg.norm(np.log(sig[sig_pos]), axis=1)
    ok_p = good & np.isfinite(hencky)
    if max_hencky is not None:
        ok_p &= hencky <= max_hencky
    return ok_p, hencky


def _stress_columns_for_frame(
    F_f: np.ndarray, vol0: np.ndarray, ok_p: np.ndarray,
    columns: str = "corotated",
) -> np.ndarray:
    """Volume-weighted (P, 2, 3, 3) stress columns, invalid particles set to F = I.

    Particles failing ok_p still enter the node sums (the engine consumed them
    too), but every node touching one is dropped, so the substituted identity
    never reaches a surviving row.
    """
    Fs = np.where(ok_p[:, None, None], F_f, np.eye(3)[None])
    s_mu, s_lam = ELASTIC_COLUMNS[columns](Fs)
    J = np.linalg.det(Fs)
    Vcur = J * vol0                          # current volume, J V^0
    return Vcur[:, None, None, None] * np.stack([s_mu, s_lam], axis=1)


def assemble_elastic_grid(
    x: np.ndarray,
    F: np.ndarray,
    vol0: np.ndarray,
    mass: np.ndarray,
    accel: np.ndarray,
    g: np.ndarray,
    n_grid: int,
    grid_lim: float,
    frames: Sequence[int] | None = None,
    collider_planes: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    collider_margin_cells: float = 3.0,
    min_support_mass_frac: float = 0.25,
    valid_frac_min: float = 1.0 - 1.0e-12,
    max_hencky: float | None = None,
    min_particles: int = 20,
    window_taper_cells: float | None = None,
    window_modes: int = 4,
) -> ElasticGridSystem:
    """Assemble interior-free-node rows for the fixed-corotated pair.

    Parameters
    ----------
    x, F, accel : (T, P, 3), (T, P, 3, 3), (T, P, 3)
        Current positions, deformation gradient, and MATERIAL acceleration per
        frame. accel is the caller's choice (trajectory finite difference is the
        default elsewhere in this tree) so the load convention stays visible.
    vol0, mass : (P,)
        Reference particle volume and particle mass. Current volume is J vol0;
        see the module docstring on the updated-Lagrangian pairing.
    g : (3,)
        Gravity vector, the same one the forward run used.
    frames :
        Frame indices to emit rows for; default every frame.
    collider_planes :
        (point, inward normal) pairs for every collider the run used.
    collider_margin_cells :
        Cells of clearance required from each plane. Three cells covers the
        two-hop particle-node-particle reach of the quadratic stencil.
    min_support_mass_frac :
        Keep a node only if its support mass is at least this fraction of the
        largest support mass in the frame.
    valid_frac_min :
        Keep a node only if this fraction of its support mass comes from
        particles with finite, non-inverted F.
    max_hencky :
        Optional bound on ||log sigma(F)||, the rotation-invariant strain
        measure. None leaves it off; only finiteness and J > 0 are required.
    window_taper_cells, window_modes :
        When window_taper_cells is set, the kept node rows of each frame are
        combined into window_modes smooth rows per direction instead of one row
        per node. See ``_window_coefficients``: the combination is exact;
        this makes the elastic recovery sub-percent. None emits raw node
        rows, which suits a manufactured test where the load is exact.
    """
    x = np.asarray(x, dtype=float)
    F = np.asarray(F, dtype=float)
    accel = np.asarray(accel, dtype=float)
    vol0 = np.asarray(vol0, dtype=float)
    mass = np.asarray(mass, dtype=float)
    g = np.asarray(g, dtype=float).reshape(3)
    T = x.shape[0]
    frame_list = list(range(T)) if frames is None else [int(f) for f in frames]

    node_ok = _collider_node_mask(
        n_grid, grid_lim, collider_planes, collider_margin_cells)

    rows_A: list[np.ndarray] = []
    rows_b: list[float] = []
    rows_node: list[int] = []
    rows_dir: list[int] = []
    rows_frame: list[int] = []
    rows_frac: list[float] = []
    hencky_all: list[np.ndarray] = []
    n_before = 0
    frames_used: list[int] = []

    for f in frame_list:
        ok_p, hencky = _particle_validity(F[f], x[f], accel[f], max_hencky)
        if ok_p.sum() < min_particles:
            continue
        Vsig = _stress_columns_for_frame(F[f], vol0, ok_p)

        terms = _frame_nodal_terms(
            x[f], Vsig, mass, ok_p, n_grid, grid_lim,
            body=mass[:, None] * (g[None, :] - accel[f]))
        A_acc, b_acc = terms["A"], terms["b_inst"]
        m_tot, m_ok = terms["m_tot"], terms["m_ok"]

        supported = m_tot > 0.0
        n_before += 3 * int(supported.sum())
        m_max = m_tot.max() if supported.any() else 0.0
        frac = np.where(supported, m_ok / np.maximum(m_tot, 1e-300), 0.0)
        keep = (supported & node_ok
                & (m_tot >= min_support_mass_frac * m_max)
                & (frac >= valid_frac_min))
        idx = np.where(keep)[0]
        if idx.size == 0:
            continue
        frames_used.append(f)
        hencky_all.append(hencky[ok_p])

        if window_taper_cells is not None:
            coeffs = _window_coefficients(
                keep, n_grid, grid_lim, window_taper_cells, window_modes)
            if coeffs is None:
                frames_used.pop()
                hencky_all.pop()
                continue
            for d in range(3):
                rows_A.append(coeffs[:, idx] @ A_acc[d, idx, :])
                rows_b.append(coeffs[:, idx] @ b_acc[d, idx])
                nm = coeffs.shape[0]
                rows_node.append(np.full(nm, -1, dtype=np.int64))
                rows_dir.append(np.full(nm, d))
                rows_frame.append(np.full(nm, f))
                rows_frac.append(
                    np.full(nm, float(m_tot[idx].mean() / max(m_max, 1e-300))))
            continue

        for d in range(3):
            rows_A.append(A_acc[d, idx, :])
            rows_b.append(b_acc[d, idx])
            rows_node.append(idx)
            rows_dir.append(np.full(idx.size, d))
            rows_frame.append(np.full(idx.size, f))
            rows_frac.append(m_tot[idx] / max(m_max, 1e-300))

    if not rows_A:
        return ElasticGridSystem(
            A=np.zeros((0, 2)), b=np.zeros(0), node_id=np.zeros(0, dtype=int),
            node_dir=np.zeros(0, dtype=int), node_frame=np.zeros(0, dtype=int),
            node_mass_frac=np.zeros(0), n_rows=0,
            n_rows_before_gating=n_before, row_survival=0.0,
            strain_coverage=(0.0, 0.0), frames_used=[])

    A = np.concatenate(rows_A, axis=0)
    b = np.concatenate([np.asarray(r) for r in rows_b], axis=0)
    hen = np.concatenate(hencky_all) if hencky_all else np.zeros(1)
    return ElasticGridSystem(
        A=A,
        b=b,
        node_id=np.concatenate(rows_node),
        node_dir=np.concatenate(rows_dir),
        node_frame=np.concatenate(rows_frame),
        node_mass_frac=np.concatenate(rows_frac),
        n_rows=int(A.shape[0]),
        n_rows_before_gating=n_before,
        row_survival=(A.shape[0] / n_before) if n_before else 0.0,
        strain_coverage=(float(np.percentile(hen, 5)),
                         float(np.percentile(hen, 99))),
        frames_used=frames_used,
    )


def _temporal_window(
    nw: int, frame_dt: float, power: int
) -> tuple[np.ndarray, np.ndarray]:
    """chi(s) = sin^(2m)(pi s) sampled on nw frames, and its time derivative.

    The time-weak load is a sum over sampled frames standing in for a time
    integral, so the temporal weight sets the quadrature order. Euler-Maclaurin
    kills the h^(2j) error terms of the trapezoid rule only when the odd
    derivatives of the integrand vanish at both ends of the window, and the
    integrand of the momentum piece is chi' times the momentum functional. With
    chi = sin^(2m) the first 2m - 1 derivatives of chi vanish at both ends, so
    the convergence is h^(2m): with m = 1 an order h^2 term dominates; m = 2
    removes it.
    """
    m = max(int(power), 1)
    s = np.arange(nw) / (nw - 1.0)
    span = (nw - 1.0) * frame_dt
    sn = np.sin(np.pi * s)
    chi = sn ** (2 * m)
    dchi = (2 * m) * np.pi * sn ** (2 * m - 1) * np.cos(np.pi * s) / span
    return chi, dchi


def assemble_elastic_timeweak(
    x: np.ndarray,
    F: np.ndarray,
    v: np.ndarray,
    vol0: np.ndarray,
    mass: np.ndarray,
    g: np.ndarray,
    frame_dt: float,
    n_grid: int,
    grid_lim: float,
    frames: Sequence[int] | None = None,
    window_frames: int = 8,
    window_stride: int | None = None,
    collider_planes: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    collider_margin_cells: float = 3.0,
    min_support_mass_frac: float = 0.25,
    valid_frac_min: float = 1.0 - 1.0e-12,
    max_hencky: float | None = None,
    min_particles: int = 20,
    window_taper_cells: float | None = 1.0,
    window_modes: int = 4,
    time_power: int = 2,
    columns: str = "corotated",
) -> ElasticGridSystem:
    """Time-weak variant: no acceleration data, so no time differentiation.

    Why this exists. The instantaneous nodal balance needs the acceleration, and
    a dump supplies only the trajectory finite difference of v. That costs twice
    over: the difference itself aliases the elastic oscillation, and the gap
    between the interpolated particle acceleration and the nodal one enters
    divided by dt. Integrating the weak form in time against a test function
    that vanishes at both ends of the window removes the acceleration entirely,

        INT dt sum_p m_p a_p . w chi
           = - INT dt sum_p m_p [ v_p (v_p . grad w) chi + v_p w chi' ],

    which is the M2 time-weak load already used for the granular space-time form
    in ``galerkin_spacetime.py``. The remaining data are v, x, F: no derivative
    of a measured quantity anywhere.

    The temporal weight is chi(s) = sin^(2 time_power)(pi s) on s in [0, 1]
    across each window; see ``_temporal_window`` for why the power sets the
    accuracy. Windows overlap by ``window_stride``. A node must pass every
    check in every frame of a window, since the row sums those frames.
    """
    x = np.asarray(x, dtype=float)
    F = np.asarray(F, dtype=float)
    vol0 = np.asarray(vol0, dtype=float)

    def columns_fn(f: int):
        ok_p, hencky = _particle_validity(F[f], x[f], np.asarray(v)[f],
                                          max_hencky)
        if ok_p.sum() < min_particles:
            return None
        return _stress_columns_for_frame(F[f], vol0, ok_p, columns=columns), None, ok_p, hencky

    return assemble_columns_timeweak(
        x, v, mass, g, frame_dt, n_grid, grid_lim, columns_fn, n_columns=2,
        frames=frames, window_frames=window_frames, window_stride=window_stride,
        collider_planes=collider_planes,
        collider_margin_cells=collider_margin_cells,
        min_support_mass_frac=min_support_mass_frac,
        valid_frac_min=valid_frac_min,
        window_taper_cells=window_taper_cells, window_modes=window_modes,
        time_power=time_power)


def assemble_columns_timeweak(
    x: np.ndarray,
    v: np.ndarray,
    mass: np.ndarray,
    g: np.ndarray,
    frame_dt: float,
    n_grid: int,
    grid_lim: float,
    columns_fn,
    n_columns: int,
    frames: Sequence[int] | None = None,
    window_frames: int = 8,
    window_stride: int | None = None,
    collider_planes: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    collider_margin_cells: float = 3.0,
    min_support_mass_frac: float = 0.25,
    valid_frac_min: float = 1.0 - 1.0e-12,
    window_taper_cells: float | None = 1.0,
    window_modes: int = 4,
    time_power: int = 2,
) -> ElasticGridSystem:
    """Time-weak grid-consistent assembly for ANY law linear in theta.

    This is the shared engine. A caller supplies

        columns_fn(frame) -> (Vsig, Vsig_known, ok_p, diag) or None

    with Vsig the (P, n_columns, 3, 3) product of current particle volume and
    each Cauchy stress column, Vsig_known the same product for the part of the
    stress that is DATA rather than unknown (a granular pressure closure, for
    instance) or None, ok_p the per-particle validity flags, and diag any
    per-particle scalar to report as coverage. Returning None skips the frame.

    Everything downstream, the node filtering, the collider clearance, the smooth
    spatial windows and the temporal weight, is law independent.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    mass = np.asarray(mass, dtype=float)
    g = np.asarray(g, dtype=float).reshape(3)
    T = x.shape[0]
    n_nodes = n_grid ** 3
    frame_list = list(range(T)) if frames is None else [int(f) for f in frames]
    nw = max(int(window_frames), 3)
    stride = nw // 2 if window_stride is None else max(int(window_stride), 1)

    node_ok = _collider_node_mask(
        n_grid, grid_lim, collider_planes, collider_margin_cells)

    rows_A: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    rows_node: list[np.ndarray] = []
    rows_dir: list[np.ndarray] = []
    rows_frame: list[np.ndarray] = []
    rows_frac: list[np.ndarray] = []
    hencky_all: list[np.ndarray] = []
    n_before = 0
    frames_used: list[int] = []

    cache: dict[int, dict | None] = {}

    def frame_terms(f: int) -> dict | None:
        if f in cache:
            return cache[f]
        got = columns_fn(f)
        if got is None:
            cache[f] = None
            return None
        Vsig, Vsig_known, ok_p, diag = got
        t = _frame_nodal_terms(x[f], Vsig, mass, ok_p, n_grid, grid_lim,
                               v_f=v[f], Vsig_known=Vsig_known)
        m_tot = t["m_tot"]
        supported = m_tot > 0.0
        m_max = m_tot.max() if supported.any() else 0.0
        frac = np.where(supported, t["m_ok"] / np.maximum(m_tot, 1e-300), 0.0)
        t["keep"] = (supported & node_ok
                     & (m_tot >= min_support_mass_frac * m_max)
                     & (frac >= valid_frac_min))
        t["m_frac"] = m_tot / max(m_max, 1e-300)
        t["hencky"] = np.asarray(diag)[ok_p] if diag is not None else np.zeros(1)
        cache[f] = t
        return t

    for w0 in range(0, max(len(frame_list) - nw + 1, 0), stride):
        win = frame_list[w0:w0 + nw]
        # the per-frame nodal terms are dense over the grid, so only the frames
        # a future window still needs are kept; without this a few hundred
        # frames of a 48^3 run would hold gigabytes
        for f_old in [f for f in cache if f < win[0]]:
            del cache[f_old]
        terms = [frame_terms(f) for f in win]
        if any(t is None for t in terms):
            continue
        keep = np.ones(n_nodes, dtype=bool)
        for t in terms:
            keep &= t["keep"]
        idx = np.where(keep)[0]
        n_before += 3 * int((terms[0]["m_tot"] > 0.0).sum())
        if idx.size == 0:
            continue

        chi, dchi = _temporal_window(nw, frame_dt, time_power)

        A_w = np.zeros((3, idx.size, n_columns))
        b_w = np.zeros((3, idx.size))
        for c, t in zip(chi, terms, strict=False):
            A_w += c * frame_dt * t["A"][:, idx, :]
        for c, dc, t in zip(chi, dchi, terms, strict=False):
            # b = INT dt [ chi (m g N)  +  chi (m v (v.gradN))  +  chi' (m v N) ]
            #     minus the known part of the internal force, if any
            b_w += frame_dt * (
                c * (g[:, None] * t["m_tot"][None, idx] + t["b_conv"][:, idx])
                + dc * t["b_mom"][:, idx])
            if "b_known" in t:
                b_w -= frame_dt * c * t["b_known"][:, idx]

        frames_used.extend(win)
        hencky_all.append(terms[len(terms) // 2]["hencky"])
        f_label = win[len(win) // 2]
        if window_taper_cells is not None:
            coeffs = _window_coefficients(
                keep, n_grid, grid_lim, window_taper_cells, window_modes)
            if coeffs is None:
                continue
            C = coeffs[:, idx]
            nm = C.shape[0]
            for d in range(3):
                rows_A.append(C @ A_w[d])
                rows_b.append(C @ b_w[d])
                rows_node.append(np.full(nm, -1, dtype=np.int64))
                rows_dir.append(np.full(nm, d))
                rows_frame.append(np.full(nm, f_label))
                rows_frac.append(np.full(nm, float(terms[0]["m_frac"][idx].mean())))
        else:
            for d in range(3):
                rows_A.append(A_w[d])
                rows_b.append(b_w[d])
                rows_node.append(idx)
                rows_dir.append(np.full(idx.size, d))
                rows_frame.append(np.full(idx.size, f_label))
                rows_frac.append(terms[0]["m_frac"][idx])

    if not rows_A:
        return ElasticGridSystem(
            A=np.zeros((0, n_columns)), b=np.zeros(0),
            node_id=np.zeros(0, dtype=int),
            node_dir=np.zeros(0, dtype=int), node_frame=np.zeros(0, dtype=int),
            node_mass_frac=np.zeros(0), n_rows=0,
            n_rows_before_gating=n_before, row_survival=0.0,
            strain_coverage=(0.0, 0.0), frames_used=[])
    A = np.concatenate(rows_A, axis=0)
    b = np.concatenate(rows_b, axis=0)
    hen = np.concatenate(hencky_all) if hencky_all else np.zeros(1)
    return ElasticGridSystem(
        A=A, b=b,
        node_id=np.concatenate(rows_node),
        node_dir=np.concatenate(rows_dir),
        node_frame=np.concatenate(rows_frame),
        node_mass_frac=np.concatenate(rows_frac),
        n_rows=int(A.shape[0]),
        n_rows_before_gating=n_before,
        row_survival=(A.shape[0] / n_before) if n_before else 0.0,
        strain_coverage=(float(np.percentile(hen, 5)),
                         float(np.percentile(hen, 99))),
        frames_used=sorted(set(frames_used)),
    )


def aggregate_elastic_rows(
    system: ElasticGridSystem,
    n_grid: int,
    node_block: int = 1,
    frame_block: int = 1,
) -> ElasticGridSystem:
    """Sum rows over blocks of nodes and frames, giving wider test functions.

    Summing rows is exact. Each kept row is the exact discrete balance for one
    node basis function N_i and one direction, so the sum over a set S of kept
    nodes is the exact balance for the test function w = sum_{i in S} N_i, and
    the sum over frames is the exact time-integrated balance. No node outside
    the kept set enters, so the collider and validity filtering still holds.

    Why widen. A single node basis is the narrowest test function the discrete
    space allows, which makes its row maximally sensitive to the one term the
    dump cannot supply exactly: the load carries the interpolated PARTICLE
    acceleration where the exact statement carries the nodal one, and the gap is
    the P2G-then-G2P projection residual divided by dt. That residual varies on
    the dx and substep scales, so it averages down under wider test functions
    while the exact balance does not move. Summing kept rows is exact, so
    widening costs nothing.

    node_block = 1 and frame_block = 1 return the system unchanged.
    """
    if node_block <= 1 and frame_block <= 1:
        return system
    if system.n_rows == 0:
        return system
    nb = max(int(node_block), 1)
    fb = max(int(frame_block), 1)

    ix = system.node_id // (n_grid * n_grid)
    iy = (system.node_id // n_grid) % n_grid
    iz = system.node_id % n_grid
    nbg = (n_grid + nb - 1) // nb
    block = (ix // nb) * nbg * nbg + (iy // nb) * nbg + (iz // nb)

    frames_sorted = sorted(set(system.frames_used))
    fpos = {f: i for i, f in enumerate(frames_sorted)}
    fblk = np.array([fpos[int(f)] // fb for f in system.node_frame])

    key = np.stack([block, system.node_dir, fblk], axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n_out = int(inv.max()) + 1

    A = np.stack([np.bincount(inv, weights=system.A[:, k], minlength=n_out)
                  for k in range(system.A.shape[1])], axis=1)
    b = np.bincount(inv, weights=system.b, minlength=n_out)
    cnt = np.bincount(inv, minlength=n_out)
    frac = np.bincount(inv, weights=system.node_mass_frac,
                       minlength=n_out) / np.maximum(cnt, 1)
    # representative labels, the first row that landed in each group
    first = np.full(n_out, -1, dtype=np.int64)
    order = np.arange(inv.size)[::-1]
    first[inv[order]] = order
    return ElasticGridSystem(
        A=A,
        b=b,
        node_id=system.node_id[first],
        node_dir=system.node_dir[first],
        node_frame=system.node_frame[first],
        node_mass_frac=frac,
        n_rows=n_out,
        n_rows_before_gating=system.n_rows_before_gating,
        row_survival=system.row_survival,
        strain_coverage=system.strain_coverage,
        frames_used=system.frames_used,
    )


def moduli_to_E_nu(mu: float, lam: float) -> tuple[float, float]:
    nu = lam / (2.0 * (lam + mu))
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    return float(E), float(nu)


def E_nu_to_moduli(E: float, nu: float) -> tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(mu), float(lam)


def solve_elastic_grid(
    system: ElasticGridSystem, ridge_lambda: float = 0.0
) -> dict[str, float]:
    """Least-squares (optionally ridge) solve for (mu, lambda).

    Works for any number of columns. theta and theta_sd are always returned;
    when there are exactly two columns the elastic names (mu, lam, E, nu) are
    added, since that is what the elastic call sites read.

    Columns can differ in magnitude by orders of magnitude on a nearly
    incompressible motion, so the normal equations are formed on
    column-normalized data and the scaling is undone afterwards. cond(A^T A) is
    reported on the RAW columns, which is the number the gate scripts record.
    """
    A, b = system.A, system.b
    if A.shape[0] < 2:
        raise ValueError("elastic solve needs at least two rows")
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    As = A / scale
    M = As.T @ As + ridge_lambda * np.eye(A.shape[1])
    theta_s = np.linalg.solve(M, As.T @ b)
    theta = theta_s / scale

    r = A @ theta - b
    bn = float(np.linalg.norm(b))
    AtA = A.T @ A
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float((sv[0] / sv[-1]) ** 2) if sv[-1] > 0.0 else float("inf")
    # posterior on the scaled system, mapped back
    dof = float(np.trace(As @ np.linalg.solve(M, As.T))) if A.shape[0] < 4000 \
        else float(A.shape[1])
    sigma2 = float(r @ r) / max(A.shape[0] - dof, 1.0)
    cov_s = sigma2 * np.linalg.inv(M)
    cov = cov_s / np.outer(scale, scale)

    out = {
        "theta": [float(t) for t in theta],
        "theta_sd": [float(np.sqrt(max(cov[k, k], 0.0)))
                     for k in range(A.shape[1])],
        "cond_AtA": cond,
        "cond_AtA_scaled": float(np.linalg.cond(As.T @ As)),
        "residual_rel": float(np.linalg.norm(r) / bn) if bn > 0 else float("inf"),
        "n_rows": int(A.shape[0]),
        "trace_AtA": float(np.trace(AtA)),
    }
    if A.shape[1] == 2:
        # the elastic pair, named for readability at the call sites that want it
        mu, lam = float(theta[0]), float(theta[1])
        E, nu = moduli_to_E_nu(mu, lam)
        out.update({"mu": mu, "lam": lam, "E": E, "nu": nu,
                    "mu_sd": out["theta_sd"][0], "lam_sd": out["theta_sd"][1]})
    return out

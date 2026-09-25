"""Tests for the grid-consistent elastic (fixed-corotated) weak form.

The designated bug classes for this module are the factor of two in the mu
column, the J pairing between current volume and Cauchy stress, and the sign of
the acceleration load. Each has a test below that fails by order one when the
bug is introduced.

The end-to-end test is a manufactured-acceleration patch test: a particle cloud
carrying a prescribed non-uniform deformation gradient fixes the exact nodal
internal force for a chosen (mu, lambda); per-particle accelerations are then
solved so that the interpolated load reproduces that force at every supported
node. The assembler and solver must return the chosen moduli to solver
precision, because in that construction the discrete residual is exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from ident.weakform.elastic_grid import (
    E_nu_to_moduli,
    assemble_elastic_grid,
    bspline_stencil,
    corotated_cauchy_columns,
    moduli_to_E_nu,
    polar_rotation,
    solve_elastic_grid,
)

N_GRID = 12
GRID_LIM = 1.0
G_VEC = np.array([0.0, 0.0, -9.81])


def _cloud(n_grid: int = N_GRID, grid_lim: float = GRID_LIM, half: float = 0.25):
    """A uniformly seeded cube of particles at pitch dx/2, centred in the box."""
    dx = grid_lim / n_grid
    h = dx / 2.0
    c = grid_lim * 0.5
    ax = np.arange(-half, half + 0.5 * h, h)
    P = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + c
    vol0 = np.full(len(P), h ** 3)
    mass = 1000.0 * vol0
    return P, vol0, mass


def _manufactured_F(X: np.ndarray, amp: float = 0.06) -> np.ndarray:
    """A smooth, non-uniform, non-rigid F field with a spatially varying J.

    Both stress columns need spatial variation to produce a nonzero interior
    nodal force: a uniform stress has zero divergence and would make the patch
    test degenerate.
    """
    k = 2.0 * np.pi
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    Gm = np.zeros((len(X), 3, 3))
    Gm[:, 0, 0] = np.sin(k * x)
    Gm[:, 1, 1] = np.cos(k * y) * 0.5
    Gm[:, 2, 2] = np.sin(k * (x + z)) * 0.7
    Gm[:, 0, 1] = 0.4 * np.sin(k * z)
    Gm[:, 1, 2] = 0.3 * np.cos(k * x)
    Gm[:, 2, 0] = 0.2 * np.sin(k * y)
    return np.eye(3)[None] + amp * Gm


def _exact_nodal_force(x, F, vol0, n_grid, grid_lim, mu, lam):
    """sum_p V_p sigma_p[d,:] . grad N_i(x_p) with sigma = mu s_mu + lam s_lam."""
    nid, N, gradN, valid = bspline_stencil(x, n_grid, grid_lim)
    s_mu, s_lam = corotated_cauchy_columns(F)
    J = np.linalg.det(F)
    Vcur = J * vol0
    Vsig = Vcur[:, None, None] * (mu * s_mu + lam * s_lam)
    n_nodes = n_grid ** 3
    f = np.zeros((3, n_nodes))
    for s in range(27):
        m = valid[:, s]
        for d in range(3):
            f[d] += np.bincount(
                nid[m, s],
                weights=np.einsum("nj,nj->n", Vsig[m, d, :], gradN[m, s]),
                minlength=n_nodes)
    return f, nid, N, valid


def _manufactured_accel(mu, lam, amp=0.06, n_grid=N_GRID, grid_lim=GRID_LIM):
    """Cloud plus accelerations that make the discrete residual exactly zero."""
    X, vol0, mass = _cloud(n_grid, grid_lim)
    F = _manufactured_F(X, amp)
    f, nid, N, valid = _exact_nodal_force(X, F, vol0, n_grid, grid_lim, mu, lam)

    n_nodes = n_grid ** 3
    m_tot = np.zeros(n_nodes)
    for s in range(27):
        m = valid[:, s]
        m_tot += np.bincount(nid[m, s], weights=mass[m] * N[m, s],
                             minlength=n_nodes)
    sup = np.where(m_tot > 0.0)[0]

    # M[node, p] = m_p N_node(x_p); solve M c_d = f_d for the body-force
    # density c = g - a, then a = g - c.
    M = np.zeros((sup.size, len(X)))
    remap = -np.ones(n_nodes, dtype=int)
    remap[sup] = np.arange(sup.size)
    for s in range(27):
        m = valid[:, s]
        rows = remap[nid[m, s]]
        cols = np.where(m)[0]
        np.add.at(M, (rows, cols), mass[m] * N[m, s])

    c = np.linalg.lstsq(M, f[:, sup].T, rcond=None)[0]      # (P, 3)
    resid = np.linalg.norm(M @ c - f[:, sup].T) / max(
        np.linalg.norm(f[:, sup]), 1e-300)
    assert resid < 1e-8, f"manufactured load not reproducible, rel {resid:.2e}"
    accel = G_VEC[None, :] - c
    return X, F, vol0, mass, accel


# --------------------------------------------------------------------------
# B-spline stencil: must match warp-mpm's p2g exactly
# --------------------------------------------------------------------------

def test_stencil_partition_of_unity_and_zero_gradient_sum():
    X, _, _ = _cloud()
    _, N, gradN, valid = bspline_stencil(X, N_GRID, GRID_LIM)
    assert np.allclose(N.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(gradN.sum(axis=1), 0.0, atol=1e-9)
    assert valid.all(), "interior cloud should never leave the grid"


def test_stencil_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    X = 0.5 + rng.uniform(-0.1, 0.1, (40, 3))
    nid, _N, gradN, _ = bspline_stencil(X, N_GRID, GRID_LIM)
    eps = 1e-6
    for d in range(3):
        dX = np.zeros(3)
        dX[d] = eps
        nid_p, N_p, _, _ = bspline_stencil(X + dX, N_GRID, GRID_LIM)
        nid_m, N_m, _, _ = bspline_stencil(X - dX, N_GRID, GRID_LIM)
        # same base cell for this displacement size, so stencil slots line up
        assert np.array_equal(nid_p, nid) and np.array_equal(nid_m, nid)
        fd = (N_p - N_m) / (2.0 * eps)
        assert np.allclose(fd, gradN[:, :, d], atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# Stress columns
# --------------------------------------------------------------------------

def test_stress_columns_vanish_at_identity_and_stay_symmetric():
    F = np.repeat(np.eye(3)[None], 5, axis=0)
    s_mu, s_lam = corotated_cauchy_columns(F)
    assert np.allclose(s_mu, 0.0, atol=1e-12)
    assert np.allclose(s_lam, 0.0, atol=1e-12)
    X, _, _ = _cloud()
    Fv = _manufactured_F(X)
    s_mu, s_lam = corotated_cauchy_columns(Fv)
    for s in (s_mu, s_lam):
        assert np.allclose(s, np.transpose(s, (0, 2, 1)), atol=1e-12)


def test_stress_columns_reproduce_the_engine_kirchhoff_formula():
    """mu s_mu + lam s_lam times J must equal the fork's kirchoff_stress_FCR."""
    X, _, _ = _cloud()
    F = _manufactured_F(X)
    mu, lam = E_nu_to_moduli(2.0e5, 0.3)
    s_mu, s_lam = corotated_cauchy_columns(F)
    J = np.linalg.det(F)
    tau_cols = J[:, None, None] * (mu * s_mu + lam * s_lam)
    R = polar_rotation(F)
    tau_engine = (2.0 * mu * (F - R) @ np.transpose(F, (0, 2, 1))
                  + lam * (J * (J - 1.0))[:, None, None] * np.eye(3)[None])
    tau_engine = 0.5 * (tau_engine + np.transpose(tau_engine, (0, 2, 1)))
    assert np.allclose(tau_cols, tau_engine, rtol=1e-11, atol=1e-9)


def test_polar_rotation_is_a_proper_rotation():
    rng = np.random.default_rng(1)
    F = np.eye(3)[None] + rng.uniform(-0.3, 0.3, (30, 3, 3))
    R = polar_rotation(F)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3)[None],
                       atol=1e-12)
    assert np.all(np.linalg.det(R) > 0.99)


# --------------------------------------------------------------------------
# Manufactured patch test: exact recovery
# --------------------------------------------------------------------------

def test_manufactured_accelerations_recover_the_moduli_exactly():
    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, accel = _manufactured_accel(mu_t, lam_t)
    sysm = assemble_elastic_grid(
        X[None], F[None], vol0, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        min_support_mass_frac=0.0)
    assert sysm.n_rows > 100
    out = solve_elastic_grid(sysm)
    assert abs(out["mu"] / mu_t - 1.0) < 1e-8
    assert abs(out["lam"] / lam_t - 1.0) < 1e-8
    assert out["residual_rel"] < 1e-9
    E, nu = moduli_to_E_nu(out["mu"], out["lam"])
    assert abs(E / 2.0e5 - 1.0) < 1e-8
    assert abs(nu - 0.3) < 1e-8


def test_manufactured_patch_test_is_sensitive_to_the_factor_of_two():
    """Halving the mu column must double the recovered mu: an order-one failure.

    This is the negative control for the canonical bug of this codebase.
    """
    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, accel = _manufactured_accel(mu_t, lam_t)
    sysm = assemble_elastic_grid(
        X[None], F[None], vol0, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        min_support_mass_frac=0.0)
    sysm.A[:, 0] *= 0.5
    out = solve_elastic_grid(sysm)
    assert abs(out["mu"] / mu_t - 2.0) < 1e-6


def test_manufactured_patch_test_is_sensitive_to_the_load_sign():
    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, accel = _manufactured_accel(mu_t, lam_t)
    # flip the load sign: b -> -b sends (mu, lam) -> (-mu, -lam)
    sysm = assemble_elastic_grid(
        X[None], F[None], vol0, mass, 2.0 * G_VEC[None, None, :] - accel[None],
        G_VEC, N_GRID, GRID_LIM, min_support_mass_frac=0.0)
    out = solve_elastic_grid(sysm)
    assert out["mu"] < 0.0
    assert abs(out["mu"] / (-mu_t) - 1.0) < 1e-6


def test_j_pairing_is_load_bearing():
    """Pairing REFERENCE volume with Cauchy stress instead of current volume
    biases the recovery by order (J - 1); with a 6 percent strain field that is
    a visible, not a rounding, error."""
    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, accel = _manufactured_accel(mu_t, lam_t, amp=0.12)
    good = assemble_elastic_grid(
        X[None], F[None], vol0, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        min_support_mass_frac=0.0)
    ref = solve_elastic_grid(good)
    assert abs(ref["mu"] / mu_t - 1.0) < 1e-8
    # mimic the wrong pairing by re-assembling with vol0 / J
    J = np.linalg.det(F)
    bad = assemble_elastic_grid(
        X[None], F[None], vol0 / J, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        min_support_mass_frac=0.0)
    wrong = solve_elastic_grid(bad)
    assert abs(wrong["mu"] / mu_t - 1.0) > 1e-3


# --------------------------------------------------------------------------
# Node gating
# --------------------------------------------------------------------------

def test_collider_adjacent_nodes_are_excluded():
    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, accel = _manufactured_accel(mu_t, lam_t)
    dx = GRID_LIM / N_GRID
    floor_z = 3.0 * dx
    margin = 2.0
    sysm = assemble_elastic_grid(
        X[None], F[None], vol0, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        collider_planes=[((0.0, 0.0, floor_z), (0.0, 0.0, 1.0))],
        collider_margin_cells=margin, min_support_mass_frac=0.0)
    assert sysm.n_rows > 0
    node_z = (sysm.node_id % N_GRID) * dx
    assert node_z.min() >= floor_z + margin * dx - 1e-12
    # and the surviving rows still recover the moduli exactly
    out = solve_elastic_grid(sysm)
    assert abs(out["mu"] / mu_t - 1.0) < 1e-7


def test_mass_gate_and_row_bookkeeping():
    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, accel = _manufactured_accel(mu_t, lam_t)
    loose = assemble_elastic_grid(
        X[None], F[None], vol0, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        min_support_mass_frac=0.0)
    tight = assemble_elastic_grid(
        X[None], F[None], vol0, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        min_support_mass_frac=0.9)
    assert tight.n_rows < loose.n_rows
    assert loose.n_rows_before_gating == tight.n_rows_before_gating
    assert loose.n_rows == loose.n_rows_before_gating   # nothing gated at 0.0
    assert 0.0 < tight.row_survival < 1.0
    assert tight.node_mass_frac.min() >= 0.9 - 1e-12


def test_non_finite_particles_gate_their_nodes_out():
    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, accel = _manufactured_accel(mu_t, lam_t)
    Fb = F.copy()
    Fb[0] = np.nan
    sysm = assemble_elastic_grid(
        X[None], Fb[None], vol0, mass, accel[None], G_VEC, N_GRID, GRID_LIM,
        min_support_mass_frac=0.0)
    nid, _, _, valid = bspline_stencil(X[:1], N_GRID, GRID_LIM)
    poisoned = set(nid[0][valid[0]].tolist())
    assert poisoned.isdisjoint(set(sysm.node_id.tolist()))
    assert np.isfinite(sysm.A).all() and np.isfinite(sysm.b).all()


def test_empty_system_is_reported_not_raised():
    X, vol0, mass = _cloud()
    F = np.repeat(np.eye(3)[None], len(X), axis=0)
    sysm = assemble_elastic_grid(
        X[None], F[None], vol0, mass, np.zeros((1, len(X), 3)), G_VEC,
        N_GRID, GRID_LIM,
        collider_planes=[((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        collider_margin_cells=100.0)
    assert sysm.n_rows == 0
    assert sysm.row_survival == 0.0
    with pytest.raises(ValueError):
        solve_elastic_grid(sysm)


def test_moduli_conversions_round_trip():
    for E, nu in ((1.0e5, 0.2), (3.0e5, 0.25), (2.0e5, 0.3)):
        mu, lam = E_nu_to_moduli(E, nu)
        E2, nu2 = moduli_to_E_nu(mu, lam)
        assert abs(E2 / E - 1.0) < 1e-12
        assert abs(nu2 - nu) < 1e-12


# --------------------------------------------------------------------------
# The law-independent core: any number of columns, plus a known stress part
# --------------------------------------------------------------------------

def test_single_column_solve_and_known_stress_part():
    """Moving a column into the known stress part shifts b by exactly that column.

    The sand and water legs of the NCLaw comparison both ride this path: one
    unknown, and for sand a pressure term that is data rather than unknown. The
    identity below is exact by construction, so it pins the b_known plumbing and
    the sign with it, and a K = 1 system must not go looking for the elastic
    pair's names.
    """
    from ident.weakform.elastic_grid import assemble_columns_timeweak

    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, _ = _manufactured_accel(mu_t, lam_t)
    T = 8
    xs = np.repeat(X[None], T, axis=0)
    vs = np.full((T, len(X), 3), 0.05)
    s_mu, s_lam = corotated_cauchy_columns(F)
    Vcur = np.linalg.det(F) * vol0
    both = Vcur[:, None, None, None] * np.stack([s_mu, lam_t * s_lam], axis=1)
    only_mu = both[:, :1]
    known = both[:, 1]

    kw = dict(n_columns=2, window_frames=T, min_support_mass_frac=0.0,
              window_taper_cells=None)
    sys_two = assemble_columns_timeweak(
        xs, vs, mass, G_VEC, 1.0e-3, N_GRID, GRID_LIM,
        lambda f: (both, None, np.ones(len(X), dtype=bool), None), **kw)
    kw["n_columns"] = 1
    sys_one = assemble_columns_timeweak(
        xs, vs, mass, G_VEC, 1.0e-3, N_GRID, GRID_LIM,
        lambda f: (only_mu, known, np.ones(len(X), dtype=bool), None), **kw)

    assert sys_one.n_rows == sys_two.n_rows > 0
    assert sys_one.A.shape[1] == 1 and sys_two.A.shape[1] == 2
    assert np.allclose(sys_one.A[:, 0], sys_two.A[:, 0], rtol=1e-12, atol=0.0)
    assert np.allclose(sys_one.b, sys_two.b - sys_two.A[:, 1],
                       rtol=1e-10, atol=1e-14 * np.abs(sys_two.b).max())

    out = solve_elastic_grid(sys_one)
    assert len(out["theta"]) == 1
    assert "mu" not in out and "lam" not in out
    assert len(solve_elastic_grid(sys_two)["theta"]) == 2
    assert "mu" in solve_elastic_grid(sys_two)


def test_columns_fn_may_skip_a_frame():
    """Returning None from columns_fn drops the frame, and any window
    containing it, without raising."""
    from ident.weakform.elastic_grid import assemble_columns_timeweak

    mu_t, lam_t = E_nu_to_moduli(2.0e5, 0.3)
    X, F, vol0, mass, _ = _manufactured_accel(mu_t, lam_t)
    T = 10
    xs = np.repeat(X[None], T, axis=0)
    vs = np.zeros((T, len(X), 3))
    s_mu, s_lam = corotated_cauchy_columns(F)
    Vsig = (np.linalg.det(F) * vol0)[:, None, None, None] * np.stack(
        [s_mu, s_lam], axis=1)

    def columns_fn(f):
        return None if f == 0 else (Vsig, None, np.ones(len(X), dtype=bool), None)

    sysm = assemble_columns_timeweak(
        xs, vs, mass, G_VEC, 1.0e-3, N_GRID, GRID_LIM, columns_fn,
        n_columns=2, window_frames=4, min_support_mass_frac=0.0,
        window_taper_cells=None)
    assert sysm.n_rows > 0
    assert 0 not in sysm.frames_used

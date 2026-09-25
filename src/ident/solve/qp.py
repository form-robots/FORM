"""Constrained QP solve via OSQP (MATH_REFERENCE.md Section 8.1).

minimize |A theta - b|^2 + lambda theta^T G theta   subject to   C theta >= h

with admissibility (mu >= mu_min on a log-I grid), the Mode P nonnegativity
pattern (sigmoid coefficients c_g >= 0), and optional monotonicity (mu
non-decreasing in I via dphi_dlogI rows). Gate logic uses this and the
closed-form ridge only; cvxpy is exploratory.

The collinear sigmoid basis of Mode P is deliberately ill-conditioned; the
nonnegativity and admissibility constraints plus L2 (or L1) regularization are
what make the friction-rise recovery well-posed, which is why the
unconstrained ridge alone returns unphysical negative coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

from ident.features.base import Dictionary


@dataclass
class QPResult:
    theta: np.ndarray
    status: str
    cond_AtA: float
    effective_rank: int
    residual_rel: float
    mu_min_active: bool


def constrained_solve(
    A: np.ndarray,
    b: np.ndarray,
    dictionary: Dictionary,
    lam: float = 1.0e-6,
    G: np.ndarray | None = None,
    mu_min: float = 0.05,
    I_constraint_grid: np.ndarray | None = None,
    nonnegativity: bool = True,
    monotonic: bool = False,
    l1: float = 0.0,
) -> QPResult:
    """Solve the constrained ridge/QP for theta.

    nonnegativity uses the dictionary metadata pattern (Mode P: c_g >= 0).
    mu_min admissibility and monotonicity are enforced on I_constraint_grid.
    l1 > 0 adds an L1 penalty on the nonnegative coefficients (sparse
    selection); only valid together with nonnegativity so |c| = c.
    """
    K = dictionary.K
    if G is None:
        G = np.eye(K)
    if I_constraint_grid is None:
        I_constraint_grid = np.logspace(-4, 0, 40)

    P = 2.0 * (A.T @ A + lam * G)
    q = -2.0 * (A.T @ b)

    # L1 on nonnegative coefficients = linear term lam_l1 * sum c_g
    if l1 > 0.0:
        nn = np.array(dictionary.metadata["nonnegativity"], dtype=bool)
        q = q + l1 * nn.astype(float)

    # symmetrize P for OSQP
    P = 0.5 * (P + P.T)

    rows_C = []
    lo = []
    # admissibility: phi(I_i) theta >= mu_min
    Phi_c = dictionary.phi(I_constraint_grid)            # (m, K)
    rows_C.append(Phi_c)
    lo.append(np.full(Phi_c.shape[0], mu_min))
    # nonnegativity: theta_k >= 0 for flagged columns
    if nonnegativity:
        nn = np.array(dictionary.metadata["nonnegativity"], dtype=bool)
        if nn.any():
            E = np.eye(K)[nn]
            rows_C.append(E)
            lo.append(np.zeros(E.shape[0]))
    # monotonicity: dphi_dlogI(I_i) theta >= 0
    if monotonic:
        D_c = dictionary.dphi_dlogI(I_constraint_grid)
        rows_C.append(D_c)
        lo.append(np.zeros(D_c.shape[0]))

    C = np.vstack(rows_C)
    l = np.concatenate(lo)
    u = np.full(l.shape[0], np.inf)

    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(C), l=l, u=u,
               verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000,
               polish=True)
    res = prob.solve()
    theta = np.asarray(res.x, dtype=float)
    if theta is None or not np.all(np.isfinite(theta)):
        theta = np.zeros(K)

    sv = np.linalg.svd(A, compute_uv=False)
    sv2 = sv * sv
    cond = float(sv2[0] / sv2[-1]) if sv2[-1] > 0 else np.inf
    eff_rank = int(np.sum(sv2 > 1e-8 * sv2[0]))
    r = A @ theta - b
    bnorm = float(np.linalg.norm(b))
    res_rel = float(np.linalg.norm(r)) / bnorm if bnorm > 0 else np.inf
    mu_at_grid = Phi_c @ theta
    mu_min_active = bool(np.any(mu_at_grid <= mu_min + 1e-6))

    return QPResult(theta=theta, status=str(res.info.status), cond_AtA=cond,
                    effective_rank=eff_rank, residual_rel=res_rel,
                    mu_min_active=mu_min_active)

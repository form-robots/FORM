"""Shared definitions, coordinate and sign helpers, defaults, and seeds.

The mathematical reference is docs/MATH_REFERENCE.md, and G0 checks these conventions.
In Config A, x is horizontal, z is vertical and positive upward, and y is out of plane.
The in-plane gravity vector is (0, -G_MAG). Scalar-gravity formulas with explicit
signs are centralized in this module; assembly formulas use the gravity vector.

Pressure sign conventions also pass through this module. Pressure from a
dumped 3D Cauchy stress uses the full 3D trace, not the in-plane trace.
"""

from __future__ import annotations

import subprocess

import numpy as np


def git_rev() -> str | None:
    """Short git commit of the repo, or None if not a git repo / git unavailable.
    Used to stamp every gate results.json for reproducibility (CLAUDE.md contract)."""
    try:
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


# ---------------- physical constants and defaults ----------------

G_MAG: float = 9.81
EPS_GAMMA_DEFAULT: float = 0.02  # 1/s, shear-rate regularization
SEED_DEFAULT: int = 0

# Flowing mask defaults (MATH_REFERENCE.md Section 4)
GAMMA_MIN_DEFAULT: float = 0.5  # 1/s
I_MIN_DEFAULT: float = 1.0e-4

# Validity mask constants, in units of grain diameter d
C_F_DEFAULT: float = 10.0  # distance to front
C_S_DEFAULT: float = 5.0   # distance to free surface and base
C_H_DEFAULT: float = 8.0   # flowing-layer thickness

# Ground-truth mu(I) tabulation contract (dump schema)
LOG10_I_TABLE_MIN: float = -4.0
LOG10_I_TABLE_MAX: float = 0.0
MU_TABLE_POINTS: int = 256

LN10: float = float(np.log(10.0))


def gravity_vector_inplane() -> np.ndarray:
    """In-plane gravity vector g = (g_x, g_z) = (0, -G_MAG)."""
    return np.array([0.0, -G_MAG])


# ---------------- kinematic definitions ----------------

def sym(L: np.ndarray) -> np.ndarray:
    """D = sym(grad v) for L with convention L_ij = d v_i / d x_j."""
    return 0.5 * (L + np.swapaxes(L, -1, -2))


def equivalent_shear_rate(D: np.ndarray, eps_gamma: float = EPS_GAMMA_DEFAULT) -> np.ndarray:
    """|gamma_dot|_eps = sqrt(2 D : D + eps_gamma^2).

    D has shape (..., d, d). The factor of 2 here is pinned by G0; for simple
    shear v_x = gamma * z this returns gamma (up to eps_gamma).
    """
    dd = np.einsum("...ij,...ij->...", D, D)
    return np.sqrt(2.0 * dd + eps_gamma * eps_gamma)


def inertial_number(gamma_dot_eps: np.ndarray, p: np.ndarray, d: float, rho_s: float) -> np.ndarray:
    """I = |gamma_dot|_eps d / sqrt(p / rho_s). Returns +inf where p <= 0."""
    gamma_dot_eps = np.asarray(gamma_dot_eps, dtype=float)
    p = np.asarray(p, dtype=float)
    out = np.full(np.broadcast_shapes(gamma_dot_eps.shape, p.shape), np.inf)
    ok = p > 0.0
    out[ok] = (gamma_dot_eps * d * np.sqrt(rho_s / np.where(ok, p, 1.0)))[ok]
    return out


def flow_direction(D: np.ndarray, eps_gamma: float = EPS_GAMMA_DEFAULT) -> np.ndarray:
    """2 D / |gamma_dot|_eps, the tensor multiplying mu(I) p in the law."""
    gd = equivalent_shear_rate(D, eps_gamma)
    return 2.0 * D / gd[..., None, None]


# ---------------- pressure conventions ----------------

def pressure_from_cauchy_3d_trace(sigma: np.ndarray) -> np.ndarray:
    """p = -(sigma_xx + sigma_yy + sigma_zz) / 3 for sigma of shape (..., 3, 3).

    The 2D trace is forbidden; sigma_yy exists in plane strain and enters here.
    """
    sigma = np.asarray(sigma)
    if sigma.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) Cauchy stress, got {sigma.shape}")
    tr = sigma[..., 0, 0] + sigma[..., 1, 1] + sigma[..., 2, 2]
    return -tr / 3.0


def hydrostatic_pressure_p0(rho: np.ndarray, h: np.ndarray, z: np.ndarray) -> np.ndarray:
    """P0 closure value: p = rho * G_MAG * (h - z), z vertical upward."""
    return np.asarray(rho) * G_MAG * (np.asarray(h) - np.asarray(z))


def p1_integrand(rho: np.ndarray, a_z: np.ndarray) -> np.ndarray:
    """Integrand of the P1 closure: dp/dz = rho (g_z - a_z) with g_z = -G_MAG,
    integrated downward from the free surface, gives p(z) = INT_z^h rho (a_z + G_MAG) dz'.

    Keep this sign convention in the helper rather than duplicating it in closures.
    """
    return np.asarray(rho) * (np.asarray(a_z) + G_MAG)


# ---------------- reference mu(I) laws (ground truth bookkeeping) ----------------

def pouliquen_mu(I: np.ndarray, mu_s: float, delta_mu: float, I0: float) -> np.ndarray:
    """mu(I) = mu_s + delta_mu * I / (I + I0). Used for sim truth tables only."""
    I = np.asarray(I, dtype=float)
    return mu_s + delta_mu * I / (I + I0)


def mu_table_grid() -> np.ndarray:
    """The pinned 256-point log10 I grid in [-4, 0] for ground-truth tables."""
    return np.linspace(LOG10_I_TABLE_MIN, LOG10_I_TABLE_MAX, MU_TABLE_POINTS)

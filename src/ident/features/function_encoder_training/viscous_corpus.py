"""Viscous (apparent-viscosity) corpus for the fluid function encoder.

The basis is trained to span physically plausible generalized-Newtonian /
viscoplastic apparent viscosities eta_app(gd) on s = log10 gd in [-1, 2]
(shear rate gd in [0.1, 100] /s, the realized pour band). Families: Newtonian
(constant), shear-thinning power-law, Carreau, Bingham, and Herschel-Bulkley.
The yield 1/gd divergence is regularized (eps) so eta_app is bounded, matching
the simulator's kirchoff_stress_newtonian. The weak-form viscous basis multiplies
eta_app(gd), so the learned Phi_k(gd) spans these shapes and the linear solve
recovers eta_app(gd) = sum_k theta_k Phi_k(gd), the viscous analogue of mu(I).

Each law is normalized to unit weighted L2 on the s-grid: the basis spans shapes;
magnitude is carried by theta at solve time. Pure numpy.
"""
from __future__ import annotations
import numpy as np

S_MIN, S_MAX, N_S = -1.0, 2.0, 256
EPS_GD = 0.05


def s_grid() -> np.ndarray:
    return np.linspace(S_MIN, S_MAX, N_S)


def gd_grid() -> np.ndarray:
    return 10.0 ** s_grid()


def _yield_term(gd, tau_y):
    return tau_y / np.sqrt(gd * gd + EPS_GD * EPS_GD)


def sample_eta_law(rng: np.random.Generator):
    """Return (eta_app on s-grid, descriptor); shape-normalized to unit L2."""
    gd = gd_grid()
    kind = rng.choice(["newtonian", "powerlaw", "carreau", "bingham", "herschel"])
    if kind == "newtonian":
        eta = rng.uniform(0.5, 50.0)
        e = np.full_like(gd, eta); desc = {"kind": kind, "eta": eta}
    elif kind == "powerlaw":  # shear-thinning eta = K gd^(n-1), n<1
        K = 10.0 ** rng.uniform(0.0, 2.0); n = rng.uniform(0.3, 0.95)
        e = K * np.power(gd, n - 1.0); desc = {"kind": kind, "K": K, "n": n}
    elif kind == "carreau":
        eta0 = 10.0 ** rng.uniform(1.0, 3.0); einf = rng.uniform(0.1, 2.0)
        lam = 10.0 ** rng.uniform(-1.0, 1.0); n = rng.uniform(0.2, 0.9)
        e = einf + (eta0 - einf) * np.power(1.0 + (lam * gd) ** 2, (n - 1.0) / 2.0)
        desc = {"kind": kind, "eta0": eta0, "einf": einf, "lam": lam, "n": n}
    elif kind == "bingham":
        eta = rng.uniform(1.0, 40.0); tau_y = 10.0 ** rng.uniform(1.0, 2.8)
        e = eta + _yield_term(gd, tau_y); desc = {"kind": kind, "eta": eta, "tau_y": tau_y}
    else:  # herschel-bulkley
        K = 10.0 ** rng.uniform(0.0, 1.8); n = rng.uniform(0.3, 0.95)
        tau_y = 10.0 ** rng.uniform(1.0, 2.8)
        e = K * np.power(gd, n - 1.0) + _yield_term(gd, tau_y)
        desc = {"kind": kind, "K": K, "n": n, "tau_y": tau_y}
    # normalize to unit L2 on the (uniform) s-grid -> basis spans shapes
    nrm = np.sqrt(np.mean(e * e))
    return (e / max(nrm, 1e-30)).astype(np.float64), desc


def build_corpus(n_materials: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    E = np.empty((n_materials, N_S)); descs = []
    for m in range(n_materials):
        e, d = sample_eta_law(rng); E[m] = e; descs.append(d)
    return E, descs

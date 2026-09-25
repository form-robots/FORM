"""mu(I) material corpus and the L2_w weight (docs/FUNCTION_ENCODER.md Section 1).

The basis is trained to span a family of physically plausible granular
friction laws on s = log10 I in [-4, 0]. The corpus is intentionally broader
than the single Pouliquen form so the learned basis generalizes (and so the
out-of-family probe has something to fail on): Pouliquen sigmoids, constants,
saturating power laws, and two-component (double-sigmoid) laws.

Pure numpy; no torch here (the network consumes these samples).
"""

from __future__ import annotations

import numpy as np

S_MIN, S_MAX, N_S = -4.0, 0.0, 256


def s_grid() -> np.ndarray:
    return np.linspace(S_MIN, S_MAX, N_S)


def I_grid() -> np.ndarray:
    return 10.0 ** s_grid()


def sample_mu_law(rng: np.random.Generator):
    """Return (mu_values_on_s_grid, descriptor) for one corpus material."""
    I = I_grid()
    kind = rng.choice(["pouliquen", "pouliquen", "pouliquen", "constant",
                       "powerlaw", "double"])
    if kind == "constant":
        mu_s = rng.uniform(0.2, 0.55)
        mu = np.full_like(I, mu_s)
        desc = {"kind": kind, "mu_s": mu_s}
    elif kind == "pouliquen":
        mu_s = rng.uniform(0.18, 0.45)
        dmu = rng.uniform(0.05, 0.55)
        I0 = 10.0 ** rng.uniform(-2.5, 0.0)
        mu = mu_s + dmu * I / (I + I0)
        desc = {"kind": kind, "mu_s": mu_s, "delta_mu": dmu, "I0": I0}
    elif kind == "powerlaw":
        mu_s = rng.uniform(0.2, 0.45)
        a = rng.uniform(0.1, 0.6)
        n = rng.uniform(0.3, 1.0)
        mu = mu_s + a * np.power(np.clip(I, 1e-12, None), n)
        mu = np.minimum(mu, 1.2)
        desc = {"kind": kind, "mu_s": mu_s, "a": a, "n": n}
    else:  # double sigmoid
        mu_s = rng.uniform(0.18, 0.4)
        d1 = rng.uniform(0.05, 0.3)
        d2 = rng.uniform(0.05, 0.3)
        I1 = 10.0 ** rng.uniform(-3.0, -1.0)
        I2 = 10.0 ** rng.uniform(-1.0, 0.0)
        mu = mu_s + d1 * I / (I + I1) + d2 * I / (I + I2)
        desc = {"kind": kind, "mu_s": mu_s, "d1": d1, "d2": d2, "I1": I1, "I2": I2}
    return mu.astype(np.float64), desc


def build_corpus(n_materials: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    mus = np.empty((n_materials, N_S))
    descs = []
    for m in range(n_materials):
        mu, d = sample_mu_law(rng)
        mus[m] = mu
        descs.append(d)
    return mus, descs


def weight_vector(realized_hist=None, mix: float = 0.5) -> np.ndarray:
    """w(s): mix of a realized flowing-I histogram and uniform-in-log.

    realized_hist: optional (edges, counts) on log10 I; if None, uniform only.
    Returns a length-N_S nonnegative weight that integrates (trapezoid on s)
    to 1.
    """
    s = s_grid()
    uniform = np.ones_like(s)
    if realized_hist is not None:
        edges, counts = realized_hist
        # edges are I (logspace); map to s and interpolate counts onto s_grid
        s_edges = np.log10(edges)
        s_cent = 0.5 * (s_edges[:-1] + s_edges[1:])
        dens = np.interp(s, s_cent, counts, left=0.0, right=0.0)
        dens = dens / max(dens.max(), 1e-12)
        w = mix * dens + (1.0 - mix) * (uniform / uniform.max())
    else:
        w = uniform
    # normalize to unit trapezoid mass on s
    ds = np.gradient(s)
    w = w / max(np.sum(w * ds), 1e-12)
    return w

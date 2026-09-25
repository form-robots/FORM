"""Least squares through a FUNCTION-ENCODER basis: the third row of the NCLaw table.

The comparison table puts three identification methods on the same grid-20
NCLaw-suite trajectories:

  1. least squares with the KNOWN constitutive form (the suite's own identify
     legs: one friction coefficient, one elastic pair, one bulk modulus),
  2. gradient descent through a differentiable simulator
     (experiments.diffsim),
  3. least squares with a trained function-encoder basis in place of the known
     form, which is this file.

Row 3 keeps everything about row 1 except the dictionary: same dump, same masks,
same grid-consistent time-weak assembly, same convex solve. The unknown is a
FUNCTION (mu(I) for sand, W'(I1bar) for jelly, eta_app(gamma_dot) for the two
viscous surrogates) expanded in K trained basis functions, so the solve stays
linear in theta and convex. The recovered curve is then baked into the engine's
tabulated material and re-simulated, which turns a curve error into a
trajectory error.

Per-material outcomes, numbers in report.md:

  sand      mu(I) through the trained granular basis, identified and rolled out
            on the cube (reconstruction) and their blub mesh (generalization).
  jelly     one-invariant hyperelastic basis plus one volumetric column,
            identification only; the engine has no tabulated hyperelastic
            material, so the rollout cell is UNSUPPORTED.
  plasticine, water
            viscous surrogates; the truth is an elastoplastic solid and an
            inviscid EOS fluid, so these rows measure how far a viscous
            surrogate carries the rollout.

Regularization, chosen once and applied to every material and every variant: the
family prior where the trained table ships one (the granular basis carries
theta_mean and theta_cov from the corpus), a plain ridge where it does not. The
weight is picked by leave-one-window-out cross-validation of the weak-form rows,
with ties inside CV_TIE_TOL of the minimum going to the LARGER weight. Nothing
in the rule looks at the truth. The results json records the full weight sweep
with its cross-validation scores.

Stages, resumable, one material at a time:

  identify   assemble and solve; write identify_<material>.json plus the curve
  rollout    bake the curve into the engine table, re-simulate, score the
             NCLaw position MSE (sand only; other materials record the reason)
  report     out/fe_ls_baseline/report.md, results_<material>.json, one figure
             per identified curve

Artifacts, all under out/fe_ls_baseline (see experiments.fe_ls.artifact_dir for
which out/ tree that is): identify_<m>.json, rollout_<m>.json, legs/, dumps/,
fe_curve_<m>.png, results_<m>.json, report.md, run.log.

Run:
  .venv/bin/python -m experiments.fe_ls identify --material sand
  .venv/bin/python -m experiments.fe_ls rollout  --material sand
  .venv/bin/python -m experiments.fe_ls report   --material all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.nclaw import suite

from . import ENGINE_ROOT, STAGING_ROOT, artifact_dir

ENGINE = ENGINE_ROOT
OUT = artifact_dir("fe_ls_baseline")
DIFFSIM_OUT = artifact_dir("diffsim_baseline")
DUMPS = OUT / "dumps"
WEIGHTS = ENGINE / "fe-weights"
N_GRID = 20                      # the diff-sim baseline's grid, and this table's

# Trained bases. The granular table is the only one that ships a family prior
# (theta_mean, theta_cov from features/function_encoder_training/prior.py); the
# others get a plain ridge, which the report states.
FE_GRANULAR = WEIGHTS / "granular_mu_i.npz"
FE_HYPER = WEIGHTS / "hyperelastic_1inv.npz"
FE_VISCOUS = WEIGHTS / "viscous.npz"

CV_TIE_TOL = 1.05                # a weight within 5 percent of the best CV score
                                 # counts as tied, and the larger weight wins
REG_GRID = np.concatenate([[0.0], np.logspace(-9.0, 1.0, 21)])

# Time-weak window for the viscous surrogate legs, in sampled frames. The
# suite's own windows (26 for plasticine, 16 for water) leave two and one
# surviving windows on these rows, and one window is one held-out unit, so the
# cross-validation rule cannot run. These are the longest windows that leave at
# least three: 5 surviving windows for plasticine and 4 for water.
VISCOUS_WINDOW = {"plasticine": 16, "water": 10}

# Held-out mesh per material, NCLaw's own pairing, at the mild throw (the
# preset throw drives the stiff sand blub through the walls).
HELD_OUT = {m: (suite.NCLAW_HELD_OUT_SHAPE[m], "mild") for m in suite.NCLAW_PUBLISHED}

MATERIALS = ("sand", "jelly", "plasticine", "water")

# Reasons a rollout cell is empty.
ROLLOUT_UNSUPPORTED = {
    "jelly": ("UNSUPPORTED: the warp engine has no tabulated hyperelastic "
              "material, so a recovered W'(I1bar) curve cannot be re-simulated "
              "without adding one."),
}


def _log_factory(path: Path):
    """Print and append to a log file, truncating the 256-entry table lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a")

    def log(*args):
        msg = " ".join(str(a) for a in args)
        if len(msg) > 400:
            msg = msg[:400] + f" ... [{len(msg)} chars truncated]"
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    return log


def _rel_figure(p: Path) -> str:
    """A figure path as the results json records it: "out/fe_ls_baseline/x.png"
    wherever OUT resolved, absolute if it sits outside any out/ tree."""
    try:
        return str(p.relative_to(OUT.parent.parent))
    except ValueError:
        return str(p)


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# dictionaries
# ---------------------------------------------------------------------------

def load_granular_fe():
    """The granular mu(I) basis and its family prior (theta_mean, theta_cov)."""
    from ident.features.function_encoder import FunctionEncoderDict
    d = np.load(FE_GRANULAR)
    fe = FunctionEncoderDict(d["s_grid"], d["table"])
    prior = None
    if "theta_mean" in d.files:
        prior = (np.asarray(d["theta_mean"], float), np.asarray(d["theta_cov"], float))
    return fe, prior


def load_table_fe(path: Path, key: str):
    """A one-input trained table as (grid, table, K) with cubic interpolation.

    The hyperelastic and viscous tables are not mu(I) dictionaries, so they do
    not go through FunctionEncoderDict (whose input coordinate is log10 I);
    their own abscissa is used, clamped at both ends exactly as that class
    clamps, so the out-of-support behaviour matches.
    """
    from scipy.interpolate import CubicSpline
    d = np.load(path)
    grid = np.asarray(d[key], float)
    table = np.asarray(d["table"], float)
    spline = CubicSpline(grid, table, axis=0, bc_type="natural")

    def phi(x):
        return spline(np.clip(np.asarray(x, float), grid[0], grid[-1]))

    return phi, grid, int(table.shape[1])


# ---------------------------------------------------------------------------
# the regularized convex solve, shared by every material
# ---------------------------------------------------------------------------

def _prior_rows(A: np.ndarray, b: np.ndarray, weight: float,
                prior: tuple[np.ndarray, np.ndarray] | None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Augment a system with the regularizer as extra rows.

    With a family prior the penalty is (theta - theta_bar)^T (w Sigma^-1)
    (theta - theta_bar), which equals the least squares of the augmented rows
    L = chol(w Sigma^-1) against L theta_bar; this is exactly what
    ident/gates/fe_joint_prior.py does, so no solver changes. Without a prior it
    is an isotropic ridge of the same weight, scaled by trace(A^T A) / K so the
    weight is dimensionless in both cases.
    """
    if weight <= 0.0:
        return A, b
    K = A.shape[1]
    if prior is not None:
        theta_bar, cov = prior
        L = np.linalg.cholesky(weight * np.linalg.inv(cov))
        return np.vstack([A, L]), np.concatenate([b, L @ theta_bar])
    lam = weight * np.trace(A.T @ A) / K
    L = np.sqrt(lam) * np.eye(K)
    return np.vstack([A, L]), np.concatenate([b, np.zeros(K)])


def _fit(A: np.ndarray, b: np.ndarray, weight: float,
         prior: tuple[np.ndarray, np.ndarray] | None,
         qp: dict | None) -> np.ndarray:
    """One convex fit: least squares, or the constrained QP when qp is given."""
    Aa, ba = _prior_rows(A, b, weight, prior)
    if qp is None:
        return np.linalg.lstsq(Aa, ba, rcond=None)[0]
    from ident.solve.qp import constrained_solve
    res = constrained_solve(Aa, ba, qp["dictionary"], lam=qp.get("lam", 1.0e-6),
                            G=qp.get("G"), mu_min=qp.get("mu_min", 0.05),
                            I_constraint_grid=qp.get("grid"),
                            nonnegativity=False, monotonic=True)
    return np.asarray(res.theta, float)


def regularized_solve(A: np.ndarray, b: np.ndarray,
                      prior: tuple[np.ndarray, np.ndarray] | None,
                      curve_fn, groups: np.ndarray | None = None,
                      qp: dict | None = None, log=print) -> dict:
    """Sweep the regularization weight, pick it by cross-validation, report.

    groups labels each row with the time-weak window it came from. The rows of
    one window are the same nodal balance read through several smooth spatial
    modes, so they are strongly correlated and only a whole window is a
    meaningful held-out unit; the score is therefore leave-one-window-out.
    curve_fn(theta) returns the recovered curve on the reporting grid, so the
    sweep can carry the curve at every weight without knowing what it means.
    """
    scale = float(np.linalg.norm(A)) + 1.0e-300
    An, bn = A / scale, b / scale
    bnorm = float(np.linalg.norm(bn)) + 1.0e-300
    folds = [] if groups is None else list(np.unique(groups))
    sweep = []
    for w in REG_GRID:
        theta = _fit(An, bn, float(w), prior, qp)
        resid = float(np.linalg.norm(An @ theta - bn) / bnorm)
        curve = np.asarray(curve_fn(theta), float)
        cv = None
        if len(folds) >= 3:
            num = den = 0.0
            for f in folds:
                te = groups == f
                th = _fit(An[~te], bn[~te], float(w), prior, qp)
                num += float(np.sum((An[te] @ th - bn[te]) ** 2))
                den += float(np.sum(bn[te] ** 2))
            cv = float(np.sqrt(num / max(den, 1e-300)))
        sweep.append({"weight": float(w), "residual_rel": resid, "cv": cv,
                      "theta": theta.tolist(),
                      "curve_min": float(np.min(curve)),
                      "curve_max": float(np.max(curve))})

    scored = [s for s in sweep if s["cv"] is not None]
    if scored:
        best = min(s["cv"] for s in scored)
        tied = [s for s in scored if s["cv"] <= CV_TIE_TOL * best]
        pick = max(tied, key=lambda s: s["weight"])
        rule = (f"leave-one-window-out cross-validation over {len(folds)} windows; "
                f"weights within {CV_TIE_TOL} of the best score count as tied and "
                "the largest of those is taken")
    else:
        pick = sweep[0]
        rule = "no cross-validation possible (fewer than three windows); unregularized"
    theta = np.asarray(pick["theta"], float)

    # posterior covariance at the selected weight, on the raw column scale. This
    # is a VARIANCE statement about the rows that survived; the curve error
    # below is dominated by bias (a misspecified family, an unexcited part of
    # the support), which no posterior of this form can see.
    K = A.shape[1]
    M = np.zeros((K, K))
    if pick["weight"] > 0.0:
        if prior is not None:
            M = pick["weight"] * np.linalg.inv(prior[1])
        else:
            M = (pick["weight"] * np.trace(An.T @ An) / K) * np.eye(K)
    Hinv = np.linalg.inv(An.T @ An + M)
    r = An @ theta - bn
    dof = max(A.shape[0] - float(np.trace(An @ Hinv @ An.T)), 1.0)
    sigma2 = float(r @ r) / dof
    cov = sigma2 * Hinv / (scale ** 2)

    sv = np.linalg.svd(A, compute_uv=False)
    log(f"[solve] rows={A.shape[0]} K={K} kind={'qp' if qp else 'ridge'} "
        f"weight={pick['weight']:.3g} cv={pick['cv']} "
        f"residual={pick['residual_rel']:.4f} cond={float((sv[0]/sv[-1])**2):.3e}")
    return {
        "theta": theta.tolist(),
        "theta_cov": cov.tolist(),
        "theta_sd": np.sqrt(np.clip(np.diag(cov), 0.0, None)).tolist(),
        "solver": "constrained QP (OSQP), monotone and mu >= mu_min" if qp
                  else "closed-form least squares",
        "reg_weight": float(pick["weight"]),
        "reg_rule": rule,
        "reg_sweep": sweep,
        "cv_score": pick["cv"],
        "residual_rel": pick["residual_rel"],
        "residual_rel_unregularized": sweep[0]["residual_rel"],
        "cond_AtA": float((sv[0] / sv[-1]) ** 2) if sv[-1] > 0 else float("inf"),
        "effective_rank": int((sv > sv[0] * 1e-10).sum()),
        "n_rows": int(A.shape[0]),
        "n_cv_folds": len(folds),
        "row_scale": scale,
    }


def curve_errors(y_hat: np.ndarray, y_true: np.ndarray, x_realized: np.ndarray,
                 y_hat_realized: np.ndarray, y_true_realized: np.ndarray) -> dict:
    """Curve error two ways: on the reporting grid, which weights every decade of
    the support equally, and once per surviving particle-frame, which weights it
    by where the material went. The two differ when the realized support is
    skewed."""
    ref = float(np.sqrt(np.mean(y_true ** 2))) + 1.0e-300
    lo, hi = np.percentile(x_realized, [5, 95])
    inside = (x_realized >= lo) & (x_realized <= hi)
    refr = float(np.sqrt(np.mean(y_true_realized ** 2))) + 1.0e-300
    return {
        "relL2_on_grid": float(np.sqrt(np.mean((y_hat - y_true) ** 2)) / ref),
        "relL2_realized": float(
            np.sqrt(np.mean((y_hat_realized - y_true_realized) ** 2)) / refr),
        "relL2_realized_p5_p95": float(
            np.sqrt(np.mean((y_hat_realized[inside] - y_true_realized[inside]) ** 2))
            / refr) if inside.any() else None,
        "max_abs_err_on_grid": float(np.max(np.abs(y_hat - y_true))),
    }


# ---------------------------------------------------------------------------
# sand: Mode F mu(I) beside the suite's constant-friction Mode C
# ---------------------------------------------------------------------------

def identify_friction_fe(arr: dict, fe, prior, window_frames: int = 10,
                         frame_stride: int = 2, margin_cells: float = 3.0,
                         eps_gamma: float = 0.02, gd_min: float = 1.0,
                         yield_frac_min: float = 0.97, log=print) -> dict:
    """Mode F friction in 3D: K function-encoder columns where Mode C has one.

    Runs beside suite.identify_friction and keeps all of its checks: the
    pressure is DATA (the 3D stress trace), a particle enters only where it is
    shearing under positive pressure, a cohesionless particle at or below zero
    pressure is stress free and therefore modelled rather than invalid, the
    frame list is the longest contiguous shearing run after the post-impact
    kinetic-energy check, and a node must pass in every frame of the time-weak
    window. The single column V p (2 D / |gd|)
    becomes K columns V phi_k(I) p (2 D / |gd|), with

        I = |gamma_dot|_eps d / sqrt(p / rho_s)

    from the conventions helper at the dump's own grain diameter and grain
    density. mu(I) = sum_k theta_k phi_k(I) stays linear in theta.
    """
    from common.conventions import (
        equivalent_shear_rate,
        inertial_number,
        pressure_from_cauchy_3d_trace,
        sym,
    )
    from ident.weakform.elastic_grid import assemble_columns_timeweak

    if not arr["meta"].has_pressure:
        return {"refused": True, "reason": (
            "no oracle pressure: the dump's pressure_source is "
            f"{arr['meta'].pressure_source!r}, so the stress trace this leg needs "
            "is absent")}

    d_grain = float(arr["meta"].grain_diameter)
    rho_s = float(arr["meta"].rho_s)
    D = sym(arr["L"])
    gd = equivalent_shear_rate(D, eps_gamma)
    pres = pressure_from_cauchy_3d_trace(arr["stress"])
    eye = np.eye(3)[None]
    I_acc: list[np.ndarray] = []
    W_acc: list[np.ndarray] = []
    K = fe.K
    I_lo, I_hi = fe.metadata["support"]

    def columns_fn(f: int):
        finite = np.isfinite(pres[f]) & np.isfinite(D[f]).all(axis=(1, 2))
        at_yield = finite & (pres[f] > 0.0) & (gd[f] > gd_min)
        free = finite & (pres[f] <= 0.0)
        ok = at_yield | free
        if at_yield.sum() < 50:
            return None
        Vp = arr["volume"][f]
        gsafe = np.where(gd[f] > 0.0, gd[f], 1.0)
        flow = 2.0 * D[f] / gsafe[:, None, None]
        I = inertial_number(gd[f], pres[f], d_grain, rho_s)  # noqa: E741 (the inertial number)
        phi = fe.phi(np.where(np.isfinite(I), I, 1.0))       # clamped off support
        w = np.where(at_yield, Vp * pres[f], 0.0)
        Vsig = (w[:, None] * phi)[:, :, None, None] * flow[:, None, :, :]
        Vsig_known = -w[:, None, None] * eye
        I_acc.append(I[at_yield])
        # the rollout's weight: dissipation V p |gd|
        W_acc.append((Vp * pres[f] * gd[f])[at_yield])
        return Vsig, Vsig_known, ok, gd[f]

    ke = 0.5 * np.einsum("p,fpi->f", arr["mass"], arr["v"] ** 2)
    k_peak = int(np.argmax(ke))
    frames: list[int] = []
    ke_frac_used = None
    for frac in (0.1, 0.2, 0.5):
        decayed = np.flatnonzero(ke <= frac * ke[k_peak])
        k_start = int(decayed[decayed > k_peak][0]) if np.any(decayed > k_peak) else k_peak
        all_frames = [f for f in range(0, arr["x"].shape[0], frame_stride) if f >= k_start]
        counts = [int(((gd[f] > gd_min) & (pres[f] > 0.0)).sum()) for f in all_frames]
        frames = suite._longest_run(all_frames, counts, 50)
        if len(frames) >= window_frames:
            ke_frac_used = frac
            break
    if len(frames) < window_frames:
        return {"refused": True, "reason": (
            f"only {len(frames)} contiguous shearing frames, fewer than the "
            f"{window_frames}-frame window")}

    t0 = time.time()
    sysm = assemble_columns_timeweak(
        arr["x"], arr["v"], arr["mass"], arr["g"], arr["frame_dt"] * frame_stride,
        arr["n_grid"], arr["grid_lim"], columns_fn, n_columns=K, frames=frames,
        window_frames=window_frames,
        collider_planes=suite.wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells, valid_frac_min=yield_frac_min)
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating}

    I_realized = np.concatenate(I_acc)
    weights = np.concatenate(W_acc)
    finite_I = np.isfinite(I_realized)
    I_realized, weights = I_realized[finite_I], weights[finite_I]
    q = np.percentile(I_realized, [1, 5, 25, 50, 75, 95, 99])
    I_grid = np.logspace(np.log10(max(q[1], I_lo)), np.log10(min(q[5], I_hi)), 80)

    mu_true = suite.friction_to_mu(float(arr["meta"].law_params["friction_angle"]))
    I_clamped = np.clip(I_realized, I_lo, I_hi)
    Phi = fe.phi(I_grid)
    # The QP constrains the whole tabulated range: the return map reads the
    # table anywhere in [1e-4, 1] during its bisection and takes the first
    # entry as the static yield threshold, so a curve that is only admissible
    # where the data lives is not a law the engine can run.
    con_grid = np.logspace(np.log10(I_lo), np.log10(I_hi), 80)
    gram_grid = np.logspace(np.log10(I_lo), np.log10(I_hi), 257)
    qp_cfg = {"dictionary": fe, "lam": 1.0e-6, "mu_min": 0.05,
              "grid": con_grid,
              "G": fe.gram((gram_grid, np.ones(gram_grid.size)))}

    def one_variant(qp: dict | None) -> dict:
        res = regularized_solve(sysm.A, sysm.b, prior, lambda th: fe.phi(I_grid) @ th,
                                groups=np.asarray(sysm.node_frame), qp=qp, log=log)
        theta = np.asarray(res["theta"], float)
        cov = np.asarray(res["theta_cov"], float)
        mu_hat = Phi @ theta
        mu_sd = np.sqrt(np.clip(np.einsum("ik,kl,il->i", Phi, cov, Phi), 0.0, None))
        mu_realized = fe.phi(I_clamped) @ theta
        errs = curve_errors(mu_hat, np.full_like(I_grid, mu_true), I_clamped,
                            mu_realized, np.full_like(I_clamped, mu_true))
        errs["relL2_dissipation_weighted"] = float(
            np.sqrt(np.sum(weights * (mu_realized - mu_true) ** 2) / np.sum(weights))
            / mu_true)
        mu_med = float((fe.phi(np.array([np.median(I_realized)])) @ theta)[0])
        mu_q = fe.phi(np.clip(q, I_lo, I_hi)) @ theta
        mu_table = fe.phi(10.0 ** np.linspace(-4.0, 0.0, 256)) @ theta
        res.update({
            "curve": {"I": I_grid.tolist(), "mu_hat": mu_hat.tolist(),
                      "mu_sd": mu_sd.tolist(), "mu_true": [mu_true] * I_grid.size},
            "curve_error": errs,
            "mu_at_median_I": mu_med,
            "mu_at_I_quantiles": {k: float(v) for k, v in zip(
                ["p1", "p5", "p25", "p50", "p75", "p95", "p99"], mu_q, strict=True)},
            "friction_angle_at_median_I": suite.mu_to_friction(mu_med),
            "mu_static_table_value": float(mu_table[0]),
            "table_min_increment": float(np.min(np.diff(mu_table))),
            "monotone_on_table": bool(np.min(np.diff(mu_table)) >= -1.0e-4),
            "baked_table": bake_mu_table(fe, theta),
        })
        log(f"[sand FE {'qp' if qp else 'ridge'}] mu(median I="
            f"{np.median(I_realized):.3e}) = {mu_med:.4f} (truth {mu_true:.4f}), "
            f"curve relL2 grid={errs['relL2_on_grid']:.3f} "
            f"dissipation-weighted={errs['relL2_dissipation_weighted']:.3f}, "
            f"table mu(1e-4)={mu_table[0]:.3f}, monotone="
            f"{res['monotone_on_table']}")
        return res

    variants = {"ridge": one_variant(None), "qp": one_variant(qp_cfg)}
    out = {
        "refused": False, "mode": "F", "wall_seconds": time.time() - t0,
        "headline": "qp",
        "headline_reason": (
            "the monotone, mu >= 0.05 constrained solve is the headline because "
            "the fork's tabulated mu(I) return map bisects a residual that is "
            "monotone only when mu(I) is non-decreasing; an unconstrained curve "
            "is not a law the engine can integrate, and the unregularized fit "
            "is not even nonnegative"),
        "variants": variants,
        "K": K, "dictionary": FE_GRANULAR.name,
        "dictionary_support": [float(I_lo), float(I_hi)],
        "prior": "family prior (theta_mean, theta_cov)" if prior else "ridge",
        "n_rows": variants["qp"]["n_rows"],
        "n_rows_before_gating": sysm.n_rows_before_gating,
        "row_survival": sysm.row_survival,
        "n_shearing_frames": len(frames), "ke_frac_used": ke_frac_used,
        "window_frames": window_frames, "frame_stride": frame_stride,
        "yield_frac_min": yield_frac_min, "gd_min": gd_min,
        "eps_gamma": eps_gamma,
        "shear_rate_coverage": list(sysm.strain_coverage),
        "mu_true_constant": mu_true,
        "I_quantiles": {k: float(v) for k, v in
                        zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], q, strict=True)},
        "I_fraction_outside_support": float(
            ((I_realized < I_lo) | (I_realized > I_hi)).mean()),
        "I_hist_edges": np.logspace(-4, 0, 41).tolist(),
        "I_hist_counts": np.histogram(I_clamped, bins=np.logspace(-4, 0, 41))[0].tolist(),
        "unregularized_reference": {
            "theta": variants["ridge"]["reg_sweep"][0]["theta"],
            "curve_min": variants["ridge"]["reg_sweep"][0]["curve_min"],
            "curve_max": variants["ridge"]["reg_sweep"][0]["curve_max"],
            "residual_rel": variants["ridge"]["reg_sweep"][0]["residual_rel"],
            "baked_table": bake_mu_table(
                fe, np.asarray(variants["ridge"]["reg_sweep"][0]["theta"], float)),
        },
    }
    # the headline variant's fields are lifted to the top level so the report and
    # results json read the same keys for every material
    for key in ("theta", "theta_sd", "theta_cov", "reg_weight", "reg_rule",
                "reg_sweep", "cv_score", "residual_rel",
                "residual_rel_unregularized", "cond_AtA", "effective_rank",
                "solver", "n_cv_folds", "curve", "curve_error", "mu_at_median_I",
                "mu_at_I_quantiles", "friction_angle_at_median_I",
                "mu_static_table_value",
                "monotone_on_table", "baked_table"):
        out[key] = variants[out["headline"]][key]
    return out


# ---------------------------------------------------------------------------
# jelly: one-invariant hyperelastic basis
# ---------------------------------------------------------------------------

def identify_hyperelastic_fe(arr: dict, window_frames: int = 26,
                             frame_stride: int = 2, margin_cells: float = 3.0,
                             log=print) -> dict:
    """W'(I1bar) through the trained one-invariant basis, plus one volumetric column.

    Same grid-consistent time-weak assembly the suite's elastic leg uses, with
    the two fixed-corotated columns replaced by K deviatoric basis columns and
    one volumetric column:

        sigma_k   = (2 / J) phi_k(I1bar - 3) dev(bbar),   k = 1..K
        sigma_vol = (J - 1) I

    so tau_dev = 2 W1 dev(bbar) with W1 = sum_k theta_k phi_k, exactly the
    recover_fe reading in sim/hyperelastic.py, and theta_vol is the bulk
    modulus. Two references matter and are both reported. The truth is fixed
    corotated, which is NOT in this basis's family (neo-Hookean, Yeoh, Gent), so
    the small-strain equivalents are the reference targets: W1 -> mu / 2 and
    theta_vol -> lam + 2 mu / 3, the latter because the corotated deviatoric
    column carries pressure of its own while the basis's deviatoric columns
    carry none.
    """
    from ident.weakform.elastic_grid import _particle_validity, assemble_columns_timeweak

    if "F" not in arr:
        return {"refused": True, "reason": "dump carries no deformation gradient"}
    phi_fn, x_grid, K = load_table_fe(FE_HYPER, "x_grid")
    F, vol0 = arr["F"], arr["vol0"]
    x_acc: list[np.ndarray] = []

    def columns_fn(f: int):
        ok_p, hencky = _particle_validity(F[f], arr["x"][f], arr["v"][f], None)
        if ok_p.sum() < 20:
            return None
        Fs = np.where(ok_p[:, None, None], F[f], np.eye(3)[None])
        J = np.linalg.det(Fs)
        bmat = Fs @ np.transpose(Fs, (0, 2, 1))
        bbar = np.power(np.clip(J, 1e-6, None), -2.0 / 3.0)[:, None, None] * bmat
        I1bar = bbar[:, 0, 0] + bbar[:, 1, 1] + bbar[:, 2, 2]
        eye = np.eye(3)[None]
        devb = bbar - (I1bar / 3.0)[:, None, None] * eye
        phi = phi_fn(I1bar - 3.0)                            # (P, K), clamped
        x_acc.append((I1bar - 3.0)[ok_p])
        sig = np.concatenate([
            (2.0 / J)[:, None, None, None] * phi[:, :, None, None] * devb[:, None, :, :],
            (J - 1.0)[:, None, None, None] * eye[:, None, :, :]], axis=1)
        return (J * vol0)[:, None, None, None] * sig, None, ok_p, hencky

    frames = list(range(0, arr["x"].shape[0], frame_stride))
    t0 = time.time()
    sysm = assemble_columns_timeweak(
        arr["x"], arr["v"], arr["mass"], arr["g"], arr["frame_dt"] * frame_stride,
        arr["n_grid"], arr["grid_lim"], columns_fn, n_columns=K + 1, frames=frames,
        window_frames=window_frames,
        collider_planes=suite.wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells)
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating}

    lp = arr["meta"].law_params
    E_t, nu_t = float(lp["E"]), float(lp["nu"])
    mu_t = E_t / (2.0 * (1.0 + nu_t))
    lam_t = E_t * nu_t / ((1.0 + nu_t) * (1.0 - 2.0 * nu_t))
    W1_ref = 0.5 * mu_t
    bulk_ref = lam_t + 2.0 * mu_t / 3.0

    x_realized = np.concatenate(x_acc)
    x_realized = x_realized[np.isfinite(x_realized)]
    q = np.percentile(x_realized, [1, 5, 25, 50, 75, 95, 99])
    x_grid_rep = np.linspace(0.0, float(min(q[5], x_grid[-1])), 80)
    res = regularized_solve(sysm.A, sysm.b, None,
                            lambda th: phi_fn(x_grid_rep) @ th[:K],
                            groups=np.asarray(sysm.node_frame), log=log)
    theta = np.asarray(res["theta"], float)
    cov = np.asarray(res["theta_cov"], float)[:K, :K]
    Phi = phi_fn(x_grid_rep)
    W1 = Phi @ theta[:K]
    W1_sd = np.sqrt(np.clip(np.einsum("ik,kl,il->i", Phi, cov, Phi), 0.0, None))
    x_clamped = np.clip(x_realized, x_grid[0], x_grid[-1])
    W1_realized = phi_fn(x_clamped) @ theta[:K]
    errs = curve_errors(W1, np.full_like(x_grid_rep, W1_ref), x_clamped,
                        W1_realized, np.full_like(x_clamped, W1_ref))
    bulk_hat = float(theta[K])
    W1_med = float((phi_fn(np.array([np.median(x_realized)])) @ theta[:K])[0])
    res.update({
        "refused": False, "mode": "F", "wall_seconds": time.time() - t0,
        "K": K, "dictionary": FE_HYPER.name,
        "dictionary_support": [float(x_grid[0]), float(x_grid[-1])],
        "prior": "ridge (this table ships no family prior)",
        "n_rows_before_gating": sysm.n_rows_before_gating,
        "row_survival": sysm.row_survival,
        "strain_coverage": list(sysm.strain_coverage),
        "window_frames": window_frames, "frame_stride": frame_stride,
        "x_quantiles": {k: float(v) for k, v in
                        zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], q, strict=True)},
        "x_fraction_outside_support": float(
            ((x_realized < x_grid[0]) | (x_realized > x_grid[-1])).mean()),
        "curve": {"x": x_grid_rep.tolist(), "W1_hat": W1.tolist(),
                  "W1_sd": W1_sd.tolist(), "W1_true": [W1_ref] * x_grid_rep.size},
        "curve_error": errs,
        "W1_at_median_strain": W1_med,
        "W1_reference_small_strain": W1_ref,
        "shear_modulus_from_curve": 2.0 * W1_med,
        "shear_modulus_truth": mu_t,
        "shear_modulus_rel_err": abs(2.0 * W1_med / mu_t - 1.0),
        "bulk_coefficient": bulk_hat,
        "bulk_reference_lam_plus_2mu_3": bulk_ref,
        "bulk_rel_err": abs(bulk_hat / bulk_ref - 1.0),
        "x_hist_edges": np.linspace(0.0, float(x_grid[-1]), 41).tolist(),
        "x_hist_counts": np.histogram(x_clamped,
                                      bins=np.linspace(0.0, float(x_grid[-1]), 41))[0].tolist(),
    })
    log(f"[jelly FE] W1(median x={np.median(x_realized):.3e}) = {W1_med:.4e} "
        f"(reference mu/2 = {W1_ref:.4e}, shear modulus error "
        f"{100 * res['shear_modulus_rel_err']:.1f} percent), bulk = {bulk_hat:.4e} "
        f"(reference {bulk_ref:.4e}, {100 * res['bulk_rel_err']:.1f} percent)")
    return res


# ---------------------------------------------------------------------------
# plasticine and water: the viscous surrogate
# ---------------------------------------------------------------------------

def identify_viscous_fe(arr: dict, window_frames: int = 16, frame_stride: int = 2,
                        margin_cells: float = 3.0, eps_gamma: float = 0.02,
                        gd_min: float = 0.1, valid_frac_min: float = 0.97,
                        log=print) -> dict:
    """eta_app(gamma_dot) through the trained viscous basis, pressure as data.

    sigma = -p I + 2 eta_app(|gamma_dot|) D with the pressure read from the 3D
    stress trace, so the columns are V phi_k(log10 |gamma_dot|) 2 D and the
    pressure term is a known load. The truth of both materials this leg is run
    on is NOT viscous (an elastoplastic solid and an inviscid EOS fluid), so the
    recovered curve is a surrogate by construction; the report scores it by
    rollout only.
    """
    from common.conventions import equivalent_shear_rate, pressure_from_cauchy_3d_trace, sym
    from ident.weakform.elastic_grid import assemble_columns_timeweak

    if not arr["meta"].has_pressure:
        return {"refused": True, "reason": "no oracle pressure in the dump"}
    phi_fn, s_grid, K = load_table_fe(FE_VISCOUS, "s_grid")
    D = sym(arr["L"])
    gd = equivalent_shear_rate(D, eps_gamma)
    pres = pressure_from_cauchy_3d_trace(arr["stress"])
    eye = np.eye(3)[None]
    gd_acc: list[np.ndarray] = []

    def columns_fn(f: int):
        finite = np.isfinite(pres[f]) & np.isfinite(D[f]).all(axis=(1, 2))
        ok = finite & (gd[f] > gd_min)
        if ok.sum() < 50:
            return None
        Vp = arr["volume"][f]
        phi = phi_fn(np.log10(np.maximum(gd[f], 1e-12)))
        w = np.where(ok, Vp, 0.0)
        Vsig = (w[:, None] * phi)[:, :, None, None] * (2.0 * D[f])[:, None, :, :]
        Vsig_known = -(w * pres[f])[:, None, None] * eye
        gd_acc.append(gd[f][ok])
        return Vsig, Vsig_known, ok, gd[f]

    frames = list(range(0, arr["x"].shape[0], frame_stride))
    t0 = time.time()
    sysm = assemble_columns_timeweak(
        arr["x"], arr["v"], arr["mass"], arr["g"], arr["frame_dt"] * frame_stride,
        arr["n_grid"], arr["grid_lim"], columns_fn, n_columns=K, frames=frames,
        window_frames=window_frames,
        collider_planes=suite.wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells, valid_frac_min=valid_frac_min)
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating}

    gd_realized = np.concatenate(gd_acc)
    q = np.percentile(gd_realized, [1, 5, 25, 50, 75, 95, 99])
    s_lo = max(np.log10(max(q[1], 1e-12)), s_grid[0])
    s_hi = min(np.log10(q[5]), s_grid[-1])
    gd_rep = np.logspace(s_lo, s_hi, 80)
    res = regularized_solve(sysm.A, sysm.b, None,
                            lambda th: phi_fn(np.log10(gd_rep)) @ th,
                            groups=np.asarray(sysm.node_frame), log=log)
    theta = np.asarray(res["theta"], float)
    cov = np.asarray(res["theta_cov"], float)
    Phi = phi_fn(np.log10(gd_rep))
    eta = Phi @ theta
    eta_sd = np.sqrt(np.clip(np.einsum("ik,kl,il->i", Phi, cov, Phi), 0.0, None))
    s_clamped = np.clip(np.log10(np.maximum(gd_realized, 1e-12)), s_grid[0], s_grid[-1])
    eta_realized = phi_fn(s_clamped) @ theta
    eta_med = float((phi_fn(np.array([np.median(s_clamped)])) @ theta)[0])
    res.update({
        "refused": False, "mode": "F", "wall_seconds": time.time() - t0,
        "K": K, "dictionary": FE_VISCOUS.name,
        "dictionary_support": [float(10.0 ** s_grid[0]), float(10.0 ** s_grid[-1])],
        "prior": "ridge (this table ships no family prior)",
        "n_rows_before_gating": sysm.n_rows_before_gating,
        "row_survival": sysm.row_survival,
        "window_frames": window_frames, "frame_stride": frame_stride,
        "gd_quantiles": {k: float(v) for k, v in
                         zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], q, strict=True)},
        "gd_fraction_outside_support": float(
            ((gd_realized < 10.0 ** s_grid[0])
             | (gd_realized > 10.0 ** s_grid[-1])).mean()),
        "curve": {"gd": gd_rep.tolist(), "eta_hat": eta.tolist(),
                  "eta_sd": eta_sd.tolist()},
        "eta_at_median_rate": eta_med,
        "eta_min_on_support": float(eta.min()),
        "eta_max_on_support": float(eta.max()),
        "eta_negative_on_support": bool(eta.min() < 0.0),
        "eta_negative_fraction_on_support": float((eta < 0.0).mean()),
        "eta_realized_median": float(np.median(eta_realized)),
        "gd_hist_edges": np.logspace(s_grid[0], s_grid[-1], 41).tolist(),
        "gd_hist_counts": np.histogram(
            10.0 ** s_clamped, bins=np.logspace(s_grid[0], s_grid[-1], 41))[0].tolist(),
        "truth_is_not_viscous": True,
    })
    log(f"[viscous FE] eta_app(median rate {np.median(gd_realized):.3g} 1/s) = "
        f"{eta_med:.4g} Pa s, range on support "
        f"[{res['eta_min_on_support']:.4g}, {res['eta_max_on_support']:.4g}]")
    return res


# ---------------------------------------------------------------------------
# baking a recovered curve into the engine's table
# ---------------------------------------------------------------------------

def bake_mu_table(fe, theta: np.ndarray) -> dict:
    """The recovered mu(I) on the engine's 256-point log10 I grid in [-4, 0].

    The fork's tabulated mu(I) return map (material 13) reads the table with a
    clamped linear interpolation on s = log10 I over [smin, smax] and uses the
    FIRST entry as the static yield threshold, so the grid here is exactly the
    dump-schema grid (common.conventions LOG10_I_TABLE_MIN/MAX, MU_TABLE_POINTS)
    and the ends are the curve's own ends. The bake clips negative samples to
    zero, reports the count, and the caller refuses a rollout whose clip is
    material.
    """
    from common.conventions import LOG10_I_TABLE_MAX, LOG10_I_TABLE_MIN, MU_TABLE_POINTS
    s = np.linspace(LOG10_I_TABLE_MIN, LOG10_I_TABLE_MAX, MU_TABLE_POINTS)
    mu = fe.phi(10.0 ** s) @ np.asarray(theta, float)
    n_neg = int((mu < 0.0).sum())
    return {"table": np.clip(mu, 0.0, None).tolist(),
            "smin": float(LOG10_I_TABLE_MIN), "smax": float(LOG10_I_TABLE_MAX),
            "n_points": int(MU_TABLE_POINTS),
            "n_negative_clipped": n_neg,
            "mu_min": float(mu.min()), "mu_max": float(mu.max())}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def truth_dump(material: str, shape: str, vel: str) -> Path:
    return suite.dump_path(material, shape, "truth", vel, N_GRID)


def stage_identify(material: str, force: bool = False, log=print) -> dict:
    """FE identification plus the known-form row on the same dump, both timed."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"identify_{material}.json"
    if path.exists() and not force:
        log(f"[identify] reuse {path.name} (exists)")
        return json.loads(path.read_text())
    dump = truth_dump(material, "cube", "preset")
    if not dump.exists():
        raise SystemExit(f"missing {dump}; generate the grid-{N_GRID} truth first")
    arr = suite._load_arrays(dump)
    res: dict = {"material": material, "dump": dump.name, "n_grid": arr["n_grid"],
                 "n_particles": int(arr["x"].shape[1]),
                 "n_frames": int(arr["x"].shape[0])}

    t0 = time.time()
    if material == "sand":
        fe, prior = load_granular_fe()
        res["fe"] = identify_friction_fe(arr, fe, prior, log=log)
    elif material == "jelly":
        res["fe"] = identify_hyperelastic_fe(arr, log=log)
    else:
        res["fe"] = identify_viscous_fe(
            arr, window_frames=VISCOUS_WINDOW[material], log=log)
    res["fe"]["wall_seconds_total"] = time.time() - t0

    # the known-form row, on the SAME dump, timed the same way
    t0 = time.time()
    known: dict = {}
    if material == "sand":
        k = suite.identify_friction(arr, window_frames=suite.WINDOW_FRAMES["sand"], log=log)
        known = {"leg": "identify_friction (Mode C, one column)",
                 "mu_c": k.get("mu_c"), "friction_angle": k.get("friction_angle"),
                 "truth_friction_angle": float(arr["meta"].law_params["friction_angle"]),
                 "mu_true": suite.friction_to_mu(
                     float(arr["meta"].law_params["friction_angle"])),
                 "n_rows": k.get("n_rows"), "cond_AtA": k.get("cond_AtA"),
                 "residual_rel": k.get("residual_rel"), "refused": k.get("refused")}
        if known["mu_c"]:
            known["mu_rel_err"] = abs(known["mu_c"] / known["mu_true"] - 1.0)
    elif material == "jelly":
        k = suite.identify_elastic(arr, window_frames=26, log=log)
        lp = arr["meta"].law_params
        known = {"leg": "identify_elastic (two corotated columns)",
                 "mu": k.get("mu"), "lam": k.get("lam"), "E": k.get("E"), "nu": k.get("nu"),
                 "truth_E": float(lp["E"]), "truth_nu": float(lp["nu"]),
                 "n_rows": k.get("n_rows"), "cond_AtA": k.get("cond_AtA"),
                 "residual_rel": k.get("residual_rel"), "refused": k.get("refused")}
        if known["E"]:
            known["E_rel_err"] = abs(known["E"] / float(lp["E"]) - 1.0)
    elif material == "water":
        k = suite.identify_eos(arr, window_frames=suite.WINDOW_FRAMES["water"], log=log)
        bulk_t = suite.bulk_from_E_nu(suite.MATERIALS["water"]["truth"]["E"],
                                      suite.MATERIALS["water"]["truth"]["nu"])
        known = {"leg": "identify_eos (one volumetric column)",
                 "bulk_modulus": k.get("bulk_modulus"), "truth_bulk_modulus": bulk_t,
                 "n_rows": k.get("n_rows"), "cond_AtA": k.get("cond_AtA"),
                 "residual_rel": k.get("residual_rel"), "refused": k.get("refused")}
        if k.get("bulk_modulus"):
            known["bulk_rel_err"] = abs(k["bulk_modulus"] / bulk_t - 1.0)
    elif material == "plasticine":
        k = suite.identify_elastic(arr, window_frames=26, log=log)
        y = (suite.identify_yield(arr, k["mu"], log=log)
             if not k.get("refused", True) else {"refused": True})
        lp = arr["meta"].law_params
        known = {"leg": "identify_elastic + identify_yield",
                 "mu": k.get("mu"), "lam": k.get("lam"), "E": k.get("E"), "nu": k.get("nu"),
                 "yield_stress": y.get("yield_stress"),
                 "truth_E": float(lp["E"]), "truth_nu": float(lp["nu"]),
                 "truth_yield_stress": float(lp["yield_stress"]),
                 "n_rows": k.get("n_rows"), "cond_AtA": k.get("cond_AtA"),
                 "refused": k.get("refused") or y.get("refused")}
        if known["E"]:
            known["E_rel_err"] = abs(known["E"] / float(lp["E"]) - 1.0)
    known["wall_seconds"] = time.time() - t0
    res["known_form"] = known

    path.write_text(json.dumps(res, indent=2, default=float))
    log(f"[identify] wrote {path}")
    return res


def _rollout(material_key: str, shape: str, vel: str, theta: dict, tag: str,
             truth: Path, force: bool, log=print) -> dict:
    """One warp rollout seeded from the truth dump's own frame-0 cloud.

    Seeding from the dump (suite.cloud_from_dump) makes the rollout differ from
    the truth in the constitutive law alone: same particles, reference volumes,
    initial velocities, grid, horizon and time step.
    """
    DUMPS.mkdir(parents=True, exist_ok=True)
    pred = DUMPS / f"{tag}.npz"
    cloud = suite.cloud_from_dump(truth)
    if not pred.exists() or force:
        suite.run_scene(material_key, shape, pred, theta=theta, vel=vel,
                        cloud=cloud, log=log)
    score = suite.nclaw_position_mse(truth, pred, strict=False)
    score.pop("per_frame", None)
    score["dump"] = pred.name
    score["truth_dump"] = truth.name
    score["n_particles_truth"] = int(cloud["pts"].shape[0])
    score["particle_counts_match"] = bool(score["n_particles"] == cloud["pts"].shape[0])
    score.update(_inbox_score(truth, pred))
    # The dump writer truncates a non-finite rollout, so the metric scores only
    # surviving frames and can flatter a divergent law. Record divergence; the
    # partial score is not comparable.
    n_expected = int(np.load(truth)["x"].shape[0])
    score["n_frames_expected"] = n_expected
    score["diverged"] = bool(score["n_frames"] < n_expected)
    if score["diverged"]:
        score["reason"] = (
            f"the rollout went non-finite and was truncated at frame "
            f"{score['n_frames']} of {n_expected}; the position MSE below covers "
            "only the frames that exist and is not comparable to a converged leg")
        log(f"[rollout] {tag}: DIVERGED at frame {score['n_frames']} of {n_expected}")
        return score
    log(f"[rollout] {tag}: MSE={score['mse']:.4e} (in-box "
        f"{score['mse_inbox']:.4e}) final={score['mse_final_frame']:.4e} "
        f"RMS={score['rmse_mm']:.3f} mm over {score['n_frames']} frames, "
        f"{score['n_particles']} particles")
    return score


def _inbox_score(truth: Path, pred: Path, grid_lim: float = 1.0,
                 frame_step: int = 5, pad: float = 0.01) -> dict:
    """The same metric over the particles the TRUTH keeps inside the box.

    The truth seeds 106 of 6711 blub particles outside the unit domain, 64 of
    them far outside the domain, where the fork's clamp, not the physics, moves
    them, and they dominate the mean. This variant drops them; it equals the
    NCLaw metric when nothing escapes.
    """
    tx = np.load(truth)["x"]
    rx = np.load(pred)["x"]
    nf = min(tx.shape[0], rx.shape[0])
    n = min(tx.shape[1], rx.shape[1])
    tx, rx = tx[:nf, :n], rx[:nf, :n]
    keep = ~(((tx < -pad) | (tx > grid_lim + pad)).any(-1)).any(0)
    diff = (tx[:, keep] - rx[:, keep]) / grid_lim
    per_frame = (diff ** 2).mean(axis=(1, 2))
    return {"mse_inbox": float(per_frame[::frame_step].mean()),
            "mse_inbox_final_frame": float(per_frame[-1]),
            "rmse_inbox_mm": float(np.sqrt((diff ** 2).sum(-1).mean()) * 1e3),
            "n_particles_inbox": int(keep.sum()),
            "n_particles_escaped_in_truth": int((~keep).sum())}


def diffsim_leg(material: str) -> list[tuple[str, str, dict]]:
    """The gradient-through-simulator method's rollout leg, when its fit exists.

    Reads the best-init theta (already in warp's own engine arguments) from
    out/diffsim_baseline/results_<material>.json. Absent or unfinished fits
    contribute no leg; the report marks the cell pending instead.
    """
    path = DIFFSIM_OUT / f"results_{material}.json"
    if not path.exists():
        return []
    rec = json.loads(path.read_text())
    ds = rec.get("diffsim") or {}
    theta = ds.get("theta_engine")
    if not theta:
        return []
    mat_key = {"water": "water"}.get(material, material)
    legs = [("diffsim", mat_key, dict(theta))]
    refine = DIFFSIM_OUT / f"refine_{material}.json"
    if refine.exists():
        best = (json.loads(refine.read_text()) or {}).get("best_theta") or {}
        if best:
            th = dict(best)
            # the refine record stores Lame coefficients; the engine takes (E, nu)
            if "mu" in th and "lam" in th:
                mu, lam = th.pop("mu"), th.pop("lam")
                th["E"] = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
                th["nu"] = lam / (2.0 * (lam + mu))
            legs.append(("diffsim_refined", mat_key, th))
    return legs


def leg_specs(material: str, ident: dict) -> tuple[list[tuple[str, str, dict]], dict]:
    legs, notes = _leg_specs_base(material, ident)
    legs = list(legs) + diffsim_leg(material)
    return legs, notes


def _leg_specs_base(material: str, ident: dict) -> tuple[list[tuple[str, str, dict]], dict]:
    """(leg, engine material, theta) for every rollout of one material, plus notes.

    Legs, in the order the report prints them:
      fe                 the headline FE curve (monotone constrained, for sand)
      fe_ridge           the same rows solved without the constraints, when its
                         table is a law at all; the ablation that measures what
                         the constraints contribute
      known_form         the suite's own identified scalar, same dump
      truth_theta        the truth law, same engine: the same-engine replay
                         error at the truth law
      flat_table_floor   a CONSTANT table at the truth mu through the tabulated
                         material: the error of the tabulated return map at the
                         truth mu
    """
    notes: dict = {}
    if material == "sand":
        fe = ident["fe"]
        baked = fe.get("baked_table")
        if baked is None:
            return [], {"status": "FE identification refused; nothing to roll out"}
        mu_true = ident["known_form"]["mu_true"]
        legs = [("fe", "sand_table",
                 {"eta_table": baked["table"], "eta_table_smin": baked["smin"],
                  "eta_table_smax": baked["smax"]})]
        ridge = fe["variants"]["ridge"]["baked_table"]
        if ridge["n_negative_clipped"] == 0:
            legs.append(("fe_ridge", "sand_table",
                         {"eta_table": ridge["table"], "eta_table_smin": ridge["smin"],
                          "eta_table_smax": ridge["smax"]}))
        else:
            notes["fe_ridge"] = {"refused": True, "reason": (
                f"{ridge['n_negative_clipped']} of {ridge['n_points']} table "
                "samples of the unconstrained fit are negative, so it is not a "
                "friction law and is not simulated")}
        un = fe["unregularized_reference"]["baked_table"]
        notes["fe_unregularized"] = {"refused": True, "reason": (
            f"the unregularized fit is non-physical: {un['n_negative_clipped']} of "
            f"{un['n_points']} table samples are negative (mu spans "
            f"{un['mu_min']:.3g} to {un['mu_max']:.3g}), so it is not simulated")}
        legs += [
            ("known_form", "sand",
             {"friction_angle": ident["known_form"]["friction_angle"]}),
            ("truth_theta", "sand",
             {"friction_angle": ident["known_form"]["truth_friction_angle"]}),
            ("flat_table_floor", "sand_table",
             {"eta_table": [mu_true] * baked["n_points"],
              "eta_table_smin": baked["smin"], "eta_table_smax": baked["smax"]}),
        ]
        return legs, notes

    kf_j = ident.get("known_form", {})
    if material == "jelly":
        # the FE leg is unsupported in the engine (no tabulated hyperelastic);
        # only the known-form and truth legs exist for this material
        if kf_j.get("refused", True):
            return [], {"status": "known-form identification refused"}
        return [("known_form", "jelly", {"E": kf_j["E"], "nu": kf_j["nu"]}),
                ("truth_theta", "jelly",
                 {"E": kf_j["truth_E"], "nu": kf_j["truth_nu"]})], notes

    fe = ident["fe"]
    if fe.get("refused") or "curve" not in fe:
        return [], {"status": "FE identification refused; nothing to roll out"}
    eta_tab, smin, smax = bake_eta_table(fe)
    truth = suite.MATERIALS[material]["truth"]
    bulk = suite.bulk_from_E_nu(truth["E"], truth["nu"])
    # suite._wave_speed sizes the time step from (E, nu) for every engine except
    # "fluid", and the tabulated viscous material is not that engine, so E is set
    # here to make the p-wave speed the FLUID's: lam + 2 mu = E f(nu) must equal
    # the bulk modulus the EOS carries. Left at the material default the
    # step would be sized from a slower wave than the run has.
    nu_v = float(suite.MATERIALS["visc_table"]["truth"]["nu"])
    f_nu = nu_v / ((1.0 + nu_v) * (1.0 - 2.0 * nu_v)) + 1.0 / (1.0 + nu_v)
    E_v = bulk / f_nu
    notes["viscous_surrogate_caveat"] = (
        "The truth here is not a viscous fluid (plasticine is an elastoplastic "
        "solid, water an inviscid EOS fluid), and the rollout material is a "
        "weakly compressible fluid whose bulk modulus is supplied as data from "
        "the truth. This row measures how far a viscous surrogate identified "
        "from the trajectory carries the rollout.")
    # a negative apparent viscosity is not a fluid, and the material factory
    # refuses one outright. The identified curve is therefore reported as
    # refused, and what gets simulated is that curve clipped at zero, named as
    # such: a different law from the one identified.
    leg = "fe"
    if fe.get("eta_negative_on_support"):
        leg = "fe_clipped"
        notes["fe"] = {"refused": True, "reason": (
            "the identified apparent viscosity is negative over "
            f"{100 * fe['eta_negative_fraction_on_support']:.0f} percent of the "
            f"realized rate support (minimum {fe['eta_min_on_support']:.3g} Pa s), "
            "which is not a fluid; the weak-form residual at the selected weight "
            f"is {fe['residual_rel']:.3f} and the held-out score is "
            f"{fe['cv_score']:.3f}, so these rows reject the viscous family. "
            "The simulated leg below is this curve clipped at zero, which is a "
            "different law.")}
    legs = [(leg, "visc_table",
             {"eta_table": eta_tab, "eta_table_smin": smin, "eta_table_smax": smax,
              "bulk_modulus": bulk, "E": E_v, "nu": nu_v})]
    # known-form and truth legs for the non-sand materials, so the comparison
    # table has all methods and the replay reference on every material, same
    # subprocess isolation and metric as the FE legs
    kf = ident.get("known_form", {})
    if material == "plasticine" and not kf.get("refused", True):
        legs += [("known_form", "plasticine",
                  {"E": kf["E"], "nu": kf["nu"], "yield_stress": kf["yield_stress"]}),
                 ("truth_theta", "plasticine",
                  {"E": kf["truth_E"], "nu": kf["truth_nu"],
                   "yield_stress": kf["truth_yield_stress"]})]
    elif material == "water" and not kf.get("refused", True):
        legs += [("known_form", "water", {"bulk_modulus": kf["bulk_modulus"]}),
                 ("truth_theta", "water", {"bulk_modulus": kf["truth_bulk_modulus"]})]
    return legs, notes


def stage_rollout(material: str, force: bool = False, isolate: bool = True,
                  log=print) -> dict:
    """Roll out every leg and score the NCLaw metric, one child process per leg.

    Each rollout runs in its own process. A law the engine cannot integrate
    leaves the grid, and the fork's out-of-range write is a SIGSEGV; isolation
    records an unstable leg without killing the run.
    """
    ipath = OUT / f"identify_{material}.json"
    if not ipath.exists():
        raise SystemExit(f"missing {ipath}; run the identify stage first")
    ident = json.loads(ipath.read_text())
    rpath = OUT / f"rollout_{material}.json"
    scores: dict = json.loads(rpath.read_text()) if rpath.exists() else {}

    if material in ROLLOUT_UNSUPPORTED:
        # the FE leg alone is unsupported; the known-form and truth legs
        # still run so the comparison table has this material's LS column
        scores["fe"] = {"refused": True, "reason": ROLLOUT_UNSUPPORTED[material]}
        log(f"[rollout] {material} fe leg: {ROLLOUT_UNSUPPORTED[material]}")

    legs, notes = leg_specs(material, ident)
    if material in ROLLOUT_UNSUPPORTED:
        legs = [(leg, mk, th) for leg, mk, th in legs
                if leg in ("known_form", "truth_theta", "diffsim", "diffsim_refined")]
    scores.update(notes)
    shape_h, vel_h = HELD_OUT[material]
    scenes = [("cube", "preset", "reconstruction"), (shape_h, vel_h, "generalization")]

    for leg, _mat_key, _theta in legs:
        for shape, vel, role in scenes:
            truth = truth_dump(material, shape, vel)
            if not truth.exists():
                log(f"[rollout] skip {leg}/{shape}: no truth dump {truth.name}")
                continue
            key = f"{leg}@{shape}"
            if key in scores and not force:
                log(f"[rollout] reuse {key}")
                continue
            legfile = OUT / "legs" / f"{material}_{leg}_{shape}_{vel}.json"
            if isolate:
                legfile.parent.mkdir(parents=True, exist_ok=True)
                if legfile.exists() and not force:
                    legfile.unlink()
                cmd = [sys.executable, "-m", "experiments.fe_ls", "rollout-one",
                       "--material", material, "--leg", leg, "--shape", shape,
                       "--vel", vel]
                if force:
                    cmd.append("--force")
                t0 = time.time()
                proc = subprocess.run(cmd, cwd=ENGINE, capture_output=True, text=True)
                wall = time.time() - t0
                if legfile.exists():
                    s = json.loads(legfile.read_text())
                else:
                    s = {"crashed": True, "returncode": proc.returncode,
                         "reason": (
                             f"the rollout process exited with code "
                             f"{proc.returncode} and wrote no score. Signal 11 "
                             "(returncode -11 or 139) is the fork writing "
                             "outside the grid after the law drove particles out "
                             "of the domain, which is how an unstable law fails "
                             "on this machine."),
                         "stderr_tail": proc.stderr[-1500:],
                         "stdout_tail": proc.stdout[-1500:]}
                    log(f"[rollout] {key}: CRASHED rc={proc.returncode}")
                s["wall_seconds"] = wall
            else:
                s = rollout_one(material, leg, shape, vel, force=force, log=log)
            s.update({"leg": leg, "role": role})
            scores[key] = s
            rpath.write_text(json.dumps(scores, indent=2, default=float))
    rpath.write_text(json.dumps(scores, indent=2, default=float))
    return scores


def rollout_one(material: str, leg: str, shape: str, vel: str,
                force: bool = False, log=print) -> dict:
    """One leg of one scene, in this process; the child of stage_rollout."""
    ident = json.loads((OUT / f"identify_{material}.json").read_text())
    legs, _ = leg_specs(material, ident)
    match = [spec for spec in legs if spec[0] == leg]
    if not match:
        raise SystemExit(f"no leg {leg!r} for {material}")
    _, mat_key, theta = match[0]
    truth = truth_dump(material, shape, vel)
    tag = f"{material}_{shape}_{vel}_g{N_GRID}_{leg}"
    s = _rollout(mat_key, shape, vel, theta, tag, truth, force, log=log)
    s["engine_material"] = mat_key
    out = OUT / "legs" / f"{material}_{leg}_{shape}_{vel}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2, default=float))
    return s


def bake_eta_table(fe_res: dict, n_points: int = 256) -> tuple[list, float, float]:
    """The recovered eta_app curve on the engine's uniform log10 rate grid.

    The tabulated viscous material clamps outside [smin, smax], so the grid is
    the trained basis's own support and negative viscosities are clipped to zero
    (the material factory refuses them outright, and a negative apparent
    viscosity is not a fluid).
    """
    phi_fn, s_grid, _ = load_table_fe(FE_VISCOUS, "s_grid")
    s = np.linspace(float(s_grid[0]), float(s_grid[-1]), n_points)
    eta = phi_fn(s) @ np.asarray(fe_res["theta"], float)
    return np.clip(eta, 0.0, None).tolist(), float(s_grid[0]), float(s_grid[-1])


# ---------------------------------------------------------------------------
# figures and report
# ---------------------------------------------------------------------------

def _figure(material: str, ident: dict) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fe = ident.get("fe", {})
    if fe.get("refused") or "curve" not in fe:
        return None
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [2.4, 1.0]})
    if material == "sand":
        x = np.asarray(fe["curve"]["I"])
        y = np.asarray(fe["curve"]["mu_hat"])
        sd = np.asarray(fe["curve"]["mu_sd"])
        yt = np.asarray(fe["curve"]["mu_true"])
        edges = np.asarray(fe["I_hist_edges"])
        counts = np.asarray(fe["I_hist_counts"])
        xlabel, ylabel = "inertial number I", r"$\mu$"
        tlabel = f"truth (Drucker-Prager cone, $\\mu$ = {yt[0]:.3f})"
        logx = True
    elif material == "jelly":
        x = np.asarray(fe["curve"]["x"])
        y = np.asarray(fe["curve"]["W1_hat"])
        sd = np.asarray(fe["curve"]["W1_sd"])
        yt = np.asarray(fe["curve"]["W1_true"])
        edges = np.asarray(fe["x_hist_edges"])
        counts = np.asarray(fe["x_hist_counts"])
        xlabel, ylabel = r"$\bar{I}_1 - 3$", r"$W'(\bar{I}_1)$  [Pa]"
        tlabel = f"corotated small-strain reference $\\mu/2$ = {yt[0]:.3g} Pa"
        logx = False
    else:
        x = np.asarray(fe["curve"]["gd"])
        y = np.asarray(fe["curve"]["eta_hat"])
        sd = np.asarray(fe["curve"]["eta_sd"])
        yt = None
        edges = np.asarray(fe["gd_hist_edges"])
        counts = np.asarray(fe["gd_hist_counts"])
        xlabel, ylabel = r"$|\dot\gamma|$  [1/s]", r"$\eta_{app}$  [Pa s]"
        tlabel = None
        logx = True

    ax[0].plot(x, y, lw=2.2, color="#1c7ed6", label="FE least squares")
    ax[0].fill_between(x, y - sd, y + sd, color="#1c7ed6", alpha=0.2,
                       label="posterior +/- 1 sd")
    if yt is not None:
        ax[0].plot(x, yt, lw=1.8, ls="--", color="#212529", label=tlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[0].set_title(f"(a) {material}: function-encoder curve on the realized support "
                    f"(K = {fe['K']}, {fe['n_rows']} rows)")
    if yt is not None:
        span = float(np.abs(yt[0]))
        ax[0].set_ylim(min(0.0, y.min() - 0.2 * span), max(y.max(), 2.0 * span))
    ylo, yhi = ax[0].get_ylim()
    if (y - sd).min() < ylo or (y + sd).max() > yhi:
        ax[0].text(0.02, 0.04, "the posterior band runs past both axis limits",
                   transform=ax[0].transAxes, fontsize=8, color="#1c7ed6")

    centres = 0.5 * (edges[:-1] + edges[1:])
    ax[1].bar(centres, counts, width=np.diff(edges), color="#adb5bd",
              align="center")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("particle-frames")
    ax[1].set_title("(b) realized support (clamped to the basis)", fontsize=9)
    ax[1].grid(alpha=0.3, axis="y")
    if logx:
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
    ax[0].set_xlim(x.min(), x.max())
    ax[1].set_xlim(x.min(), x.max())
    fig.tight_layout()
    path = OUT / f"fe_curve_{material}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _clipped_eta_summary(fe_res: dict) -> tuple[float, float]:
    """Median and maximum of the nonnegative table the viscous leg simulates."""
    tab = np.asarray(bake_eta_table(fe_res)[0], float)
    return float(np.median(tab)), float(tab.max())


def _mse_rows(material: str, scores: dict) -> list[tuple[str, str, dict]]:
    rows = []
    for key, val in scores.items():
        if not isinstance(val, dict) or "mse" not in val:
            continue
        rows.append((val.get("leg", key), val.get("role", ""), val))
    order = {"fe": 0, "fe_clipped": 0, "fe_ridge": 1, "known_form": 2,
             "truth_theta": 3, "flat_table_floor": 4}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1] != "reconstruction"))
    return rows


def stage_report(materials: list[str], log=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    all_res: dict = {}
    lines = ["# Least squares through a function-encoder basis: the NCLaw comparison row",
             "",
             "Third method of the morning table. Same grid-20 NCLaw-suite cube "
             "trajectories, same",
             "masks and same grid-consistent time-weak assembly as the known-form "
             "least-squares row;",
             "the constitutive form is replaced by a trained function-encoder basis, so the "
             "unknown is",
             "a FUNCTION and the solve stays convex and linear in theta. The simulator is "
             "never",
             "differentiated. The regularization weight is chosen per material and "
             "per variant by",
             "leave-one-window-out cross-validation of the weak-form rows, weights "
             f"within {CV_TIE_TOL} of",
             "the best score counting as tied and the largest of those taken; the "
             "whole sweep,",
             "cross-validation score included, is in the results json. Every rollout "
             "is seeded from",
             "the truth dump's own frame-0 cloud, so a rollout differs from its truth "
             "trajectory in",
             "the constitutive law alone.", "",
             "The posterior band on each curve is a variance statement about the "
             "rows that survived",
             "filtering. The curve errors below are dominated by bias, a misspecified "
             "family or a part",
             "of the support the trajectory never visited, which no posterior of "
             "this form can see;",
             "the two numbers answer different questions and the band is not an "
             "error bar on the",
             "truth.", ""]

    for material in materials:
        ipath = OUT / f"identify_{material}.json"
        if not ipath.exists():
            log(f"[report] skip {material}: no identify json")
            continue
        ident = json.loads(ipath.read_text())
        rpath = OUT / f"rollout_{material}.json"
        scores = json.loads(rpath.read_text()) if rpath.exists() else {}
        fig = _figure(material, ident)
        fe = ident.get("fe", {})
        results = {
            "schema_version": "fe-ls-baseline-1.0",
            "material": material,
            "git_rev_video2sim": _git_rev(STAGING_ROOT),
            "git_rev_mpm_engine": _git_rev(ENGINE),
            "n_grid": ident.get("n_grid"),
            "dump": ident.get("dump"),
            "n_particles": ident.get("n_particles"),
            "mode": "F",
            "pressure_source": "true_mpm_trace",
            "dictionary_mode": fe.get("dictionary"),
            "dictionary_K": fe.get("K"),
            "regularization_rule": fe.get("reg_rule"),
            "regularization_weight": fe.get("reg_weight"),
            "regularization_kind": fe.get("prior"),
            "row_count_before_gating": fe.get("n_rows_before_gating"),
            "row_count_after_gating": fe.get("n_rows"),
            "row_survival": fe.get("row_survival"),
            "condition_number": fe.get("cond_AtA"),
            "effective_rank": fe.get("effective_rank"),
            "residual_rel": fe.get("residual_rel"),
            "residual_rel_unregularized": fe.get("residual_rel_unregularized"),
            "theta_hat": fe.get("theta"),
            "theta_sd": fe.get("theta_sd"),
            "curve_error": fe.get("curve_error"),
            "identify_wall_seconds": fe.get("wall_seconds_total"),
            "known_form": ident.get("known_form"),
            "mse": {k: v for k, v in scores.items() if isinstance(v, dict) and "mse" in v},
            "rollout_notes": {k: v for k, v in scores.items()
                              if not (isinstance(v, dict) and "mse" in v)},
            "figure": None if fig is None else _rel_figure(fig),
            "reg_sweep": fe.get("reg_sweep"),
        }
        for key in ("I_quantiles", "x_quantiles", "gd_quantiles",
                    "I_fraction_outside_support", "x_fraction_outside_support",
                    "gd_fraction_outside_support", "mu_at_median_I",
                    "mu_at_I_quantiles", "mu_static_table_value",
                    "monotone_on_table", "solver", "cv_score",
                    "mu_true_constant", "friction_angle_at_median_I",
                    "W1_at_median_strain", "W1_reference_small_strain",
                    "shear_modulus_from_curve", "shear_modulus_truth",
                    "shear_modulus_rel_err", "bulk_coefficient",
                    "bulk_reference_lam_plus_2mu_3", "bulk_rel_err",
                    "eta_at_median_rate", "eta_min_on_support", "eta_max_on_support",
                    "baked_table"):
            if key in fe:
                results[key] = fe[key]
        (OUT / f"results_{material}.json").write_text(
            json.dumps(results, indent=2, default=float))
        all_res[material] = results

        lines += [f"## {material}", ""]
        if fe.get("refused"):
            lines += [f"FE identification REFUSED: {fe.get('reason')}", ""]
            continue
        lines += [
            f"- dictionary {fe['dictionary']}, K = {fe['K']}, "
            f"{fe['prior']}, weight {fe['reg_weight']:.3g}",
            f"- rows {fe['n_rows']} of {fe['n_rows_before_gating']} before filtering "
            f"({100 * fe['row_survival']:.2f} percent), cond(A^T A) "
            f"{fe['cond_AtA']:.3e}, residual {fe['residual_rel']:.4f} "
            f"(unregularized {fe['residual_rel_unregularized']:.4f})",
            f"- identify wall time {fe['wall_seconds_total']:.1f} s "
            f"(known-form leg on the same dump: "
            f"{ident['known_form'].get('wall_seconds', float('nan')):.1f} s)",
        ]
        ce = fe.get("curve_error", {})
        if material == "sand":
            lines += [
                f"- realized I: p5 {fe['I_quantiles']['p5']:.3e}, median "
                f"{fe['I_quantiles']['p50']:.3e}, p95 {fe['I_quantiles']['p95']:.3e}; "
                f"{100 * fe['I_fraction_outside_support']:.1f} percent of "
                "particle-frames outside the basis support (clamped)",
                f"- mu at the median realized I: {fe['mu_at_median_I']:.4f} "
                f"(truth {fe['mu_true_constant']:.4f}); curve relL2 "
                f"{ce['relL2_on_grid']:.3f} on the p5-p95 grid, "
                f"{ce['relL2_dissipation_weighted']:.3f} dissipation-weighted",
                f"- the recovered curve invents a rate dependence across the "
                f"sampled band: mu = {fe['mu_at_I_quantiles']['p5']:.3f} at the 5th "
                f"percentile of I, {fe['mu_at_I_quantiles']['p50']:.3f} at the "
                f"median, {fe['mu_at_I_quantiles']['p95']:.3f} at the 95th, and "
                f"{fe['mu_static_table_value']:.3f} at the bottom of the table, "
                "which is the static yield threshold the return map reads. The "
                "median value is close to the known form's single coefficient; "
                "the rollout error comes from the ends",
                f"- known form on the same dump: mu = "
                f"{ident['known_form']['mu_c']:.4f} "
                f"({100 * ident['known_form']['mu_rel_err']:.2f} percent), "
                f"phi = {ident['known_form']['friction_angle']:.2f} deg "
                f"(truth {ident['known_form']['truth_friction_angle']:.1f})",
            ]
        elif material == "jelly":
            lines += [
                f"- realized strain I1bar - 3: p5 {fe['x_quantiles']['p5']:.3e}, "
                f"median {fe['x_quantiles']['p50']:.3e}, p95 "
                f"{fe['x_quantiles']['p95']:.3e}",
                f"- W'(median strain) = {fe['W1_at_median_strain']:.4e} Pa against the "
                f"corotated small-strain reference mu/2 = "
                f"{fe['W1_reference_small_strain']:.4e} Pa: shear modulus "
                f"{fe['shear_modulus_from_curve']:.4e} vs {fe['shear_modulus_truth']:.4e} "
                f"({100 * fe['shear_modulus_rel_err']:.1f} percent)",
                f"- volumetric coefficient {fe['bulk_coefficient']:.4e} against "
                f"lam + 2 mu / 3 = {fe['bulk_reference_lam_plus_2mu_3']:.4e} "
                f"({100 * fe['bulk_rel_err']:.1f} percent)",
                f"- curve relL2 {ce['relL2_on_grid']:.3f} on the 0 to p95 grid, "
                f"{ce['relL2_realized_p5_p95']:.3f} over the p5-p95 realized strain",
                f"- known form on the same dump: E = {ident['known_form']['E']:.4e} "
                f"({100 * ident['known_form']['E_rel_err']:.2f} percent), nu = "
                f"{ident['known_form']['nu']:.4f}",
            ]
        else:
            lines += [
                f"- realized shear rate: p5 {fe['gd_quantiles']['p5']:.3g}, median "
                f"{fe['gd_quantiles']['p50']:.3g}, p95 {fe['gd_quantiles']['p95']:.3g} 1/s",
                f"- eta_app at the median rate {fe['eta_at_median_rate']:.4g} Pa s, "
                f"range [{fe['eta_min_on_support']:.4g}, {fe['eta_max_on_support']:.4g}]",
                "- the truth is not a viscous fluid, so there is no truth curve to "
                "compare against; this row is a surrogate, see the rollout note",
                f"- the simulated law, the curve clipped at zero: median "
                f"{_clipped_eta_summary(fe)[0]:.3g}, maximum "
                f"{_clipped_eta_summary(fe)[1]:.3g} Pa s over the tabulated rate "
                "range. Near zero the leg is the weakly compressible EOS fluid at "
                "the truth bulk modulus, which for water is the truth's own family",
            ]
        rows = _mse_rows(material, scores)
        if rows:
            lines += ["",
                      "| leg | scene | role | position MSE | MSE in-box | final frame "
                      "| frames |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for leg, role, val in rows:
                nf = f"{val['n_frames']} / {val.get('n_frames_expected', '')}"
                if val.get("diverged"):
                    lines += [f"| {leg} | {val['truth_dump']} | {role} | DIVERGED | "
                              f"DIVERGED | DIVERGED | {nf} |"]
                    continue
                lines += [f"| {leg} | {val['truth_dump']} | {role} | {val['mse']:.3e} | "
                          f"{val['mse_inbox']:.3e} | {val['mse_final_frame']:.3e} | "
                          f"{nf} |"]
            esc = max(int(v.get("n_particles_escaped_in_truth", 0)) for _, _, v in rows)
            if esc:
                lines += ["", f"In-box MSE excludes the {esc} particles the TRUTH "
                          "trajectory puts outside the unit domain (see "
                          "_inbox_score); the two columns agree wherever nothing "
                          "escapes. A DIVERGED leg went non-finite and was "
                          "truncated, and its partial score is not comparable."]
        for key, val in scores.items():
            if isinstance(val, dict) and "mse" not in val:
                lines += ["", f"{key}: {val.get('reason', val.get('status', val))}"]
            elif not isinstance(val, dict):
                lines += ["", f"{key}: {val}"]
        if fig is not None:
            lines += ["", f"![{material} curve]({fig.name})"]
        lines += [""]

    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    log(f"[report] wrote {OUT / 'report.md'}")
    return all_res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["identify", "rollout", "report", "all",
                                      "rollout-one"])
    ap.add_argument("--material", default="sand", choices=[*MATERIALS, "all"])
    ap.add_argument("--leg", default=None, help="rollout-one: which leg")
    ap.add_argument("--shape", default=None, help="rollout-one: which shape")
    ap.add_argument("--vel", default=None, help="rollout-one: which throw")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run rollouts in this process instead of one child each")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    log = _log_factory(OUT / "run.log")
    if a.stage == "rollout-one":
        if not (a.leg and a.shape and a.vel):
            raise SystemExit("rollout-one needs --leg, --shape and --vel")
        rollout_one(a.material, a.leg, a.shape, a.vel, force=a.force, log=log)
        return
    mats = list(MATERIALS) if a.material == "all" else [a.material]
    for m in mats:
        if a.stage in ("identify", "all"):
            stage_identify(m, force=a.force, log=log)
        if a.stage in ("rollout", "all"):
            stage_rollout(m, force=a.force, isolate=not a.no_isolate, log=log)
    if a.stage in ("report", "all"):
        stage_report(mats, log=log)


if __name__ == "__main__":
    main()

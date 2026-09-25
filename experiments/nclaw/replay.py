"""Elastic-state replay: identification of plastic laws from positions alone.

A plastic material's stored deformation gradient is the elastic part, because
the return map overwrites F every step. Positions can only give the total
deformation, and once the material flows the two diverge. Measured on NCLaw's
plasticine dataset throw: the stored elastic deviatoric strain caps at 0.021
while the total strain reaches 1.7 by the last frame. An estimator that reads
sigma(F_total) is reading a state the constitutive law never sees.

This module rebuilds the elastic state from the measured motion. Between
consecutive frames the local deformation increment is a moving least squares
fit over current-frame neighbours, which stays well conditioned at any
accumulated deformation (the reference-frame fit the tier dump carries does
not). The elastic gradient is then replayed through the material's own return
map,

    F_e[n+1] = project(F_incr[n] @ F_e[n]),

per-particle algebra on measured kinematics: no simulator, no momentum time
stepping, nothing differentiated. On the plasticine dataset the replayed
strain matches the stored elastic F through the flow phase.

The return maps mirror NCLaw's own (nclaw/material/preset.py), the same
formulas tests/test_composed_material.py verifies the engine against.

The unknown return-map parameter is then a LINEAR momentum fit at a fixed
replay: particles the projection clipped sit exactly on the yield surface, so
their deviatoric stress is the unknown parameter times a known direction,
while every unclipped particle's full stress follows from the elastic pair
and the replayed strain and enters as data. Because the replay itself needs a
candidate parameter, the estimate is iterated to self consistency: replay at
y, fit y', repeat until they agree. The elastic pair is either fit on the
pre-yield window, where total and elastic deformation still coincide
(plasticine), or assumed and stated (sand, whose throw never loads it
observably).
"""
from __future__ import annotations

import time

import numpy as np

SIG_FLOOR = 0.05                     # their clamp_min on singular values
_EYE = np.eye(3)[None]


# ---------------------------------------------------------------------------
# Measured deformation increments
# ---------------------------------------------------------------------------

def incremental_gradients(x: np.ndarray, k: int = 16, ridge: float = 1.0e-10
                          ) -> np.ndarray:
    """Per-frame-pair deformation increments from positions, (T-1, N, 3, 3).

    For each particle at frame n, the k nearest neighbours AT FRAME n define a
    local frame-to-frame map x[n] -> x[n+1]; the least squares gradient of that
    map is F_incr with F_total[n+1] = F_incr[n] @ F_total[n]. Current-frame
    neighbourhoods keep the fit local; the reference-frame MLS in the tier
    dump loses locality at large flow.
    """
    from scipy.spatial import cKDTree
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape[:2]
    out = np.empty((T - 1, N, 3, 3))
    for n in range(T - 1):
        _, idx = cKDTree(x[n]).query(x[n], k=k + 1)
        rel0 = x[n][idx[:, 1:]] - x[n][:, None, :]
        rel1 = x[n + 1][idx[:, 1:]] - x[n + 1][:, None, :]
        A = np.einsum("pki,pkj->pij", rel0, rel0) + ridge * np.eye(3)
        C = np.einsum("pki,pkj->pij", rel1, rel0)
        out[n] = C @ np.linalg.inv(A)
    return out


def grid_affine_increments(x: np.ndarray, dt: float, n_grid: int,
                           grid_lim: float, mass: np.ndarray,
                           iters: int = 2) -> np.ndarray:
    """Per-step APIC affine matrices C from positions, (T-1, N, 3, 3).

    This is the engine-kernel observer. Both engines advect particles by
    x[n+1] = x[n] + dt v[n+1], so the backward position difference IS the
    particle velocity. The affine matrix the F update consumes is then
    rebuilt with the engine's own quadratic B-spline transfers: scatter the
    velocities to the grid (P2G), read the APIC moment back (G2P), and
    iterate that fixed point so the scatter uses the current C estimate.
    F[n+1] = (I + dt C[n]) F[n] with C[n] this function's entry n, which
    pairs the increment with the step it advanced, x[n] -> x[n+1].

    Observation density sets the accuracy: the grid field is recoverable
    only where several particles fall in a cell. At one particle per cell
    the per-step relative error is 9 to 16 percent and the compounded
    replay fails the momentum fit's residual check.
    """
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape[:2]
    dx = grid_lim / n_grid
    inv_d = 4.0 / dx ** 2
    offs = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]

    def kernel(xp):
        base = np.floor(xp / dx - 0.5).astype(int)
        fx = xp / dx - base
        w = np.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2,
                      0.5 * (fx - 0.5) ** 2], 0)
        return base, w

    out = np.empty((T - 1, N, 3, 3))
    for n in range(T - 1):
        vp = (x[n + 1] - x[n]) / dt
        base, w = kernel(x[n])
        stencil = []
        for (i, j, k) in offs:
            wijk = w[i, :, 0] * w[j, :, 1] * w[k, :, 2]
            gi = np.clip(base + [i, j, k], 0, n_grid - 1)
            stencil.append((wijk, gi, gi * dx - x[n]))
        C = np.zeros((N, 3, 3))
        for _ in range(iters):
            mom = np.zeros((n_grid, n_grid, n_grid, 3))
            mg = np.zeros((n_grid,) * 3)
            for wijk, gi, dpos in stencil:
                val = mass[:, None] * wijk[:, None] * (
                    vp + np.einsum("pab,pb->pa", C, dpos))
                np.add.at(mom, (gi[:, 0], gi[:, 1], gi[:, 2]), val)
                np.add.at(mg, (gi[:, 0], gi[:, 1], gi[:, 2]), mass * wijk)
            gv = mom / np.maximum(mg, 1e-12)[..., None]
            C = np.zeros((N, 3, 3))
            for wijk, gi, dpos in stencil:
                C += inv_d * wijk[:, None, None] * np.einsum(
                    "pa,pb->pab", gv[gi[:, 0], gi[:, 1], gi[:, 2]], dpos)
        out[n] = C
    return out


# ---------------------------------------------------------------------------
# Return maps, numpy mirrors of NCLaw's preset.py
# ---------------------------------------------------------------------------

def _mu_lam(E: float, nu: float) -> tuple[float, float]:
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def _svd_log(F: np.ndarray):
    U, sig, Vh = np.linalg.svd(F)
    eps = np.log(np.clip(sig, SIG_FLOOR, None))
    return U, eps, Vh


def _recompose(U: np.ndarray, eps: np.ndarray, Vh: np.ndarray) -> np.ndarray:
    return np.einsum("pij,pj,pjk->pik", U, np.exp(eps), Vh)


def project_von_mises(F: np.ndarray, E: float, nu: float, sigma_y: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """VonMisesPlasticity.forward. Returns (projected F, clipped mask)."""
    mu, _ = _mu_lam(E, nu)
    U, eps, Vh = _svd_log(F)
    hat = eps - eps.mean(-1, keepdims=True)
    norm = np.linalg.norm(hat, axis=-1)
    dgamma = norm - sigma_y / (2.0 * mu)
    clipped = dgamma > 0.0
    eps = np.where(clipped[:, None],
                   eps - (dgamma / np.maximum(norm, 1e-30))[:, None] * hat, eps)
    return _recompose(U, eps, Vh), clipped


def project_drucker_prager(F: np.ndarray, E: float, nu: float,
                           friction_angle: float, cohesion: float = 0.0
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DruckerPragerPlasticity.forward. Returns (F, on_cone mask, tension mask)."""
    import math
    mu, lam = _mu_lam(E, nu)
    s = math.sin(math.radians(friction_angle))
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)
    U, eps, Vh = _svd_log(F)
    tr = eps.sum(-1)
    hat = eps - (tr / 3.0)[:, None]
    norm = np.linalg.norm(hat, axis=-1)
    shifted = tr - 3.0 * cohesion
    tension = shifted >= 0.0
    dgamma = norm + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * shifted * alpha
    on_cone = (~tension) & (dgamma > 0.0)
    compress = eps - (np.clip(dgamma, 0.0, None)
                      / np.maximum(norm, 1e-30))[:, None] * hat
    eps = np.where(tension[:, None], np.full_like(eps, cohesion),
                   np.where(on_cone[:, None], compress, eps))
    return _recompose(U, eps, Vh), on_cone, tension


def replay_elastic(Fincr: np.ndarray, project) -> tuple[np.ndarray, np.ndarray]:
    """F_e[n+1] = project(F_incr[n] @ F_e[n]) from identity, plus flow flags.

    ``project`` maps a batch of gradients to (projected batch, mask, ...); the
    first mask names the particles the projection moved that step, which is
    the at-yield set of the momentum fit. Shapes: (T, N, 3, 3) and
    (T, N) with frame 0 all-identity and no flow.
    """
    T1, N = Fincr.shape[:2]
    Fe = np.empty((T1 + 1, N, 3, 3))
    flow = np.zeros((T1 + 1, N), dtype=bool)
    Fe[0] = np.eye(3)
    cur = Fe[0]
    for n in range(T1):
        res = project(Fincr[n] @ cur)
        cur = res[0]
        Fe[n + 1] = cur
        flow[n + 1] = res[1]
    return Fe, flow


def hencky_cauchy(Fe: np.ndarray, E: float, nu: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cauchy stress of SigmaElasticity, split for the linear fit.

    Returns (mean_stress, dev_direction N_hat, dev_norm_kirchhoff, J) with
    Cauchy sigma = mean_stress I + dev_norm_kirchhoff * N_hat / J, so a
    particle the return map holds at the yield surface has dev_norm_kirchhoff
    equal to the yield parameter times a known factor and N_hat known.
    """
    mu, lam = _mu_lam(E, nu)
    U, eps, _ = _svd_log(Fe)
    tr = eps.sum(-1)
    J = np.exp(tr)
    hat = eps - (tr / 3.0)[..., None]
    norm = np.linalg.norm(hat, axis=-1)
    N = np.einsum("...ij,...j,...kj->...ik", U,
                  hat / np.maximum(norm, 1e-30)[..., None], U)
    mean = (2.0 * mu / 3.0 + lam) * tr / J
    return mean, N, 2.0 * mu * norm, J


# ---------------------------------------------------------------------------
# One linear momentum fit at a fixed replay
# ---------------------------------------------------------------------------

def _fit_scale_on_flow_set(arr: dict, Fe: np.ndarray, flow: np.ndarray,
                           E: float, nu: float, column_norm: np.ndarray,
                           window_frames: int, frame_stride: int,
                           margin_cells: float, valid_frac_min: float,
                           log=print) -> dict:
    """Fit the one coefficient scaling the flowing particles' deviatoric stress.

    On the flow set the unknown column is V * column_norm * N_hat / J; every
    other particle's full stress follows from the elastic pair at the replayed
    strain and is data. column_norm carries the law-specific known factor, so
    the solved coefficient IS the physical parameter (yield stress in Pa for
    von Mises, the cone coefficient alpha for Drucker-Prager).
    """
    from experiments.nclaw.suite import _longest_run, wall_planes
    from ident.weakform.elastic_grid import assemble_columns_timeweak, solve_elastic_grid

    mean, N, dev_norm, J = hencky_cauchy(Fe, E, nu)
    vol = arr["vol0"][None, :] * J                     # current particle volume

    def columns_fn(f: int):
        finite = np.isfinite(N[f]).all(axis=(1, 2)) & np.isfinite(mean[f])
        at_yield = finite & flow[f]
        if int(at_yield.sum()) < 50:
            return None
        Vp = vol[f]
        w = np.where(at_yield, Vp * column_norm[f] / np.maximum(J[f], 1e-30), 0.0)
        Vsig = w[:, None, None, None] * N[f][:, None, :, :]
        known = (Vp * mean[f])[:, None, None] * _EYE
        elastic_dev = np.where(at_yield, 0.0, Vp * dev_norm[f] / np.maximum(J[f], 1e-30))
        known = known + elastic_dev[:, None, None] * N[f]
        return Vsig, known, finite, dev_norm[f]

    all_frames = list(range(0, arr["x"].shape[0], frame_stride))
    counts = [int(flow[f].sum()) for f in all_frames]
    frames = _longest_run(all_frames, counts, 50)
    if len(frames) < window_frames:
        return {"refused": True, "n_rows": 0, "n_rows_before_gating": 0,
                "reason": (f"only {len(frames)} contiguous frames with 50 flowing "
                           f"particles, fewer than the {window_frames}-frame window"),
                "wall_assemble_s": 0.0, "wall_fit_s": 0.0}
    t_asm = time.time()
    sysm = assemble_columns_timeweak(
        arr["x"], arr["v"], arr["mass"], arr["g"],
        arr["frame_dt"] * frame_stride, arr["n_grid"], arr["grid_lim"],
        columns_fn, n_columns=1, frames=frames, window_frames=window_frames,
        collider_planes=wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells, valid_frac_min=valid_frac_min)
    wall_assemble_s = time.time() - t_asm
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": 0.0}
    t_fit = time.time()
    out = solve_elastic_grid(sysm)
    out["wall_assemble_s"], out["wall_fit_s"] = wall_assemble_s, time.time() - t_fit
    out.update({"refused": False, "n_flowing_frames": len(frames),
                "row_survival": sysm.row_survival,
                "n_rows_before_gating": sysm.n_rows_before_gating})
    return out


# ---------------------------------------------------------------------------
# Self-consistent estimators
# ---------------------------------------------------------------------------

def identify_yield_replay(arr: dict, mu_hat: float, lam_hat: float,
                          y0: float = 1.0e4, max_iter: int = 8,
                          rel_tol: float = 0.01, mls_k: int = 16,
                          window_frames: int = 26, frame_stride: int = 2,
                          margin_cells: float = 3.0, valid_frac_min: float = 0.9,
                          residual_bar: float = 0.15, log=print) -> dict:
    """von Mises yield stress from positions alone, elastic pair given.

    Replay at a candidate yield stress, fit the yield stress the momentum rows
    prefer at that replay, and iterate to self consistency. On the clipped set
    the deviatoric Kirchhoff norm equals the yield stress, so column_norm is 1
    and the solved coefficient is the yield stress itself.
    """
    E = float(mu_hat * (3.0 * lam_hat + 2.0 * mu_hat) / (lam_hat + mu_hat))
    nu = float(lam_hat / (2.0 * (lam_hat + mu_hat)))
    t_recon = time.time()
    Fincr = incremental_gradients(arr["x"], k=mls_k)
    wall_reconstruct_s = time.time() - t_recon
    wall_assemble_s = wall_fit_s = 0.0
    y, path = float(y0), []
    out: dict = {}

    def _timed(d: dict) -> dict:
        return {**d, "wall_reconstruct_s": wall_reconstruct_s,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": wall_fit_s}

    for it in range(max_iter):
        t_recon2 = time.time()
        Fe, flow = replay_elastic(
            Fincr, lambda F: project_von_mises(F, E, nu, y))
        wall_reconstruct_s += time.time() - t_recon2
        ones = np.ones(arr["x"].shape[:2])
        out = _fit_scale_on_flow_set(
            arr, Fe, flow, E, nu, ones, window_frames, frame_stride,
            margin_cells, valid_frac_min, log=log)
        wall_assemble_s += out.get("wall_assemble_s", 0.0)
        wall_fit_s += out.get("wall_fit_s", 0.0)
        if out.get("refused"):
            out.update(_timed({"estimator": "replay_yield", "iterations": path}))
            return out
        y_new = float(out["theta"][0])
        path.append({"candidate": y, "fit": y_new,
                     "residual_rel": float(out["residual_rel"])})
        log(f"[replay] yield iter {it}: candidate {y:.4e} -> fit {y_new:.4e} "
            f"resid {float(out['residual_rel']):.3f}")
        if not np.isfinite(y_new) or y_new <= 0.0:
            out.update(_timed({"refused": True, "estimator": "replay_yield",
                        "iterations": path,
                        "reason": f"non-physical yield fit {y_new!r}"}))
            return out
        done = abs(y_new - y) <= rel_tol * y
        y = y_new
        if done:
            break
    out.update(_timed({"yield_stress": y, "estimator": "replay_yield",
                "iterations": path, "mls_k": mls_k,
                "residual_bar": residual_bar,
                "elastic_pair_used": {"E": E, "nu": nu}}))
    if float(out["residual_rel"]) > residual_bar:
        out.update({"refused": True,
                    "reason": (f"self-consistent fit y={y:.4e} at relative "
                               f"residual {float(out['residual_rel']):.3f} against "
                               f"a bar of {residual_bar}")})
    log(f"[replay] yield={y:.4e} resid={float(out['residual_rel']):.3f} "
        f"refused={out['refused']}")
    return out


def identify_friction_replay(arr: dict, E: float, nu: float,
                             phi0: float = 30.0, max_iter: int = 8,
                             rel_tol: float = 0.01, mls_k: int = 16,
                             window_frames: int = 10, frame_stride: int = 2,
                             margin_cells: float = 3.0, valid_frac_min: float = 0.9,
                             residual_bar: float = 0.15, log=print) -> dict:
    """Drucker-Prager friction angle from positions alone, elastic pair ASSUMED.

    On the cone the return map sets ||dev eps|| = -alpha (3 lam + 2 mu) /
    (2 mu) tr(eps), so the deviatoric Kirchhoff norm is alpha times the known
    factor -(3 lam + 2 mu) tr(eps): column_norm carries that factor and the
    solved coefficient is alpha, converted back to a friction angle. Tension
    particles are stress free by the same map and enter as zero-stress data.
    """
    import math
    mu, lam = _mu_lam(E, nu)
    t_recon = time.time()
    Fincr = incremental_gradients(arr["x"], k=mls_k)
    wall_reconstruct_s = time.time() - t_recon
    wall_assemble_s = wall_fit_s = 0.0
    phi, path = float(phi0), []
    out: dict = {}

    def _timed(d: dict) -> dict:
        return {**d, "wall_reconstruct_s": wall_reconstruct_s,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": wall_fit_s}

    def alpha_of(p: float) -> float:
        s = math.sin(math.radians(p))
        return math.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)

    def phi_of(a: float) -> float:
        s = 3.0 * a / (2.0 * math.sqrt(2.0 / 3.0) + a)
        return math.degrees(math.asin(min(max(s, 0.0), 0.999)))

    for it in range(max_iter):
        t_recon2 = time.time()
        Fe, on_cone = replay_elastic(
            Fincr, lambda F: project_drucker_prager(F, E, nu, phi))
        wall_reconstruct_s += time.time() - t_recon2
        tr = np.log(np.clip(np.linalg.svd(Fe, compute_uv=False),
                            SIG_FLOOR, None)).sum(-1)
        column_norm = -(3.0 * lam + 2.0 * mu) * tr
        out = _fit_scale_on_flow_set(
            arr, Fe, on_cone, E, nu, column_norm, window_frames, frame_stride,
            margin_cells, valid_frac_min, log=log)
        wall_assemble_s += out.get("wall_assemble_s", 0.0)
        wall_fit_s += out.get("wall_fit_s", 0.0)
        if out.get("refused"):
            out.update(_timed({"estimator": "replay_friction", "iterations": path}))
            return out
        a_new = float(out["theta"][0])
        if not np.isfinite(a_new) or a_new <= 0.0:
            out.update(_timed({"refused": True, "estimator": "replay_friction",
                        "iterations": path,
                        "reason": f"non-physical cone coefficient {a_new!r}"}))
            return out
        phi_new = phi_of(a_new)
        path.append({"candidate_deg": phi, "fit_deg": phi_new,
                     "alpha_fit": a_new,
                     "residual_rel": float(out["residual_rel"])})
        log(f"[replay] friction iter {it}: candidate {phi:.2f} -> fit "
            f"{phi_new:.2f} deg resid {float(out['residual_rel']):.3f}")
        done = abs(phi_new - phi) <= rel_tol * max(phi, 1.0)
        phi = phi_new
        if done:
            break
    out.update(_timed({"friction_angle": phi, "alpha": alpha_of(phi),
                "estimator": "replay_friction", "iterations": path,
                "mls_k": mls_k, "residual_bar": residual_bar,
                "elastic_pair_assumed": {"E": E, "nu": nu}}))
    if float(out["residual_rel"]) > residual_bar:
        out.update({"refused": True,
                    "reason": (f"self-consistent fit phi={phi:.2f} deg at relative "
                               f"residual {float(out['residual_rel']):.3f} against "
                               f"a bar of {residual_bar}")})
    log(f"[replay] friction={phi:.2f} deg resid={float(out['residual_rel']):.3f} "
        f"refused={out['refused']}")
    return out


# ---------------------------------------------------------------------------
# Pre-yield window for the elastic pair
# ---------------------------------------------------------------------------

def pre_yield_frames(arr: dict, strain_bar: float = 0.03,
                     min_frames: int = 12) -> list[int]:
    """Frames where the p95 total deviatoric strain is still below the bar.

    Before yield the total deformation IS the elastic deformation, so the
    tier's own reference-frame MLS F is valid there and the standard elastic
    assembly applies. The default bar is a generic upper bound on the elastic
    strain of a stiff plastic solid; callers with a recovered yield refine it
    to 0.8 times the measured cap and refit.
    """
    from experiments.nclaw.suite import _hencky_dev_norm
    T = arr["F"].shape[0]
    p95 = np.array([np.percentile(_hencky_dev_norm(arr["F"][f]), 95)
                    for f in range(T)])
    over = np.flatnonzero(p95 > strain_bar)
    end = int(over[0]) if over.size else T
    return list(range(0, max(end, min_frames)))

"""Closed-form weak-form viscosity identification from a recorded pour.

Primary channel (the one used on ep0001): the TIME-WEAK brink lubrication balance --
see the block comment above brink_forcing. The measured receiver curve V(t), the cup
pose chain, and the caliper rim geometry close

    V(t) - V(t0) = (1/eta) INT F dt',   F = (rho g / 3) sin(a(t)) INT h(y,t)^3 dy

which is linear in 1/eta and solved as one robust convex least squares. No simulator
is run, differentiated, or iterated anywhere in the fit, and no derivative of data
is ever taken.

Also here, gated by a manufactured solution (--selftest): the slender viscous jet
weak form. Physics (Trouton; world z ascending, flow downward, speed u > 0):
    mass       A_t + d(Au)/ds = 0
    momentum   rho (u_t + u u_s) = rho g gamma - sigma k_s + 3 eta (A u_s)_s / A
multiplied by A phi(z) and integrated with phi -> 0 at both ends, so the viscous and
capillary terms integrate by parts and no third derivative touches data; per frame
the joint (Q^2, eta*Q) solve is one 2-parameter least squares. The real-data jet and
spout-film extraction was removed from pour_perception.py -- ep0001's optics defeat
it (through-wall refraction distorts the jet widths, the film has no silhouette; the
extractors live in git history, 40ce313) -- so the estimator remains here purely as
the selftest target: the recovery gate shows the method is sound where the fields
are observable.

Run:
  python experiments/pour/pour_weakform_identify.py               # brink fit on ep0001
  python experiments/pour/pour_weakform_identify.py --t-fit 2.6 3.6
  python experiments/pour/pour_weakform_identify.py --selftest    # manufactured solution

Outputs (out/pour_wf/<episode>/):
  identify.json      eta_hat, statistical CI, the fill-tolerance systematic
                     (eta_v0_range), V0 scan, prefix sweep, diagnostics
  identify.png       fit, forcing, residuals, prefix sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "out" / "pour_wf"

RHO = 1260.0                       # glycerol
G = 9.81
SIGMA = 0.063                      # N/m at 20 C (literature constant, not fitted)
MIN_BIN_PX = 4                     # jet bins thinner than this are unmeasured
MOUND_FACTOR = 1.6                 # width blow-up marking the pool/impact mound
SMOOTH_ZBINS = 7                   # Savitzky-Golay window along z (odd)
SMOOTH_TFRAMES = 5                 # temporal smoothing window (odd)


# --------------------------------------------------------------------------------------
# profile conditioning
# --------------------------------------------------------------------------------------
@dataclass
class Frames:
    """Conditioned jet kinematics on the common z grid (F, Z)."""
    t: np.ndarray            # (F,) seconds on the pour clock
    z: np.ndarray            # (Z,) world heights, ascending
    A: np.ndarray            # (F, Z) cross-section area, NaN outside the jet window
    R: np.ndarray            # (F, Z) radius
    gamma: np.ndarray        # (F, Z) cos(centerline, vertical)
    valid: np.ndarray        # (F, Z) bool
    z_lip: np.ndarray        # (F,)
    z_pool: np.ndarray       # (F,) top of the impact mound (bins below are cut)


def _savgol_nan(y: np.ndarray, window: int, deriv: int = 0, dx: float = 1.0):
    """Savitzky-Golay on a NaN-padded 1D array: fit only where a full window of
    finite samples exists; NaN elsewhere."""
    from scipy.signal import savgol_filter

    out = np.full_like(y, np.nan)
    finite = np.isfinite(y)
    # longest finite runs
    idx = np.where(finite)[0]
    if len(idx) < window:
        return out
    splits = np.where(np.diff(idx) > 1)[0]
    for run in np.split(idx, splits + 1):
        if len(run) >= window:
            out[run] = savgol_filter(y[run], window, polyorder=2, deriv=deriv,
                                     delta=dx)
    return out


def condition(obs: dict, t_window: tuple[float, float],
              block_s: float = 0.25) -> Frames:
    """Select active-jet frames in the time window, average them in `block_s` blocks
    (the jet is quasi-steady at that scale and profile noise attenuates the viscous
    column quadratically -- errors-in-variables), and condition the profiles."""
    t = obs["t"]
    zg = obs["jet_z"]
    dz = float(zg[1] - zg[0])
    sel = (t >= t_window[0]) & (t <= t_window[1]) & (obs["n_jet_px"] > 60)
    if sel.sum() < 5:
        raise SystemExit(f"only {sel.sum()} usable jet frames in window {t_window}")
    w = obs["jet_width"][sel].copy()
    chi = obs["jet_chi"][sel].copy()
    npx = obs["jet_npx"][sel]
    lip = obs["lip"][sel]
    ts = t[sel]
    w[npx < MIN_BIN_PX] = np.nan
    chi[npx < MIN_BIN_PX] = np.nan

    if block_s > 0:
        edges = np.arange(ts[0], ts[-1] + block_s, block_s)
        bw, bchi, blip, bt = [], [], [], []
        for a, bnd in itertools.pairwise(edges):
            m = (ts >= a) & (ts < bnd)
            if m.sum() < max(3, int(0.4 * block_s * 60)):
                continue
            cov = np.mean(np.isfinite(w[m]), axis=0)
            wm = np.nanmean(np.where(np.isfinite(w[m]), w[m], np.nan), axis=0)
            cm = np.nanmean(np.where(np.isfinite(chi[m]), chi[m], np.nan), axis=0)
            wm[cov < 0.6] = np.nan
            cm[cov < 0.6] = np.nan
            bw.append(wm)
            bchi.append(cm)
            blip.append(lip[m].mean(axis=0))
            bt.append(0.5 * (a + bnd))
        if len(bw) < 2:
            raise SystemExit("block averaging starved; widen the time window")
        w, chi = np.asarray(bw), np.asarray(bchi)
        lip, ts = np.asarray(blip), np.asarray(bt)

    F, Z = w.shape
    z_pool = np.full(F, -np.inf)
    for f in range(F):
        good = np.isfinite(w[f])
        if good.sum() < 6:
            w[f] = np.nan
            continue
        ref = np.nanmedian(w[f][good])
        # walk up from the bottom: cut the mound (width blow-up at the pool)
        kbot = np.argmax(good)
        k = kbot
        while k < Z and (not good[k] or w[f, k] > MOUND_FACTOR * ref):
            z_pool[f] = zg[k]
            k += 1
        w[f, :k] = np.nan
        chi[f, :k] = np.nan
    # smooth along z; derivatives via the same fits
    R = 0.5 * _savgol_nan_2d(w, SMOOTH_ZBINS, dx=dz)
    dchi = _savgol_nan_2d(chi, SMOOTH_ZBINS, deriv=1, dx=dz)
    gamma = 1.0 / np.sqrt(1.0 + dchi**2)
    A = np.pi * R**2
    valid = np.isfinite(A) & np.isfinite(gamma) & (R > 0.001)
    A[~valid] = np.nan
    return Frames(t=ts, z=zg, A=A, R=R, gamma=np.where(valid, gamma, np.nan),
                  valid=valid, z_lip=lip[:, 2], z_pool=z_pool)


def _savgol_nan_2d(y: np.ndarray, window: int, deriv: int = 0, dx: float = 1.0):
    return np.stack([_savgol_nan(row, window, deriv, dx) for row in y])


def curvature(fr: Frames, f: int, dz: float) -> np.ndarray:
    """Slender mean curvature k = 1/R - R_ss (R_ss ~ gamma^2 R_zz)."""
    Rzz = _savgol_nan(fr.R[f], SMOOTH_ZBINS, deriv=2, dx=dz)
    return 1.0 / fr.R[f] - fr.gamma[f] ** 2 * Rzz


# --------------------------------------------------------------------------------------
# weak-form assembly: joint (flux, viscosity), closed form
# --------------------------------------------------------------------------------------
def bump_tests(z: np.ndarray, zlo: float, zhi: float, n: int = 5):
    """Compactly supported C1 bumps phi_j spanning [zlo, zhi]; phi and phi_z on z."""
    centers = np.linspace(zlo, zhi, n + 2)[1:-1]
    half = (zhi - zlo) / (n + 1)
    phis = []
    for c in centers:
        x = np.clip((z - c) / half, -1.0, 1.0)
        phi = (1.0 - x**2) ** 2
        dphi = -4.0 * x * (1.0 - x**2) / half
        outside = (z < zlo) | (z > zhi)
        phi[outside] = 0.0
        dphi[outside] = 0.0
        phis.append((phi, dphi))
    return phis


def frame_rows(fr: Frames, f: int, window: tuple[float, float], n_tests: int,
               u_t_row=None):
    """Weak-form integrals for one frame, factored so the flux never needs a prior:
    with steady mass conservation u = Q/A, each test function gives the row

        -rho*I1_j * x  - CT_j * y  =  rho*g*I2_j - CAP_j - UT_j
        x = Q^2,  y = eta*Q,  TAU_j (Bingham column) multiplies tau_y

    I1_j = INT (1/A)_z phi dz            (inertia shape)
    I2_j = INT A phi dz                  (gravity)
    CAP_j = sigma INT k (A_z phi + A phi_z) dz
    CT_j = -3 INT A gamma (1/A)_z phi_z dz          (viscous shape; times eta*Q)
    TAU_j = sqrt(3) INT A phi_z dz
    UT_j = INT rho A u_t phi / gamma dz  (unsteady; from a previous pass, else 0)

    Returns dict of arrays over tests, or None if the window is starved."""
    dz = float(fr.z[1] - fr.z[0])
    valid_z = np.where(fr.valid[f], fr.z, np.nan)
    zlo = max(fr.z_pool[f] + window[0], np.nanmin(valid_z))
    zhi = min(fr.z_lip[f] - window[1], np.nanmax(valid_z))
    if not np.isfinite(zlo) or zhi - zlo < 8 * dz:
        return None
    A, gam = fr.A[f], fr.gamma[f]
    Ainv_z = _savgol_nan(1.0 / A, SMOOTH_ZBINS, deriv=1, dx=dz)
    A_z = _savgol_nan(A, SMOOTH_ZBINS, deriv=1, dx=dz)
    kap = curvature(fr, f, dz)
    ok = (fr.valid[f] & np.isfinite(Ainv_z) & np.isfinite(kap)
          & (fr.z >= zlo) & (fr.z <= zhi))
    if u_t_row is not None:
        ok &= np.isfinite(u_t_row)
    if ok.sum() < 8:
        return None
    out = {k: [] for k in ("I1", "I2", "CAP", "CT", "TAU", "UT")}
    for phi, dphi in bump_tests(fr.z, zlo, zhi, n_tests):
        m = ok & (phi > 0)
        if m.sum() < 6:
            continue
        out["I1"].append(np.sum(Ainv_z[m] * phi[m]) * dz)
        out["I2"].append(np.sum(A[m] * phi[m]) * dz)
        out["CAP"].append(SIGMA * np.sum(kap[m] * (A_z[m] * phi[m]
                                                   + A[m] * dphi[m])) * dz)
        out["CT"].append(-3.0 * np.sum(A[m] * gam[m] * Ainv_z[m] * dphi[m]) * dz)
        out["TAU"].append(np.sqrt(3.0) * np.sum(A[m] * dphi[m]) * dz)
        out["UT"].append(0.0 if u_t_row is None
                         else RHO * np.sum(A[m] * u_t_row[m] * phi[m] / gam[m]) * dz)
    if not out["I1"]:
        return None
    return {k: np.asarray(v) for k, v in out.items()}


def solve_frame(rows) -> tuple[float, float]:
    """Per-frame convex solve for (x, y) = (Q^2, eta*Q). Returns (Q, eta)."""
    Xm = np.stack([-RHO * rows["I1"], -rows["CT"]], axis=1)
    rhs = RHO * G * rows["I2"] - rows["CAP"] - rows["UT"]
    th, *_ = np.linalg.lstsq(Xm, rhs, rcond=None)
    x, y = float(th[0]), float(th[1])
    if x <= 0:
        return np.nan, np.nan
    Q = np.sqrt(x)
    return Q, y / Q


def solve_all(fr: Frames, window: tuple[float, float], n_tests: int,
              passes: int = 2):
    """Two closed-form passes: (1) per-frame steady (Q, eta); (2) same with the
    unsteady term from pass-1 velocity fields. Then a robust GLOBAL eta over all
    rows at the per-frame fluxes, plus the (eta, tau_y) Bingham fit and covariance.
    """
    from scipy.ndimage import uniform_filter1d

    Fn, _Z = fr.A.shape
    dt = float(np.median(np.diff(fr.t))) if Fn > 1 else 1.0 / 60.0
    Q = np.full(Fn, np.nan)
    eta_f = np.full(Fn, np.nan)
    U_t = [None] * Fn
    for p in range(passes):
        for f in range(Fn):
            rows = frame_rows(fr, f, window, n_tests, u_t_row=U_t[f])
            if rows is None or len(rows["I1"]) < 2:
                continue
            Q[f], eta_f[f] = solve_frame(rows)
        if p + 1 < passes:
            U = np.where(np.isfinite(Q[:, None]), Q[:, None] / fr.A, np.nan)
            bad = ~np.isfinite(U)
            Uf = np.where(bad, 0.0, U)
            wgt = uniform_filter1d((~bad).astype(float), SMOOTH_TFRAMES, axis=0)
            Us = uniform_filter1d(Uf, SMOOTH_TFRAMES, axis=0) / np.maximum(wgt, 1e-9)
            Us[wgt < 0.5] = np.nan
            Ut = np.gradient(Us, axis=0) / dt
            U_t = [Ut[f] for f in range(Fn)]

    # global eta at fixed per-frame flux: rows b = eta * c over all frames
    bs, cs, taus, fidx = [], [], [], []
    for f in range(Fn):
        if not np.isfinite(Q[f]):
            continue
        rows = frame_rows(fr, f, window, n_tests, u_t_row=U_t[f])
        if rows is None:
            continue
        b = (rows["UT"] - RHO * rows["I1"] * Q[f] ** 2 - RHO * G * rows["I2"]
             + rows["CAP"])
        c = rows["CT"] * Q[f]
        bs.append(b)
        cs.append(c)
        taus.append(rows["TAU"])
        fidx.append(np.full(len(b), f))
    if not bs:
        raise SystemExit("no usable weak-form rows (window too tight?)")
    b = np.concatenate(bs)
    c = np.concatenate(cs)
    tau = np.concatenate(taus)
    fidx = np.concatenate(fidx)
    eta, se, keep = robust_ratio(b, c)
    th, cov = bingham_fit(b, c, tau, keep)
    return dict(Q=Q, eta_frames=eta_f, eta=eta, se=se, keep=keep, b=b, c=c,
                tau=tau, fidx=fidx, bingham=(th, cov))


def robust_ratio(b, c, robust_iters: int = 3):
    """eta = argmin |c*eta - b|^2 with sigma clipping. Returns (eta, se, keep)."""
    keep = np.isfinite(b) & np.isfinite(c) & (np.abs(c) > 0)
    eta = np.nan
    for _ in range(robust_iters):
        if keep.sum() < 4:
            break
        eta = float(np.sum(c[keep] * b[keep]) / np.sum(c[keep] ** 2))
        r = b - c * eta
        med = np.median(r[keep])
        s = 1.4826 * np.median(np.abs(r[keep] - med))
        keep_new = keep & (np.abs(r - med) < 3.0 * max(s, 1e-12))
        if keep_new.sum() == keep.sum():
            break
        keep = keep_new
    eta = float(np.sum(c[keep] * b[keep]) / np.sum(c[keep] ** 2))
    r = b[keep] - c[keep] * eta
    se = float(np.sqrt(np.sum(r**2) / max(keep.sum() - 1, 1) / np.sum(c[keep] ** 2)))
    return eta, se, keep


def bingham_fit(b, c, tau, keep):
    """(eta, tau_y) joint fit + covariance from row residuals."""
    Xm = np.stack([c[keep], tau[keep]], axis=1)
    th, *_ = np.linalg.lstsq(Xm, b[keep], rcond=None)
    r = b[keep] - Xm @ th
    dof = max(len(r) - 2, 1)
    cov = np.linalg.inv(Xm.T @ Xm) * float(np.sum(r**2) / dof)
    return th, cov


# --------------------------------------------------------------------------------------
# manufactured-solution selftest
# --------------------------------------------------------------------------------------
def selftest(eta_true: float = 1.3, Q_true: float = 3.2e-5, noise: float = 0.01,
             seed: int = 0):
    """Generate profiles by solving the steady slender-jet ODE (sigma = 0) at a known
    viscosity, add width noise, and require the estimator to recover eta and Q.
    Gate at 1% width noise; at 2% the 8-seed spread is mean -5%, sd 13% (15 rows):
    unbiased but noise-limited."""
    from scipy.integrate import solve_bvp

    global SIGMA
    sigma_saved = SIGMA
    SIGMA = 0.0
    L = 0.12
    u0 = 0.22

    def odes(s, y):
        # rho u u_s = rho g + 3 eta (A u_s)_s / A  with A = Q/u reduces to
        # u_ss = u_s^2/u + (rho u u_s - rho g) / (3 eta)
        u, du = y
        return np.vstack([du, du**2 / u + (RHO * u * du - RHO * G) / (3 * eta_true)])

    def bc(ya, yb):
        return np.array([ya[0] - u0, yb[1] - G / yb[0]])

    s = np.linspace(0, L, 200)
    y0 = np.vstack([np.sqrt(u0**2 + 2 * G * s), G / np.sqrt(u0**2 + 2 * G * s)])
    sol = solve_bvp(odes, bc, s, y0, tol=1e-8, max_nodes=20000)
    assert sol.success, sol.message
    z_lip = 0.30
    zg = np.arange(z_lip - L, z_lip, 0.0025)
    u_prof = sol.sol(z_lip - zg)[0]
    A_prof = Q_true / u_prof
    rng = np.random.default_rng(seed)
    Fn = 40
    w = 2 * np.sqrt(A_prof / np.pi)
    W = np.tile(w, (Fn, 1)) * (1 + noise * rng.standard_normal((Fn, len(zg))))
    obs = dict(
        t=np.arange(Fn) / 60.0, jet_z=zg, jet_width=W,
        jet_chi=np.zeros_like(W), jet_npx=np.full(W.shape, 50, dtype=int),
        n_jet_px=np.full(Fn, 50 * len(zg)),
        lip=np.tile([0.0, 0.0, z_lip], (Fn, 1)))
    fr = condition(obs, (-1.0, 2.0), block_s=0.25)
    sol = solve_all(fr, window=(0.008, 0.010), n_tests=5)
    SIGMA = sigma_saved
    th, cov = sol["bingham"]
    q_err = 100 * (np.nanmedian(sol["Q"]) / Q_true - 1)
    e_err = 100 * (sol["eta"] / eta_true - 1)
    print(f"selftest: Q recovered {np.nanmedian(sol['Q']) * 1e6:.2f} uL/s vs "
          f"{Q_true * 1e6:.2f} ({q_err:+.1f}%) | eta {sol['eta']:.3f} vs {eta_true} "
          f"({e_err:+.1f}%) +- {sol['se']:.3f} | Bingham eta={th[0]:.3f} "
          f"tau_y={th[1]:.2f} (sd {np.sqrt(cov[1, 1]):.2f}) | rows kept "
          f"{int(sol['keep'].sum())}/{len(sol['b'])}")
    assert abs(e_err) < 8.0, "selftest eta error above 8%"
    assert abs(q_err) < 5.0, "selftest Q error above 5%"
    return sol["eta"]


# --------------------------------------------------------------------------------------
# brink (spout-lip) weak form: the primary real-data channel
#
# The optical field channels are defeated by this episode's optics (module docstring);
# what remains trustworthy is integral: the receiver's graduation curve V(t), the
# onset, the cup kinematics, and the caliper geometry. Those still close a weak form:
#
#   mass       V_src(t) = V0 - V_rcv(t) - V_flight(t)         (conservation)
#   geometry   V_src -> horizontal free-surface level L(t) in the TILTED cup
#              (lattice inversion of the caliper spec; Re_bulk < 1 so the interior
#              surface is horizontal to ~1 deg), and L(t) -> head profile over the
#              measured rim curve: h_i(t) per transverse strip, plus h(s) upstream
#              along the spout substrate.
#   momentum   depth-averaged lubrication balance per transverse strip at the brink,
#              written in the HEAD (the level height over the measured rim curve):
#                3 eta q_i / h_i^2 = rho g sin(a) h_i
#              with q_i = Q * h_i^3 / sum h^3 (leading-order transverse split) and
#              Q(t) = dV_rcv/dt read from the receiver graduation curve.
#
# Everything except eta is measured or caliper geometry, and eta enters linearly.
# The balance is imposed time-integrated (brink_forcing / fit_eta_integrated), so no
# derivative of data is ever taken. Stated systematics (reported, not
# hidden): (1) the brink drawdown -- the free surface dips in the last few depths
# before the lip, so the head-based h overestimates the brink depth and the O(1)
# discharge coefficient is absorbed into eta_hat; (2) the flow sits between the
# viscous and inertial regimes (the inertia share is reported per frame). The twin
# run at eta_hat is the arbiter of the combined effect.
# --------------------------------------------------------------------------------------


def smooth_series(t, y, s_win: float = 0.35):
    """NaN-tolerant moving quadratic fit; returns (y_s, dy/dt) on the same t."""
    ys = np.full_like(y, np.nan, dtype=float)
    dys = np.full_like(y, np.nan, dtype=float)
    ok = np.isfinite(y)
    for i, ti in enumerate(t):
        m = ok & (np.abs(t - ti) <= s_win / 2)
        if m.sum() < 5:
            continue
        c = np.polyfit(t[m] - ti, y[m], 2)
        ys[i], dys[i] = c[2], c[1]
    return ys, dys


def brink_forcing(obs: dict, v0_ml: float, t_fit: tuple[float, float],
                  eta_flight: float | None = None):
    """The TIME-WEAK assembly: no derivative of data is ever taken. Per readable
    frame, compute the lubrication forcing

        F(t) = (rho g / 3) sin(a(t)) * INT h(y,t)^3 dy      [head over the rim curve]

    from the smoothed level state + caliper geometry, so the model predicts
        V(t) - V(t0) = (1/eta) INT_t0^t F dt'
    which is fit to the MEASURED V(t) -- linear in 1/eta, residuals in mL.
    eta_flight, when given, sizes the in-flight correction from the model flux.
    Returns dict with t, V_meas (mL), cumF (mL*Pa), diagnostics."""
    sys.path.insert(0, str(REPO / "examples"))
    sys.path.insert(0, str(REPO / "experiments" / "pour"))
    from pour_perception import build_cavity_lattice, rim_curve_local

    from warpmpm.colliders.glass import quat_to_mat

    t = obs["t"]
    Vr, _ = smooth_series(t, obs["rcv_vol"] * 1e6, s_win=0.30)      # mL
    lattice, cell = build_cavity_lattice()
    rim = rim_curve_local()
    ys = rim[:, 1]
    keep_y = np.abs(ys) <= 0.019
    rim, ys = rim[keep_y], ys[keep_y]

    m_win = (t >= t_fit[0]) & (t <= t_fit[1]) & np.isfinite(Vr)
    idx = np.where(m_win)[0]
    out = dict(t=[], V=[], Fq=[], h_tip=[], sina=[], Q_model=[], level=[])
    for i in idx:
        pos, quat = obs["cup_pos"][i], obs["cup_quat"][i]
        rot = quat_to_mat(quat)
        v_fl = 0.0
        if eta_flight is not None and out["Fq"]:
            q_prev = out["Q_model"][-1]                 # mL/s at the previous frame
            lip_z = obs["lip"][i][2]
            pool_z = obs["rcv_level"][i]
            if not np.isfinite(pool_z):
                pool_z = obs["table_z"] + 0.0044
            fall = max(float(lip_z - pool_z), 0.0)
            v_fl = q_prev * np.sqrt(2 * fall / G)       # mL in ballistic flight
        v_src = (v0_ml - Vr[i] - v_fl) * 1e-6
        if v_src <= 1e-6:
            Fq = 0.0
            level = np.nan
            h_tip = 0.0
            sina = np.nan
        else:
            wz = np.sort(lattice @ rot[2] + pos[2])
            k = int(np.clip(round(v_src / cell), 1, len(wz) - 1))
            level = float(wz[k])
            rim_w = rim @ rot.T + pos
            tangent = rot @ np.array([1.0, 0.0, 0.0])
            tangent /= np.linalg.norm(tangent)
            sina = abs(float(tangent[2]))
            cosa = np.sqrt(max(1.0 - sina**2, 1e-6))
            h_perp = np.clip(level - rim_w[:, 2], 0.0, None) * cosa
            H3 = float(np.trapezoid(h_perp**3, ys))
            h_tip = float(h_perp.max())
            Fq = RHO * G * sina * H3 / 3.0              # eta * Q  [Pa * m^3/s]
        out["t"].append(t[i])
        out["V"].append(Vr[i])
        out["Fq"].append(Fq)
        out["h_tip"].append(h_tip)
        out["sina"].append(sina)
        out["level"].append(level)
        out["Q_model"].append(1e6 * Fq / max(eta_flight or 1.4, 1e-3))
    o = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    # cumulative model integral in mL * Pa.s (trapezoid over the readable frames)
    dt_seg = np.diff(o["t"])
    fmid = 0.5 * (o["Fq"][1:] + o["Fq"][:-1]) * 1e6     # mL/s * Pa.s
    o["cumF"] = np.concatenate([[0.0], np.cumsum(fmid * dt_seg)])
    return o


def fit_eta_integrated(o: dict, t_hi: float | None = None):
    """Closed-form 1/eta fit of V(t) - V(t0) = (1/eta) cumF(t), with sigma clipping.
    Returns (eta, se, keep, dV, cumF)."""
    m = np.isfinite(o["V"])
    if t_hi is not None:
        m &= o["t"] <= t_hi
    dV = o["V"][m] - o["V"][m][0]
    cF = o["cumF"][m] - o["cumF"][m][0]
    keep = np.ones(len(dV), bool)
    inv = np.nan
    for _ in range(3):
        if keep.sum() < 6 or np.sum(cF[keep] ** 2) <= 0:
            break
        inv = np.sum(cF[keep] * dV[keep]) / np.sum(cF[keep] ** 2)
        r = dV - cF * inv
        med = np.median(r[keep])
        s = 1.4826 * np.median(np.abs(r[keep] - med))
        keep_new = keep & (np.abs(r - med) < 3.5 * max(s, 1e-9))
        if keep_new.sum() == keep.sum():
            break
        keep = keep_new
    inv = np.sum(cF[keep] * dV[keep]) / np.sum(cF[keep] ** 2)
    eta = 1.0 / inv
    r = dV[keep] - cF[keep] * inv
    se_inv = np.sqrt(np.sum(r**2) / max(keep.sum() - 1, 1) / np.sum(cF[keep] ** 2))
    se = float(se_inv * eta**2)
    return float(eta), se, keep, dV, cF


def run_brink(episode: str, v0_ml: float, t_fit, v0_tol: float = 10.0):
    out = OUT_ROOT / episode
    obs = dict(np.load(out / "observations.npz"))
    # two passes: the in-flight correction needs a flux scale
    o = brink_forcing(obs, v0_ml, tuple(t_fit))
    eta0, *_ = fit_eta_integrated(o)
    o = brink_forcing(obs, v0_ml, tuple(t_fit), eta_flight=eta0)
    eta, se, keep, dV, cF = fit_eta_integrated(o)
    rms = float(np.sqrt(np.mean((dV[keep] - cF[keep] / eta) ** 2)))
    res = dict(channel="brink-integrated", eta=eta, eta_se=se,
               n_frames=len(dV), n_kept=int(keep.sum()),
               rms_mL=rms, v0_ml=v0_ml, t_fit=list(t_fit),
               h_tip_max_mm=float(np.nanmax(o["h_tip"]) * 1e3),
               dV_total_mL=float(dV[-1]))
    # V0 scan: the head enters cubed, so eta_hat is sensitive to the fill (about 1.4%
    # per mL on ep0001) while the residual barely moves across the scan: the window
    # holds about a dozen distinct level readings and, after smoothing, a few
    # independent samples, so the rms cannot rank V0. The fill therefore stays a
    # protocol prior and its tolerance is carried as a systematic on eta_hat.
    v0_scan = []
    for v0 in np.arange(v0_ml - 30.0, v0_ml + 30.0 + 1e-9, 5.0):
        ov = brink_forcing(obs, float(v0), tuple(t_fit), eta_flight=eta0)
        ev, sv, kv, dVv, cFv = fit_eta_integrated(ov)
        rv = float(np.sqrt(np.mean((dVv[kv] - cFv[kv] / ev) ** 2)))
        v0_scan.append(dict(v0=float(v0), eta=ev, se=sv, rms_mL=rv))
    res["v0_scan"] = v0_scan
    res["v0_tol_ml"] = float(v0_tol)
    res["eta_v0_range"] = [
        fit_eta_integrated(brink_forcing(obs, v0_ml + d, tuple(t_fit), eta_flight=eta0))[0]
        for d in (-v0_tol, v0_tol)]
    # prefix sweep: eta identified from data up to T only
    prefix = []
    for t_hi in np.arange(t_fit[0] + 0.3, t_fit[1] + 1e-9, 0.1):
        try:
            ep_, sp_, kp_, dVp, _ = fit_eta_integrated(o, t_hi=float(t_hi))
        except Exception:
            continue
        if np.isfinite(ep_) and len(dVp) >= 8:
            prefix.append(dict(t_hi=float(t_hi), eta=ep_, se=sp_,
                               n=int(kp_.sum()), dV_mL=float(dVp[-1])))
    res["prefix_sweep"] = prefix
    (out / "identify.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("prefix_sweep", "v0_scan")}, indent=2))
    print("v0 scan:", " ".join(f"{d['v0']:.0f}:{d['eta']:.2f}({d['rms_mL']:.1f}mL)"
                               for d in v0_scan))
    lo, hi = res["eta_v0_range"]
    print(f"fill prior {v0_ml:.0f} +- {v0_tol:.0f} mL -> eta_hat in [{lo:.2f}, {hi:.2f}] Pa.s "
          f"({100 * (lo / eta - 1):+.0f}% / {100 * (hi / eta - 1):+.0f}%), the dominant "
          f"systematic (statistical: +-{100 * se / eta:.1f}%)")
    plot_brink_integrated(o, eta, keep, dV, cF, res, out / "identify.png")
    print("wrote", out / "identify.json", "and", out / "identify.png")
    return res


def plot_brink_integrated(o, eta, keep, dV, cF, res, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(o["t"], o["V"], ".", ms=3, color="k", label="measured V_rcv [mL]")
    m = np.isfinite(o["V"])
    ax.plot(o["t"][m], o["V"][m][0] + (o["cumF"][m] - o["cumF"][m][0]) / eta, "-",
            lw=1.8, color="tab:blue", label=f"time-weak model, eta={eta:.2f} Pa.s")
    ax.set_xlabel("t - t_send [s]")
    ax.set_ylabel("mL")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(o["t"], np.asarray(o["h_tip"]) * 1e3, ".-", ms=3, label="head [mm]")
    ax.plot(o["t"], o["Fq"] * 1e6 / eta, "--",
            label="model flux at eta [mL/s]")
    ax.set_xlabel("t - t_send [s]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(cF[keep], dV[keep], ".", ms=4, alpha=0.7)
    ax.plot(cF[~keep], dV[~keep], "x", ms=4, color="tab:red", alpha=0.6)
    xx = np.linspace(0, cF.max(), 10)
    ax.plot(xx, xx / eta, "k-", lw=1, label="slope = 1/eta")
    ax.set_xlabel("cumulative forcing [mL Pa.s]")
    ax.set_ylabel("transferred [mL]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    pre = res.get("prefix_sweep", [])
    if pre:
        tt = [p["t_hi"] for p in pre]
        ee = [p["eta"] for p in pre]
        ss = [p["se"] for p in pre]
        ax.errorbar(tt, ee, yerr=ss, fmt="o-", ms=3, lw=1, capsize=2)
        ax.axhline(eta, color="k", lw=0.8, ls=":")
        ax.axhline(1.41, color="tab:green", lw=0.8, ls="--",
                   label="Segur-Oberstar 20 C")
        ax.set_xlabel("fit uses data up to t [s]")
        ax.set_ylabel("eta [Pa.s]")
        ax.set_title("identification vs data seen")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="ep0001")
    ap.add_argument("--selftest", action="store_true",
                    help="manufactured-solution gate for the slender-jet estimator")
    ap.add_argument("--v0-ml", type=float, default=300.0,
                    help="initial fill (the protocol's metered 300 mL)")
    ap.add_argument("--t-fit", type=float, nargs=2, default=(2.6, 3.6),
                    help="pour-clock fit window (s after t_send); the default is "
                         "the quasi-steady mid-drain (early = slosh, late = "
                         "flight/film-drain)")
    ap.add_argument("--v0-tol", type=float, default=10.0,
                    help="fill uncertainty (mL) carried as the eta_hat systematic")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        sys.exit(0)
    run_brink(args.episode, args.v0_ml, args.t_fit, args.v0_tol)

"""Grid-consistent (Bubnov-Galerkin) weak-form assembly on the MPM grid basis.

The discrete weak residual is unbiased only when the test functions live in the
same discrete space the forward solver uses and the internal force is assembled
with the same gradient operator (docs/MATH_REFERENCE.md Section 2.5, following
EUCLID). For warp-mpm that space is the quadratic B-spline grid basis N_i used
in P2G/G2P; independent analytic bump functions sampled at particles carry a
patch-scale bias on this data.

This assembles the per-grid-node, per-direction momentum residual using the MPM
B-spline weights and gradients (reconstructed exactly from the dump's particle
positions and the grid spacing), restricted to plane strain by summing the 3D
basis over the out-of-plane node layer (sum_y N = 1, sum_y dN/dy = 0). The MPM
discrete balance per node i and direction d is, exactly,

    sum_p V_p sigma_p[d,:] . grad N_i(x_p) = sum_p m_p (g_d - a_d) N_i(x_p)

With sigma = -p I + sum_k theta_k p phi_k(I) (2D/|gd|) this is linear in theta:

    A[(i,d),k] = sum_p V_p phi_k p (2D/|gd|)_p[d,:] . grad N_i(x_p)
    b[(i,d)]   = sum_p m_p (g_d - a_d) N_i(x_p) + sum_p V_p p_p grad_d N_i(x_p)

(the pressure term does not drop: these test functions are not divergence free,
so pressure enters b as data). Rows are emitted per frame (instantaneous MPM
balance) for nodes whose contributing particles are predominantly flowing and
at yield, so the linear mu(I) model holds for every contributing particle.

NOTE: exactness additionally wants the GRID acceleration a_i; here a is the
particle trajectory acceleration interpolated implicitly through the sum, which
is consistent to the temporal discretization. This module is oracle-specific
(it uses the grid); the real-data path reconstructs fields on a chosen basis
and uses test = that basis (MATH_REFERENCE 6.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.conventions import (
    EPS_GAMMA_DEFAULT,
    equivalent_shear_rate,
    gravity_vector_inplane,
    inertial_number,
    pressure_from_cauchy_3d_trace,
    sym,
)
from ident.features.base import Dictionary
from ident.io.schema import Dump
from ident.masks.flowing import flowing_mask, gate_transient_cutoff


@dataclass
class GridSystem:
    A: np.ndarray            # (n_rows, K)
    b: np.ndarray            # (n_rows,) acceleration-form load
    node_flow_frac: np.ndarray   # (n_rows,) flowing weight fraction at the node
    n_rows: int                  # rows AFTER the flow_frac/mass gate (row_count_after_gating)
    I_observed: tuple[float, float]
    n_rows_before_gating: int = 0    # candidate rows (mass-supported nodes x 2 dirs) BEFORE the gate
    row_survival: float = 0.0        # n_rows / n_rows_before_gating


def _bspline_weights_1d(fx: np.ndarray):
    """Quadratic B-spline weights and d/d(gridpos) derivatives, 3 nodes.

    Matches warp-mpm p2g: w0=0.5(1.5-fx)^2, w1=0.75-(fx-1)^2, w2=0.5(fx-0.5)^2;
    dw0=fx-1.5, dw1=-2(fx-1), dw2=fx-0.5 (derivatives w.r.t. grid position).
    """
    w = np.stack([0.5 * (1.5 - fx) ** 2,
                  0.75 - (fx - 1.0) ** 2,
                  0.5 * (fx - 0.5) ** 2], axis=-1)
    dw = np.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], axis=-1)
    return w, dw


def assemble_grid_consistent(
    dump: Dump,
    dictionary: Dictionary,
    eps_gamma: float = EPS_GAMMA_DEFAULT,
    frame_stride: int = 4,
    flow_frac_min: float = 0.97,
    min_support_mass_frac: float = 0.5,
    gate_clearance_time: float = 0.0,
    pressure: np.ndarray | None = None,
    flowing_override: np.ndarray | None = None,
    accel: np.ndarray | None = None,
    g_vec: np.ndarray | None = None,
    min_node_z: float | None = None,
) -> GridSystem:
    """Grid-consistent assembly.

    pressure: optional (F, P) array to use instead of the 3D stress trace
        (for the G1P closure ablation: pass P0 or P1 here). I is recomputed
        from it; the b pressure term and A both use it.
    flowing_override: optional (F, P) boolean to fix the flowing-at-yield set
        (so closures are compared on the SAME nodes). If None it is computed
        from the (possibly overridden) pressure.
    accel: optional (F, P, 2) in-plane MATERIAL acceleration to use instead of
        the trajectory finite difference of dump.v. Needed when the dump's
        particles are fixed Eulerian quadrature points (e.g. the reconstructed
        field of the G3 perception path), where the trajectory FD would give
        only the Eulerian dv/dt and miss the convective term; pass
        a = dv/dt + L@v.
    """
    meta = dump.meta
    ax, az = meta.in_plane_axes
    cfg = meta.extra.get("config", {})
    n_grid = cfg["n_grid"]
    grid_lim = cfg["grid_lim"]
    inv_dx = n_grid / grid_lim
    d, rho_s = meta.grain_diameter, meta.rho_s
    K = dictionary.K
    # default pinned vertical gravity; inclined-plane scenes pass tilted g.
    g = gravity_vector_inplane() if g_vec is None else np.asarray(g_vec, dtype=float)
    gate_cut = gate_transient_cutoff(d, gate_clearance_time)

    # per-frame fields
    x_ip = dump.x[..., [ax, az]]
    v_ip = dump.v[..., [ax, az]]
    L_ip = dump.L[:, :, [ax, az]][:, :, :, [ax, az]]
    D_ip = sym(L_ip)
    gd = equivalent_shear_rate(D_ip, eps_gamma)
    p = pressure_from_cauchy_3d_trace(dump.stress) if pressure is None else np.asarray(pressure)
    I = inertial_number(gd, p, d, rho_s)
    rho = dump.mass[None, :] / np.maximum(dump.volume, 1e-30)
    frame_dt = meta.frame_dt
    F = meta.n_frames
    if accel is not None:
        a_ip = np.asarray(accel, dtype=float)
    else:
        a_ip = np.zeros_like(v_ip)
        if F >= 3:
            a_ip[1:-1] = (v_ip[2:] - v_ip[:-2]) / (2.0 * frame_dt)
        a_ip[0] = (v_ip[1] - v_ip[0]) / frame_dt
        a_ip[-1] = (v_ip[-1] - v_ip[-2]) / frame_dt

    rows_A: list[np.ndarray] = []
    rows_b: list[float] = []
    rows_frac: list[float] = []
    I_lo, I_hi = np.inf, 0.0
    n_before = 0
    n_nodes = n_grid * n_grid

    for f in range(0, F, frame_stride):
        if dump.times[f] < gate_cut:
            continue
        act = dump.active[f]
        if not np.any(act):
            continue
        xs = x_ip[f][act]
        if flowing_override is not None:
            flow = flowing_override[f][act] & (p[f][act] > 0) & np.isfinite(I[f][act])
        else:
            flow = flowing_mask(gd[f][act], I[f][act]) & (p[f][act] > 0) & np.isfinite(I[f][act])
        if flow.sum() < 50:
            continue
        gp = xs * inv_dx
        base = np.floor(gp - 0.5).astype(int)
        fx = gp - base
        wx, dwx = _bspline_weights_1d(fx[:, 0])
        wz, dwz = _bspline_weights_1d(fx[:, 1])

        vol = dump.volume[f][act]
        mass = dump.mass[act]
        pp = p[f][act]
        Dp = D_ip[f][act]
        gdp = gd[f][act]
        flow_dir = 2.0 * Dp / gdp[:, None, None]      # (n,2,2)
        Ip = I[f][act]
        # clamp non-finite / non-positive I (p <= 0 particles) to keep phi
        # finite; such particles have ~zero stress (vol*p ~ 0) so the clamp
        # value is immaterial to the assembled force
        Ip_phi = np.clip(np.nan_to_num(Ip, posinf=1e6, neginf=0.0), 1e-12, 1e6)
        Phi = dictionary.phi(Ip_phi)    # (n, K)
        ap = a_ip[f][act]

        # accumulators over flat node ids
        A_x = np.zeros((n_nodes, K)); A_z = np.zeros((n_nodes, K))
        b_x = np.zeros(n_nodes); b_z = np.zeros(n_nodes)
        w_flow = np.zeros(n_nodes); w_tot = np.zeros(n_nodes)

        body = mass[:, None] * (g[None, :] - ap)        # (n,2): m(g-a)
        for i in range(3):
            nx = base[:, 0] + i
            for j in range(3):
                nz = base[:, 1] + j
                valid_node = (nx >= 0) & (nx < n_grid) & (nz >= 0) & (nz < n_grid)
                nid = np.where(valid_node, nx * n_grid + nz, 0)
                N = wx[:, i] * wz[:, j]
                gNx = dwx[:, i] * wz[:, j] * inv_dx
                gNz = wx[:, i] * dwz[:, j] * inv_dx
                m_ = valid_node
                # internal-force stress-power per basis column k, direction d:
                # (2D/|gd|)[d,0]*gNx + (2D/|gd|)[d,1]*gNz, times V phi_k p
                fd_x = flow_dir[:, 0, 0] * gNx + flow_dir[:, 0, 1] * gNz   # (n,)
                fd_z = flow_dir[:, 1, 0] * gNx + flow_dir[:, 1, 1] * gNz
                coefx = (vol * pp)[:, None] * Phi * fd_x[:, None]          # (n,K)
                coefz = (vol * pp)[:, None] * Phi * fd_z[:, None]
                for k in range(K):
                    np.add.at(A_x[:, k], nid[m_], coefx[m_, k])
                    np.add.at(A_z[:, k], nid[m_], coefz[m_, k])
                # b: m(g-a)_d N + V p grad_d N
                np.add.at(b_x, nid[m_], (body[:, 0] * N + vol * pp * gNx)[m_])
                np.add.at(b_z, nid[m_], (body[:, 1] * N + vol * pp * gNz)[m_])
                np.add.at(w_tot, nid[m_], (mass * N)[m_])
                np.add.at(w_flow, nid[m_], (mass * N * flow)[m_])

        frac = np.where(w_tot > 0, w_flow / np.maximum(w_tot, 1e-30), 0.0)
        max_mass = w_tot.max() if w_tot.size else 0.0
        n_before += 2 * int((w_tot > 0).sum())          # candidate rows before the flow/mass gate
        good = (frac >= flow_frac_min) & (w_tot >= min_support_mass_frac * max_mass)
        if min_node_z is not None:
            # drop nodes within the basal boundary layer: the rough-floor
            # collider exerts a contact force not present in the rho(g-a) load,
            # so near-bed nodes would bias mu. node z = (nid % n_grid) * dx.
            node_z = (np.arange(n_nodes) % n_grid) / inv_dx
            good = good & (node_z >= min_node_z)
        gidx = np.where(good)[0]
        for nid_ in gidx:
            rows_A.append(A_x[nid_]); rows_b.append(b_x[nid_]); rows_frac.append(frac[nid_])
            rows_A.append(A_z[nid_]); rows_b.append(b_z[nid_]); rows_frac.append(frac[nid_])
        if gidx.size:
            # I coverage from flowing particles this frame (finite only, and
            # 99th percentile so the low-pressure tail does not dominate)
            If = Ip[flow]
            If = If[np.isfinite(If)]
            if If.size:
                I_lo = min(I_lo, float(np.percentile(If, 5)))
                I_hi = max(I_hi, float(np.percentile(If, 90)))

    if not rows_A:
        return GridSystem(np.zeros((0, K)), np.zeros(0), np.zeros(0), 0, (0.0, 0.0),
                          n_rows_before_gating=n_before, row_survival=0.0)
    A = np.array(rows_A)
    b = np.array(rows_b)
    frac = np.array(rows_frac)
    if not np.isfinite(I_lo):
        I_lo = 0.0
    return GridSystem(A=A, b=b, node_flow_frac=frac, n_rows=len(b),
                      I_observed=(I_lo, I_hi), n_rows_before_gating=n_before,
                      row_survival=(len(b) / n_before if n_before else 0.0))

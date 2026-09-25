"""Probe the dumped velocity-gradient index convention and pressure sanity.

The dump stores a per-particle velocity gradient L. The identifier needs to
know whether L_ij = dv_i/dx_j (so the material acceleration is dv/dt + L @ v)
or the transpose (dv/dt + L^T @ v). Rather than trust the kernel, this probe
reconstructs the spatial gradient G_ij = dv_i/dx_j from a local least-squares
fit of the velocity field over each particle's neighbors at a single frame,
and reports which of L or L^T matches G. The verdict is decisive and
data-driven.

Also reports:
  - div(v) from L in the bulk (incompressibility consistency)
  - a static-column pressure check: P0 = rho g (h - z) against the 3D stress
    trace before the column has moved much.

Run as a script on any dump; prints the verdict string to embed in the dump
metadata (sim/column_scene already embeds the verified string and this probe
is the guard).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ident.io.schema import load_dump  # noqa: E402
from common.conventions import G_MAG, pressure_from_cauchy_3d_trace  # noqa: E402

L_CONVENTION_STRING = "L_ij=dv_i/dx_j; material accel a = dv/dt + L@v"


def _local_velocity_gradient(x3, v3, sample_idx, k=24):
    """Least-squares dv_i/dx_j over k nearest neighbours (full 3D)."""
    grads = np.empty((len(sample_idx), 3, 3))
    for n, i in enumerate(sample_idx):
        d2 = np.sum((x3 - x3[i]) ** 2, axis=1)
        nbr = np.argsort(d2)[: k + 1]  # includes self
        dx = x3[nbr] - x3[i]           # (k+1, 3)
        dv = v3[nbr] - v3[i]           # (k+1, 3)
        # solve dv ~ dx @ G^T  (rows: samples) => G_ij = dv_i/dx_j
        GT, *_ = np.linalg.lstsq(dx, dv, rcond=None)  # (3,3): dv = dx @ GT
        grads[n] = GT.T
    return grads


def probe(dump_path: str | Path, frame_index: int | None = None) -> dict:
    dump = load_dump(dump_path)
    ax_x, ax_z = dump.meta.in_plane_axes

    # pick a frame with real motion
    speed = np.linalg.norm(dump.v, axis=2)
    frame_motion = np.array([speed[f, dump.active[f]].mean() if dump.active[f].any() else 0.0
                             for f in range(dump.meta.n_frames)])
    fi = frame_index if frame_index is not None else int(np.argmax(frame_motion))

    m = dump.active[fi]
    x3 = dump.x[fi, m]
    v3 = dump.v[fi, m]
    L = dump.L[fi, m]

    rng = np.random.default_rng(0)
    pool = np.where(np.linalg.norm(v3, axis=1) > 0.05)[0]
    if len(pool) < 30:
        pool = np.arange(len(x3))
    sample = rng.choice(pool, size=min(200, len(pool)), replace=False)

    G_fit = _local_velocity_gradient(x3, v3, sample)
    L_s = L[sample]
    scale = np.maximum(np.linalg.norm(G_fit.reshape(len(sample), -1), axis=1), 1e-9)
    err_L = np.linalg.norm((L_s - G_fit).reshape(len(sample), -1), axis=1) / scale
    err_LT = np.linalg.norm(
        (np.swapaxes(L_s, 1, 2) - G_fit).reshape(len(sample), -1), axis=1
    ) / scale
    med_L, med_LT = float(np.median(err_L)), float(np.median(err_LT))
    verdict = "L == dv_i/dx_j (a = dv/dt + L@v)" if med_L < med_LT else \
              "L == (dv_i/dx_j)^T (a = dv/dt + L^T@v)"

    # div(v) from L (in-plane trace + full trace) in the bulk
    div_full = np.abs(L[:, 0, 0] + L[:, 1, 1] + L[:, 2, 2])
    div_inplane = np.abs(L[:, ax_x, ax_x] + L[:, ax_z, ax_z])

    # static-column pressure check at the first frame (column barely moved)
    m0 = dump.active[0]
    x0 = dump.x[0, m0]
    z = x0[:, ax_z]
    p0_stress = pressure_from_cauchy_3d_trace(dump.stress[0, m0])
    # column free surface per x bin
    bin_w = 4.0 * dump.meta.grain_diameter
    xb = np.floor((x0[:, ax_x] - x0[:, ax_x].min()) / bin_w).astype(int)
    h = np.full(xb.max() + 1, -np.inf)
    np.maximum.at(h, xb, z)
    p0_hydro = dump.meta.rho_bulk * G_MAG * (h[xb] - z)
    # the first frame is stress-free (F=I), so check the early settled frame
    fi_static = min(20, dump.meta.n_frames - 1)
    ms = dump.active[fi_static]
    zs = dump.x[fi_static, ms][:, ax_z]
    ps = pressure_from_cauchy_3d_trace(dump.stress[fi_static, ms])
    xs = dump.x[fi_static, ms][:, ax_x]
    xbs = np.floor((xs - xs.min()) / bin_w).astype(int)
    hs = np.full(xbs.max() + 1, -np.inf)
    np.maximum.at(hs, xbs, zs)
    p_hydro_s = dump.meta.rho_bulk * G_MAG * (hs[xbs] - zs)
    # compare only deep, interior points
    deep = (hs[xbs] - zs) > 5 * dump.meta.grain_diameter
    if deep.sum() > 20:
        rel_p = np.median(np.abs(ps[deep] - p_hydro_s[deep]) / np.maximum(p_hydro_s[deep], 1.0))
    else:
        rel_p = float("nan")

    return {
        "frame_index": fi,
        "median_err_vs_L": med_L,
        "median_err_vs_LT": med_LT,
        "verdict": verdict,
        "matches_expected": med_L < med_LT and med_L < 0.25,
        "div_full_median": float(np.median(div_full)),
        "div_inplane_median": float(np.median(div_inplane)),
        "static_pressure_rel_err_median": float(rel_p),
        "static_frame": fi_static,
    }


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "out" / "dumps" / "smoke.npz")
    res = probe(path)
    print(json.dumps(res, indent=2))
    print("EXPECTED CONVENTION:", L_CONVENTION_STRING)
    print("PROBE", "OK" if res["matches_expected"] else "MISMATCH")

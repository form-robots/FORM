"""Function-encoder identification on NCLaw's own trajectories.

The same unknown-form dictionaries as the same-engine study (baseline.py),
pointed at the ingested cross-engine dumps and rolled out on their five scenes
with their metric, under the same engine-compatibility flags the known-form
cross-engine rows use (experiments.nclaw.compare). The opponent row at equal
task difficulty is NCLaw's learned network, which also fits a full function
from one trajectory; the known-form rows bound both from above.

Per material:
  sand        mu(I) through the trained granular basis, the accepted curve
              baked into the engine's tabulated material and rolled out.
  jelly       W'(I1bar) through the trained one-invariant basis plus one
              volumetric column. The engine has no tabulated hyperelastic
              material, so the rollout leg projects the recovered curve to its
              small-strain corotated pair and the leg name says so.
  water       the same hyperelastic family: for a fluid its deviatoric
              coefficients should return near zero and its volumetric column
              is exactly their linear law sigma = lam (J - 1) I, rolled out
              through the comparison EOS. The viscous family runs beside it.
  plasticine  every family this campaign ships is the wrong class for an
              elastoplastic solid, so the legs refuse. No trained plasticity
              basis is wired in here.

Run from the engine root:

    .venv/bin/python -m experiments.fe_ls.cross [material ...]

Writes out/fe_ls_cross/results.json and the prediction dumps beside it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.fe_ls.baseline import (
    VISCOUS_WINDOW,
    bake_mu_table,
    identify_friction_fe,
    identify_hyperelastic_fe,
    identify_viscous_fe,
    load_granular_fe,
)
from experiments.nclaw import suite

CROSS_DUMPS = suite.ROOT / "out" / "nclaw_cross_generalize" / "dumps"
OUT = suite.ROOT / "out" / "fe_ls_cross"
RESIDUAL_BAR = 0.15

# the engine-compatibility flags of the winning known-form rows, per material
FLAGS = {"jelly": {"nclaw_bc": True, "substeps": 1},
         "plasticine": {"nclaw_bc": True},
         "sand": {"nclaw_bc": True, "substeps": 1},
         "water": {"nclaw_bc": True, "nclaw_law": True}}

PUBLISHED = {"jelly": {"dataset": 2.4e-4, "time": 9.8e-4, "vel_0001": 2.4e-4,
                       "vel_0007": 2.4e-4, "vel_0008": 2.4e-4},
             "plasticine": {"dataset": 6.5e-5, "time": 1.4e-4, "vel_0001": 4.6e-5,
                            "vel_0007": 4.6e-5, "vel_0008": 4.6e-5},
             "sand": {"dataset": 2.6e-5, "time": 4.2e-5, "vel_0001": 6.5e-5,
                      "vel_0007": 6.5e-5, "vel_0008": 6.5e-5},
             "water": {"dataset": 2.0e-5, "time": 3.5e-4, "vel_0001": 1.9e-5,
                       "vel_0007": 1.9e-5, "vel_0008": 1.9e-5}}
# the fifth scene, the held-out mesh (_rollout_legs names it "shape_<mesh>",
# suite.py's own comment on NCLAW_SHAPE_CLIP_BOUND explains why the clip
# bound -- not this table -- is what varies per scene): NCLaw's geometry
# column, suite.NCLAW_PUBLISHED[material]["generalization"], keyed to match.
for _mat, _shape in suite.NCLAW_HELD_OUT_SHAPE.items():
    PUBLISHED[_mat][f"shape_{_shape}"] = suite.NCLAW_PUBLISHED[_mat]["generalization"]
del _mat, _shape


def identify_friction_fe_binned(arr: dict, fe, gd_min: float = 1.0,
                                eps_gamma: float = 0.02, n_bins: int = 32,
                                min_count: int = 100, ridge: float = 1.0e-4,
                                log=print) -> dict:
    """mu(I) from binned pointwise cone readings.

    Bin r = sqrt(J2(dev sigma))/p against I over post-impact shearing frames,
    take the median per bin (the median rejects sub-yield readings), and fit
    the basis to the medians by count-weighted ridge least squares. Momentum
    rows carry impact-phase bias, so this estimator reads the stress channel
    instead.
    """
    from common.conventions import (
        equivalent_shear_rate,
        inertial_number,
        pressure_from_cauchy_3d_trace,
        sym,
    )
    d_grain = float(arr["meta"].grain_diameter)
    rho_s = float(arr["meta"].rho_s)
    D = sym(arr["L"])
    gd = equivalent_shear_rate(D, eps_gamma)
    p = pressure_from_cauchy_3d_trace(arr["stress"])
    dev = arr["stress"] - (np.trace(arr["stress"], axis1=2, axis2=3) / 3.0
                           )[:, :, None, None] * np.eye(3)[None, None]
    r = np.sqrt(0.5 * np.sum(dev * dev, axis=(2, 3))) / np.where(p > 0, p, np.nan)

    ke = 0.5 * np.einsum("p,fpi->f", arr["mass"], arr["v"] ** 2)
    k_peak = int(np.argmax(ke))
    decayed = np.flatnonzero(ke <= 0.1 * ke[k_peak])
    k_start = int(decayed[decayed > k_peak][0]) if np.any(decayed > k_peak) else k_peak

    sel = np.zeros(p.shape, bool)
    sel[k_start:] = True
    sel &= np.isfinite(r) & (gd > gd_min) & (p > 0)
    sel_r, sel_gd, sel_p = r[sel], gd[sel], p[sel]
    floor = np.nanpercentile(sel_p, 25)
    keep = sel_p > floor
    I = inertial_number(sel_gd[keep], sel_p[keep], d_grain, rho_s)  # noqa: E741
    rr = sel_r[keep]
    s = np.log10(np.clip(I, 1e-12, None))
    edges = np.linspace(np.percentile(s, 1), np.percentile(s, 99), n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    centers, medians, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) >= min_count:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            medians.append(float(np.median(rr[m])))
            counts.append(int(m.sum()))
    if len(centers) < 4:
        return {"refused": True,
                "reason": f"only {len(centers)} populated I bins, fewer than 4"}
    Ic = 10.0 ** np.asarray(centers)
    Phi = fe.phi(Ic)
    w = np.sqrt(np.asarray(counts, float))
    A = Phi * w[:, None]
    b_vec = np.asarray(medians) * w
    theta = np.linalg.solve(A.T @ A + ridge * np.eye(Phi.shape[1]), A.T @ b_vec)
    fit = Phi @ theta
    resid = float(np.linalg.norm((fit - medians) * w) / np.linalg.norm(b_vec))
    baked = bake_mu_table(fe, theta)
    out = {"refused": False, "theta": theta.tolist(), "baked_table": baked,
           "n_bins_used": len(centers), "bin_centers_log10I": centers,
           "bin_medians": medians, "bin_counts": counts,
           "fit_residual_rel": resid, "k_start": k_start,
           "n_readings": int(keep.sum()),
           "estimator": "binned pointwise cone readings, median per I bin"}
    log(f"[fe-cross] binned-cone curve: {len(centers)} bins, "
        f"median r span {min(medians):.4f}..{max(medians):.4f} "
        f"(truth cone 0.5680), fit resid {resid:.3f}")
    return out


YIELD_BASIS = suite.ROOT / "fe-weights" / "yield_surface.npz"


def identify_yield_surface_fe(arr: dict, gd_min: float = 1.0,
                              eps_gamma: float = 0.02, n_bins: int = 32,
                              min_count: int = 100, ridge: float = 1.0e-4,
                              log=print) -> dict:
    """The yield surface h(p) as a learned function: sqrt(J2(dev tau)) = h(p).

    One unknown-form family for the whole perfect-plasticity zoo: a von Mises
    solid returns the flat curve h = sigma_y / sqrt(2), a cohesionless cone
    returns the line through the origin with slope sqrt(J2)/p, cohesion and
    caps sit in between. The estimator is the binned pointwise reading in
    Kirchhoff quantities over the post-impact shearing set, tension states
    included, because yielding under tension is exactly what separates the
    flat family from the cones. Fit by ridge least squares on the trained
    basis, weighted by bin counts; linear in theta throughout.
    """
    from common.conventions import (
        equivalent_shear_rate,
        pressure_from_cauchy_3d_trace,
        sym,
    )
    from experiments.fe_ls.baseline import load_table_fe

    phi_fn, p_grid, K = load_table_fe(YIELD_BASIS, "p_grid")
    D = sym(arr["L"])
    gd = equivalent_shear_rate(D, eps_gamma)
    J = np.linalg.det(arr["F"])
    p_k = pressure_from_cauchy_3d_trace(arr["stress"]) * J
    dev = arr["stress"] - (np.trace(arr["stress"], axis1=2, axis2=3) / 3.0
                           )[:, :, None, None] * np.eye(3)[None, None]
    sqrtJ2_k = np.sqrt(0.5 * np.sum(dev * dev, axis=(2, 3))) * J

    ke = 0.5 * np.einsum("p,fpi->f", arr["mass"], arr["v"] ** 2)
    k_peak = int(np.argmax(ke))
    decayed = np.flatnonzero(ke <= 0.1 * ke[k_peak])
    k_start = int(decayed[decayed > k_peak][0]) if np.any(decayed > k_peak) else k_peak

    sel = np.zeros(p_k.shape, bool)
    sel[k_start:] = True
    sel &= np.isfinite(p_k) & np.isfinite(sqrtJ2_k) & (gd > gd_min)
    pp, yy = p_k[sel], sqrtJ2_k[sel]
    lo, hi = np.percentile(pp, 1), np.percentile(pp, 99)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pp, edges) - 1, 0, n_bins - 1)
    centers, medians, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) >= min_count:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            medians.append(float(np.median(yy[m])))
            counts.append(int(m.sum()))
    if len(centers) < 4:
        return {"refused": True,
                "reason": f"only {len(centers)} populated pressure bins"}
    pc = np.asarray(centers)
    outside = float(((pc < p_grid[0]) | (pc > p_grid[-1])).mean())
    Phi = phi_fn(np.clip(pc, p_grid[0], p_grid[-1]))
    w = np.sqrt(np.asarray(counts, float))
    A = Phi * w[:, None]
    b_vec = np.asarray(medians) * w
    # The rollout reads the surface at impact pressures far above the observed
    # bins. A second-difference penalty on h over the trained grid keeps
    # extrapolation straight; flat and linear surfaces lie in its null space.
    Phi_g = phi_fn(p_grid)
    n_g = len(p_grid)
    D2 = (np.eye(n_g, k=0) * -2 + np.eye(n_g, k=1) + np.eye(n_g, k=-1))[1:-1]
    scale = np.linalg.norm(b_vec) / max(np.median(np.abs(medians)), 1e-12)
    C = scale * (D2 @ Phi_g)
    theta = np.linalg.solve(A.T @ A + C.T @ C + ridge * np.eye(K), A.T @ b_vec)
    fit = Phi @ theta
    resid = float(np.linalg.norm((fit - medians) * w) / np.linalg.norm(b_vec))
    # the curve on the trained grid, for the bake and for shape diagnostics
    h_grid = phi_fn(p_grid) @ theta
    dh = np.gradient(h_grid, p_grid)
    out = {"refused": False, "theta": theta.tolist(),
           "fit_residual_rel": resid, "n_bins_used": len(centers),
           "bin_centers_p_kirchhoff": centers, "bin_medians_sqrtJ2": medians,
           "bin_counts": counts, "k_start": k_start,
           "p_fraction_outside_support": outside,
           "n_readings": int(sel.sum()),
           "h_at_p0": float(np.interp(0.0, p_grid, h_grid)),
           "dh_dp_median_on_support": float(np.median(
               dh[(p_grid >= lo) & (p_grid <= hi)])),
           "curve": {"p": p_grid.tolist(), "h": h_grid.tolist()},
           "estimator": ("binned pointwise sqrt(J2) against Kirchhoff "
                         "pressure on the post-impact shearing set, "
                         "tension included")}
    log(f"[fe-cross] yield surface: {len(centers)} bins over p "
        f"[{lo:.0f}, {hi:.0f}] Pa, h(0) = {out['h_at_p0']:.1f} Pa, "
        f"median dh/dp = {out['dh_dp_median_on_support']:.4f}, "
        f"fit resid {resid:.3f}")
    return out


def _yield_surface_theta(result: dict, E: float, nu: float) -> dict:
    """Engine arguments for the tabulated-yield rollout of a recovered h(p)."""
    p = np.asarray(result["curve"]["p"], float)
    h = np.clip(np.asarray(result["curve"]["h"], float), 0.0, None)
    return {"E": float(E), "nu": float(nu), "eta_table": h.tolist(),
            "eta_table_smin": float(p[0]), "eta_table_smax": float(p[-1])}


def _load(material: str) -> dict:
    dump = CROSS_DUMPS / f"{material}_dataset_truth.npz"
    arr = suite._load_arrays(dump)
    # the ingested dumps carry empty law_params; the FE legs read truth values
    # from there only as reporting references, so fill them from the config
    if not arr["meta"].law_params:
        arr["meta"].law_params.update(suite.MATERIALS[material]["truth"])
    return arr


def _rollout_legs(material: str, legs: list[tuple[str, str, dict]],
                  device: str = "cpu", log=print) -> dict:
    """Every accepted leg on every scene, seeded from their frame-0 clouds."""
    flags = FLAGS[material]
    scenes = sorted(p.name.split(f"{material}_", 1)[1].rsplit("_truth.npz", 1)[0]
                    for p in CROSS_DUMPS.glob(f"{material}_*_truth.npz"))
    rows: dict = {}
    for scene in scenes:
        truth = CROSS_DUMPS / f"{material}_{scene}_truth.npz"
        cloud = suite.cloud_from_dump(truth)
        n_expected = int(np.load(truth)["x"].shape[0])
        cells: dict = {}
        for leg, mat_key, theta in legs:
            pred = OUT / f"{material}_{scene}_{leg}.npz"
            sim_wall_s = None
            sim_timing: dict = {}
            if not pred.exists():
                t0 = time.time()
                # xonly: these prediction dumps are scored (nclaw_position_mse
                # reads x only) and never re-seeded from, so the writer can
                # skip v/L/stress/F/volume/active/mass -- same choice
                # experiments.nclaw.compare makes for its own prediction dumps.
                suite.run_scene(mat_key, scene, pred, theta=dict(theta),
                                cloud=cloud,
                                nclaw_bc=flags.get("nclaw_bc", False),
                                nclaw_law=flags.get("nclaw_law", False),
                                substeps=flags.get("substeps"), device=device,
                                log=log, xonly=True, timing=sim_timing)
                sim_wall_s = time.time() - t0
                log(f"[fe-cross] {scene}/{leg} simulated in {sim_wall_s:.0f}s")
            s = suite.nclaw_position_mse(truth, pred, strict=False)
            cell = {k: s[k] for k in ("mse", "mse_final_frame", "rmse_mm", "n_frames")}
            cell["diverged"] = bool(s["n_frames"] < n_expected)
            cell["sim_wall_s"] = sim_wall_s          # None => reused a cached rollout
            cell["sim_timing"] = sim_timing or None  # setup/step/snapshot/finalize
            if cell["diverged"]:
                cell["reason"] = (f"non-finite at frame {s['n_frames']} of "
                                  f"{n_expected}; partial score not comparable")
            published = PUBLISHED[material].get(scene)  # None => not tabulated above
            cell["published"] = published
            if cell["diverged"]:
                label = "diverged"
            elif published is not None:
                cell["margin_vs_published"] = published / s["mse"]
                label = f"{cell['margin_vs_published']:.1f}x vs published"
            else:
                label = "no published number for this scene"
            log(f"[fe-cross] {scene}/{leg}: MSE {s['mse']:.3e} ({label})")
            cells[leg] = cell
        rows[scene] = cells
    return rows


def run_material(material: str, device: str = "cpu", log=print) -> dict:
    t_material = time.time()
    arr = _load(material)
    res: dict = {"material": material, "source": f"{material}_dataset_truth.npz",
                 "tier": "full_channels", "flags": FLAGS[material], "device": device,
                 "identify": {}, "legs_rolled": [], "scenes": {}}
    legs: list[tuple[str, str, dict]] = []

    if material == "sand":
        fe, prior = load_granular_fe()
        t0 = time.time()
        ident = identify_friction_fe(arr, fe, prior, log=log)
        ident["wall_seconds"] = time.time() - t0
        res["identify"]["granular_fe"] = {
            k: ident.get(k) for k in
            ("refused", "reason", "residual_rel", "n_rows", "curve_error",
             "wall_seconds", "K")}
        if not ident.get("refused", True) and ident.get("baked_table"):
            b = ident["baked_table"]
            legs.append(("fe_mu_of_I", "sand_table",
                         {"eta_table": b["table"], "eta_table_smin": b["smin"],
                          "eta_table_smax": b["smax"]}))
            # control: a CONSTANT table at the true cone level through the same
            # tabulated material, separating curve error from the gap between
            # the tabulated return map and their Drucker-Prager integrator
            mu_true = suite.friction_to_mu(
                float(suite.MATERIALS["sand"]["truth"]["friction_angle"]))
            legs.append(("flat_table_truth_mu", "sand_table",
                         {"eta_table": [mu_true] * b["n_points"],
                          "eta_table_smin": b["smin"],
                          "eta_table_smax": b["smax"]}))
        t0 = time.time()
        ys = identify_yield_surface_fe(arr, log=log)
        ys["wall_seconds"] = time.time() - t0
        res["identify"]["yield_surface_fe"] = {
            k: ys.get(k) for k in
            ("refused", "reason", "fit_residual_rel", "n_bins_used", "h_at_p0",
             "dh_dp_median_on_support", "wall_seconds", "estimator")}
        if not ys.get("refused", True):
            tr_s = suite.MATERIALS["sand"]["truth"]
            legs.append(("fe_yield_surface", "yield_table",
                         _yield_surface_theta(ys, tr_s["E"], tr_s["nu"])))
            res["identify"]["yield_surface_fe"]["elastic_pair"] = (
                "fixed at the configured values (same as the known-form sand "
                "row)")
        t0 = time.time()
        binned = identify_friction_fe_binned(arr, fe, log=log)
        binned["wall_seconds"] = time.time() - t0
        res["identify"]["granular_fe_binned_cone"] = {
            k: binned.get(k) for k in
            ("refused", "reason", "n_bins_used", "fit_residual_rel",
             "bin_medians", "wall_seconds", "estimator")}
        if not binned.get("refused", True):
            bb = binned["baked_table"]
            legs.append(("fe_binned_cone", "sand_table",
                         {"eta_table": bb["table"], "eta_table_smin": bb["smin"],
                          "eta_table_smax": bb["smax"]}))
        res["identify"]["granular_fe_full"] = {
            k: v for k, v in ident.items() if k not in ("curve",)}

    elif material in ("jelly", "water"):
        from ident.weakform.elastic_grid import moduli_to_E_nu
        t0 = time.time()
        ident = identify_hyperelastic_fe(arr, log=log)
        ident["wall_seconds"] = time.time() - t0
        accepted = (not ident.get("refused", True)
                    and float(ident.get("residual_rel", 1.0)) <= RESIDUAL_BAR)
        res["identify"]["hyperelastic_fe"] = {
            k: ident.get(k) for k in
            ("refused", "residual_rel", "n_rows", "shear_modulus_from_curve",
             "shear_modulus_rel_err", "bulk_coefficient", "bulk_rel_err",
             "wall_seconds", "K")}
        res["identify"]["hyperelastic_fe"]["accepted"] = accepted
        if material == "jelly" and accepted:
            mu_h = float(ident["shear_modulus_from_curve"])
            lam_h = float(ident["bulk_coefficient"]) - 2.0 * mu_h / 3.0
            E_h, nu_h = moduli_to_E_nu(mu_h, lam_h)
            legs.append(("fe_projected_corotated", "jelly",
                         {"E": float(E_h), "nu": float(nu_h)}))
            res["identify"]["projection"] = {
                "note": ("the engine has no tabulated hyperelastic material; "
                         "the leg rolls the curve's small-strain corotated "
                         "pair, W1 -> mu/2 and volumetric -> lam + 2mu/3"),
                "E": float(E_h), "nu": float(nu_h)}
        if material == "water":
            # for a fluid the family's deviatoric part should vanish and the
            # volumetric column is exactly their law: lam is read directly
            lam_h = float(ident["bulk_coefficient"])
            res["identify"]["lam_from_volumetric_column"] = lam_h
            res["identify"]["lam_truth"] = 57692.0
            if accepted or float(ident.get("residual_rel", 1.0)) <= 0.5:
                legs.append(("fe_volumetric_eos", "water",
                             {"E": suite.lam_to_E(lam_h, 0.3), "nu": 0.3}))
            t0 = time.time()
            visc = identify_viscous_fe(
                arr, window_frames=VISCOUS_WINDOW["water"], log=log)
            visc["wall_seconds"] = time.time() - t0
            res["identify"]["viscous_fe"] = {
                k: visc.get(k) for k in
                ("refused", "reason", "residual_rel", "wall_seconds")}

    elif material == "plasticine":
        t0 = time.time()
        visc = identify_viscous_fe(
            arr, window_frames=VISCOUS_WINDOW["plasticine"], log=log)
        visc["wall_seconds"] = time.time() - t0
        res["identify"]["viscous_fe"] = {
            k: visc.get(k) for k in
            ("refused", "reason", "residual_rel", "wall_seconds")}
        t0 = time.time()
        hyper = identify_hyperelastic_fe(arr, log=log)
        hyper["wall_seconds"] = time.time() - t0
        res["identify"]["hyperelastic_fe"] = {
            k: hyper.get(k) for k in
            ("refused", "residual_rel", "n_rows", "shear_modulus_from_curve",
             "shear_modulus_rel_err", "bulk_coefficient", "wall_seconds")}
        res["identify"]["hyperelastic_fe"]["accepted"] = (
            not hyper.get("refused", True)
            and float(hyper.get("residual_rel", 1.0)) <= RESIDUAL_BAR)
        if res["identify"]["hyperelastic_fe"]["accepted"]:
            # the stored F is the elastic state, so the hyperelastic family
            # fits it cleanly at every frame; what no shipped basis covers is
            # the yield. Rolling the recovered elastic function WITHOUT a
            # yield cap measures that gap, and the leg name says so.
            from ident.weakform.elastic_grid import moduli_to_E_nu
            mu_h = float(hyper["shear_modulus_from_curve"])
            lam_h = float(hyper["bulk_coefficient"]) - 2.0 * mu_h / 3.0
            E_h, nu_h = moduli_to_E_nu(mu_h, lam_h)
            legs.append(("fe_elastic_only_no_yield", "plasticine",
                         {"E": float(E_h), "nu": float(nu_h),
                          "yield_stress": 1.0e9}))
            res["identify"]["projection"] = {"E": float(E_h), "nu": float(nu_h)}
            # the full unknown-form law: FE elastic pair plus the learned
            # yield surface h(p) through the tabulated-yield material
            t0 = time.time()
            ys = identify_yield_surface_fe(arr, log=log)
            ys["wall_seconds"] = time.time() - t0
            res["identify"]["yield_surface_fe"] = {
                k: ys.get(k) for k in
                ("refused", "reason", "fit_residual_rel", "n_bins_used",
                 "h_at_p0", "dh_dp_median_on_support", "wall_seconds",
                 "estimator")}
            if not ys.get("refused", True):
                legs.append(("fe_yield_surface", "yield_table",
                             _yield_surface_theta(ys, E_h, nu_h)))
        res["note"] = ("the hyperelastic family recovers the elastic function "
                       "because the stored F is the elastic state; no trained "
                       "plasticity basis is wired into this campaign, so the "
                       "elastic-only rollout leg measures the cost of the "
                       "missing yield")

    res["legs_rolled"] = [leg for leg, _, _ in legs]
    t_rollout = time.time()
    if legs:
        res["scenes"] = _rollout_legs(material, legs, device=device, log=log)
    wall_rollout_s = time.time() - t_rollout

    # identify_total_s: every identify_*_fe wall_seconds this material ran,
    # sand's binned-cone and yield-surface legs included.
    identify_total_s = sum(v.get("wall_seconds", 0.0) or 0.0
                           for v in res["identify"].values() if isinstance(v, dict))
    eval_sim_timings = [cell["sim_timing"] for cells in res["scenes"].values()
                       for cell in cells.values() if cell.get("sim_timing")]
    wall_simulate_eval_s = sum(t.get("total_s", 0.0) for t in eval_sim_timings)
    simulate_eval_detail_s = {
        k: sum(t.get(k, 0.0) for t in eval_sim_timings)
        for k in ("setup_s", "step_s", "snapshot_s", "finalize_s")}
    res["timing_breakdown_s"] = {
        "identify_total_s": identify_total_s,
        "rollout_wall_s": wall_rollout_s,          # includes cache checks/scoring
        "simulate_eval_s": wall_simulate_eval_s,   # sum of fresh rollouts only
        "simulate_eval_detail_s": simulate_eval_detail_s,
        "material_total_s": time.time() - t_material,
    }
    log(f"[fe-cross] {material} timing: identify {identify_total_s:.1f}s, "
        f"simulate {wall_simulate_eval_s:.1f}s {simulate_eval_detail_s}, "
        f"total {res['timing_breakdown_s']['material_total_s']:.1f}s")
    return res


def main(argv: list[str] | None = None) -> None:
    raw = argv if argv is not None else sys.argv[1:]
    device = next((a.split("=", 1)[1] for a in raw if a.startswith("--device=")),
                  "cpu")
    mats = [a for a in raw if not a.startswith("--device=")] or \
        ["sand", "jelly", "water", "plasticine"]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    print(f"[fe-cross] device={device}")
    t0 = time.time()
    for m in mats:
        results[m] = run_material(m, device=device)
        path.write_text(json.dumps(results, indent=2, default=float))
        print(f"[fe-cross] {m} recorded -> {path} "
              f"({results[m]['timing_breakdown_s']['material_total_s']:.1f}s)")
    print(f"[fe-cross] {len(mats)} material(s) in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()

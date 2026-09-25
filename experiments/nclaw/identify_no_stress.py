"""Identification without the stress channel, on NCLaw's own trajectories.

The full-channel path identifies with every stored channel. This module runs
the same identification with the stress channel excluded: the stored kinematics
(x, v, L, F, volume, mass) are used exactly as before, and anything the
full-channel path read off the stress tensor is either replaced by a stated
model or refused.

What each material loses, measured on their dumps:

jelly, water
    nothing. The elastic momentum fit reads x, v, F, volume and mass, and the
    volumetric fit reads J from F, so neither ever touched the stress channel.
    Both are rerun at this tier to confirm that.
plasticine
    nothing in the elastic pair, and nothing in the strain-cap reading of the
    yield stress either, which takes the saturation of the deviatoric Hencky
    strain of the stored elastic F. This module adds the estimator that skips
    the strain cap: the momentum fit with one yield column on the
    fast-shearing set, reported next to it.
sand
    the pressure. The friction fit needs pressure as data, and the
    full-channel path read it from the 3D stress trace. Three sources are
    offered here and all three are reported:

    hencky_F      the Hencky volumetric relation on the stored elastic F at the
                  configured E and nu. This is how NCLaw's own
                  DruckerPragerPlasticity gets pressure (preset.py: E and nu
                  are held fixed and the friction angle is the fitted
                  parameter), so it is both stress-free and like-for-like with
                  their model. Their stress channel is elasticity(F) of the
                  same F. This source should reproduce the full-channel
                  pressure; the agreement is measured and reported.
    basal_scaled  pressure measured within one grid cell of the floor, which is
                  what a force plate under the pile gives, with the
                  depth-below-surface shape scaled to match that basal level.
                  This is the one variant that reads the stress channel, and it
                  reads it only inside that band.
    depth         the pure closure, density times gravity times depth below the
                  per-column free surface, from positions alone.

The deviatoric stress the yield-set check and the cone-level estimator need is
reconstructed from F for the hencky_F source, by the same relation and the same
fixed elastic pair. The two closure sources supply pressure only, so their fits
select the yield set on kinematics alone and have no cone level to fall back
on.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# lower bound on singular values in the log; their preset.py clamps at 0.05,
# this bound only guards the log
SIG_FLOOR = 1.0e-6
PRESSURE_SOURCES = ("hencky_F", "basal_scaled", "depth")

# A variant leg carries the value its own estimator produced, refusal or not.
_REFUSED_LEG_NOTE = (
    "this estimator refused; the leg rolls out the refused value to measure "
    "its cost. A refusal falls back to the known-class prior, which here "
    "equals the truth row.")


# ---------------------------------------------------------------------------
# Stress-free pressure from the stored deformation gradient
# ---------------------------------------------------------------------------

def hencky_stress_parts(F: np.ndarray, E: float, nu: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cauchy pressure, deviatoric Cauchy stress and J from F at fixed (E, nu).

    The engine's ``kirchoff_stress_drucker_prager`` returns
    U center V^T F^T with center_ii = (2 mu log sig_i + lam sum log sig) / sig_i,
    which multiplies out to the Kirchhoff stress

        tau = U diag(2 mu eps_i + lam tr eps) U^T,   eps = log sig(F),

    the same Hencky form NCLaw's SigmaElasticity uses. Hence

        tr tau = (2 mu + 3 lam) tr eps,   dev tau = 2 mu U diag(eps_hat) U^T,

    and Cauchy is tau / J. Only the elastic pair enters, and for their sand it
    is a configured constant.
    """
    from ident.weakform.elastic_grid import E_nu_to_moduli
    mu, lam = E_nu_to_moduli(float(E), float(nu))
    U, sig, _ = np.linalg.svd(np.asarray(F, dtype=float))
    eps = np.log(np.clip(np.abs(sig), SIG_FLOOR, None))
    tr = eps.sum(axis=-1)
    J = np.prod(np.clip(np.abs(sig), SIG_FLOOR, None), axis=-1)
    p = -(2.0 * mu + 3.0 * lam) * tr / (3.0 * J)
    eps_hat = eps - (tr / 3.0)[..., None]
    dev = (2.0 * mu / J)[..., None, None] * np.einsum(
        "...ij,...j,...kj->...ik", U, eps_hat, U)
    return p, dev, J


def column_surface_pressure(x: np.ndarray, rho: float, g_z: float, cell: float
                            ) -> np.ndarray:
    """rho |g| (h - z) with h the free surface of the particle column at (x, y).

    The hydrostatic depth closure of the granular-collapse experiments in this
    repository, here on the evolving cloud: bin the particles by cell in the
    two horizontal directions, take the
    highest particle of each bin as that column's surface, and read the depth
    below it. Positions alone; no stress and no deformation gradient.
    """
    x = np.asarray(x, dtype=float)
    T, P = x.shape[0], x.shape[1]
    p = np.zeros((T, P))
    gz = abs(float(g_z))
    for f in range(T):
        ix = np.floor(x[f, :, 0] / cell).astype(np.int64)
        iy = np.floor(x[f, :, 1] / cell).astype(np.int64)
        ix -= ix.min()
        iy -= iy.min()
        bid = ix * (iy.max() + 1) + iy
        surf = np.full(int(bid.max()) + 1, -np.inf)
        np.maximum.at(surf, bid, x[f, :, 2])
        p[f] = rho * gz * np.maximum(surf[bid] - x[f, :, 2], 0.0)
    return p


def basal_scaled_pressure(x: np.ndarray, p_measured: np.ndarray, floor_z: float,
                          cell: float, rho: float, g_z: float
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """The depth shape scaled per frame to the pressure measured at the base.

    A force plate under the pile measures pressure where the material touches
    it and nowhere else. ``p_measured`` is therefore read only inside one cell
    of the floor; the shape of the field above the band is the depth closure,
    and one scalar per frame lifts that shape onto the measured basal level. The
    scale is a ratio of medians over the band; a frame with too few band
    particles falls back to the median scale over frames with enough band
    particles.
    """
    x = np.asarray(x, dtype=float)
    p_depth = column_surface_pressure(x, rho, g_z, cell)
    band = x[:, :, 2] <= floor_z + cell
    ok = band & np.isfinite(p_measured) & (p_measured > 0.0) & (p_depth > 0.0)
    scale = np.full(x.shape[0], np.nan)
    for f in range(x.shape[0]):
        sel = ok[f]
        if int(sel.sum()) >= 20:
            scale[f] = (np.median(p_measured[f][sel])
                        / max(np.median(p_depth[f][sel]), 1e-30))
    good = np.isfinite(scale)
    fallback = float(np.median(scale[good])) if good.any() else 1.0
    scale_used = np.where(good, scale, fallback)
    diag = {
        "band_cells": 1.0, "floor_z": float(floor_z),
        "n_frames_with_band_measurement": int(good.sum()),
        "n_frames": int(x.shape[0]),
        "scale_median": float(np.median(scale_used)),
        "scale_p05": float(np.percentile(scale_used, 5)),
        "scale_p95": float(np.percentile(scale_used, 95)),
        "band_particles_median": float(np.median(ok.sum(axis=1))),
        "fallback_scale": fallback,
    }
    return scale_used[:, None] * p_depth, diag


def pressure_agreement(p_model: np.ndarray, p_true: np.ndarray,
                       weight: np.ndarray | None = None) -> dict[str, float]:
    """How a pressure model compares with the stored stress trace. Diagnosis only.

    Records the closure's bias as a number. No fit at this tier reads it.
    """
    m = np.asarray(p_model, dtype=float)
    t = np.asarray(p_true, dtype=float)
    sel = np.isfinite(m) & np.isfinite(t) & (t > 0.0)
    if weight is not None:
        sel &= np.asarray(weight, dtype=bool)
    if not sel.any():
        return {"n": 0}
    ratio = m[sel] / t[sel]
    return {
        "n": int(sel.sum()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p05": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "rel_err_median": float(np.median(np.abs(ratio - 1.0))),
        "p_model_median": float(np.median(m[sel])),
        "p_true_median": float(np.median(t[sel])),
    }


# ---------------------------------------------------------------------------
# Yield stress from the momentum balance, without the strain cap
# ---------------------------------------------------------------------------

def hencky_yield_parts(F: np.ndarray, mu: float, lam: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The yield column of a von Mises solid: direction, volumetric part, norms.

    Returns N, the unit-norm flow direction U diag(eps_hat) U^T / ||eps_hat|| as
    a Cauchy-stress direction; sigma_vol, the mean Cauchy stress
    (2 mu / 3 + lam) tr(eps) / J that the elastic pair fixes as data; the norm
    ||eps_hat||; and J. With y the yield stress, the Cauchy stress of a particle
    at yield is sigma = sigma_vol I + y N / J, which is what makes y the single
    unknown of a linear momentum fit.
    """
    U, sig, _ = np.linalg.svd(np.asarray(F, dtype=float))
    s = np.clip(np.abs(sig), SIG_FLOOR, None)
    eps = np.log(s)
    tr = eps.sum(axis=-1)
    J = np.prod(s, axis=-1)
    eps_hat = eps - (tr / 3.0)[..., None]
    hat_norm = np.linalg.norm(eps_hat, axis=-1)
    N = np.einsum("...ij,...j,...kj->...ik", U,
                  eps_hat / np.maximum(hat_norm, 1e-30)[..., None], U)
    sigma_vol = (2.0 * mu / 3.0 + lam) * tr / J
    return N, sigma_vol, hat_norm, J


def identify_yield_momentum(arr: dict, mu_hat: float, lam_hat: float,
                            window_frames: int = 26, frame_stride: int = 2,
                            margin_cells: float = 3.0, gd_min: float = 1.0,
                            eps_gamma: float = 0.02, flow_frac_min: float = 0.9,
                            residual_bar: float = 0.15, log=print) -> dict:
    """von Mises yield stress as the one unknown of a momentum fit.

    At yield the return map holds the deviatoric Kirchhoff stress at norm
    ``yield_stress`` and leaves its direction alone, so with the elastic pair
    already recovered the stress of a flowing particle is

        tau = (2 mu / 3 + lam) tr(eps) I + y N,
        N = U diag(eps_hat) U^T / ||eps_hat||,   eps = log sig(F),

    one unknown y multiplying a known direction, with the volumetric part as
    data. The set that enters is the kinematically flowing one, particles whose
    equivalent shear rate clears ``gd_min``: a particle below yield carries
    elastic deviatoric stress this model does not describe, so it invalidates
    its nodes.
    """
    from common.conventions import equivalent_shear_rate, sym
    from experiments.nclaw.suite import _longest_run, wall_planes
    from ident.weakform.elastic_grid import assemble_columns_timeweak, solve_elastic_grid

    N, p_vol, hat_norm, J = hencky_yield_parts(arr["F"], mu_hat, lam_hat)
    gd = equivalent_shear_rate(sym(arr["L"]), eps_gamma)
    eye = np.eye(3)[None]

    def columns_fn(f: int):
        finite = np.isfinite(N[f]).all(axis=(1, 2)) & np.isfinite(p_vol[f])
        flowing = finite & (gd[f] > gd_min) & (hat_norm[f] > 1e-9)
        if int(flowing.sum()) < 50:
            return None
        Vp = arr["volume"][f]
        w = np.where(flowing, Vp / np.maximum(J[f], 1e-30), 0.0)
        Vsig = w[:, None, None, None] * N[f][:, None, :, :]
        Vsig_known = np.where(flowing, Vp * p_vol[f], 0.0)[:, None, None] * eye
        return Vsig, Vsig_known, flowing, hat_norm[f]

    all_frames = list(range(0, arr["x"].shape[0], frame_stride))
    counts = [int((gd[f] > gd_min).sum()) for f in all_frames]
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
        collider_margin_cells=margin_cells, valid_frac_min=flow_frac_min)
    wall_assemble_s = time.time() - t_asm
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "wall_assemble_s": wall_assemble_s, "wall_fit_s": 0.0}
    t_fit = time.time()
    out = solve_elastic_grid(sysm)
    out["wall_assemble_s"], out["wall_fit_s"] = wall_assemble_s, time.time() - t_fit
    y = float(out["theta"][0])
    out.update({"yield_stress": y, "estimator": "momentum_yield_column",
                "refused": False, "gd_min": gd_min,
                "flow_frac_min": flow_frac_min,
                "n_flowing_frames": len(frames),
                "row_survival": sysm.row_survival,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "residual_bar": residual_bar,
                "hencky_dev_coverage": list(sysm.strain_coverage)})
    if not np.isfinite(y) or y <= 0.0 or float(out["residual_rel"]) > residual_bar:
        out.update({"refused": True,
                    "reason": (f"yield column fit y={y:.4e} at relative residual "
                               f"{float(out['residual_rel']):.3f} against a bar of "
                               f"{residual_bar}")})
    log(f"[ident] yield-momentum y={y:.4e} rows={sysm.n_rows} "
        f"resid={float(out['residual_rel']):.3f} refused={out['refused']}")
    return out


# ---------------------------------------------------------------------------
# The tier's identify stage
# ---------------------------------------------------------------------------

def _sand_pressures(arr: dict, material_truth: dict, basal_dump: Path | None,
                    log=print) -> dict[str, dict]:
    """The three pressure fields for the friction fit, each with its provenance."""
    from common.conventions import pressure_from_cauchy_3d_trace
    cell = arr["grid_lim"] / arr["n_grid"]
    E, nu = float(material_truth["E"]), float(material_truth["nu"])
    p_F, dev_F, _ = hencky_stress_parts(arr["F"], E, nu)
    out: dict[str, dict] = {
        "hencky_F": {
            "p": p_F, "dev": dev_F,
            "provenance": ("stored elastic F through the Hencky volumetric and "
                           f"deviatoric relation at the configured E={E:g}, nu={nu:g}; "
                           "no stress channel read"),
            "reads_stress": False,
        },
    }
    p_depth = column_surface_pressure(arr["x"], arr["meta"].rho_bulk, arr["g"][2], cell)
    out["depth"] = {
        "p": p_depth, "dev": None,
        "provenance": ("rho g times depth below the per-column free surface of the "
                       "position cloud, one cell columns; positions alone"),
        "reads_stress": False,
    }
    if basal_dump is not None:
        d = np.load(basal_dump)
        T, P = arr["x"].shape[0], arr["x"].shape[1]
        stress = d["stress"].astype(np.float64).reshape(T, P, 3, 3)
        p_true = pressure_from_cauchy_3d_trace(stress)
        floor_z = float(arr["meta"].extra.get("bound_cells", 3)) * cell
        p_basal, diag = basal_scaled_pressure(
            arr["x"], p_true, floor_z, cell, arr["meta"].rho_bulk, arr["g"][2])
        out["basal_scaled"] = {
            "p": p_basal, "dev": None, "diagnostics": diag,
            "provenance": ("stress trace read ONLY within one grid cell of the floor "
                           f"(z <= {floor_z:.4g} m), the depth-below-surface shape "
                           "scaled per frame to that basal level"),
            "reads_stress": True,
        }
        # diagnosis only: how each pressure model compares with the stored trace
        for key, entry in out.items():
            entry["agreement_with_stress_trace"] = pressure_agreement(
                entry["p"], p_true)
            log(f"[ident] pressure {key}: median ratio to the stress trace "
                f"{entry['agreement_with_stress_trace'].get('ratio_median')}")
    return out


def stage_identify_no_stress(material: str, dump: str | Path,
                             tag: str | None = None, nclaw_law: bool = False,
                             window_frames: int | None = None,
                             basal_dump: str | Path | None = None,
                             nclaw_bc: bool = False,
                             substeps: int | None = None,
                             device: str = "cpu",
                             log=print) -> dict:
    """Identify from one trajectory with the stress channel excluded.

    Returns the same shape ``suite.stage_identify`` returns, so a runner can
    swap one for the other, plus ``theta_variants``: the extra estimators this
    tier makes available (sand's pressure sources, plasticine's momentum yield
    column), each already in the engine's own arguments so a rollout can be run
    per variant.
    """
    from experiments.nclaw import suite

    dump = Path(dump)
    if window_frames is None:
        window_frames = suite.WINDOW_FRAMES[material]
    arr = suite._load_arrays(dump)
    if arr["meta"].has_pressure:
        raise ValueError(
            f"{dump.name} still carries an oracle pressure "
            f"(pressure_source={arr['meta'].pressure_source!r}); run this on the "
            "tier dump written by experiments/nclaw/strip_channels.py so a leg "
            "that reads the stress trace cannot do so by accident")
    truth = suite.MATERIALS[material]["truth"]
    ident: dict[str, Any] = {
        "tier": arr["meta"].extra.get("tier", "no_stress"),
        "source_dump": dump.name,
        "n_grid": arr["n_grid"],
        "channel_provenance": arr["meta"].extra.get("channel_provenance", {}),
        "pressure_source_of_dump": arr["meta"].pressure_source,
    }
    variants: dict[str, dict] = {}
    walls: dict[str, float] = {}
    scan_results: list[dict] = []   # every scan_parameter() call's own dict, for timing

    tier = arr["meta"].extra.get("tier", "no_stress")
    t0 = time.time()
    if tier == "positions_only" and material in ("plasticine", "sand"):
        # Stored F is elastic only; positions give total deformation, and the
        # two diverge under flow. The replay estimators run and typically
        # refuse (reconstruction errors trip the residual check); the reported
        # parameter comes from the rollout scan with the elastic pair assumed
        # and stated.
        from experiments.nclaw.replay import (
            identify_friction_replay,
            identify_yield_replay,
        )
        from experiments.nclaw.rollout_scan import scan_parameter
        E_a, nu_a = float(truth["E"]), float(truth["nu"])
        mu_a = E_a / (2.0 * (1.0 + nu_a))
        lam_a = E_a * nu_a / ((1.0 + nu_a) * (1.0 - 2.0 * nu_a))
        assumed_note = ("elastic pair assumed at the configured value at this "
                        "tier: the throw loads it only in a short impact window "
                        "and the reconstructed state there biases the fit "
                        "(mu +40 percent on plasticine)")
        ident["assumed_parameters"] = ["E", "nu"]
        if material == "plasticine":
            ident["elastic"] = {"refused": True, "reason": assumed_note}
            t1 = time.time()
            ident["yield_replay"] = identify_yield_replay(
                arr, mu_a, lam_a, log=log)
            walls["yield_replay_s"] = time.time() - t1
            t1 = time.time()
            ident["yield"] = scan_parameter(
                material, dump, "yield_stress",
                coarse=[1000, 2000, 4000, 8000, 16000, 32000],
                theta_base={"E": E_a, "nu": nu_a}, mode="mul",
                refine_rounds=[[0.6, 0.8, 1.25, 1.6], [0.9, 1.1]],
                nclaw_bc=nclaw_bc, nclaw_law=nclaw_law, substeps=substeps,
                device=device, log=log)
            walls["yield_scan_s"] = time.time() - t1
            scan_results.append(ident["yield"])
            yr = ident["yield_replay"]
            if yr.get("yield_stress") is not None:
                base, _, _ = suite.theta_for_engine(material, ident, nclaw_law=nclaw_law)
                variants["replay"] = {
                    "theta": {**base, "yield_stress": yr["yield_stress"]},
                    "refused": bool(yr.get("refused", True)),
                    "note": _REFUSED_LEG_NOTE if yr.get("refused", True) else "",
                    "provenance": ("weak-form momentum fit on the elastic state "
                                   "reconstructed by identify_yield_replay from "
                                   "positions alone, elastic pair assumed; "
                                   "reported despite refusal to measure its "
                                   "rollout cost against the scan"),
                    "diagnostics": {k: yr.get(k) for k in
                                    ("yield_stress", "residual_rel", "refused",
                                     "reason", "elastic_pair_used")},
                }
        else:
            t1 = time.time()
            ident["friction_replay"] = identify_friction_replay(
                arr, E=E_a, nu=nu_a, log=log)
            walls["friction_replay_s"] = time.time() - t1
            t1 = time.time()
            ident["friction"] = scan_parameter(
                material, dump, "friction_angle",
                coarse=[15, 20, 25, 30, 35, 40],
                theta_base={"E": E_a, "nu": nu_a}, mode="add",
                refine_rounds=[[-3, -2, -1, 1, 2, 3], [-0.5, 0.5]],
                nclaw_bc=nclaw_bc, nclaw_law=nclaw_law, substeps=substeps,
                device=device, log=log)
            walls["friction_scan_s"] = time.time() - t1
            scan_results.append(ident["friction"])
            fr = ident["friction_replay"]
            if fr.get("friction_angle") is not None:
                base, _, _ = suite.theta_for_engine(material, ident, nclaw_law=nclaw_law)
                variants["replay"] = {
                    "theta": {**base, "friction_angle": fr["friction_angle"]},
                    "refused": bool(fr.get("refused", True)),
                    "note": _REFUSED_LEG_NOTE if fr.get("refused", True) else "",
                    "provenance": ("weak-form momentum fit on the elastic state "
                                   "reconstructed by identify_friction_replay from "
                                   "positions alone, elastic pair assumed; "
                                   "reported despite refusal to measure its "
                                   "rollout cost against the scan"),
                    "diagnostics": {k: fr.get(k) for k in
                                    ("friction_angle", "alpha", "residual_rel",
                                     "refused", "reason")},
                }
    elif material in ("jelly", "plasticine"):
        ident["elastic"] = suite.identify_elastic(
            arr, window_frames=window_frames,
            columns="hencky" if material == "plasticine" else "corotated",
            log=log)
        if material == "plasticine" and not ident["elastic"].get("refused", True):
            mu_hat, lam_hat = ident["elastic"]["mu"], ident["elastic"]["lam"]
            ident["yield"] = suite.identify_yield(arr, mu_hat, log=log)
            t1 = time.time()
            ident["yield_momentum"] = identify_yield_momentum(
                arr, mu_hat, lam_hat, window_frames=window_frames, log=log)
            walls["yield_momentum_s"] = time.time() - t1
            ym = ident["yield_momentum"]
            base, _, _ = suite.theta_for_engine(material, ident, nclaw_law=nclaw_law)
            variants["yield_momentum"] = {
                "theta": {**base, "yield_stress": ym.get("yield_stress")},
                "refused": bool(ym.get("refused", True)),
                "note": _REFUSED_LEG_NOTE if ym.get("refused", True) else "",
                "provenance": ("elastic pair from the momentum fit, yield stress from "
                               "the momentum fit with one yield column on the "
                               "kinematically flowing set"),
                "diagnostics": {k: ym.get(k) for k in
                                ("yield_stress", "n_rows", "residual_rel", "cond_AtA",
                                 "row_survival", "refused", "reason")},
            }
    elif material == "sand":
        t1 = time.time()
        fields = _sand_pressures(arr, truth, Path(basal_dump) if basal_dump else None,
                                 log=log)
        walls["sand_pressure_reconstruct_s"] = time.time() - t1
        for key in PRESSURE_SOURCES:
            if key not in fields:
                continue
            entry = fields[key]
            t1 = time.time()
            fr = suite.identify_friction(
                arr, window_frames=window_frames, pressure=entry["p"],
                dev_stress=entry["dev"], pressure_label=key, log=log)
            walls[f"friction_{key}_s"] = time.time() - t1
            fr["pressure_provenance"] = entry["provenance"]
            fr["pressure_reads_stress"] = entry["reads_stress"]
            for extra in ("diagnostics", "agreement_with_stress_trace"):
                if extra in entry:
                    fr[f"pressure_{extra}"] = entry[extra]
            if key == "hencky_F":
                ident["friction"] = fr
            else:
                ident[f"friction_{key}"] = fr
                base, _, _ = suite.theta_for_engine(material, {}, nclaw_law=nclaw_law)
                refused = bool(fr.get("refused", True))
                angle = (fr.get("friction_angle_solve") if refused
                         else fr.get("friction_angle"))
                variants[key] = {
                    "theta": {**base, "friction_angle": angle},
                    "refused": refused,
                    "note": _REFUSED_LEG_NOTE if refused else "",
                    "provenance": entry["provenance"],
                    "diagnostics": {k: fr.get(k) for k in
                                    ("friction_angle", "friction_angle_solve",
                                     "mu_c", "mu_c_solve", "n_rows", "residual_rel",
                                     "cond_AtA", "refused", "reason")},
                }
    elif material == "water":
        ident["eos"] = suite.identify_eos(
            arr, window_frames=window_frames,
            form="linear" if nclaw_law else "power_law", log=log)
        if tier == "positions_only":
            # The weak-form fit does not refuse, so it stays primary; the scan
            # runs as a variant leg to measure what the rollout objective
            # recovers.
            from experiments.nclaw.rollout_scan import scan_parameter
            nu_w = float(truth["nu"])
            t1 = time.time()
            sc = scan_parameter(
                material, dump, "E",
                coarse=[40000, 60000, 80000, 100000, 120000, 140000],
                theta_base={"nu": nu_w}, mode="mul",
                refine_rounds=[[0.8, 0.9, 1.1, 1.25], [0.95, 1.05]],
                nclaw_bc=nclaw_bc, nclaw_law=nclaw_law, substeps=substeps,
                device=device, log=log)
            walls["eos_scan_s"] = time.time() - t1
            scan_results.append(sc)
            variants["rollout_scan"] = {
                "theta": {"E": sc["E"], "nu": nu_w},
                "refused": False,
                "note": "",
                "provenance": ("stiffness by the rollout scan on the identify "
                               "trajectory; nu carried at the configured value "
                               "(their law reads lam alone)"),
                "diagnostics": {k: sc.get(k) for k in
                                ("E", "mse_at_best", "n_rollouts", "scan")},
            }
    walls["identify_total_s"] = time.time() - t0

    # Standardized breakdown across every tier: field reconstruction (MLS/FD
    # state rebuilt from positions, or sand's pressure closures rebuilt from
    # F), equation assembly, the linear/closed-form fit itself (including any
    # weak-form attempt that ultimately refuses), and simulator rollouts run
    # as part of identification (the derivative-free scans; NOT the later
    # scoring rollouts compare.py runs separately and never counts here).
    fit_dict_keys = ("elastic", "yield", "yield_replay", "yield_momentum",
                     "friction", "friction_replay", "eos")
    sub_dicts = [ident[k] for k in fit_dict_keys if isinstance(ident.get(k), dict)]
    sub_dicts += [v for k, v in ident.items()
                 if k.startswith("friction_") and k not in fit_dict_keys
                 and isinstance(v, dict)]
    reconstruct_s = (walls.get("sand_pressure_reconstruct_s", 0.0)
                     + sum(d.get("wall_reconstruct_s", 0.0) for d in sub_dicts))
    simulate_s = sum(v for k, v in walls.items() if k.endswith("_scan_s"))
    # every candidate rollout the scan(s) actually simulated (not the ones
    # reused from out/nclaw_suite/scan/*.npz), split into setup/step/
    # snapshot/finalize; scan_parameter runs these on run_scene's default
    # device ("cpu"), not whatever --device the rest of this comparison uses.
    simulate_search_detail_s = {
        k: sum(r.get("wall_simulate_detail_s", {}).get(k, 0.0) for r in scan_results)
        for k in ("setup_s", "step_s", "snapshot_s", "finalize_s")}
    ident["timing_breakdown_s"] = {
        "reconstruct_fields_s": reconstruct_s,
        "assemble_equations_s": sum(d.get("wall_assemble_s", 0.0) for d in sub_dicts),
        "fit_parameters_s": sum(d.get("wall_fit_s", 0.0) for d in sub_dicts),
        "simulate_s": simulate_s,
        "simulate_search_detail_s": simulate_search_detail_s,
        "simulate_search_device": device if scan_results else None,
        "identify_total_s": walls["identify_total_s"],
    }

    theta, refused, param_estimator = suite.theta_for_engine(
        material, ident, nclaw_law=nclaw_law)
    ident.update({
        "theta_engine": theta,
        "refused_parameters": refused,
        "parameter_estimator": param_estimator,
        "theta_variants": variants,
        "wall_times_s": walls,
        "truth": truth,
        "nclaw_law": suite.MATERIALS[material].get("nclaw_law") if nclaw_law else None,
    })
    suite.OUT.mkdir(parents=True, exist_ok=True)
    name = (f"identify_no_stress_{material}.json" if tag is None
            else f"identify_no_stress_{material}_{tag}.json")
    (suite.OUT / name).write_text(json.dumps(ident, indent=2, default=float))
    log(f"[ident] no-stress theta={theta} refused={refused} "
        f"variants={sorted(variants)} in {walls['identify_total_s']:.1f}s")
    return ident

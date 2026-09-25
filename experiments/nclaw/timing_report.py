"""Timing / loss / parameter-error table across every (material, tier) run.

Reads the ``compare_<material><TAGS[material]><TIER_SUFFIX[tier]>.json`` files
``compare.py`` writes (now carrying a ``timing_breakdown_s`` block and a
per-scene ``sim_wall_s`` on every rollout cell) and renders three markdown
tables:

1. **Timing** -- wall time split into reconstructing fields (MLS/finite
   difference state rebuild, or sand's pressure closures), assembling the
   weak-form equations, fitting the parameters (the linear solve, including
   any weak-form attempt that ends in refusal), simulate-search (rollouts run
   INSIDE identification by the derivative-free scans for sand/plasticine
   positions-only and water's scan variant), and simulate-eval (the later
   rollouts used to score the comparison table, timed separately and never
   counted in identification time). The five components sum to
   ``identify_total_s`` plus ``simulate_eval_s``.
2. **Loss** -- per-scene position MSE (dataset, time, vel mean, shape) plus
   "Mean Loss", defined HERE as the equally-weighted mean of {time, vel mean,
   shape} -- NCLaw's own time/velocity/geometry axes -- excluding the
   dataset/reconstruction scene. This aggregation is not computed anywhere
   else in this repo; the formula is stated explicitly so it can be checked
   against whatever external convention it is being compared to.
3. **Parameter recovery error** -- recovered vs. truth, percent error per
   parameter actually fit (assumed/refused parameters are marked, not scored).

A cell's ``sim_wall_s`` is ``None`` when compare.py reused a cached rollout
npz rather than re-simulating; such runs under-report simulate-eval time.
Delete the relevant npz files before rerunning compare.py if you need a
from-scratch timing measurement (see run_euclid_compare.sh).

Run:  .venv/bin/python -m experiments.nclaw.timing_report [--out out/nclaw_cross_compare/timing_report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.nclaw.no_stress_table import SCENE_ROLE, TAGS, TIER_SUFFIX, _role  # noqa: E402

OUT = ROOT / "out" / "nclaw_cross_compare"
MATERIALS = ["jelly", "plasticine", "sand", "water"]
TIERS = ["full", "nostress", "positionsonly"]
TIMING_COLS = ["reconstruct_fields_s", "assemble_equations_s", "fit_parameters_s",
              "simulate_search_s", "simulate_eval_s"]


def _load(material: str, tier: str) -> dict | None:
    path = OUT / f"compare_{material}{TAGS[material]}{TIER_SUFFIX[tier]}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _mean_loss(res: dict) -> tuple[dict[str, float | None], float | None]:
    """Per-role recovered MSE, and the equally-weighted mean of {time, vel, shape}."""
    by_role: dict[str, list[float]] = {}
    for scene, cells in res["scenes"].items():
        role = _role(scene)
        by_role.setdefault(role, []).append(cells["recovered"]["mse"])
    per_axis = {role: float(np.mean(v)) for role, v in by_role.items()}
    axes = [per_axis[r] for r in ("time", "vel", "shape") if r in per_axis]
    mean_loss = float(np.mean(axes)) if len(axes) == 3 else None
    return per_axis, mean_loss


def _param_rows(res: dict) -> list[tuple[str, float, float, float | None, str]]:
    """(name, truth, recovered, percent error, estimator) per reported parameter.

    Includes assumed/refused parameters too (recovered == truth there by
    construction, so error comes out ~0 -- that equality is itself the
    signal that the parameter was never actually fit, not estimated).
    ``estimator`` comes from ``theta_for_engine`` (suite.py), computed fresh
    each run from what the code actually did -- not a hardcoded guess:
    "assumed" (never attempted, or attempted and refused), "weak_form" (the
    convex momentum-balance solve), "closed_form_strain_cap" (plasticine's
    yield-plateau read), "cone_level_reading" (sand's mode-of-ratio fallback
    when the weak-form residual is too high but a cone level exists), or
    "rollout_scan" (the derivative-free simulator search).
    """
    truth, recovered = res["theta_truth"], res["theta_recovered"]
    estimator = res.get("identify_diagnostics", {}).get("parameter_estimator") or {}
    rows = []
    for k, t in truth.items():
        if k not in recovered:
            continue
        r = recovered[k]
        err = 100.0 * (r - t) / t if t else None
        rows.append((k, t, r, err, estimator.get(k, "?")))
    return rows


# (material, tier) pairs where the rollout scan actually runs (jelly never
# scans at any tier; sand/plasticine/water scan only at positions-only).
SCAN_BEARING = {("plasticine", "positionsonly"), ("sand", "positionsonly"),
                ("water", "positionsonly")}


def _fmt(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def _fmt_s(x, applicable=True):
    """Scientific-notation seconds, or '-' when the component has no code
    path for this combination (not just a small measured value)."""
    if not applicable or x is None:
        return "-"
    return f"{x:.2e}"


def render() -> str:
    lines = []

    lines += ["## Timing breakdown (seconds, scientific notation; '-' = no "
             "code path for this combination, not just a small value)", "",
             "| material | tier | reconstruct | assemble | fit | "
             "simulate (search) | search device | simulate (eval) | eval device | "
             "identify total | identify+eval total |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for material in MATERIALS:
        for tier in TIERS:
            res = _load(material, tier)
            if res is None:
                lines.append(f"| {material} | {tier} | (missing) | - | - | - | - | - | - | - | - |")
                continue
            tb = res.get("timing_breakdown_s") or {}
            recon, asm, fit, sim_s, sim_e = (tb.get(k, 0.0) for k in TIMING_COLS)
            total = tb.get("identify_total_s", res.get("wall_identify_s", 0.0))
            scan_here = (material, tier) in SCAN_BEARING
            search_dev = tb.get("simulate_search_device") if scan_here else None
            eval_dev = tb.get("simulate_eval_device") or res.get("device", "?")
            lines.append(
                f"| {material} | {tier} | "
                f"{_fmt_s(recon, tier != 'full')} | {_fmt_s(asm)} | {_fmt_s(fit)} | "
                f"{_fmt_s(sim_s, scan_here)} | {search_dev or '-'} | "
                f"{_fmt_s(sim_e)} | {eval_dev} | "
                f"{_fmt_s(total)} | {_fmt_s(total + sim_e)} |")

    lines += ["", "## Timing detail (sub-components, seconds)", "",
             "Reconstruct-tier-dump: fd_velocity_s, neighbor_search_s (one "
             "cKDTree query, not one per frame), mls_solve_s (both the L and "
             "F least-squares fits, both against the SAME frame-0-fixed "
             "neighbour set -- the pre-re-neighbour baseline, restored "
             "2026-09-05), volume_mass_s, npz_write_s "
             "(np.savez, uncompressed). None if a cached tier dump on disk "
             "made write_tier_dump skip the rebuild. Simulate detail: "
             "setup_s, step_s (physics substeps), snapshot_s, finalize_s "
             "(np.savez, uncompressed), summed over every rollout the "
             "run/scan actually simulated (cached npz reuse contributes 0).",
             ""]
    for material in MATERIALS:
        for tier in TIERS:
            res = _load(material, tier)
            if res is None:
                continue
            tb = res.get("timing_breakdown_s") or {}
            recon_detail = tb.get("reconstruct_tier_dump_detail_s")
            search_detail = tb.get("simulate_search_detail_s")
            eval_detail = tb.get("simulate_eval_detail_s")
            has_detail = recon_detail or (search_detail and any(search_detail.values())) \
                or (eval_detail and any(eval_detail.values()))
            if not has_detail:
                continue
            lines.append(f"**{material} / {tier}**")
            if recon_detail:
                lines.append(f"- reconstruct (tier dump): {recon_detail}")
            if search_detail and any(search_detail.values()):
                lines.append(f"- simulate (search, device={tb.get('simulate_search_device')}): {search_detail}")
            if eval_detail and any(eval_detail.values()):
                lines.append(f"- simulate (eval, device={tb.get('simulate_eval_device')}): {eval_detail}")
            lines.append("")

    lines += ["", "## Loss (position MSE, recovered parameters)", "",
             "| material | tier | dataset | time | vel mean | shape | "
             "Mean Loss (time+vel+shape)/3 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for material in MATERIALS:
        for tier in TIERS:
            res = _load(material, tier)
            if res is None:
                lines.append(f"| {material} | {tier} | (missing) | | | | |")
                continue
            per_axis, mean_loss = _mean_loss(res)
            lines.append(
                f"| {material} | {tier} | "
                f"{per_axis.get('dataset', float('nan')):.3e} | "
                f"{per_axis.get('time', float('nan')):.3e} | "
                f"{per_axis.get('vel', float('nan')):.3e} | "
                f"{per_axis.get('shape', float('nan')):.3e} | "
                f"{'' if mean_loss is None else f'{mean_loss:.3e}'} |")

    lines += ["", "## Recovered-parameter error vs. truth", "",
             "\"method\" is computed fresh each run from what the code "
             "actually did (theta_for_engine), not a hardcoded label: "
             "**assumed** = never attempted, or attempted and refused (falls "
             "back to the truth prior, which is why truth==recovered); "
             "**weak_form** = the convex momentum-balance solve; "
             "**closed_form_strain_cap** = plasticine's yield-plateau read of "
             "the stored F (no assemble/solve); **cone_level_reading** = "
             "sand's mode-of-ratio fallback when the weak-form residual is "
             "too high but a cone level is observable; **rollout_scan** = "
             "the derivative-free simulator search.",
             "",
             "| material | tier | parameter | ground truth | recovered | error (%) | method |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for material in MATERIALS:
        for tier in TIERS:
            res = _load(material, tier)
            if res is None:
                lines.append(f"| {material} | {tier} | (missing) | - | - | - | - |")
                continue
            rows = _param_rows(res)
            for i, (k, t, r, err, meth) in enumerate(rows):
                mat_cell = material if i == 0 else ""
                tier_cell = tier if i == 0 else ""
                err_str = "-" if err is None else f"{err:+.2f}%"
                lines.append(f"| {mat_cell} | {tier_cell} | {k} | "
                            f"{t:.4e} | {r:.4e} | {err_str} | {meth} |")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT / "timing_report.md"))
    a = ap.parse_args(argv)
    text = render()
    print(text)
    Path(a.out).write_text(text + "\n")
    print(f"\n[timing_report] wrote {a.out}")


if __name__ == "__main__":
    main()

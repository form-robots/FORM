"""Markdown tables for the FE/LS unknown-form legs rolled on NCLaw's own
trajectories (experiments.fe_ls.cross), mirroring
experiments/nclaw/no_stress_table.py and timing_report.py for
out/nclaw_cross_compare/.

Reads out/fe_ls_cross/results.json and renders, per material, one column per
accepted leg (``legs_rolled``) against NCLaw's own published MSE for that
scene. "margin" is the published MSE divided by the best (lowest-MSE) leg
for that row -- with several candidate unknown-form families per material,
there is no single "recovered" column the way compare.py has one, so the
margin heralds whichever family actually won that scene. vel_0001/0007/0008
collapse to a "vel mean" row and every shape_<mesh> scene would collapse to
"shape mean" if a material had more than one (none currently do).

Run:  .venv/bin/python -m experiments.fe_ls.cross_table
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

OUT = ROOT / "out" / "fe_ls_cross"
MATERIALS = ["jelly", "plasticine", "sand", "water"]

# one line per leg, from experiments.fe_ls.cross's own docstrings/comments --
# what each identifies and, for sand/plasticine, what it isolates by omission
LEG_DESCRIPTIONS = {
    "fe_projected_corotated": (
        "hyperelastic W'(I1bar) fit through the trained one-invariant basis "
        "plus one volumetric column; the engine has no tabulated hyperelastic "
        "material, so the rollout leg projects the recovered curve to its "
        "small-strain corotated pair (mu, lam)."),
    "fe_volumetric_eos": (
        "the same hyperelastic family, but for a fluid the deviatoric "
        "coefficients should return near zero -- only the volumetric column "
        "is used, rolled out through the comparison EOS sigma = lam (J-1) I."),
    "fe_elastic_only_no_yield": (
        "the recovered elastic (mu, lam) pair rolled out WITHOUT any yield "
        "cap (yield_stress fixed absurdly high); the stored F is the elastic "
        "state so this family fits it cleanly, and rolling it alone measures "
        "the cost of the missing plasticity."),
    "fe_yield_surface": (
        "the learned yield surface h(p) (sqrt(J2(dev tau)) = h(p)), one "
        "unknown-form family covering the whole perfect-plasticity zoo (flat "
        "= von Mises, line through origin = cohesionless cone, etc.), read "
        "from binned pointwise Kirchhoff stress over the post-impact "
        "shearing set and rolled through the tabulated-yield material -- for "
        "plasticine, on top of the fe_elastic_only_no_yield pair above."),
    "fe_mu_of_I": (
        "mu(I) fit through the trained granular basis by a monotone, "
        "mu >= 0.05 constrained solve (the headline sand fit), baked into "
        "the engine's tabulated mu(I) material and rolled out."),
    "flat_table_truth_mu": (
        "control leg: a CONSTANT mu table at the true cone level, through "
        "the same tabulated material as fe_mu_of_I -- isolates curve-fitting "
        "error from the gap between the engine's tabulated return map and "
        "NCLaw's own Drucker-Prager integrator."),
    "fe_binned_cone": (
        "mu(I) read directly off binned pointwise cone ratios "
        "(r = sqrt(J2(dev sigma))/p vs. I, median per bin) rather than fit "
        "through the weak-form dictionary -- a cheaper, more direct read of "
        "the same curve as fe_mu_of_I."),
}


# leg -> identify() block(s) whose wall_seconds fund it. hyperelastic_fe funds
# BOTH plasticine legs (the elastic pair is fit once, then reused with and
# without the yield surface on top) -- summing identify_s across legs of the
# same material therefore double-counts that block; it is still the honest
# answer to "what did fitting THIS leg cost". flat_table_truth_mu has no
# identify entry: it reads the true cone level directly, no fit at all.
LEG_IDENTIFY_KEYS = {
    "fe_mu_of_I": ["granular_fe"],
    "flat_table_truth_mu": [],
    "fe_yield_surface": ["yield_surface_fe"],
    "fe_binned_cone": ["granular_fe_binned_cone"],
    "fe_projected_corotated": ["hyperelastic_fe"],
    "fe_volumetric_eos": ["hyperelastic_fe"],
    "fe_elastic_only_no_yield": ["hyperelastic_fe"],
}
# plasticine's fe_yield_surface also needs the elastic pair underneath it
PLASTICINE_YIELD_EXTRA_KEYS = ["hyperelastic_fe"]

SIM_DETAIL_COLS = ["setup_s", "step_s", "snapshot_s", "finalize_s"]

# identify() blocks that are a diagnostic re-dump of ANOTHER block's result
# rather than a distinct call -- sand's "granular_fe_full" copies almost every
# field (wall_seconds included) out of the same `ident` object "granular_fe"
# was already built from, so summing wall_seconds over every identify block
# blindly (as results.json's own timing_breakdown_s.identify_total_s does)
# double-counts that one fit. Excluded when we recompute the total here.
IDENTIFY_DEDUPE_EXCLUDE = {"granular_fe_full"}


def _identify_total_s(res: dict) -> float:
    """Wall time of the DISTINCT identify_*_fe calls this material ran.

    Recomputed from res["identify"] rather than trusted from
    timing_breakdown_s.identify_total_s, which sums every block including
    diagnostic copies (see IDENTIFY_DEDUPE_EXCLUDE) and so over-reports sand
    by exactly its granular_fe wall time (double-counted).
    """
    return sum(float(v.get("wall_seconds") or 0.0)
              for k, v in res.get("identify", {}).items()
              if isinstance(v, dict) and k not in IDENTIFY_DEDUPE_EXCLUDE)


def _role(scene: str) -> str:
    if scene.startswith("vel"):
        return "vel"
    if scene.startswith("shape"):
        return "shape"
    return scene


def _fmt(x) -> str:
    return f"{x:.1e}" if isinstance(x, (int, float)) else ""


def _fmt_margin(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.1f}x" if x < 10 else f"{x:.0f}x"


def _row(label: str, legs: list[str], cells_by_leg: dict, published: float | None) -> str:
    vals = []
    for leg in legs:
        c = cells_by_leg.get(leg)
        if c is None:
            vals.append("")
        elif c.get("diverged"):
            vals.append("diverged")
        else:
            vals.append(_fmt(c["mse"]))
    finite = [(leg, cells_by_leg[leg]["mse"]) for leg in legs
              if leg in cells_by_leg and not cells_by_leg[leg].get("diverged")]
    margin = None
    if finite and published is not None:
        _, best_mse = min(finite, key=lambda kv: kv[1])
        margin = published / best_mse if best_mse else None
    return "| " + " | ".join([label] + vals + [_fmt(published), _fmt_margin(margin)]) + " |"


def material_table(material: str, res: dict) -> str:
    legs = res["legs_rolled"]
    scenes = res["scenes"]
    head = ["scene"] + legs + ["published", "margin (best leg)"]
    lines = [f"### {material}", "",
             "| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]

    by_role: dict[str, list[tuple]] = {}
    for scene, cells_by_leg in scenes.items():
        published = next((cells_by_leg[leg]["published"] for leg in legs
                          if leg in cells_by_leg), None)
        by_role.setdefault(_role(scene), []).append((scene, cells_by_leg, published))
        lines.append(_row(scene, legs, cells_by_leg, published))

    for role, group in by_role.items():
        if len(group) < 2:
            continue
        mean_cells = {}
        for leg in legs:
            vals = [cbl[leg]["mse"] for _, cbl, _ in group
                    if leg in cbl and not cbl[leg].get("diverged")]
            if vals:
                mean_cells[leg] = {"mse": float(np.mean(vals)), "diverged": False}
        pub_vals = [p for _, _, p in group if p is not None]
        mean_pub = float(np.mean(pub_vals)) if pub_vals else None
        lines.append(_row(f"{role} mean", legs, mean_cells, mean_pub))

    lines.append("")
    lines.append("legs:")
    for leg in legs:
        desc = LEG_DESCRIPTIONS.get(leg, "(no description)")
        lines.append(f"- **{leg}**: {desc}")
    lines.append("")
    for key, block in res.get("identify", {}).items():
        if not isinstance(block, dict):
            continue
        summary = {k: v for k, v in block.items() if not isinstance(v, (list, dict))}
        if not summary:
            continue
        tag = (" [diagnostic copy, excluded from the identify total below -- "
              "its wall_seconds duplicates another block's]"
              if key in IDENTIFY_DEDUPE_EXCLUDE else "")
        lines.append(f"identify[{key}]{tag}: {json.dumps(summary, default=float)}")
    tb = res.get("timing_breakdown_s", {})
    lines.append(f"timing: identify {_identify_total_s(res):.1f}s, "
                f"rollout {tb.get('rollout_wall_s', 0.0):.1f}s, "
                f"simulate-eval {tb.get('simulate_eval_s', 0.0):.1f}s, "
                f"material total {tb.get('material_total_s', 0.0):.1f}s "
                f"(device={res.get('device')})")
    if res.get("note"):
        lines.append(f"note: {res['note']}")
    return "\n".join(lines)


def render(materials: list[str]) -> str:
    results = json.loads((OUT / "results.json").read_text())
    parts = [material_table(m, results[m]) for m in materials if m in results]
    return "\n\n".join(parts)


def _identify_keys_for_leg(material: str, leg: str) -> list[str]:
    keys = list(LEG_IDENTIFY_KEYS.get(leg, []))
    if material == "plasticine" and leg == "fe_yield_surface":
        keys = PLASTICINE_YIELD_EXTRA_KEYS + keys
    return keys


def _leg_timing(material: str, res: dict, leg: str) -> dict:
    identify = res.get("identify", {})
    identify_s = sum(float(identify.get(k, {}).get("wall_seconds") or 0.0)
                     for k in _identify_keys_for_leg(material, leg))

    scenes = res.get("scenes", {})
    n_scenes = n_fresh = 0
    rollout_s = 0.0
    detail = {c: 0.0 for c in SIM_DETAIL_COLS}
    for cells_by_leg in scenes.values():
        cell = cells_by_leg.get(leg)
        if cell is None:
            continue
        n_scenes += 1
        if cell.get("sim_wall_s") is not None:
            n_fresh += 1
            rollout_s += cell["sim_wall_s"]
            for c in SIM_DETAIL_COLS:
                detail[c] += (cell.get("sim_timing") or {}).get(c, 0.0)
    return {"identify_s": identify_s, "rollout_s": rollout_s,
            "n_fresh": n_fresh, "n_scenes": n_scenes, "detail": detail}


def timing_table(materials: list[str]) -> str:
    results = json.loads((OUT / "results.json").read_text())
    lines = ["## Timing breakdown (seconds)", "",
             "\"identify total\" is recomputed from res[\"identify\"] rather than "
             "read from timing_breakdown_s.identify_total_s -- that stored field "
             "sums wall_seconds over EVERY identify block including diagnostic "
             "copies, which double-counts sand (\"granular_fe_full\" re-dumps "
             "\"granular_fe\"'s own result, wall_seconds included); see "
             "IDENTIFY_DEDUPE_EXCLUDE. \"material total\" is a directly measured "
             "wall-clock span (time.time() across the whole run_material call), "
             "not a sum of the other columns, so it is unaffected and can be "
             "larger than identify + rollout + simulate-eval.",
             "",
             "| material | identify total | rollout wall | simulate-eval | "
             "material total | device |",
             "| --- | --- | --- | --- | --- | --- |"]
    for m in materials:
        res = results.get(m)
        if res is None:
            lines.append(f"| {m} | (missing) | | | | |")
            continue
        tb = res.get("timing_breakdown_s", {})
        lines.append(f"| {m} | {_identify_total_s(res):.1f} | "
                     f"{tb.get('rollout_wall_s', 0.0):.1f} | "
                     f"{tb.get('simulate_eval_s', 0.0):.1f} | "
                     f"{tb.get('material_total_s', 0.0):.1f} | {res.get('device', '?')} |")

    lines += ["", "## Per-leg timing breakdown (seconds)", "",
             "\"identify\" is the wall time of the identify_*_fe block(s) that "
             "fund that leg's parameters -- for plasticine's fe_yield_surface "
             "this includes the shared hyperelastic_fe fit underneath it, so "
             "summing identify time across a material's legs double-counts "
             "that shared block; flat_table_truth_mu has none (a constant at "
             "the true cone level, not fit at all). \"rollout (fresh)\" and "
             "the setup/step/snapshot/finalize detail sum only scenes that "
             "were actually simulated this run -- a cached prediction npz on "
             "disk contributes 0 and is counted in \"cached\" instead.",
             "",
             "| material | leg | identify | rollout (fresh) | setup | step | "
             "snapshot | finalize | scenes fresh/cached/total |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in materials:
        res = results.get(m)
        if res is None:
            continue
        for leg in res.get("legs_rolled", []):
            t = _leg_timing(m, res, leg)
            d = t["detail"]
            n_cached = t["n_scenes"] - t["n_fresh"]
            lines.append(
                f"| {m} | {leg} | {t['identify_s']:.1f} | {t['rollout_s']:.1f} | "
                f"{d['setup_s']:.1f} | {d['step_s']:.1f} | {d['snapshot_s']:.1f} | "
                f"{d['finalize_s']:.1f} | {t['n_fresh']}/{n_cached}/{t['n_scenes']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--materials", default=",".join(MATERIALS))
    ap.add_argument("--out", default=str(OUT / "table_full.md"))
    ap.add_argument("--timing-out", default=str(OUT / "timing_report.md"))
    a = ap.parse_args(argv)
    mats = [m.strip() for m in a.materials.split(",") if m.strip()]

    text = render(mats)
    print(text)
    Path(a.out).write_text(text + "\n")

    timing_text = timing_table(mats)
    Path(a.timing_out).write_text(timing_text + "\n")
    print(f"\n[fe_ls cross_table] wrote {a.out} and {a.timing_out}")


if __name__ == "__main__":
    main()

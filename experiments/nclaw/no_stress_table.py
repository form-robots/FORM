"""Per-tier table: our rollouts, a live full-channel reference, their published cells.

Reads the ``compare_<material>*.json`` files ``compare.py`` writes and prints
one markdown table per material for the requested tier (``full``, the
default full-channel run itself; ``nostress``, from ``compare.py
--no-stress``; or ``positionsonly``, from ``compare.py --positions-only``).
For any tier but ``full``, each row also carries a "full-channel recovered"
column read live from that material's own full-channel ``compare_*.json`` on
disk, so a degraded-tier row always sits next to the row it degrades from,
computed from your own run rather than a frozen snapshot; the column reads
blank if that file has not been produced yet. The published column is
NCLaw's own, from ``suite.NCLAW_PUBLISHED``.

Run:  .venv/bin/python -m experiments.nclaw.no_stress_table [--tier full|nostress|positionsonly]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "out" / "nclaw_cross_compare"

# per-material flag tag of the comparison, matching compare.py's CANONICAL
TAGS = {"jelly": "_nclawbc_sub1", "plasticine": "_nclawbc",
        "sand": "_nclawbc_nclawlaw_sub1", "water": "_nclawbc_nclawlaw"}

# tier name (as given on --tier) to the filename suffix compare.py writes;
# "full" is the full-channel run itself, which carries no tier suffix
TIER_SUFFIX = {"full": "", "nostress": "_nostress", "positionsonly": "_positionsonly"}

# scene name -> published-table axis. vel_0001/0007/0008 collapse to "vel",
# and every shape_<mesh> scene (one held-out mesh per material) collapses to
# "shape"; both are collapsed the same way scenes are grouped for the mean row
SCENE_ROLE = {"dataset": "reconstruction", "time": "time",
              "vel": "velocity", "shape": "generalization"}


def _role(scene: str) -> str:
    if scene.startswith("vel"):
        return "vel"
    if scene.startswith("shape"):
        return "shape"
    return scene


def _full_channel_reference(material: str) -> dict[str, float] | None:
    """Per-SCENE recovered MSE from that material's own full-channel run
    (e.g. vel_0001 maps to vel_0001, not to a vel-group average -- the tiers
    share scene names 1:1, so match on the exact scene), or None if the run
    has not been produced yet (out/nclaw_cross_compare/compare_
    <material><TAGS[material]>.json missing) -- the caller prints a blank
    column rather than treating this as an error, same convention as an
    unpublished cell.
    """
    path = OUT / f"compare_{material}{TAGS[material]}.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    return {scene: cells["recovered"]["mse"] for scene, cells in res["scenes"].items()}


def material_rows(material: str, tier: str) -> dict:
    from experiments.nclaw.suite import NCLAW_PUBLISHED
    path = OUT / f"compare_{material}{TAGS[material]}{TIER_SUFFIX[tier]}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    res = json.loads(path.read_text())
    published = NCLAW_PUBLISHED[material]
    legs = [k for k in next(iter(res["scenes"].values())) if k == "truth_theta"
            or k.startswith("recovered")]
    rows = []
    for scene, cells in res["scenes"].items():
        role = _role(scene)
        pub = published.get(SCENE_ROLE.get(role))
        row = {"scene": scene, "role": role, "published": pub}
        for leg in legs:
            row[leg] = cells[leg]["mse"]
        row["margin_vs_published"] = (pub / cells["recovered"]["mse"]
                                      if pub is not None else None)
        rows.append(row)
    return {"result": res, "rows": rows, "legs": legs}


def _fmt(x: float | None) -> str:
    return f"{x:.1e}" if x is not None else ""


def _fmt_margin(x: float | None) -> str:
    if x is None:
        return ""
    # single-digit margins round to nothing useful at .0f (0.79x and 1.16x
    # both show "1x"); one decimal below 10x, whole numbers above
    return f"{x:.1f}x" if x < 10 else f"{x:.0f}x"


def render(material: str, tier: str = "nostress") -> str:
    got = material_rows(material, tier)
    res, rows, legs = got["result"], got["rows"], got["legs"]
    recovered_legs = [leg for leg in legs if leg.startswith("recovered")]
    # a tier's own full-channel run has no full-channel row to sit next to
    show_full_channel = tier != "full"
    full_channel = _full_channel_reference(material) if show_full_channel else None

    head = ["scene", "our sim, correct properties"] + recovered_legs
    if show_full_channel:
        head.append("full-channel recovered")
    head += ["their published", "margin"]
    lines = [f"### {material} ({tier})", "",
             "| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]

    def row_cells(label: str, truth_theta: float, leg_vals: dict[str, float],
                  full_channel_val: float | None, published: float | None,
                  margin: float | None) -> str:
        cells = [label, _fmt(truth_theta)] + [_fmt(leg_vals[leg]) for leg in recovered_legs]
        if show_full_channel:
            cells.append(_fmt(full_channel_val))
        cells += [_fmt(published), _fmt_margin(margin)]
        return "| " + " | ".join(cells) + " |"

    by_role: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)
        fc = full_channel.get(r["scene"]) if full_channel else None
        lines.append(row_cells(r["scene"], r["truth_theta"], r, fc,
                                r["published"], r["margin_vs_published"]))

    # a mean row for any role backed by more than one scene (the vel_000x
    # triplet, or a material with more than one held-out mesh); the
    # full-channel reference is the mean of the same scenes' OWN full-channel
    # numbers, not a separately-computed role average
    for role, group in by_role.items():
        if len(group) < 2:
            continue
        mean_truth = float(np.mean([r["truth_theta"] for r in group]))
        mean_legs = {leg: float(np.mean([r[leg] for r in group])) for leg in recovered_legs}
        pub = group[0]["published"]
        margin = pub / mean_legs["recovered"] if pub is not None else None
        fc_vals = ([full_channel[r["scene"]] for r in group if r["scene"] in full_channel]
                  if full_channel else [])
        mean_fc = float(np.mean(fc_vals)) if fc_vals else None
        lines.append(row_cells(f"{role} mean", mean_truth, mean_legs, mean_fc, pub, margin))

    lines += ["", f"theta: {json.dumps(res['theta_recovered'], default=float)}",
              f"refused: {res['identify_diagnostics']['refused_parameters']}",
              f"identify wall time: {res['wall_identify_s']:.1f} s "
              f"({json.dumps(res.get('wall_times_s'), default=float)})",
              f"provenance: {json.dumps(res.get('channel_provenance'), default=float)}"]
    for name, spec in (res.get("theta_variants") or {}).items():
        lines += ["", f"variant {name}: theta "
                  f"{json.dumps(spec['theta'], default=float)}",
                  f"  provenance: {spec.get('provenance')}",
                  f"  refused: {spec.get('refused')} {spec.get('note', '')}",
                  f"  diagnostics: {json.dumps(spec.get('diagnostics'), default=float)}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", default="nostress", choices=sorted(TIER_SUFFIX))
    ap.add_argument("--materials", default="jelly,plasticine,sand,water")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = "\n\n".join(render(m, a.tier)
                       for m in a.materials.split(",") if m.strip())
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

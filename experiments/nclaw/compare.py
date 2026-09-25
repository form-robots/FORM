"""Engine-gap decomposition on NCLaw-generated trajectories.

For one material, given ingested truth dumps under
mpm_engine/out/nclaw_cross_generalize/dumps/<material>_<scene>_truth.npz:

  1. identify theta from the DATASET scene alone (the protocol's rule);
  2. per scene, roll our engine from THEIR frame-0 cloud at (a) the truth
     parameters and (b) the recovered parameters, at their own frame cadence;
  3. score both against their trajectory in their metric.

Row (a) runs our simulator with the correct material properties and compares it to their trajectory; whatever mismatch remains is simulator difference, and no identification method can score below it. The ratio
(b)/(a) is the identification excess, the only part of a cross-engine cell
that identification quality controls.

--nclaw-bc turns on the engine's NCLaw-compatibility grid mode (their
approach-only wall clamp, eps-softened mass division, free-fall velocity on
empty nodes, MLS transfer and their particle position clamp); the truth-theta
leg measures what this grid mode is worth. --nclaw-law additionally rolls the material out on
NCLaw's own constitutive pair for it (engine material 14) and identifies in that
form, which matters wherever their law is not a reparameterization of ours:
their water is a linear volumetric EOS with no deviatoric term at all.
--substeps=N fixes the substep count per dumped frame; --substeps=1 is what
matches their integrator exactly, and their sand needs it because our CFL picks
two substeps where they take one. --bisect reruns the truth-theta leg one
behavior at a time to attribute the MSE drop.

--no-stress runs the same comparison with the stress channel excluded from
identification: the dataset scene is copied to a tier dump by
experiments/nclaw/strip_channels.py, which keeps every stored kinematic channel
bitwise and zeroes the stress, and the identification goes through
experiments/nclaw/identify_no_stress.py, which states a pressure model where the
full-channel path read the stress trace. --positions-only is the harder tier
below it: positions and times measured, velocities by finite difference, L and F
by moving least squares -- but only the dataset scene is ever identified from,
so only its tier dump is materialized; every other scene's rollout is seeded
with a frame-0 velocity derived in memory from that scene's own truth dump
(strip_channels.positions_only_seed_cloud), never written to disk. Both tiers
roll out the primary recovered parameters and every variant estimator the tier
offers, each in its own leg.

Run:  .venv/bin/python -m experiments.nclaw.compare sand --trajectories=/path/to/their/dumps plasticine \
          [--nclaw-bc] [--nclaw-law] [--substeps=1] [--bisect]
          [--no-stress | --positions-only]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_TRAJECTORIES = ROOT / "out" / "nclaw_cross_generalize" / "dumps"
OUT = ROOT / "out" / "nclaw_cross_compare"

# One behavior at a time, against the engine's default grid path. "off" is the
# baseline (no mode at all); "full" is every behavior at their dataset settings.
OFF = {"freeslip_bound": 0, "mass_eps": 0.0, "empty_node_gravity": False,
       "mls_transfer": False, "particle_clip_cells": -1.0}
BISECT_LEGS = {
    "off": False,
    "freeslip_walls": {**OFF, "freeslip_bound": 3},
    "mass_eps": {**OFF, "mass_eps": 1.0e-7},
    "empty_node_gravity": {**OFF, "empty_node_gravity": True},
    "mls_transfer": {**OFF, "mls_transfer": True},
    "particle_clip": {**OFF, "particle_clip_cells": 0.5},
    "full": True,
    # leave-one-out, to attribute the residual the walls alone leave behind
    "full_no_mls": {"mls_transfer": False},
    "full_no_mass_eps": {"mass_eps": 0.0},
    "full_no_empty_gravity": {"empty_node_gravity": False},
    "full_no_clip": {"particle_clip_cells": -1.0},
    "walls_and_mls": {**OFF, "freeslip_bound": 3, "mls_transfer": True},
}


def main(material: str, trajectories: str | Path | None = None,
         nclaw_bc: bool = False, bisect: bool = False,
         nclaw_law: bool = False, substeps: int | None = None,
         tier: str | None = None, device: str = "cpu") -> None:
    DUMPS = Path(trajectories) if trajectories else DEFAULT_TRAJECTORIES
    print(f"[compare] trajectories from {DUMPS}")
    from experiments.nclaw.suite import (
        MATERIALS,
        cloud_from_dump,
        nclaw_position_mse,
        run_scene,
        stage_identify,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.name.split(f"{material}_", 1)[1].rsplit("_truth.npz", 1)[0]
                    for p in DUMPS.glob(f"{material}_*_truth.npz"))
    if "dataset" not in scenes:
        raise SystemExit(f"no {material}_dataset_truth.npz under {DUMPS}")
    theta_true = dict(MATERIALS[material]["truth"])

    if bisect:
        # attribution run: the dataset scene at truth theta, one behavior at a
        # time. The grid mode changes no identification input, so theta is fixed.
        truth = DUMPS / f"{material}_dataset_truth.npz"
        cloud = cloud_from_dump(truth)
        rows = {}
        for leg, mode in BISECT_LEGS.items():
            pred = OUT / f"{material}_dataset_truth_theta_bis_{leg}.npz"
            if not pred.exists():
                t0 = time.time()
                run_scene(material, "dataset", pred, theta=dict(theta_true),
                          cloud=cloud, nclaw_bc=mode, nclaw_law=nclaw_law,
                          substeps=substeps, device=device)
                print(f"[bisect] {leg} simulated in {time.time() - t0:.0f}s")
            s = nclaw_position_mse(truth, pred)
            rows[leg] = {k: s[k] for k in
                         ("mse", "mse_final_frame", "rmse_mm", "n_frames")}
            print(f"[bisect] {leg}: MSE {s['mse']:.4e} "
                  f"(RMS {s['rmse_mm']:.2f} mm)")
        base = rows["off"]["mse"]
        for leg, cell in rows.items():
            cell["ratio_to_off"] = cell["mse"] / base if base > 0 else None
        path = OUT / f"bisect_{material}.json"
        path.write_text(json.dumps(
            {"material": material, "scene": "dataset", "theta": theta_true,
             "legs": BISECT_LEGS, "rows": rows}, indent=2, default=float))
        print(f"[bisect] wrote {path}")
        return

    tier_tag = {"no_stress": "_nostress", "positions_only": "_positionsonly"}
    tag = (("_nclawbc" if nclaw_bc else "") + ("_nclawlaw" if nclaw_law else "")
           + (f"_sub{substeps}" if substeps is not None else "")
           + (tier_tag[tier] if tier else ""))
    print(f"[compare] {material}: scenes {scenes} nclaw_bc={nclaw_bc} "
          f"nclaw_law={nclaw_law} tier={tier or 'full_channels'}")

    dataset = DUMPS / f"{material}_dataset_truth.npz"
    variants: dict[str, dict] = {}
    wall_reconstruct_tier_s = 0.0
    tier_dump_timing: dict = {}
    t_ident = time.time()
    if tier is None:
        ident = stage_identify(material, dump=dataset,
                              tag=f"cross_{material}{tag}", nclaw_law=nclaw_law)
        identify_dump = dataset
    else:
        from experiments.nclaw.identify_no_stress import stage_identify_no_stress
        from experiments.nclaw.strip_channels import write_tier_dump
        t_tier = time.time()
        identify_dump = write_tier_dump(dataset, tier, timing=tier_dump_timing)
        wall_reconstruct_tier_s = time.time() - t_tier
        ident = stage_identify_no_stress(
            material, dump=identify_dump, tag=f"cross_{material}{tag}",
            nclaw_law=nclaw_law, nclaw_bc=nclaw_bc, substeps=substeps,
            device=device,
            # read for the basal-plate variant only, and only inside one cell
            # of the comparison; every other use of this file at this tier is
            # diagnosis.
            # The positions-only tier reads no stress anywhere, so no basal dump.
            basal_dump=(dataset if material == "sand" and tier == "no_stress"
                        else None))
        variants = ident.get("theta_variants", {})
    wall_identify = time.time() - t_ident
    theta_rec = ident["theta_engine"]
    print(f"[compare] recovered {theta_rec} vs truth {theta_true} "
          f"({wall_identify:.1f}s), variants {sorted(variants)}")

    legs: dict[str, dict] = {"truth_theta": theta_true, "recovered": theta_rec}
    for name, spec in variants.items():
        legs[f"recovered_{name}"] = spec["theta"]

    rows = {}
    for scene in scenes:
        truth = DUMPS / f"{material}_{scene}_truth.npz"
        # a positions-only rollout is seeded from a derived frame-0 velocity,
        # not the measured one, so a scene other than dataset pays for that
        # degradation too -- computed in memory from its own truth dump (see
        # strip_channels.positions_only_seed_cloud), no tier file needed. The
        # no-stress tier keeps the stored velocities, so its cloud is the
        # trajectory's own regardless of scene.
        if tier != "positions_only":
            cloud = cloud_from_dump(truth)
        elif scene == "dataset":
            cloud = cloud_from_dump(identify_dump)     # the dataset tier dump itself
        else:
            from experiments.nclaw.strip_channels import positions_only_seed_cloud
            cloud = positions_only_seed_cloud(truth)
        cells = {}
        for leg, theta in legs.items():
            pred = OUT / f"{material}_{scene}_{leg}{tag}.npz"
            sim_wall_s = None
            sim_timing: dict = {}
            if not pred.exists():
                t0 = time.time()
                run_scene(material, scene, pred, theta=dict(theta), cloud=cloud,
                          nclaw_bc=nclaw_bc, nclaw_law=nclaw_law,
                          substeps=substeps, device=device, xonly=True,
                          timing=sim_timing)
                sim_wall_s = time.time() - t0
                print(f"[compare] {scene}/{leg} simulated in {sim_wall_s:.0f}s")
            s = nclaw_position_mse(truth, pred)
            cells[leg] = {k: s[k] for k in
                          ("mse", "mse_final_frame", "rmse_mm", "n_frames")}
            cells[leg]["sim_wall_s"] = sim_wall_s   # None => reused a cached rollout
            cells[leg]["sim_timing"] = sim_timing or None  # setup/step/snapshot/finalize
            print(f"[compare] {scene}/{leg}: MSE {s['mse']:.3e} "
                  f"(RMS {s['rmse_mm']:.2f} mm, {s['n_frames']} frames)")
        floor = cells["truth_theta"]["mse"]
        for leg in [k for k in cells if k.startswith("recovered")]:
            cells[f"identification_excess_{leg}"] = (cells[leg]["mse"] / floor
                                                    if floor > 0 else None)
        cells["identification_excess"] = cells["identification_excess_recovered"]
        rows[scene] = cells

    eval_sim_timings = [cell["sim_timing"] for cells in rows.values()
                       for leg, cell in cells.items()
                       if isinstance(cell, dict) and cell.get("sim_timing")]
    wall_simulate_eval_s = sum(t.get("total_s", 0.0) for t in eval_sim_timings)
    simulate_eval_detail_s = {
        k: sum(t.get(k, 0.0) for t in eval_sim_timings)
        for k in ("setup_s", "step_s", "snapshot_s", "finalize_s")}
    ident_timing = dict(ident.get("timing_breakdown_s") or {})
    timing_breakdown_s = {
        "reconstruct_fields_s": wall_reconstruct_tier_s + ident_timing.get("reconstruct_fields_s", 0.0),
        # tier-dump build (fd_velocity, one frame-0 neighbour search reused
        # for both the L and F moving-least-squares fits -- the
        # pre-re-neighbour baseline, restored 2026-09-05 -- and npz write);
        # None if a stale tier dump on disk let write_tier_dump skip the
        # rebuild -- delete
        # <material>_dataset_truth_<tier>.npz under
        # out/nclaw_cross_generalize/dumps/ first if you need this populated.
        "reconstruct_tier_dump_detail_s": tier_dump_timing or None,
        # positions-only sand/plasticine only: the SEPARATE incremental-MLS
        # reconstruction inside the (refused) weak-form replay estimator --
        # not the same reconstruction as the tier dump above.
        "reconstruct_replay_s": ident_timing.get("reconstruct_fields_s", 0.0),
        "assemble_equations_s": ident_timing.get("assemble_equations_s", 0.0),
        "fit_parameters_s": ident_timing.get("fit_parameters_s", 0.0),
        "simulate_search_s": ident_timing.get("simulate_s", 0.0),
        "simulate_search_detail_s": ident_timing.get("simulate_search_detail_s"),
        # scan_parameter() now takes this comparison's --device (previously
        # always ran on CPU regardless of --device; fixed 2026-09-05).
        "simulate_search_device": ident_timing.get("simulate_search_device"),
        "simulate_eval_s": wall_simulate_eval_s,
        "simulate_eval_device": device,
        # eval rollouts split into physics-step / state-snapshot / npz-write
        # (np.savez, uncompressed -- DumpWriter.compress defaults False and
        # nothing in this pipeline overrides it) and warp/simulator setup.
        "simulate_eval_detail_s": simulate_eval_detail_s,
        "identify_total_s": wall_identify,
    }

    res = {"material": material, "nclaw_bc": nclaw_bc, "nclaw_law": nclaw_law,
           "substeps": substeps, "tier": tier or "full_channels",
           "identified_from": identify_dump.name,
           "channel_provenance": ident.get("channel_provenance"),
           "wall_identify_s": wall_identify,
           "wall_times_s": ident.get("wall_times_s"),
           "timing_breakdown_s": timing_breakdown_s,
           "device": device,
           "theta_recovered": theta_rec,
           "theta_truth": theta_true,
           "theta_variants": {k: {kk: v[kk] for kk in v if kk != "theta"}
                              | {"theta": v["theta"]} for k, v in variants.items()},
           "identify_diagnostics": {k: ident.get(k) for k in
                                    ("refused_parameters", "parameter_estimator")},
           "scenes": rows}
    path = OUT / f"compare_{material}{tag}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"[compare] wrote {path}")


# The cross-engine settings per material for reported tables. A bare
# invocation applies them; any explicit compatibility flag overrides them.
CANONICAL = {
    "jelly": {"nclaw_bc": True, "substeps": 1},
    "plasticine": {"nclaw_bc": True},
    "sand": {"nclaw_bc": True, "nclaw_law": True, "substeps": 1},
    "water": {"nclaw_bc": True, "nclaw_law": True},
}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    sub = next((int(f.split("=", 1)[1]) for f in flags
                if f.startswith("--substeps=")), None)
    traj = next((f.split("=", 1)[1] for f in flags
                 if f.startswith("--trajectories=")), None)
    tier_flag = ("no_stress" if "--no-stress" in flags else
                 "positions_only" if "--positions-only" in flags else None)
    dev = next((f.split("=", 1)[1] for f in flags
                if f.startswith("--device=")), "cpu")
    mat = args[0] if args else "plasticine"
    compat_given = any(f.startswith(("--nclaw-bc", "--nclaw-law", "--substeps"))
                       for f in flags)
    if compat_given or "--bisect" in flags:
        kw = {"nclaw_bc": "--nclaw-bc" in flags,
              "nclaw_law": "--nclaw-law" in flags, "substeps": sub}
    else:
        kw = dict(CANONICAL.get(mat, {}))
        print(f"[compare] canonical settings for {mat}: {kw}")
    main(mat, trajectories=traj, bisect="--bisect" in flags,
         tier=tier_flag, device=dev, **kw)

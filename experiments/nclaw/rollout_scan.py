"""Derivative-free system identification by rollout scan, one parameter.

The weak-form estimators refuse at the positions-only tier for the two
plastic materials. Their momentum fits need the per-particle elastic state,
and rebuilding that state from 1000-particle positions leaves spatially
correlated direction and volume errors that the residual check catches; see
replay.py. What survives
at this tier is the evaluation metric itself: roll the engine at a candidate
parameter from the tier's own frame-0 seed and score position MSE against the
measured identify trajectory. The scan is one dimensional and never
differentiates the simulator. NCLaw's sys-id baseline optimizes the same
objective with a differentiable MPM, so this uses the same information.

Elastic parameters are ASSUMED at their configured values and stated in the
result; only the plastic parameter is scanned. A 5 degree friction offset or
a 2x yield offset costs a factor 25 to 45 in MSE on the dataset throws, so a
coarse grid plus two refinement rounds resolves the parameter.
"""
from __future__ import annotations

from pathlib import Path


def scan_parameter(material: str, identify_dump: str | Path, param: str,
                   coarse: list[float], theta_base: dict,
                   refine_rounds: list[list[float]], mode: str = "add",
                   nclaw_bc: bool = True, nclaw_law: bool = False,
                   substeps: int | None = None, device: str = "cpu",
                   log=print) -> dict:
    """Best value of one parameter by position MSE against the identify dump.

    ``coarse`` is the blind first grid; each entry of ``refine_rounds`` is a
    list of offsets (mode "add") or factors (mode "mul") applied to the best
    value so far. A rerun re-scores cached rollouts under out/nclaw_suite/scan
    /scan; it does not re-simulate them -- the cache key is the candidate
    value and compatibility flags, not ``device``, so a value already
    simulated on one device is reused as-is if the same value is scanned
    again on another.
    """
    from experiments.nclaw.suite import OUT, cloud_from_dump, nclaw_position_mse, run_scene

    identify_dump = Path(identify_dump)
    cloud = cloud_from_dump(identify_dump)
    scan_dir = OUT / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    cfg = (("_nclawbc" if nclaw_bc else "") + ("_nclawlaw" if nclaw_law else "")
           + (f"_sub{substeps}" if substeps is not None else ""))
    tried: dict[float, float] = {}
    sim_timings: list[dict] = []

    def score(value: float) -> float:
        value = float(value)
        if value in tried:
            return tried[value]
        pred = scan_dir / f"scan_{material}_{param}_{value:g}{cfg}.npz"
        if not pred.exists():
            t = {}
            run_scene(material, "dataset", pred,
                      theta={**theta_base, param: value}, cloud=cloud,
                      nclaw_bc=nclaw_bc, nclaw_law=nclaw_law,
                      substeps=substeps, device=device,
                      log=lambda *a: None, timing=t)
            sim_timings.append(t)
        sc = nclaw_position_mse(identify_dump, pred, strict=False)
        import numpy as np
        n_expected = int(np.load(identify_dump)["x"].shape[0])
        mse = float("inf") if sc["n_frames"] < n_expected else float(sc["mse"])
        tried[value] = mse
        log(f"[scan] {material} {param}={value:g} mse={mse:.3e}")
        return mse

    for v in coarse:
        score(v)
    best = min(tried, key=tried.get)
    for offsets in refine_rounds:
        for o in offsets:
            score(best + o if mode == "add" else best * o)
        best = min(tried, key=tried.get)
    detail_s = {k: sum(t.get(k, 0.0) for t in sim_timings)
               for k in ("setup_s", "step_s", "snapshot_s", "finalize_s")}
    return {
        param: float(best),
        "estimator": "rollout_scan",
        "refused": False,
        "mse_at_best": tried[best],
        "scan": {f"{v:g}": tried[v] for v in sorted(tried)},
        "n_rollouts": len(tried),
        "n_simulated": len(sim_timings),   # vs n_rollouts: the rest were cached npz
        "assumed": dict(theta_base),
        "mode": mode,
        "device": device,
        "wall_simulate_s": sum(detail_s.values()),
        "wall_simulate_detail_s": detail_s,
        "objective": ("position MSE of an engine rollout against the identify "
                      "trajectory, seeded from the tier's own frame-0 state"),
    }

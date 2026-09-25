"""Identification and open-loop shaping toward a common analytic hexagonal prism.

Stages: prepare; plan --model MODEL; evaluate; verify. Run from the repo root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import minimize

from experiments.robotics.hex_shaping_check import execute
from experiments.robotics.hex_shaping_control import snapshot_sources, target_prism
from experiments.robotics.hex_shaping_pilot import shape
from experiments.robotics.plastic_shaping_study import (
    ROOT, NOMINAL, press, identify, chamfer_mm, save_json, save_npz,
)

OUT = ROOT/"out/hex_shaping_study_20260908"
TRUTH = {"A": dict(E=80000., nu=.30, yield_stress=1000.),
         "B": dict(E=240000., nu=.30, yield_stress=10000.)}
MODELS = ("nominal", "identified_A", "identified_B", "true_A", "true_B")
PROTOCOL = dict(truth=TRUTH, nominal=NOMINAL, initial_gaps_mm=[80., 85., 87.],
    max_evals=32, bounds_mm=[[40., 115.]]*3, angles_deg=[0., 60., 120.], cycles=2,
    target_diameter_m=.09, target_volume="Baseline initial particle volume",
    n_grid=48, grid_lim=.30, dt=1e-4, control_dt=.0008, ppc=2, seed=0,
    initial_size_m=[.12, .08, .06], density=1000.,
    opening_m=.16, speed_m_s=.10, inter_squeeze_release_s=.30,
    objective_release_s=.30, reported_final_release_s=1.,
    search="Bounded Nelder-Mead, common initial simplex, equal evaluation budget",
    rounding="None; selected gaps stored to 1e-6 mm",
    metric="Mean of two directed mean unsquared full-cloud NN distances, mm",
    observations=["x", "v", "elastic F", "mass", "reference volume"],
    probe=dict(depth_m=.014, speed_m_s=.08),
    validation=dict(depth_m=.024, speed_m_s=.12),
    contact="Sticky floor and rotated SDF jaws; instantaneous activation/removal",
    scope="One common geometric target, two simulated materials, known constitutive family")


def freeze(folder, device):
    snapshot_sources(folder, PROTOCOL)
    files = [Path(__file__), ROOT/"experiments/robotics/hex_shaping_check.py",
             ROOT/"experiments/nclaw/suite.py"]
    files += list((ROOT/"src/ident").rglob("*.py"))
    files += list((ROOT/"src/common").rglob("*.py"))
    hashes = json.loads((folder/"source_sha256.json").read_text())
    for p in files:
        rel = str(p.relative_to(ROOT))
        dest = folder/"source_snapshot"/rel
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed: {rel}; use a fresh output folder")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        hashes[rel] = digest
    save_json(folder/"source_sha256.json", hashes)
    save_json(folder/"device.json", dict(device=device))


def check_sources(folder):
    assert json.loads((folder/"protocol.json").read_text()) == PROTOCOL
    for rel, digest in json.loads((folder/"source_sha256.json").read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        assert hashlib.sha256((folder/"source_snapshot"/rel).read_bytes()).hexdigest() == digest, rel


def prepare(folder, device):
    freeze(folder, device)
    save_npz(folder/"target.npz", target_prism())
    fits = {}
    for name, law in TRUTH.items():
        p = folder/f"probe_{name}.npz"
        if not p.exists():
            save_npz(p, press(law, device=device))
        fits[name] = identify(dict(np.load(p)))
        save_json(folder/f"identification_{name}.json", fits[name])
        print("Identified", name, fits[name]["law"], flush=True)
    laws = {"nominal": NOMINAL, **{f"true_{n}": law for n, law in TRUTH.items()},
            **{f"identified_{n}": fit["law"] for n, fit in fits.items()}}
    save_json(folder/"models.json", laws)
    for key, law in laws.items():
        p = folder/f"validation_{key}.npz"
        if not p.exists():
            save_npz(p, press(law, device=device, depth=.024, speed=.12, record_state=False))
        print("Validated press", key, flush=True)


def plan(folder, model, device):
    check_sources(folder)
    law = json.loads((folder/"models.json").read_text())[model]
    dest = folder/"plans"/model
    dest.mkdir(parents=True, exist_ok=True)
    target = np.load(folder/"target.npz")["x"]
    rows = []

    def objective(gaps):
        gaps = np.round(gaps, 6)
        path = dest/(hashlib.sha256(gaps.tobytes()).hexdigest()[:16]+".npz")
        if not path.exists():
            save_npz(path, shape(law, gaps/1000, device=device, cycles=2))
        data = np.load(path)
        value = chamfer_mm(data["x"], target)
        rows.append(dict(gaps_mm=gaps.tolist(), predicted_error_mm=value,
                         file=str(path.relative_to(folder))))
        save_json(dest/"evaluations.json", rows)
        print(json.dumps(dict(model=model, evaluation=len(rows), error_mm=value,
                              gaps_mm=gaps.tolist())), flush=True)
        return value

    initial = np.array(PROTOCOL["initial_gaps_mm"])
    simplex = np.tile(initial, (4, 1))
    simplex[1:] -= 10*np.eye(3)
    result = minimize(objective, initial, method="Nelder-Mead", bounds=PROTOCOL["bounds_mm"],
        options=dict(maxfev=PROTOCOL["max_evals"], xatol=.25, fatol=.005, initial_simplex=simplex))
    best = min(rows, key=lambda r: r["predicted_error_mm"])
    save_json(dest/"selected.json", dict(**best, model=model, law=law,
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        evaluations=len(rows)))


def evaluate(folder, device):
    check_sources(folder)
    target = np.load(folder/"target.npz")["x"]
    results = []
    for name, law in TRUTH.items():
        for method, model in [("nominal", "nominal"), ("identified", f"identified_{name}"),
                              ("oracle", f"true_{name}")]:
            selected = json.loads((folder/"plans"/model/"selected.json").read_text())
            path = folder/f"execution_{name}_{method}.npz"
            if not path.exists():
                save_npz(path, execute(law, selected["gaps_mm"], 2, device=device))
            data = np.load(path)
            row = dict(material=name, method=method, model=model,
                gaps_mm=selected["gaps_mm"], predicted_error_mm=selected["predicted_error_mm"],
                error_03s_mm=chamfer_mm(data["x"], target),
                executed_error_mm=chamfer_mm(data["x_after_1s"], target),
                change_03_to_1s_mm=chamfer_mm(data["x"], data["x_after_1s"]),
                rms_speed_1s_m_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"]**2, axis=1)))))
            results.append(row)
            save_json(folder/"shaping_results.json", results)
            print(json.dumps(row), flush=True)
    # Apply the selected actions without retuning at each numerical setting.
    for tag, grid, dt in [("half_dt", 48, 5e-5), ("grid64", 64, 1e-4)]:
        save_npz(folder/"numerics"/f"target_{tag}.npz", target_prism(grid))
        for row in results:
            path = folder/"numerics"/f"{tag}_{row['material']}_{row['method']}.npz"
            if not path.exists():
                save_npz(path, execute(TRUTH[row["material"]], row["gaps_mm"], 2,
                                      device=device, n_grid=grid, dt=dt))
            print("Numerics", path.name, flush=True)


def verify(folder):
    check_sources(folder)
    target = np.load(folder/"target.npz")["x"]
    counts = {}
    for model in MODELS:
        dest = folder/"plans"/model
        rows = json.loads((dest/"evaluations.json").read_text())
        counts[model] = len(rows)
        for row in rows:
            data = np.load(folder/row["file"])
            assert int(data["inverted_count"]) == 0
            np.testing.assert_allclose(chamfer_mm(data["x"], target), row["predicted_error_mm"], rtol=1e-12)
        selected = json.loads((dest/"selected.json").read_text())
        assert selected["gaps_mm"] == min(rows, key=lambda r: r["predicted_error_mm"])["gaps_mm"]
    results = json.loads((folder/"shaping_results.json").read_text())
    assert len(results) == 6
    summary = dict(force_relative_l2={}, mean_shaping_error_mm={}, numerical_errors_mm={},
                   search_evaluations=counts, oracle_replay_difference_mm={})
    for name in TRUTH:
        actual = np.load(folder/f"validation_true_{name}.npz")["force"]
        summary["force_relative_l2"][name] = {}
        for method, model in [("nominal", "nominal"), ("identified", f"identified_{name}")]:
            predicted = np.load(folder/f"validation_{model}.npz")["force"]
            summary["force_relative_l2"][name][method] = float(np.linalg.norm(actual-predicted)/np.linalg.norm(actual))
    for row in results:
        data = np.load(folder/f"execution_{row['material']}_{row['method']}.npz")
        assert int(data["inverted_count"]) == 0
        assert all(np.all(np.isfinite(data[k])) for k in data.files)
        np.testing.assert_allclose(chamfer_mm(data["x_after_1s"], target), row["executed_error_mm"], rtol=1e-12)
        if row["method"] == "oracle":
            selected = json.loads((folder/"plans"/row["model"]/"selected.json").read_text())
            difference = chamfer_mm(data["x"], np.load(folder/selected["file"])["x"])
            assert difference < .05
            summary["oracle_replay_difference_mm"][row["material"]] = difference
        for tag in ["half_dt", "grid64"]:
            d = np.load(folder/"numerics"/f"{tag}_{row['material']}_{row['method']}.npz")
            t = np.load(folder/"numerics"/f"target_{tag}.npz")
            assert int(d["inverted_count"]) == 0
            assert all(np.all(np.isfinite(d[k])) for k in d.files)
            summary["numerical_errors_mm"].setdefault(tag, {})[f"{row['material']}_{row['method']}"] = chamfer_mm(d["x_after_1s"], t["x"])
    for method in ["nominal", "identified", "oracle"]:
        summary["mean_shaping_error_mm"][method] = float(np.mean([r["executed_error_mm"] for r in results if r["method"] == method]))
    summary["nominal_over_identified"] = summary["mean_shaping_error_mm"]["nominal"]/summary["mean_shaping_error_mm"]["identified"]
    save_json(folder/"summary.json", summary)
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json" and p.suffix != ".log"}
    save_json(folder/"checksums.json", hashes)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "plan", "evaluate", "verify"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=MODELS, default="nominal")
    args = parser.parse_args()
    folder = args.out.resolve()
    if args.stage == "prepare":
        prepare(folder, args.device)
    elif args.stage == "plan":
        plan(folder, args.model, args.device)
    elif args.stage == "evaluate":
        evaluate(folder, args.device)
    else:
        verify(folder)


if __name__ == "__main__":
    main()

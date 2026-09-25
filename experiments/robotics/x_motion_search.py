"""Model-only search over pose/opening sequences, followed by frozen cross-tests.

Action templates and bounds are experiment data. Every candidate uses the
same prescribed-motion executor. No swapped outcome enters the objective.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.optimize import minimize

from experiments.robotics import x_motion_shaping as core
from experiments.robotics.plastic_shaping_study import save_json, save_npz
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm

ROOT = core.ROOT


def prepare(source, spec_path, folder):
    assert not folder.exists(), "Use a fresh study directory"
    for rel, sha in core.read(source / "checksums.json").items():
        assert core.digest(source / rel) == sha, rel
    folder.mkdir(parents=True)
    inputs = folder / "inputs"
    inputs.mkdir()
    for name in ["models.json", "specimen.npz", "target.vtp"]:
        shutil.copy2(source / "inputs" / name, inputs / name)
    shutil.copy2(source / "summary.json", inputs / "baseline_summary.json")
    spec = core.read(spec_path)
    sources = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "experiments/robotics").glob("*.py"))
    for p in sources:
        dest = folder / "source_snapshot" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    protocol = dict(spec=spec, source_archive=str(source.resolve()),
        source_manifest_sha256=core.digest(source / "checksums.json"),
        scene=core.geometry.CONFIG, grid=64, dt=.00005,
        objective="Mean 3D surface error in the identified model, 1 s after withdrawal; all components, no registration",
        optimizer="Bounded Nelder-Mead; best observed candidate within budget; no convergence claim",
        execution="One prescribed-motion primitive; no force or shape feedback",
        material_and_target_changes=False, swapping_used_in_objective=False,
        development_selection="Action families may be compared using all four outcomes; these are development experiments, not blind tests",
        input_sha256={str(p.relative_to(folder)): core.digest(p) for p in inputs.iterdir()},
        source_sha256={str(p.relative_to(ROOT)): core.digest(p) for p in sources},
        packages={p: importlib.metadata.version(p) for p in
                  ["numpy", "scipy", "warp-lang", "pyvista", "matplotlib", "scikit-image"]})
    save_json(folder / "protocol.json", protocol)
    print(json.dumps(dict(prepared=str(folder), spec=spec)), flush=True)


def check(folder):
    p = core.read(folder / "protocol.json")
    assert p["scene"] == json.loads(json.dumps(core.geometry.CONFIG))
    for rel, sha in p["input_sha256"].items():
        assert core.digest(folder / rel) == sha, rel
    for rel, sha in p["source_sha256"].items():
        assert core.digest(ROOT / rel) == core.digest(folder / "source_snapshot" / rel) == sha, rel
    return p


def actions(spec, parameters):
    result = []
    for template in spec["actions"]:
        offset = np.array(template.get("offset_xy_mm", [0., 0.]))
        if "offset_parameter" in template:
            offset += parameters[template["offset_parameter"]] * np.array(template["offset_direction"])
        result.append(dict(angle_deg=template["angle_deg"], gap_mm=float(parameters[template["gap_parameter"]]),
                           offset_xy_mm=offset.tolist()))
    return result


def rollout(folder, path, law, command, phases, device, context, grid=64, dt=.00005):
    assert not path.with_suffix(".config.json").exists(), "Preserve previous attempts"
    config = dict(**context, law=law, device=device, grid=grid, dt=dt,
                  protocol_sha256=core.digest(folder / "protocol.json"))
    save_json(path.with_suffix(".config.json"), config)
    start = time.monotonic()
    initial = core.arrays(folder / "inputs/specimen.npz") if grid == 64 else dict(zip(["initial", "vol0"], core.geometry.specimen(grid)))
    data = core.simulate(law, command, phases, initial, device, grid, dt)
    save_npz(path, data)
    save_json(path.with_suffix(".phases.json"), phases)
    error = mesh_distance_mm(surface(data["x_after_1s"], data["vol0"], h=.00125),
                             pv.read(folder / "inputs/target.vtp"))
    record = dict(**config, surface_mm=error, data_sha256=core.digest(path),
        file=str(path.relative_to(folder)), elapsed_s=time.monotonic() - start,
        completed_pinches=len(command["gaps_mm"]), duration_s=float(command["time"][-1]))
    save_json(path.with_suffix(".json"), record)
    return record


def plan(folder, material, device):
    protocol = check(folder)
    assert not (folder / "execution_plan.json").exists()
    spec = protocol["spec"]
    dest = folder / "plans" / material
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / "evaluations.json").exists(), "Preserve the original optimizer history"
    model = f"identified_{material}"
    law = core.read(folder / "inputs/models.json")[model]
    rows = []
    cache = {}

    def objective(x):
        parameters = np.round(x, 6)
        key = tuple(parameters)
        cached = key in cache
        if not cached:
            act = actions(spec, parameters)
            command, phases = core.motion(act)
            context = dict(model=model, material=material, parameters=parameters.tolist(), actions=act)
            path = dest / f"trial_{len(cache):03d}.npz"
            cache[key] = rollout(folder, path, law, command, phases, device, context)
        row = dict(**cache[key], reused_cached_result=cached)
        rows.append(row)
        save_json(dest / "evaluations.json", rows)
        best = min(rows, key=lambda r: r["surface_mm"])
        save_json(dest / "best_so_far.json", best)
        print(json.dumps(dict(material=material, evaluation=len(rows), unique_simulations=len(cache),
            surface_mm=row["surface_mm"], best_mm=best["surface_mm"], parameters=parameters.tolist())), flush=True)
        return row["surface_mm"]

    initial = np.array(spec["start"][material], dtype=float)
    bounds = np.array(spec["bounds"], dtype=float)
    simplex = np.tile(initial, (len(initial) + 1, 1))
    for i, step in enumerate(spec["simplex_steps"]):
        simplex[i + 1, i] += -step if initial[i] - step >= bounds[i, 0] else step
    assert np.all(simplex >= bounds[:, 0]) and np.all(simplex <= bounds[:, 1])
    result = minimize(objective, initial, method="Nelder-Mead", bounds=bounds,
        options=dict(maxfev=spec["max_evaluations"], initial_simplex=simplex, xatol=0., fatol=0.))
    best = min(rows, key=lambda r: r["surface_mm"])
    save_json(dest / "selected.json", dict(**best, evaluations=len(rows), unique_simulations=len(cache),
        optimizer_success=bool(result.success), optimizer_message=str(result.message)))


def freeze(folder):
    p = check(folder)
    assert not (folder / "execution_plan.json").exists()
    selected = {n: core.read(folder / f"plans/{n}/selected.json") for n in "AB"}
    for n, row in selected.items():
        assert row["model"] == f"identified_{n}"
        assert row["evaluations"] == p["spec"]["max_evaluations"]
        assert core.digest(folder / row["file"]) == row["data_sha256"]
    save_json(folder / "execution_plan.json", dict(selected=selected,
        protocol_sha256=core.digest(folder / "protocol.json"),
        selection="Lowest identified-model target error only; complete plans frozen before cross-testing"))


def evaluate(folder, material, device):
    check(folder)
    frozen = core.read(folder / "execution_plan.json")
    assert frozen["protocol_sha256"] == core.digest(folder / "protocol.json")
    law = core.read(folder / "inputs/models.json")[f"true_{material}"]
    for planned in "AB":
        row = frozen["selected"][planned]
        stored = core.arrays(folder / row["file"])
        command = {k: stored[k] for k in ["time", "tool_centers", "phase_id", "pinch_id", "start_pose", "gaps_mm"]}
        phases = core.read((folder / row["file"]).with_suffix(".phases.json"))
        context = dict(material=material, planned_for=planned, selected_data_sha256=row["data_sha256"],
                       execution_plan_sha256=core.digest(folder / "execution_plan.json"))
        record = rollout(folder, folder / f"{material}_plan_{planned}.npz", law, command, phases, device, context)
        print(json.dumps(record), flush=True)


def audit(folder):
    protocol = check(folder)
    frozen = core.read(folder / "execution_plan.json")
    target = pv.read(folder / "inputs/target.vtp")
    initial = core.arrays(folder / "inputs/specimen.npz")
    rows = []
    for actual in "AB":
        for planned in "AB":
            path = folder / f"{actual}_plan_{planned}.npz"
            data = core.arrays(path)
            row = core.read(path.with_suffix(".json"))
            selected = frozen["selected"][planned]
            reference = core.arrays(folder / selected["file"])
            assert core.digest(path) == row["data_sha256"]
            assert row["law"] == core.read(folder / "inputs/models.json")[f"true_{actual}"]
            assert row["material"] == actual and row["planned_for"] == planned
            for k in ["time", "tool_centers", "phase_id", "pinch_id", "start_pose", "gaps_mm"]:
                np.testing.assert_array_equal(data[k], reference[k])
            for k in initial:
                np.testing.assert_array_equal(data[k], initial[k])
            assert all(np.all(np.isfinite(v)) for v in data.values()) and data["inverted_count"] == 0
            phases = core.read(path.with_suffix(".phases.json"))
            off_contact = [i for i, p in enumerate(phases) if p["name"].endswith(":reposition") or p["name"] == "final:wait"]
            assert np.max(abs(data["reaction_force"][np.isin(data["phase_id"], off_contact)])) == 0
            gaps = (np.linalg.norm(data["tool_centers"][:, 1] - data["tool_centers"][:, 0], axis=1) - .028) * 1000
            for i, opening in enumerate(data["gaps_mm"]):
                indices = np.flatnonzero(data["pinch_id"] == i)
                np.testing.assert_allclose(gaps[indices[-1]], opening, atol=1e-7)
                assert np.linalg.norm(data["tool_velocity"][indices], axis=-1).max() <= .020000001
            mesh = surface(data["x_after_1s"], data["vol0"], h=.00125)
            np.testing.assert_allclose(mesh_distance_mm(mesh, target), row["surface_mm"], rtol=1e-10)
            components = mesh.connectivity(extraction_mode="all")
            rows.append(dict(**row, surface_components=int(components.cell_data["RegionId"].max()) + 1,
                max_sampled_floor_penetration_mm=float(max(0, .01 - data["extent_samples"][:, 3].min()) * 1000)))
    scores = {(r["material"], r["planned_for"]): r["surface_mm"] for r in rows}
    goals = dict(matched_A=scores["A", "A"] <= 1.031, matched_B=scores["B", "B"] <= 1.326,
        contrast_A=scores["A", "B"] - scores["A", "A"] >= .30,
        contrast_B=scores["B", "A"] - scores["B", "B"] >= .5375729319678391)
    save_json(folder / "summary.json", dict(results=rows, goals=goals,
        all_goals_met=all(goals.values()), all_four_cases_retained=True,
        protocol_sha256=core.digest(folder / "protocol.json"),
        interpretation="Development comparison; shape contrasts are not an optimization objective or an independent holdout"))
    return rows


def report(folder):
    rows = audit(folder)
    spec = core.read(folder / "protocol.json")["spec"]
    meshes = {"target": pv.read(folder / "inputs/target.vtp")}
    for r in rows:
        key = f"{r['material']}_plan_{r['planned_for']}"
        d = core.arrays(folder / f"{key}.npz")
        meshes[key] = surface(d["x_after_1s"], d["vol0"], h=.00125)
    output = folder / "preview"
    output.mkdir(exist_ok=False)
    for scale in [.055, .061, .068, .076, .085, .10]:
        masks = {k: core.render(m, scale, "black", mask=True)[..., :3].mean(-1) < 128 for k, m in meshes.items()}
        if all(not (m[:10].any() or m[-10:].any() or m[:, :10].any() or m[:, -10:].any()) for m in masks.values()):
            break
    else:
        raise RuntimeError("Full surfaces do not fit")
    yy, xx = np.nonzero(np.logical_or.reduce(list(masks.values())))
    y0, y1, x0, x1 = int(yy.min()) - 10, int(yy.max()) + 11, int(xx.min()) - 10, int(xx.max()) + 11
    images = {}
    for key, mesh in meshes.items():
        assert masks[key][y0:y1, x0:x1].sum() == masks[key].sum()
        image = core.render(mesh, scale, "#b5b6b0" if key == "target" else core.COLORS[key[0]])
        images[key] = image[y0:y1, x0:x1]
        plt.imsave(output / f"{key}.png", images[key])
        mesh.save(output / f"{key}.vtp")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.3))
    for i, actual in enumerate("AB"):
        for j, planned in enumerate([None, actual, "B" if actual == "A" else "A"]):
            ax = axes[i, j]
            key = "target" if planned is None else f"{actual}_plan_{planned}"
            ax.imshow(images[key]); ax.axis("off")
            if planned is not None:
                ax.contour(masks["target"][y0:y1, x0:x1], [.5], colors="#253139", linewidths=.9, linestyles=[(0, (3, 2))])
                r = next(r for r in rows if r["material"] == actual and r["planned_for"] == planned)
                ax.text(.5, -.035, f"{r['surface_mm']:.3f} mm", transform=ax.transAxes, ha="center", va="top", fontsize=14)
            if i == 0:
                ax.set_title(["Target", "Plan using matched ID", "Plan using swapped ID"][j], pad=15, fontsize=13)
            if j == 0:
                ax.text(-.13, .5, actual, transform=ax.transAxes, ha="center", va="center", fontsize=21,
                        weight="bold", color={"A": "#2375aa", "B": "#c96932"}[actual])
    fig.suptitle(spec["title"], y=.98, fontsize=17, weight="bold", color="#202a32")
    fig.text(.5, .927, f"{len(spec['actions'])} pinches · prescribed openings and poses · no force feedback", ha="center", color="#53606a")
    fig.text(.5, .018, "1 s after withdrawal · dashed outline: target · labels: mean 3D surface error", ha="center", fontsize=11, color="#53606a")
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.10, wspace=.19, hspace=.22)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"comparison.{suffix}", dpi=220)
    plt.close(fig)
    save_json(output / "provenance.json", dict(common_camera_scale_m=scale,
        crop=[y0, y1, x0, x1], every_surface_retained=True, all_components_included=True,
        individual_registration_or_rescaling=False, rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "plan", "freeze", "evaluate", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "out/x_motion_shaping_20260911")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--material", choices=["A", "B"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.source, args.spec, args.out)
    elif args.stage in ["plan", "evaluate"]:
        globals()[args.stage](args.out, args.material, args.device)
    else:
        globals()[args.stage](args.out)

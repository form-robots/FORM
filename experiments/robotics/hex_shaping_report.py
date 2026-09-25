"""Render and audit the separate hexagonal-shaping feasibility experiment.

Run after the pilot and its numerical checks:
    .venv/bin/python -m experiments.robotics.hex_shaping_report
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.robotics.hex_shaping_pilot import OUT, ROOT
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import chamfer_mm, save_json


def image_of(data, key):
    import pyvista as pv

    floor = float(data["floor"])
    plotter = pv.Plotter(off_screen=True, window_size=(640, 560))
    plotter.set_background("white")
    plotter.add_mesh(surface(data[key], data["vol0"]), color="#cfb58c",
                     smooth_shading=True, ambient=.35, diffuse=.75,
                     specular=.12, specular_power=18)
    plotter.add_mesh(pv.Plane(center=(.15, .15, floor), direction=(0, 0, 1),
                    i_size=.17, j_size=.15), color="#eef0f0", lighting=False)
    focal = np.array([.15, .15, floor+.045])
    plotter.camera.position = focal + np.array([.9, -2.8, 2.9])*.20
    plotter.camera.focal_point = focal
    plotter.camera.up = (0, 0, 1)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = .079
    plotter.enable_anti_aliasing("ssaa")
    result = plotter.screenshot(return_img=True)
    plotter.close()
    return result


def render(folder):
    fig, axes = plt.subplots(2, 3, figsize=(8.1, 5.2))
    fig.subplots_adjust(left=.085, right=.995, top=.91, bottom=.09, wspace=.01, hspace=.035)
    for row, name in enumerate("AB"):
        single = np.load(folder/f"true_{name}_90.npz")
        repeat = np.load(folder/"repeat"/f"true_{name}_90.npz")
        for col, (data, key) in enumerate([(single, "initial"), (single, "x"), (repeat, "x")]):
            ax = axes[row, col]
            ax.imshow(image_of(data, key))
            ax.axis("off")
            if row == 0:
                ax.set_title(["Initial", "Three squeezes", "Six squeezes"][col], fontsize=11)
            if col == 0:
                ax.text(-.08, .5, f"{name}\n{1 if name == 'A' else 10} kPa",
                        ha="center", va="center", fontsize=11, transform=ax.transAxes)
    fig.text(.54, .027, "90 mm gaps at 0°, 60°, 120°; outcomes 0.30 s after release.",
             ha="center", fontsize=9)
    fig.savefig(folder/"pilot_preview.png", dpi=180, facecolor="white")
    fig.savefig(folder/"pilot_preview.pdf", facecolor="white")
    plt.close(fig)


def audit(folder):
    def physics_functions(path):
        result = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in {"shape", "jaw_pose", "box_sdf"}:
                if (isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
                result[node.name] = ast.dump(node)
        return result

    current_physics = physics_functions(ROOT/"experiments/robotics/hex_shaping_pilot.py")
    cases = []
    for group in [folder, folder/"deeper", folder/"repeat", folder/"unequal"]:
        for p in sorted(group.glob("true_*.npz")):
            d = np.load(p)
            assert int(d["inverted_count"]) == 0
            assert all(np.all(np.isfinite(d[k])) for k in d.files)
            assert len(d["x"]) == len(d["initial"]) == len(d["vol0"])
            def width(x):
                return float(1000*(np.quantile(x[:, 0], .99)-np.quantile(x[:, 0], .01)))
            cases.append(dict(file=str(p.relative_to(folder)),
                gap_sequence_mm=(d["gaps"]*1000).tolist(),
                first_pressed_x_width_mm=width(d["stage_0_pressed"]),
                first_released_x_width_mm=width(d["stage_0_released"]),
                final_x_width_mm=width(d["x"]),
                final_particle_height_mm=float(1000*(d["x"][:, 2].max()-d["floor"])),
                final_max_speed_m_s=float(np.linalg.norm(d["v"], axis=1).max())))
    assert len(cases) == 24
    numerical = []
    for name in "AB":
        base = np.load(folder/"repeat"/f"true_{name}_90.npz")
        for tag in ["half_dt", "grid64"]:
            data = np.load(folder/"numerics"/f"true_{name}_{tag}.npz")
            assert int(data["inverted_count"]) == 0
            assert all(np.all(np.isfinite(data[k])) for k in data.files)
            x = data["x"].copy()
            # Floor is 3*dx in each grid. Express both in the physical floor frame.
            x[:, 2] -= float(data["floor"])
            xb = base["x"].copy()
            xb[:, 2] -= float(base["floor"])
            numerical.append(dict(material=name, check=tag,
                floor_relative_cloud_chamfer_mm=chamfer_mm(x, xb),
                note="Floor-relative clouds; the grid change also changes particle sampling. Not a convergence estimate."))
    initial_hashes = json.loads((folder/"manuscript_before.json").read_text())
    for rel, digest in initial_hashes.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    for group in [folder, folder/"deeper", folder/"repeat", folder/"unequal", folder/"numerics"]:
        assert physics_functions(group/"source_snapshot/experiments/robotics/hex_shaping_pilot.py") == current_physics
        sources = json.loads((group/"source_sha256.json").read_text())
        for rel, digest in sources.items():
            assert hashlib.sha256((group/"source_snapshot"/rel).read_bytes()).hexdigest() == digest
            if rel.startswith("src/"):
                assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
    save_json(folder/"audit.json", dict(passed=True, feasibility_runs=cases,
                numerical_checks=numerical, manuscript_and_figure_unchanged=True,
                scientific_functions_unchanged_across_batches=True,
                archived_launch_sources_verified=True, python=sys.version,
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    print(json.dumps(dict(passed=True, feasibility_runs=len(cases),
                         numerical_checks=numerical, manuscript_unchanged=True), indent=2))


def numerical_plot(folder):
    from scipy.spatial import ConvexHull

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, name in zip(axes, "AB", strict=True):
        for tag, label, color in [("base", "48³, 0.1 ms", "#333333"),
                                  ("half_dt", "48³, 0.05 ms", "#ce7b34"),
                                  ("grid64", "64³, 0.1 ms", "#277eb4")]:
            path = (folder/"repeat"/f"true_{name}_90.npz" if tag == "base"
                    else folder/"numerics"/f"true_{name}_{tag}.npz")
            d = np.load(path)
            z = d["x"][:, 2]-d["floor"]
            xy = d["x"][abs(z-np.quantile(z, .5)) < .003, :2]*1000-150
            hull = ConvexHull(xy)
            xy = xy[np.r_[hull.vertices, hull.vertices[0]]]
            ax.plot(xy[:, 0], xy[:, 1], label=label, color=color)
        ax.set(title=f"Material {name}: six squeezes, 90 mm gaps", aspect="equal",
               xlim=(-65, 65), ylim=(-65, 65), xlabel="x (mm)", ylabel="y (mm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.savefig(folder/"numerical_cross_sections.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    folder = args.out.resolve()
    render(folder)
    if not args.render_only:
        audit(folder)
        numerical_plot(folder)
    for rel in ["experiments/robotics/hex_shaping_pilot.py",
                "experiments/robotics/hex_shaping_report.py",
                "experiments/robotics/plastic_shaping_figure.py",
                "tests/test_hex_shaping_pilot.py", "docs/hex_shaping_pilot.md"]:
        dest = folder/"source_snapshot_final"/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    # Bundle all current engine Python sources as well as each run's launch snapshot.
    for src in (ROOT/"src/warpmpm").rglob("*.py"):
        dest = folder/"source_snapshot_final"/src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json"}
    save_json(folder/"checksums.json", hashes)


if __name__ == "__main__":
    main()

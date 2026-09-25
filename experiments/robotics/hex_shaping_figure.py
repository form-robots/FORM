"""Four-panel paper figure from the verified hexagonal identification study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial.transform import Rotation

from experiments.robotics.hex_shaping_study import OUT, ROOT
from experiments.robotics.plastic_shaping_study import save_json
from experiments.robotics.plastic_shaping_figure import (
    panel_a, label_panel, surface, COLORS, METHOD_COLORS, COLUMNS, ROWS,
)


def prism(target):
    import pyvista as pv
    angles = np.deg2rad(np.arange(30, 390, 60))
    xy = np.column_stack((np.cos(angles), np.sin(angles)))*float(target["diameter"])/np.sqrt(3)+.15
    bottom = np.column_stack((xy, np.full(6, float(target["floor"]))))
    top = bottom+np.array([0, 0, float(target["height"])])
    faces = [[6, *range(5, -1, -1)], [6, *range(6, 12)]]
    faces += [[4, i, (i+1) % 6, (i+1) % 6+6, i+6] for i in range(6)]
    return pv.PolyData(np.vstack((bottom, top)), np.concatenate(faces))


def render(data, target, path, *, color, stage=None, is_target=False):
    import pyvista as pv
    floor = float(target["floor"])
    plotter = pv.Plotter(off_screen=True, window_size=(520, 480) if stage is not None else (720, 480))
    plotter.set_background("white")
    mesh = prism(target) if is_target else surface(
        data["x_after_1s"] if stage is None else data[f"stage_{stage}_pressed"], data["vol0"])
    plotter.add_mesh(mesh, color=color, smooth_shading=not is_target,
                     ambient=.35, diffuse=.75, specular=.12 if not is_target else 0)
    plotter.add_mesh(pv.Plane(center=(.15, .15, floor), direction=(0, 0, 1),
                    i_size=.20 if stage is not None else .16, j_size=.18 if stage is not None else .15),
                    color="#eef0f0", lighting=False)
    if stage is not None:
        rot = Rotation.from_quat(data[f"stage_{stage}_quat"]).as_matrix()
        half = data["half"]
        for center in data[f"stage_{stage}_centers"]:
            matrix = np.eye(4); matrix[:3, :3] = rot; matrix[:3, 3] = center
            tool = pv.Box(bounds=np.column_stack((-half, half)).ravel()).transform(matrix, inplace=False)
            plotter.add_mesh(tool, color="#737d85", opacity=.20, ambient=.6)
            plotter.add_mesh(tool.extract_feature_edges(), color="#626f7b", line_width=1.2)
            direction = np.array([.15, .15, center[2]])-center
            direction /= np.linalg.norm(direction)
            start = center-direction*.045
            start[2] = floor+2*half[2]+.009
            plotter.add_mesh(pv.Arrow(start=start, direction=direction, scale=.04,
                                     shaft_radius=.055, tip_radius=.13, tip_length=.30),
                             color="#15191d", lighting=False)
    focal = np.array([.15, .15, floor+(.075 if stage is not None else .041)])
    plotter.camera.position = focal+np.array([1., -2.8, 4.])*.20
    plotter.camera.focal_point = focal
    plotter.camera.up = (0, 0, 1)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = .150 if stage is not None else .070
    plotter.enable_anti_aliasing("ssaa")
    img = plotter.screenshot(return_img=True)
    plt.imsave(path, img)
    if stage is None and not is_target:
        plotter.clear()
        plotter.add_mesh(prism(target), color="black", lighting=False, reset_camera=False)
        mask = binary_fill_holes(np.max(plotter.screenshot(return_img=True)[:, :, :3], axis=2) < 128)
        np.save(path.with_suffix(".target.npy"), mask)
    plotter.close()
    return img


def panel_b(panel, folder):
    label_panel(panel, "(b) Prediction under a new press")
    panel.text(.5, .83, "24 mm stroke at 120 mm/s", ha="center", va="center",
               fontsize=7.4, color="#46515a")
    ax = panel.inset_axes([.15, .245, .82, .515])
    maximum = 0
    for name in "AB":
        truth = np.load(folder/f"validation_true_{name}.npz")
        pred = np.load(folder/f"validation_identified_{name}.npz")
        ax.plot(truth["disp"]*1000, truth["force"], color=COLORS[name], lw=1.2)
        ax.plot(pred["disp"]*1000, pred["force"], color=COLORS[name], lw=0,
                marker="o", markersize=2.5, markerfacecolor="white", markeredgewidth=.8, markevery=18)
        maximum = max(maximum, truth["force"].max(), pred["force"].max())
        ax.annotate(name, xy=(21.2, float(truth["force"][-1])), xytext=(0, 5),
                    textcoords="offset points", color=COLORS[name], fontsize=7.4, fontweight="bold")
    nominal = np.load(folder/"validation_nominal.npz")
    ax.plot(nominal["disp"]*1000, nominal["force"], color="#757575", lw=1.1, ls="--")
    ax.set(xlabel="Displacement (mm)", ylabel="Force (N)", xlim=(0, 24), ylim=(0, maximum*1.14))
    ax.set_xticks([0, 8, 16, 24]); ax.tick_params(labelsize=7, pad=2, length=3)
    ax.xaxis.label.set_size(7.4); ax.yaxis.label.set_size(7.4)
    ax.xaxis.labelpad = 2; ax.yaxis.labelpad = 2
    handles = [Line2D([], [], color="#333333", lw=1.2, label="True"),
               Line2D([], [], color="#333333", marker="o", markerfacecolor="white",
                      lw=0, markersize=2.5, label="Identified"),
               Line2D([], [], color="#757575", ls="--", lw=1.1, label="Nominal")]
    panel.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
                 ncol=3, fontsize=6.8, frameon=False, handlelength=1.5,
                 columnspacing=1., handletextpad=.45, borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.7); ax.grid(axis="y", alpha=.15)


def panel_c(ax, folder, images, results):
    label_panel(ax, "(c) Planned shaping actions")
    data = np.load(folder/"execution_A_identified.npz")
    target = np.load(folder/"target.npz")
    row = next(r for r in results if r["material"] == "A" and r["method"] == "identified")
    for stage, angle in enumerate([0, 60, 120]):
        path = images/f"action_{stage}.png"
        img = render(data, target, path, color=METHOD_COLORS["identified"], stage=stage)
        ax.text(COLUMNS[stage], .83, f"{stage+1} & {stage+4}: {angle}°", ha="center", va="center", fontsize=7.4)
        ia = ax.inset_axes([COLUMNS[stage]-.16, .29, .32, .48])
        ia.imshow(img); ia.axis("off")
        ax.text(COLUMNS[stage], .22, f"Gap {row['gaps_mm'][stage]:.1f} mm", fontsize=7.1, va="center", ha="center")
    ax.text(.5, .075, "Three directions, repeated once; release after each squeeze.",
            ha="center", va="bottom", fontsize=6.8, color="#46515a")
    ax.text(.5, .005, "First pass shown. Material A, identified plan.",
            ha="center", va="bottom", fontsize=6.8, color="#46515a")


def panel_d(ax, folder, images, results):
    label_panel(ax, "(d) Target and executed shapes")
    for x, label in zip(COLUMNS, ["Target", "Nominal", "Identified"], strict=True):
        ax.text(x, .83, label, ha="center", va="center", fontsize=7.4)
    target = np.load(folder/"target.npz")
    for row, name in enumerate("AB"):
        ax.text(.005, ROWS[row]+.11, name, color=COLORS[name], fontsize=8,
                fontweight="bold", va="center", ha="left")
        for col, method in enumerate(["target", "nominal", "identified"]):
            data = target if method == "target" else np.load(folder/f"execution_{name}_{method}.npz")
            path = images/f"outcome_{name}_{method}.png"
            img = render(data, target, path, is_target=method == "target",
                         color="#b5b6b0" if method == "target" else METHOD_COLORS[method])
            ia = ax.inset_axes([COLUMNS[col]-.145, ROWS[row], .29, .29])
            ia.imshow(img); ia.axis("off")
            if method != "target":
                ia.contour(np.load(path.with_suffix(".target.npy")), levels=[.5], colors=["black"],
                           linewidths=.55, linestyles=[(0, (2.4, 1.8))], zorder=5)
                result = next(r for r in results if r["material"] == name and r["method"] == method)
                ax.text(COLUMNS[col], ROWS[row]-.022, f"{result['executed_error_mm']:.2f} mm",
                        ha="center", va="top", fontsize=7.2, zorder=20)
    ax.text(.5, .025, "Shown 1 s after release. Dashed contour: common target.",
            ha="center", va="bottom", fontsize=6.8, color="#46515a")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dest", type=Path, default=OUT/"identification_plastic_shaping")
    args = parser.parse_args()
    folder, dest = args.out.resolve(), args.dest.resolve()
    summary = json.loads((folder/"summary.json").read_text())
    assert sum(summary["search_evaluations"].values()) == 160
    results = json.loads((folder/"shaping_results.json").read_text())
    assert len(results) == 6
    renderer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    images = folder/"renders"/renderer_hash[:12]; images.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(2, 2, left=.02, right=.985, bottom=.025, top=.99, wspace=.12, hspace=.12)
    a = fig.add_subplot(grid[0, 0]); panel_a(a, folder, images)
    a.texts[-1].set_text("A: E = 80 kPa, yield = 1 kPa. B: E = 240 kPa, yield = 10 kPa.")
    panel_b(fig.add_subplot(grid[0, 1]), folder)
    panel_c(fig.add_subplot(grid[1, 0]), folder, images, results)
    panel_d(fig.add_subplot(grid[1, 1]), folder, images, results)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=.02)
    fig.savefig(dest.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    for rel in ["experiments/robotics/hex_shaping_figure.py", "experiments/robotics/plastic_shaping_figure.py"]:
        p = folder/"figure_source"/rel; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, p)
    save_json(folder/"figure_provenance.json", dict(renderer_sha256=renderer_hash,
        result_sha256=hashlib.sha256((folder/"shaping_results.json").read_bytes()).hexdigest(),
        target="Exact analytic hexagonal mesh; identical for both materials",
        action="A identified execution; stages 0, 1, 2; real SDF box poses; schematic arrows",
        surface="Density isosurface .5, 2.5 mm voxels, Gaussian sigma 1.3 voxels",
        target_contour="Vector dashed black .55 pt; no white halo; fixed shared camera",
        outcome_camera=dict(focal_height_m=.041, direction=[1., -2.8, 4.], parallel_scale_m=.070),
        pdf_sha256=hashlib.sha256(dest.with_suffix(".pdf").read_bytes()).hexdigest(),
        png_sha256=hashlib.sha256(dest.with_suffix(".png").read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()

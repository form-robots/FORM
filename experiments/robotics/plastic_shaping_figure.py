"""Render the historical rectangular shaping figure from saved study data.

The active manuscript uses hex_shaping_figure.py. This renderer defaults to
the archived study directory so it cannot silently restore the old paper figure.

    .venv/bin/python -m experiments.robotics.plastic_shaping_figure

Surfaces use one fixed density threshold and physical smoothing length. Cameras,
scale, lighting and initial material appearance are shared; particles are never
rescaled or aligned to the target. Numerical errors use raw 3D particle clouds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.robotics.plastic_shaping_study import OUT

COLORS = {"A": "#2375aa", "B": "#c96932"}
DOUGH = "#cfb58c"
METHOD_COLORS = {"nominal": "#bf7858", "identified": "#4a91b4", "oracle": "#343e47"}


def surface(x, volume0, h=0.0025):
    import pyvista as pv
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    lo = np.floor((x.min(0) - 4*h) / h) * h
    hi = np.ceil((x.max(0) + 4*h) / h) * h
    shape = np.ceil((hi-lo)/h).astype(int) + 1
    idx = np.floor((x-lo)/h).astype(int)
    density = np.zeros(shape)
    np.add.at(density, tuple(idx.T), volume0 / h**3)
    density = gaussian_filter(density, sigma=1.3)
    vertices, faces, *_ = marching_cubes(density, level=0.5, spacing=(h, h, h))
    # Histogram samples represent voxel centers, hence the half-voxel offset.
    mesh = pv.PolyData(vertices + lo + h/2,
                       np.column_stack([np.full(len(faces), 3), faces]).ravel())
    return mesh


def snapshot(x, *, floor, volume0, path, color=DOUGH, target=None,
             plate_bottom=None, plate_thickness=None, press_stroke=None, boxes=None, half=None,
             closing_axis=None, mode="shape"):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True,
                         window_size=(720, 480) if mode == "shape" else (520, 480))
    plotter.set_background("white")
    mesh = surface(x, volume0)
    plotter.add_mesh(mesh, color=color, smooth_shading=True, ambient=0.35,
                     diffuse=0.75, specular=0.12, specular_power=18)
    # The finite patch makes the simulated planar support visible.
    ground = pv.Plane(center=(0.15, 0.15, floor), direction=(0, 0, 1),
                      i_size=.18 if mode == "action" else .155,
                      j_size=.15 if mode == "action" else .125)
    plotter.add_mesh(ground, color="#eef0f0", lighting=False)
    if plate_bottom is not None:
        plate = pv.Box(bounds=(0.08, 0.22, 0.10, 0.20,
                               plate_bottom, plate_bottom+plate_thickness))
        plotter.add_mesh(plate, color="#5f6d79", ambient=0.35,
                         diffuse=.6, specular=.25)
        # The recorded initial pose is a motion ghost, not a second tool.
        # Its offset is the actual press stroke, without visual exaggeration.
        if press_stroke is not None:
            raised = plate.translate((0, 0, press_stroke), inplace=False)
            plotter.add_mesh(raised, color="#9abbd0", opacity=.10, lighting=False)
            # Tubes retain a readable stroke when the snapshot is reduced to
            # publication size; thin OpenGL wireframe lines faded under SSAA.
            outline = raised.extract_feature_edges().tube(radius=.0008, n_sides=12)
            plotter.add_mesh(outline, color="#4b7d9b", opacity=1.0, lighting=False)
        # Schematic mounting hardware communicates an actuated end effector.
        # Only the plate is a simulated collider; the mount never alters data.
        top = plate_bottom + plate_thickness
        head_center = floor + .06 + plate_thickness + .037
        rod_bottom, rod_top = top+.008, head_center-.008
        for radius, height, z, color in (
                (.014, .008, top+.004, "#384855"),
                (.008, rod_top-rod_bottom, (rod_top+rod_bottom)/2, "#b8c1c8"),
                (.018, .016, head_center, "#384855")):
            mount = pv.Cylinder(center=(.15, .15, z), direction=(0, 0, 1),
                                radius=radius, height=height, resolution=48)
            plotter.add_mesh(mount, color=color, ambient=.35, diffuse=.6, specular=.3)
    if boxes is not None:
        for center in boxes:
            bounds = np.column_stack((center-half, center+half)).ravel()
            tool = pv.Box(bounds=bounds)
            # Transparent full-height tools expose the contact and specimen.
            plotter.add_mesh(tool, color="#737d85", opacity=.20, ambient=.6)
            plotter.add_mesh(tool.extract_feature_edges(), color="#626f7b", line_width=1.2)
            direction = np.zeros(3)
            direction[closing_axis] = np.sign(.15-center[closing_axis])
            # Enlarge the closing cue while keeping its tip at the same location.
            start = center.copy() - direction*.050
            start[2] = floor + 2*half[2] + .009
            plotter.add_mesh(pv.Arrow(start=start, direction=direction, scale=.045,
                                     shaft_radius=.055, tip_radius=.13, tip_length=.30),
                             color="#15191d", lighting=False)
    cameras = {
        # Shared framing covers the union of all six shown surfaces, with
        # at least 5 mm clearance in the vertical projection.
        "shape": (.039, [0.9, -2.8, 1.55], .061),
        "press": (.055, [0.9, -2.8, 1.25], .084),
        "action": (.065, [1.1, -2.5, 3.2], .107),
    }
    focal_height, direction, scale = cameras[mode]
    focal = np.array([0.15, 0.15, floor+focal_height])
    direction = np.asarray(direction)
    plotter.camera.position = focal + direction * 0.20
    plotter.camera.focal_point = focal
    plotter.camera.up = (0, 0, 1)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = scale
    plotter.enable_anti_aliasing("ssaa")
    img = plotter.screenshot(return_img=True)
    if target is not None:
        from scipy.ndimage import binary_fill_holes

        # Project the target through the same camera, then draw its outer
        # silhouette over the outcome without depth occlusion or interior edges.
        plotter.clear()
        plotter.add_mesh(surface(target, volume0), color="black", lighting=False,
                         reset_camera=False)
        target_image = plotter.screenshot(return_img=True)
        mask = binary_fill_holes(np.max(target_image[:, :, :3], axis=2) < 128)
        # Keep the projected contour separate from the raster. Its stroke is
        # specified in final figure points, so image reduction cannot thin it.
        np.save(path.with_suffix(".target.npy"), mask)
    plt.imsave(path, img)
    plotter.close()
    return img


def label_panel(ax, title):
    """Shared panel frame: title, rule and content all stay inside the grid."""
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0, .985, title, ha="left", va="top", fontsize=8.5,
            fontweight="bold", color="#202a32", transform=ax.transAxes)
    ax.plot([0, 1], [.885, .885], color="#d8dddf", lw=.55,
            transform=ax.transAxes, clip_on=False)


# Both specimen panels use the same columns, rows, and image boxes.
COLUMNS = (.20, .52, .84)
ROWS = (.51, .16)


def specimen_image(ax, img, col, row):
    ia = ax.inset_axes([COLUMNS[col]-.145, ROWS[row], .29, .265])
    ia.imshow(img)
    ia.axis("off")
    return ia


def panel_a(ax, folder, images):
    label_panel(ax, "(a) Same action, different response")
    for x, label in zip(COLUMNS, ("Initial", "Press 14 mm", "Released"), strict=True):
        ax.text(x, .83, label, ha="center", va="center", fontsize=7.4)
    for row, name in enumerate("AB"):
        data = np.load(folder / f"probe_{name}.npz")
        ax.text(.005, ROWS[row]+.11, name, color=COLORS[name], fontsize=8,
                fontweight="bold", va="center", ha="left")
        for col, key in enumerate(("initial", "pressed", "released")):
            path = images / f"probe_{name}_{key}.png"
            if path.exists():
                img = plt.imread(path)
            else:
                img = snapshot(data[key], floor=float(data["floor"]),
                               volume0=data["vol0"], path=path, mode="press",
                               plate_bottom=(float(data["floor"])+.06
                                             -(float(data["depth"]) if key == "pressed" else 0)
                                             if key in ("initial", "pressed") else None),
                               plate_thickness=4*float(data["grid_lim"])/int(data["n_grid"]),
                               press_stroke=float(data["depth"]) if key == "pressed" else None)
            specimen_image(ax, img, col, row)
    ax.text(.5, .025, r"Yield: A $1$ kPa; B $10$ kPa. Same initial appearance.",
            ha="center", va="bottom", fontsize=6.8, color="#46515a")


def panel_b(panel, folder):
    label_panel(panel, "(b) Prediction under a new press")
    panel.text(.5, .83, "24 mm stroke at 120 mm/s", ha="center", va="center",
               fontsize=7.4, color="#46515a")
    ax = panel.inset_axes([.15, .245, .82, .515])
    for name in "AB":
        truth = np.load(folder / f"validation_true_{name}.npz")
        pred = np.load(folder / f"validation_identified_{name}.npz")
        ax.plot(truth["disp"]*1000, truth["force"], color=COLORS[name], lw=1.2)
        ax.plot(pred["disp"]*1000, pred["force"], color=COLORS[name], lw=0,
                marker="o", markersize=2.5, markerfacecolor="white", markeredgewidth=.8,
                markevery=18)
        ax.text(21.2, float(truth["force"][-1])+(11 if name == "A" else -12),
                name, color=COLORS[name], fontsize=7.4, fontweight="bold")
    nominal = np.load(folder / "validation_nominal.npz")
    ax.plot(nominal["disp"]*1000, nominal["force"], color="#757575", lw=1.1, ls="--")
    ax.set(xlabel="Displacement (mm)", ylabel="Force (N)", xlim=(0, 24), ylim=(0, 230))
    ax.set_xticks([0, 8, 16, 24])
    ax.set_yticks([0, 100, 200])
    ax.tick_params(labelsize=7, pad=2, length=3)
    ax.xaxis.label.set_size(7.4)
    ax.yaxis.label.set_size(7.4)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    handles = [Line2D([], [], color="#333333", lw=1.2, label="True"),
               Line2D([], [], color="#333333", marker="o", markerfacecolor="white",
                      lw=0, markersize=2.5, label="Identified"),
               Line2D([], [], color="#757575", ls="--", lw=1.1, label="Nominal")]
    panel.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
                 ncol=3, fontsize=6.8, frameon=False, handlelength=1.5,
                 columnspacing=1.0, handletextpad=.45, borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.7)
    ax.grid(axis="y", alpha=.15)


def panel_c(ax, folder, images, results):
    label_panel(ax, "(c) Planned shaping actions")
    data = np.load(folder / "execution_A_0_identified.npz")
    volume0 = np.load(folder / "probe_A.npz")["vol0"]
    result = next(r for r in results if r["material"] == "A"
                  and r["target"] == 0 and r["method"] == "identified")
    for col, axis in enumerate(("x", "y", None)):
        key = f"{axis}_pressed" if axis else "x"
        path = images / f"action_A_identified_{axis or 'released'}.png"
        if path.exists():
            img = plt.imread(path)
        else:
            img = snapshot(data[key], floor=float(data["floor"]), volume0=volume0,
                           path=path, color=METHOD_COLORS["identified"], mode="action",
                           boxes=data[f"{axis}_boxes"] if axis else None,
                           half=data[f"{axis}_half"] if axis else None,
                           closing_axis=col if axis else None)
        ax.text(COLUMNS[col], .83, f"{col+1}. {axis} squeeze" if axis else "3. Release",
                ha="center", va="center", fontsize=7.4)
        ia = ax.inset_axes([COLUMNS[col]-.155, .30, .31, .44])
        ia.imshow(img)
        ia.axis("off")
        label = f"Gap {result['action'][col]*1000:.0f} mm" if axis else "After 0.30 s"
        ax.text(COLUMNS[col], .23, label, fontsize=7.1, va="center", ha="center")
    for x in (.36, .68):
        ax.annotate("", xy=(x+.018, .50), xytext=(x-.018, .50),
                    arrowprops=dict(arrowstyle="->", color="#83919b", lw=.85))
    ax.text(.5, .025, "Material A, identified plan, executed in the true material.",
            ha="center", va="bottom", fontsize=6.8, color="#46515a")


def panel_d(ax, folder, images, results):
    label_panel(ax, "(d) Target and executed shapes")
    for x, label in zip(COLUMNS, ("Target", "Nominal", "Identified"), strict=True):
        ax.text(x, .83, label, ha="center", va="center", fontsize=7.4)
    for row, name in enumerate("AB"):
        target = np.load(folder / f"target_{name}_0.npz")
        vol0 = np.load(folder / f"probe_{name}.npz")["vol0"]
        ax.text(.005, ROWS[row]+.11, name, color=COLORS[name], fontsize=8,
                fontweight="bold", va="center", ha="left")
        for col, method in enumerate(("target", "nominal", "identified")):
            data = target if method == "target" else np.load(folder / f"execution_{name}_0_{method}.npz")
            path = images / f"outcome_{name}_{method}.png"
            if path.exists():
                img = plt.imread(path)
            else:
                img = snapshot(data["x"], floor=float(data["floor"]), volume0=vol0, path=path,
                               color="#b7b7b1" if method == "target" else METHOD_COLORS[method],
                               target=None if method == "target" else target["x"])
            ia = specimen_image(ax, img, col, row)
            if method != "target":
                mask = np.load(path.with_suffix(".target.npy"))
                ia.contour(mask, levels=[.5], colors=["black"], linewidths=.55,
                           linestyles=[(0, (2.4, 1.8))], zorder=5)
                result = next(r for r in results if r["material"] == name
                              and r["target"] == 0 and r["method"] == method)
                ax.text(COLUMNS[col], ROWS[row]-.022, f"{result['executed_error_mm']:.2f} mm",
                        ha="center", va="top", fontsize=7.2, zorder=20)
    ax.text(.5, .025, "Dashed contour: target. Labels: executed shape error.",
            ha="center", va="bottom", fontsize=6.8, color="#46515a")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--press-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT, help="Study root")
    parser.add_argument("--dest", type=Path,
                        default=OUT / "identification_plastic_shaping")
    args = parser.parse_args()
    folder = args.out.resolve() / "study"
    renderer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    images = folder / "renders" / renderer_hash[:12]
    images.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 2.15) if args.press_only else (7.2, 4.15))
    grid = fig.add_gridspec(1 if args.press_only else 2, 2,
                           left=.02, right=.985, bottom=.025, top=.99,
                           wspace=.12, hspace=.12)
    panel_a(fig.add_subplot(grid[0, 0]), folder, images)
    panel_b(fig.add_subplot(grid[0, 1]), folder)
    if args.press_only:
        fig.savefig(folder / "press_panels.png", dpi=220)
    else:
        results = json.loads((folder / "shaping_results.json").read_text())
        if len(results) != 18:
            raise RuntimeError("Shaping evaluation is incomplete")
        panel_c(fig.add_subplot(grid[1, 0]), folder, images, results)
        panel_d(fig.add_subplot(grid[1, 1]), folder, images, results)
        dest = args.dest.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=.02)
        fig.savefig(dest.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=.02)
        (folder / "figure_provenance.json").write_text(json.dumps(dict(
            renderer_sha256=renderer_hash, shown_target_index=0,
            action_sequence=dict(material="A", method="identified", target_index=0,
                                 source="execution_A_0_identified.npz"),
            tools="Recorded gripper boxes, translucent; simulated plate with schematic mount and initial-pose outline",
            surface_voxel_spacing_m=.0025, gaussian_sigma_voxels=1.3,
            relative_density_isovalue=.5, camera="fixed orthographic",
            target_contour=dict(rendering="vector at final figure size", style="dashed",
                                black_width_pt=.55, dash_pattern=[2.4, 1.8],
                                white_surround_width_pt=0),
            outcome_camera=dict(parallel_scale_m=.061, focal_height_above_floor_m=.039,
                                window_size=[720, 480],
                                same_camera_for_all_targets_and_outcomes=True),
            pdf_sha256=hashlib.sha256(dest.with_suffix(".pdf").read_bytes()).hexdigest(),
            png_sha256=hashlib.sha256(dest.with_suffix(".png").read_bytes()).hexdigest(),
            result_sha256=hashlib.sha256((folder / "shaping_results.json").read_bytes()).hexdigest()
        ), indent=2)+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()

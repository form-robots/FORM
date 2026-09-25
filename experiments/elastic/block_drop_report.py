"""Render and summarize saved block-drop results; never run or alter simulations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.elastic.block_drop_study import ROOT, save, sha
from experiments.robotics.plastic_shaping_figure import surface

COLORS = {"A": "#287eab", "B": "#d77740"}
INK = "#23303a"


def read(path):
    return json.loads(Path(path).read_text())


def font(size, bold=False):
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size)


def render_view(x, vol, p, color, plotter):
    import pyvista as pv
    plotter.clear()
    plotter.enable_lightkit()
    plotter.set_background("white")
    mesh = surface(x, vol, h=.00125)
    mesh.compute_normals(auto_orient_normals=True, consistent_normals=True, inplace=True)
    plotter.add_mesh(mesh, color=color, smooth_shading=True,
                     ambient=.3, diffuse=.8, specular=.18, specular_power=22)
    plotter.add_mesh(pv.Plane(center=(.1, .1, p["floor_z_m"]), direction=(0, 0, 1),
                             i_size=.12, j_size=.10), color="#e9edef", lighting=False)
    # Physical 20 mm reference line on the floor; camera shared by all frames.
    plotter.add_mesh(pv.Line((.075, .050, p["floor_z_m"]+.0001),
                            (.095, .050, p["floor_z_m"]+.0001)),
                     color="#58656f", line_width=3)
    focal = np.array([.1, .1, p["floor_z_m"]+.030])
    plotter.camera.position = focal + np.array([.14, -.30, .13])
    plotter.camera.focal_point = focal
    plotter.camera.up = (0, 0, 1)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = .054
    return plotter.screenshot(return_img=True)


def summarize(root):
    p = read(root / "protocol.json")
    report = {"protocol": p, "materials": {}}
    for label in ("A", "B"):
        case = root / f"probe_{label}"
        fit = read(case / "identification.json")
        d = np.genfromtxt(case / "diagnostics.csv", delimiter=",", names=True)
        # Ratio of bounding extents of the same particle cloud, not a mesh metric.
        minimum = int(np.argmin(d["extent_z"]))
        row = {
            "true_E_pa": p["E_pa"][label], "identified_E_pa": fit["E_pa"],
            "relative_E_error_percent": 100*(fit["E_pa"]/p["E_pa"][label]-1),
            "minimum_vertical_particle_extent_mm": float(d["extent_z"][minimum]*1000),
            "maximum_extent_compression_percent": float(100*(1-d["extent_z"][minimum]/d["extent_z"][0])),
            "time_at_minimum_extent_s": float(d["t"][minimum]),
            "probe_checks": read(case / "completion.json"), "identification": fit,
            "validation": read(root / f"validation_{label}.json"),
            "refinement_identification": read(root / f"refinement_{label}/identification.json"),
            "refinement_checks": read(root / f"refinement_{label}/completion.json"),
        }
        row["refinement_E_error_percent"] = 100*(
            row["refinement_identification"]["E_pa"]/p["E_pa"][label]-1)
        # A resolution check is not a claim of continuum convergence.
        fine = np.genfromtxt(root / f"refinement_{label}/diagnostics.csv",
                            delimiter=",", names=True)
        row["coarse_fine_com_rms_mm"] = float(np.sqrt(np.mean(sum(
            (d[k]-fine[k])**2 for k in ("com_x", "com_y", "com_z"))))*1000)
        report["materials"][label] = row
    assert sha(root / "probe_A/initial.npy") == sha(root / "probe_B/initial.npy")
    report["identical_initial_particles"] = True
    report["limits"] = [
        "Noise-free synthetic particle positions, velocities, and full F; no camera reconstruction.",
        "Only E is estimated. Poisson ratio, density, geometry, contact, and law family are known.",
        "Fixed-corotated ideal elasticity, no modeled viscosity or plasticity; not calibrated rubber.",
        "Frictionless separable floor; explicit grid damping disabled, MPM still has numerical dissipation.",
        "80/96 grid comparison is a sensitivity check, not a convergence demonstration.",
        "No insertion, geometry transfer, robot-arm simulation, or hardware experiment yet.",
        "Display surfaces are smoothed reconstructions; all errors use unmodified particle positions.",
    ]
    save(root / "report.json", report)
    return report


def make_media(root, report):
    import imageio.v2 as imageio
    import pyvista as pv

    p = report["protocol"]
    media = root / "media"
    media.mkdir(exist_ok=True)
    X = {k: np.load(root / f"probe_{k}/x.npy", mmap_mode="r") for k in COLORS}
    vol = np.load(root / "probe_A/vol0.npy")
    times = np.load(root / "probe_A/time.npy")
    # Common times emphasize time-dependent differences without stretching motion.
    peak = report["materials"]["A"]["time_at_minimum_extent_s"]
    selected = [0.0, .090, peak, .150]
    selected = [int(np.argmin(abs(times-t))) for t in selected]
    names = ["Release", "Impact", "A's greatest compression", "Rebound"]
    pl = pv.Plotter(off_screen=True, window_size=(640, 540), lighting="light kit")
    pl.enable_anti_aliasing("ssaa")
    stills = {}

    def frame(fi):
        canvas = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((35, 17), "Two elastic blocks · same 40 mm drop", font=font(27, True), fill=INK)
        draw.text((1015, 23), f"t = {times[fi]:.3f} s", font=font(24), fill=INK)
        for ci, label in enumerate(COLORS):
            im = Image.fromarray(render_view(X[label][fi], vol, p, COLORS[label], pl))
            canvas.paste(im, (ci*640, 105))
            draw.text((ci*640+35, 68),
                      f"{label}   {p['E_pa'][label]/1000:.0f} kPa", font=font(25, True),
                      fill=COLORS[label])
            if fi in selected:
                stills[label, fi] = im
        draw.line((640, 105, 640, 635), fill="#dbe1e5", width=2)
        draw.text((35, 651), "60 × 40 × 25 mm · identical geometry · shared camera and scale",
                  font=font(20), fill=INK)
        draw.text((35, 685), "20× slow motion · floor line: 20 mm · colors identify the materials",
                  font=font(17), fill="#5c6870")
        return canvas

    # 0.30 simulated seconds -> 6 s of motion, plus brief endpoint holds.
    video_indices = list(range(0, len(times), 2))
    with imageio.get_writer(media / "drop_comparison.mp4", fps=50, codec="libx264",
                            quality=8, macro_block_size=2, pixelformat="yuv420p") as writer:
        first = np.asarray(frame(0))
        for _ in range(40):
            writer.append_data(first)
        for fi in video_indices:
            writer.append_data(np.asarray(frame(fi)))
        last = np.asarray(frame(video_indices[-1]))
        for _ in range(40):
            writer.append_data(last)
    for fi in selected:
        if ("A", fi) not in stills:
            frame(fi)
    pl.close()

    fig = plt.figure(figsize=(14, 8.5), facecolor="white")
    fig.text(.065, .965, "Drop first. Identify stiffness from the motion.",
             fontsize=21, weight="bold", color=INK)
    fig.text(.065, .925, "Same block geometry and release height for both materials; no forces used in fitting.",
             fontsize=12, color="#58656f")
    for row, label in enumerate(COLORS):
        fig.text(.018, .73-row*.285, label, fontsize=23, weight="bold", color=COLORS[label])
        for col, fi in enumerate(selected):
            ax = fig.add_axes([.062+col*.232, .59-row*.285, .23, .28])
            ax.imshow(stills[label, fi])
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{names[col]}\n{times[fi]:.4f} s", fontsize=10.5, color=INK)
    fig.text(.065, .285, "One stiffness estimate from each drop", fontsize=15, weight="bold", color=INK)
    for i, label in enumerate(COLORS):
        r = report["materials"][label]
        left = .065+i*.48
        fig.text(left, .235, f"{label}   True: {r['true_E_pa']/1000:.0f} kPa", fontsize=15,
                 color=COLORS[label], weight="bold")
        fig.text(left, .193, f"Identified: {r['identified_E_pa']/1000:.2f} kPa   "
                 f"({r['relative_E_error_percent']:+.2f}%)", fontsize=14, color=INK)
        fig.text(left, .15, "60 mm drop prediction: "
                 f"{r['validation']['post_090s_particle_rmse_mm']:.3f} mm RMSE", fontsize=12, color=INK)
    fig.text(.065, .078, "Simulation with supplied particle states. E is estimated; ν = 0.45 and ρ = 1000 kg/m³ are known.",
             fontsize=10.5, color="#58656f")
    fig.text(.065, .043, "Snapshots share physical scale and time. Reconstructed surfaces are for display; errors use raw 3D particles.",
             fontsize=10, color="#58656f")
    fig.savefig(media / "overview.png", dpi=160)
    fig.savefig(media / "overview.pdf")
    plt.close(fig)

    # Separate diagnostic figure keeps the visual overview compact.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for label, color in COLORS.items():
        for tag, style in [("true", "-"), ("identified", "--")]:
            d = np.genfromtxt(root / f"validation_{tag}_{label}/diagnostics.csv",
                              delimiter=",", names=True)
            axes[0].plot(d["t"], (d["com_z"]-p["floor_z_m"])*1000, style,
                         color=color, label=f"{label}: {tag}", linewidth=1.8)
        error = np.genfromtxt(root / f"validation_error_{label}.csv", delimiter=",", names=True)
        axes[1].plot(error["time_s"], error["particle_rmse_mm"], color=color, label=label)
    axes[0].set(ylabel="Center height above floor (mm)", xlabel="Time (s)",
                title="New 60 mm drop: no refitting")
    axes[1].set(ylabel="3D particle RMSE (mm)", xlabel="Time (s)", title="Prediction error")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(media / "validation.png", dpi=180)
    plt.close(fig)
    save(media / "provenance.json", {
        "renderer_sha256": sha(Path(__file__)), "report_sha256": sha(root / "report.json"),
        "surface_source_sha256": sha(ROOT / "experiments/robotics/plastic_shaping_figure.py"),
        "snapshot_times_s": [float(times[i]) for i in selected],
        "playback_slowdown": 20.0, "endpoint_hold_video_s": .8,
        "physical_scale_shared": True, "surface_voxel_m": .00125,
    })


def write_report(root, report):
    rows = []
    for label, r in report["materials"].items():
        rows.append(f"| {label} | {r['true_E_pa']/1000:.1f} | {r['identified_E_pa']/1000:.3f} | "
                    f"{r['relative_E_error_percent']:+.3f}% | "
                    f"{r['refinement_identification']['E_pa']/1000:.3f} | "
                    f"{r['validation']['post_090s_particle_rmse_mm']:.4f} |")
    text = """# Elastic block drop: identification pilot

Two fresh, identical 60 × 40 × 25 mm blocks fall from rest with their lower face
40 mm above a frictionless floor. They differ only in Young's modulus. A is
80 kPa; B is 240 kPa. Both are ideal fixed-corotated elastic materials, with
known Poisson ratio 0.45 and density 1000 kg/m³. These are chosen simulation
materials, not measurements of particular rubber products.
The A/B labels refer only to this elastic pilot, not the elastoplastic materials
in the manuscript's shaping experiment.

The estimator uses noise-free x, v and full deformation gradient F from 60–180 ms
of the drop. It estimates only E in one linear least-squares fit. Density,
geometry, gravity, constitutive family, Poisson ratio, and contact are known.
Neither true E nor stress nor contact forces enter the fit. The weak balances
exclude nodes within three grid cells of the floor, using the existing
grid-consistent identifier rather than the historical radial-window estimator.

## Results

| Material | True E (kPa) | Identified E, grid 80 (kPa) | Signed error | Identified E, grid 96 (kPa) | New-height prediction RMSE (mm) |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

Prediction uses the frozen grid-80 estimate in a new 60 mm drop. RMSE compares
corresponding particles in 3D over 90–300 ms, covering impact and rebound;
there is no alignment, rescaling, or refitting. Full-time and peak-frame errors
are also retained in report.json. Grid 96 repeats the identification probe; it
does not supply a replacement estimate for the displayed prediction.

![Same drop for two materials](media/overview.png)

[Drop comparison video, 20× slow motion](media/drop_comparison.mp4)

![Held-out drop predictions](media/validation.png)

## Checks and interpretation

All eight forward simulations retain their inputs and raw outputs. Every run
must finish all 601 frames, remain finite, and have positive det(F). Initial
particle arrays are identical between A and B at each resolution. A ballistic
center-of-mass check covers the first 40 ms before contact. Completion files
record residual floor penetration, volume changes, and state hashes.

The finer-grid comparison checks numerical sensitivity, not convergence.
Temporal split estimates are diagnostics from the same trajectory, not repeated
trials or confidence intervals. No camera recovery, noisy-data robustness,
geometry transfer, insertion control, or hardware result is demonstrated here.
Although explicit grid damping is disabled, MPM still has numerical dissipation.
The surface reconstruction is smoothed for display; quantitative comparisons
use the raw particle data. All materials and heights were fixed before running.

## Reproduction

From the repository root, use a new output directory:

```bash
.venv/bin/python -m experiments.elastic.block_drop_study init --out out/new_block_drop
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.block_drop_study probe --out out/new_block_drop --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.block_drop_study probe --out out/new_block_drop --material B --device cuda:1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.block_drop_study validate --out out/new_block_drop --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.block_drop_study validate --out out/new_block_drop --material B --device cuda:1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.block_drop_study refine --out out/new_block_drop --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.block_drop_study refine --out out/new_block_drop --material B --device cuda:1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.block_drop_report --out out/new_block_drop
```

Protocol, source snapshots, version information, raw trajectories, fitted weak
systems, and SHA-256 hashes are saved with the run. GPU atomics can cause small
floating-point differences across reruns; bitwise reproduction is not assumed.
"""
    a, b = (report["materials"][k] for k in ("A", "B"))
    interpretation = (
        "\n**Interpretation:** Stiffness identification works well in this supplied-state "
        "pilot. The center-of-mass trajectories differ between grids 80 and 96 by "
        f"{a['coarse_fine_com_rms_mm']:.3f} mm RMS for A and "
        f"{b['coarse_fine_com_rms_mm']:.3f} mm for B over 0–300 ms. These grid "
        "differences exceed the same-grid identified-versus-true prediction errors. "
        "The small prediction errors therefore do not establish physical or "
        "continuum-level bounce accuracy. Maximum particle penetration below the "
        f"floor in the grid-80 A probe is {max(0., -a['probe_checks']['min_particle_z_above_floor_mm']):.3f} mm. "
        "This pilot supports testing the proposed strip task, with its own numerical "
        "checks, rather than claiming a validated hardware material model.\n"
    )
    text = text.replace("## Checks and interpretation\n", "## Checks and interpretation\n" + interpretation)
    (root / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.out)
    write_report(args.out, report)
    make_media(args.out, report)


if __name__ == "__main__":
    main()

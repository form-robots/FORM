# Reproducing FORM

Run commands from the repository root with the main Python 3.12 environment
activated. Install the relevant data bundles first. Use a fresh output directory
for each run. Simulation commands can take hours; the CPU reference check takes
seconds to minutes.

## Numerical versions

The submitted experiments used three solver revisions. The default source tree
preserves the compensated position/deformation updates used by the manipulation
experiments. `reproduce/profiles/benchmark.patch` restores the comparison PR's
kernels; `pouring.patch` restores the frozen pouring kernels. The runner builds
isolated trees under `out/runtime/` and selects the correct profile automatically.
It does not change the main source or require a second repository. Snapshots are keyed by source checksum; changing release code creates a new
runtime without replacing code used by an existing job.

The supplied dependency lock describes the verification environment. MuJoCo
Menagerie is pinned to `feadf76d42f8a2162426f7d226a3b539556b3bf5`;
the required Panda model is bundled in `assets/panda/`, so paper runners do not
need to download Menagerie. CPU and GPU
floating-point differences, hardware, drivers, and optimizer sensitivity can
prevent bitwise replay across machines. Seeds, protocols, identified coefficients,
selected controls, raw observations, and reference outputs are retained.

## Material identification

```bash
python -m reproduce.run identify-elastic
python -m reproduce.run identify-plastic
python -m reproduce.run identify-hardware
python -m reproduce.run identify-fluid
python -m reproduce.verify
```

The elastic task estimates stiffness from stereo texture tracks. Simulated and
hardware pressing use the separated early-elastic and late-yielded weak fits.
Pouring estimates effective viscosity from the recorded 60-degree pour. These
commands use saved reconstructed observations; they do not fit to the downstream
task results.

To regenerate observations:

```bash
# Full synthetic bending, stereo rendering/tracking, and identification:
python -m experiments.elastic.strip_texture_reproduce \
  --out out/reproduced/stereo-bending --device cuda:0

# Hardware contour extraction and interior reconstruction:
python -m reproduce.observations --out out/reproduced/press-observations
# To start from the saved contours instead, add --profiles-only.

# Raw 60-degree pouring video to optical observations:
MUJOCO_GL=egl python -m experiments.pour.observations \
  --out out/reproduced/pour-observations
```

Pressing camera/force preprocessing is provided by
`experiments.robotics.press_hardware_observe`. The saved calibration and
segmentation inputs are in the hardware bundle. Final scan processing starts from
the supplied scanner's merged meshes and poses; those are acquisition outputs,
not surfaces produced by FORM.

Synthetic pressing uses `experiments.shaping_sim.generate_probe` with `--grid 320`,
a fresh `--tag`, and `--material A` or `B`: run stages `record`, `observe`, then
`select`. Its paper protocol uses a monotonic two-stage press. The durable surface
archives can also be rendered with `experiments.shaping_sim.render_observations`.
The archives store float32 surfaces, so re-rendered pixels may differ slightly
from the original float64 rasterization. The original images/tracks are bundled.

## Planning and forward simulation

```bash
python -m reproduce.run insertion --device cuda:0
python -m reproduce.run putting --device cuda:0
python -m reproduce.run shaping-sim --device cuda:0
python -m reproduce.run shaping-real --device cuda:0
python -m reproduce.run predict-pressing --device cuda:0
python -m reproduce.run pouring --device cuda:0                 # recorded 60°
python -m reproduce.run pouring --angle 52.23 --out out/reproduced/pour-100
```

Insertion, putting, and simulated shaping execute both matched and swapped laws
using the saved plans. Hardware shaping executes the three saved own-material
plans in MPM; it does not command a physical robot. `predict-pressing` predicts
the three lower-load recordings with frozen material estimates.

Repeat planning with `--replan` on insertion, putting, `shaping-sim`, or
`shaping-real`. Simulated shaping retains the 48-evaluation budget per material;
hardware shaping uses two 16-evaluation starts and three rounded-command checks
at grid 64. `python -m reproduce.run plan-pouring` inverts a simulated angle-volume
curve with a 1 mL target tolerance. The forward runner uses the identified
viscosity, fixed contact setting, and recorded motion-clock characterization.

## Hardware result metrics

```bash
python -m experiments.shaping_real.reconstruction
python -m experiments.pour.measurements
python -m experiments.pour.plot_results
```

The first command rebuilds photo-colored, completed scan surfaces and their XY
IoUs. It retains the saved rigid registration; add `--refit-alignment` to repeat
the original coarse-angle/Powell registration. It uses no scaling. The second
command re-reads the receiver photographs using the saved pixel annotations and
computes the five-trial means and sample standard deviations.

## Table II and NCLaw

The [baseline guide](../experiments/nclaw/baselines/README.md) prepares a pinned
NCLaw checkout, installs the missing Sys-ID launcher, trains both baselines, and
runs their evaluation. NCLaw requires a separate Python 3.10 environment. The
optional CUDA 12 compatibility setup is documented beside that guide.

After NCLaw has generated the dataset/time/velocity/shape truth trajectories:

```bash
python -m reproduce.run benchmark-ingest \
  --nclaw-root out/nclaw_baselines/NCLaw --material jelly
python -m reproduce.run benchmark --material jelly --tier full --device cuda:0
python -m reproduce.run benchmark --material jelly --tier no-stress --device cuda:0
python -m reproduce.run benchmark --material jelly --tier positions-only --device cuda:0
python -m reproduce.run benchmark-fe --material jelly --device cuda:0
```

Repeat for `sand`, `plasticine`, and `water`. Function-encoder weights are in
`fe-weights/`; their training implementations are in
`src/ident/features/function_encoder_training/`. Comparison and timing table
writers are `experiments.nclaw.no_stress_table`, `experiments.nclaw.timing_report`,
and `experiments.fe_ls.cross_table`. Run them in the benchmark runtime so they
read that profile's outputs.

The benchmark bundle contains the available original FE and ingestion/ID JSON
outputs. Some original known-form comparison JSONs and the exact original NCLaw
baseline revision/checkpoint provenance were not present in the available
artifacts. Do not describe new training timings as exact reproduction of the
paper's timing table. Timings require dedicated hardware and matching timing
scope; the ongoing shared-GPU training audit is a functionality check.

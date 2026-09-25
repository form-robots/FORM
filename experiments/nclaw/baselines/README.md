# NCLaw and differentiable Sys-ID baselines

These are the two external baseline rows in Table II. Both run in **NCLaw's
simulator**, with the launcher supplied here. The Sys-ID launcher was supplied by
Cheng-Hsi Hsiao; it retains each material's analytic constitutive classes and
optimizes their parameters from randomized initial values.

## Prepare

From this repository's root:

```bash
python experiments/nclaw/baselines/prepare.py
```

This clones [NCLaw](https://github.com/PingchuanMa/NCLaw) into
`out/nclaw_baselines/NCLaw`, checks out the revision in `upstream.json`, installs
`experiments/scripts/train/sysid.py`, and applies `upstream.patch`. Use
`--dest /path/to/NCLaw` for another location. Repeating preparation is safe;
conflicting edits or another revision cause an error instead of an overwrite.
Git LFS is required to retrieve upstream mesh assets.

Install NCLaw **in its own environment** following its
[pinned installation instructions](https://github.com/PingchuanMa/NCLaw/blob/ff4e27a5dfdaa0b34687a0bf8292af9fb0ada8b0/README.md#installation).
Upstream targets Python 3.10, PyTorch 2.0.1, CUDA 11.7 and its bundled Warp 0.6.1.
Do not run its `build.sh` in the FORM environment: it replaces Warp.

## Run

With that environment activated, from the prepared NCLaw directory:

```bash
(
set -e
python experiments/scripts/dataset/main.py
python experiments/scripts/train/invariant_full_meta-invariant_full_meta.py
python experiments/scripts/train/sysid.py
for scene in dataset time vel shape; do
    python "experiments/scripts/eval/$scene.py" --gt
done
)
```

The launchers accept `--gpu 0` (default) and `--cpu 0`. Use fresh output directories
for timing; `-y` overwrites existing experiments and `-r` is not a fresh run.
Upstream evaluation also renders videos and requires its rendering dependencies.

The patch enables Sys-ID in **all four** evaluations. For Sys-ID it loads
`train/sysid/ckpt/<epoch>.pt` without replacing the analytic material classes.
It also fixes upstream geometry evaluation's epoch-loop indentation, uses the
active Python interpreter, and stops on subprocess failures. Physics, optimizer,
seed, learning rates and training schedule are unchanged.

## Compare results

From the FORM repository root (standard-library Python is sufficient):

```bash
python experiments/nclaw/baselines/report.py \
    --log-root out/nclaw_baselines/NCLaw/experiments/log
```

The report reads epoch 300's `info.json` files for time, three velocity seeds
(1, 7, 8), and held-out geometry. Mean loss is
`(time + mean(velocity) + geometry) / 3`; reconstruction is excluded. Each scene's
MSE is computed by upstream `diff_mse` on every fifth saved frame, averaging
over particles and coordinates. Missing or invalid metrics are errors, not zeros.
JSON and Markdown results go to `out/nclaw_baselines/report.*`.

Training writes `baseline_timing.json` alongside checkpoints. Its synchronized
wall time covers the epoch loop, including checkpoint I/O and logging, and
excludes setup and evaluation. It records GPU, PyTorch version, seed and epochs.

## Reproduction status

The supplied launcher and evaluation routing have been checked against the pinned
source. Both models passed a one-epoch GPU smoke test on jelly; full 300-epoch
training is in progress. See [GPU verification](GPU_VERIFICATION.md) for the newer
environment, compatibility patch, audit criteria and result locations. Held-out
evaluation has not been rerun here.
The pin is a reproducible integration target, not a confirmed record of the
revision used for the original paper runs. `paper_reference.json` stores the
manuscript values separately from new measurements. Original checkpoint choice,
hardware and timing scope still need confirmation before claiming an exact match.

The local RTX 5090 verification uses PyTorch 2.8 / CUDA 12.8 and bundled Warp 0.6.1
with two compatibility fixes. It is distinct from upstream's original software stack.

Tests (no training or GPU required):

```bash
NCLAW_BASELINE_TEST_DIR="$PWD/out/nclaw_baselines/NCLaw" \
    .venv/bin/python -m pytest tests/test_nclaw_baselines.py
```

Our weak-form/FE comparison pipeline is separate; Cheng-Hsi's
[upstream PR #5](https://github.com/kks32/mpm-engine/pull/5) contains its I/O,
timing and table scripts. The paper subset of that PR is included here, with its kernel revision selected by `reproduce.run`.

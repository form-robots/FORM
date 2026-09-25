# GPU training verification

This is a CUDA 12.8 / RTX 5090 compatibility run, not an exact recreation of the
paper's original environment. Keep the original reproduction instructions in
[README.md](README.md) separate from this newer stack.

## Environment

From the MPM-engine root, using `uv`, Git LFS, GCC and a CUDA 12.8 toolkit:

```bash
python experiments/nclaw/baselines/prepare.py --cuda12
uv venv --no-project --python 3.10 out/nclaw_baselines/venv
uv pip install --python out/nclaw_baselines/venv/bin/python \
    -r experiments/nclaw/baselines/requirements-cuda12.txt
uv pip install --python out/nclaw_baselines/venv/bin/python --no-deps \
    -e out/nclaw_baselines/NCLaw -e out/nclaw_baselines/NCLaw/third_party/warp
(
    set -e
    source out/nclaw_baselines/venv/bin/activate
    cd out/nclaw_baselines/NCLaw/third_party/warp
    python build_lib.py --cuda_path /usr --mode release --fast_math True
    python build_exports.py
)
```

`/usr` is this machine's CUDA 12.8 toolkit prefix (`/usr/bin/nvcc`). Replace it
with your toolkit prefix if different. The dependency lock uses Python 3.10 and
PyTorch 2.8.0 with CUDA 12.8. Installation does not change the main repository's
environment.

`cuda12.patch` makes two compatibility changes to bundled Warp 0.6.1:

- Resolve the original four-argument `cuGetProcAddress` symbol with its matching
  versioned function-pointer type. CUDA 12 headers otherwise select the newer ABI.
- Omit unused host launch wrappers from NVRTC device source. CUDA 12.8 treats them
  as unsupported device-side launches. Warp already launches the unchanged global
  forward/backward kernels directly through its CUDA driver API.

No material equation, MPM kernel, gradient formula or optimizer setting is changed.
These fixes have been exercised by a full-trajectory, one-epoch GPU smoke test for
both Sys-ID and NCLaw on jelly. Full 300-epoch verification is recorded separately
below and must not be inferred from the smoke tests.

## Run and audit

Smoke test (one epoch per model, separate checkpoint directories):

```bash
out/nclaw_baselines/venv/bin/python experiments/nclaw/baselines/verify_gpu.py \
    --nclaw-root out/nclaw_baselines/NCLaw \
    --out out/nclaw_baselines/verification/smoke \
    --materials jelly --smoke --reuse-datasets
```

Full verification (four materials, both models, 300 epochs each):

```bash
out/nclaw_baselines/venv/bin/python experiments/nclaw/baselines/verify_gpu.py \
    --nclaw-root out/nclaw_baselines/NCLaw \
    --out out/nclaw_baselines/verification/full \
    --reuse-datasets --gpu 0
```

Use a fresh `--out` directory. Existing training directories are never overwritten.
`--reuse-datasets` accepts complete 1,001-state truth trajectories at the expected
resolution; absent datasets are generated. For a fully independent run, use a new
prepared NCLaw checkout and omit this option.

Each completed training run is checked for all 301 checkpoints, finite parameters
in every checkpoint, changed parameters, 300 finite epoch losses, and finite,
nonzero gradients. The runner writes the exact commands, environment, timing and
audit results to `verification.json`, and stops with an error record if a run fails.

The launch started on 2026-09-24 uses GPU 0 sequentially and stores its output in
`out/nclaw_baselines/verification/full/`. Driver log and PID metadata are adjacent
in `full-driver.log` and `full-process.json`. A final `"status": "passed"` with
eight audited runs establishes successful full training. While runs are incomplete,
full training is not yet verified. Other users' jobs are sharing the GPUs, so these
timings are not clean performance measurements. Training verification does not
establish agreement with Table II's held-out losses; use the evaluation and report
steps in the main guide after training finishes.

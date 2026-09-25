"""Run and audit full NCLaw/Sys-ID training in a prepared NCLaw environment."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MATERIALS = ("jelly", "sand", "plasticine", "water")
MODELS = ("sysid", "invariant_full_meta-invariant_full_meta")


def audit_checkpoints(root, epochs):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    paths = sorted((root / "ckpt").glob("*.pt"))
    expected = [f"{i:04d}.pt" for i in range(epochs + 1)]
    if [path.name for path in paths] != expected:
        raise ValueError(f"Incomplete checkpoint sequence: {root}")
    first = torch.load(paths[0], map_location="cpu", weights_only=True)
    changed = []
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        for component, params in state.items():
            for name, value in params.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite parameter: {path}: {component}.{name}")
                if path == paths[-1] and not torch.equal(value, first[component][name]):
                    changed.append(f"{component}.{name}")
    if not changed:
        raise ValueError(f"Training did not change any parameters: {root}")
    events = EventAccumulator(str(root), size_guidance={"scalars": 0}).Reload()
    losses = events.Scalars("loss/acc")
    if len(losses) != epochs or any(not float("-inf") < e.value < float("inf") for e in losses):
        raise ValueError(f"Missing or nonfinite epoch losses: {root}")
    gradients = {}
    for tag in events.Tags()["scalars"]:
        if tag.startswith("grad_norm/"):
            values = [event.value for event in events.Scalars(tag)]
            if len(values) != epochs or any(not 0 <= v < float("inf") for v in values):
                raise ValueError(f"Missing or nonfinite gradients: {root}: {tag}")
            gradients[tag] = {"min": min(values), "max": max(values)}
    if not gradients or not any(v["max"] > 0 for v in gradients.values()):
        raise ValueError(f"No nonzero training gradients: {root}")
    return {
        "checkpoints": len(paths),
        "changed_parameters": changed,
        "first_training_loss": losses[0].value,
        "last_training_loss": losses[-1].value,
        "gradient_norms": gradients,
        "timing": json.loads((root / "baseline_timing.json").read_text()),
    }


def run_job(command, log_path):
    print(f"Running {log_path.stem}", flush=True)
    started = time.perf_counter()
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1"},
        )
    if result.returncode:
        raise RuntimeError(f"Job failed ({result.returncode}); see {log_path}")
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nclaw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--materials", nargs="+", choices=MATERIALS, default=list(MATERIALS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--smoke", action="store_true", help="One epoch in separate smoke directories"
    )
    parser.add_argument(
        "--reuse-datasets", action="store_true", help="Use complete local truth dumps"
    )
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf

    root = args.nclaw_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.smoke else 300
    summary = {
        "mode": "smoke" if args.smoke else "full",
        "epochs": epochs,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.gpu),
        "runs": [],
        "timing_note": "Other GPU jobs may affect these verification timings.",
    }
    summary_path = out / "verification.json"
    if summary_path.exists():
        raise FileExistsError(f"Choose a fresh verification output directory: {out}")

    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        for material in args.materials:
            dataset = root / "experiments/log" / material / "dataset"
            if args.reuse_datasets and (dataset / "state/1000.pt").exists():
                cfg = OmegaConf.load(dataset / "hydra.yaml")
                if cfg.sim.num_steps != 1000 or cfg.sim.num_grids != 20 or not cfg.dataset:
                    raise ValueError(f"Unexpected dataset configuration: {dataset}")
                if len(list((dataset / "state").glob("*.pt"))) != 1001:
                    raise ValueError(f"Incomplete dataset: {dataset}")
            else:
                if dataset.exists():
                    raise FileExistsError(f"Dataset already exists: {dataset}")
                command = [
                    sys.executable,
                    str(root / "experiments/eval.py"),
                    f"env={material}",
                    "sim=low",
                    "render=debug",
                    "dataset=True",
                    f"name={material}/dataset",
                    f"gpu={args.gpu}",
                ]
                run_job(command, out / f"dataset-{material}.log")
            for model in args.models:
                name = f"{material}/train/{'smoke-' if args.smoke else ''}{model}"
                training = root / "experiments/log" / name
                if training.exists():
                    raise FileExistsError(f"Training output already exists: {training}")
                command = [
                    sys.executable,
                    str(root / "experiments/train.py"),
                    f"env={material}",
                    "sim=low",
                    "render=debug",
                    f"name={name}",
                    f"gpu={args.gpu}",
                    f"train.num_epochs={epochs}",
                    "env.blob.material.elasticity.requires_grad=True",
                    "env.blob.material.plasticity.requires_grad=True",
                ]
                if model == "sysid":
                    command += [
                        "env.blob.material.elasticity.random=True",
                        "env.blob.material.plasticity.random=True",
                    ]
                else:
                    command += [
                        "env/blob/material/elasticity=invariant_full_meta",
                        "env/blob/material/plasticity=invariant_full_meta",
                    ]
                elapsed = run_job(command, out / f"train-{material}-{model}.log")
                audit = audit_checkpoints(training, epochs)
                summary["runs"].append(
                    {
                        "material": material,
                        "model": model,
                        "command": command,
                        "process_wall_s": elapsed,
                        **audit,
                    }
                )
                save()
                print(f"Verified {material} / {model}: {epochs} epochs", flush=True)
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        save()
        raise
    summary["status"] = "passed"
    save()
    print(f"Verification complete: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

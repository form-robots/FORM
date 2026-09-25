"""Compare fresh NCLaw/Sys-ID generalization metrics with the manuscript baseline rows."""

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES = {"jelly": "armadillo", "sand": "blub", "plasticine": "bunny", "water": "spot"}
MODELS = {"NCLaw": "invariant_full_meta-invariant_full_meta", "Diff. Sys-ID": "sysid"}
SEEDS = (1, 7, 8)


def read_mse(path):
    value = float(json.loads(path.read_text())["mse"])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid MSE in {path}: {value}")
    return value


def collect(log_root: Path, epoch=300):
    reference = json.loads((HERE / "paper_reference.json").read_text())
    rows = []
    for label, model in MODELS.items():
        for material, shape in SHAPES.items():
            root = log_root / material / "train" / model
            end = f"{epoch:04d}/info.json"
            time_loss = read_mse(root / "time" / end)
            velocity_losses = [read_mse(root / "vel" / f"{seed:04d}" / end) for seed in SEEDS]
            shape_loss = read_mse(root / "shape" / shape / end)
            velocity_mean = sum(velocity_losses) / len(velocity_losses)
            timing_path = root / "baseline_timing.json"
            timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
            if timing is not None and (timing["epochs"] != epoch or timing.get("resumed")):
                raise ValueError(f"Timing is not a fresh {epoch}-epoch run: {timing_path}")
            rows.append(
                {
                    "method": label,
                    "material": material,
                    "time_mse": time_loss,
                    "velocity_mse": velocity_losses,
                    "velocity_mean_mse": velocity_mean,
                    "shape_mse": shape_loss,
                    "mean_loss": (time_loss + velocity_mean + shape_loss) / 3,
                    "timing": timing,
                    "paper_reference": reference[label][material],
                }
            )
    return {
        "epoch": epoch,
        "aggregation": "(time MSE + mean of three velocity MSEs + shape MSE) / 3",
        "metric": "upstream diff_mse, every fifth saved frame; reconstruction excluded",
        "validation": "Numerical comparison only; original revision and timing scope unconfirmed.",
        "rows": rows,
    }


def markdown(report):
    lines = [
        "# NCLaw baseline comparison",
        "",
        report["aggregation"],
        "",
        "| Method | Material | Mean loss | Paper loss | New / paper | "
        "Training loop (s) | Paper solve (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        ref = row["paper_reference"]
        timing = row["timing"]
        elapsed = f"{timing['training_loop_s']:.1f}" if timing else "not recorded"
        lines.append(
            f"| {row['method']} | {row['material']} | {row['mean_loss']:.6g} | "
            f"{ref['mean_loss']:.6g} | {row['mean_loss'] / ref['mean_loss']:.3g} | "
            f"{elapsed} | {ref['solve_time_s']:.1f} |"
        )
    lines += [
        "",
        "Training-loop time includes checkpoint writes and logging, excludes setup and "
        "evaluation. Its scope and hardware are not established to match "
        "the paper's solve time.",
        "",
        report["validation"],
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("out/nclaw_baselines/report"))
    args = parser.parse_args()
    report = collect(args.log_root, args.epoch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()

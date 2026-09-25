"""Baseline integration checks; upstream launchers are exercised without GPU work."""

import argparse
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest
from experiments.nclaw.baselines import prepare, report

HERE = Path(__file__).resolve().parents[1] / "experiments/nclaw/baselines"


@pytest.fixture
def fake_nclaw(monkeypatch, tmp_path):
    constants = types.ModuleType("nclaw.constants")
    constants.ENVS = list(report.SHAPES)
    constants.SHAPE_ENVS = {key: [value] for key, value in report.SHAPES.items()}
    constants.SEEDS = [1, 7, 8]
    constants.EPOCHS = [0, 300]
    constants.RENDER = "debug"
    constants.PYTHON_PATH = None
    utils = types.ModuleType("nclaw.utils")
    utils.get_root = lambda _: tmp_path / "experiments"

    def parser():
        result = argparse.ArgumentParser()
        result.add_argument("--gpu", type=int, default=0)
        result.add_argument("--cpu", type=int, default=0)
        return result

    utils.get_script_parser = parser
    utils.dict_to_hydra = lambda args: [f"{key}={value}" for key, value in args.items()]
    utils.clean_state = lambda _: None
    metrics = []

    def diff_mse(src, tar, skip_frame):
        metrics.append((src, tar, skip_frame))
        return {"mse": 0.0}

    utils.diff_mse = diff_mse
    ffmpeg = types.ModuleType("nclaw.ffmpeg")
    ffmpeg.cat_videos = lambda *_: None
    for name, module in [
        ("nclaw", types.ModuleType("nclaw")),
        ("nclaw.constants", constants),
        ("nclaw.utils", utils),
        ("nclaw.ffmpeg", ffmpeg),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        assert command[0] == sys.executable
        calls.append(command)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["script.py"])
    return calls, metrics


def test_sysid_uses_analytic_models_and_random_trainable_parameters(fake_nclaw):
    calls, _ = fake_nclaw
    runpy.run_path(str(HERE / "sysid.py"), run_name="__main__")
    assert len(calls) == 4
    for command, material in zip(calls, report.SHAPES, strict=True):
        args = dict(part.split("=", 1) for part in command[2:])
        assert args["name"] == f"{material}/train/sysid"
        assert args["sim"] == "low"
        assert not any("env/blob/material/" in key for key in args)
        for component in ("elasticity", "plasticity"):
            for field in ("requires_grad", "random"):
                assert args[f"env.blob.material.{component}.{field}"] == "True"


def test_sysid_stops_on_training_failure(fake_nclaw, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runpy.run_path(str(HERE / "sysid.py"), run_name="__main__")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "scene, predictions", [("dataset", 16), ("time", 16), ("vel", 48), ("shape", 16)]
)
def test_patched_upstream_evaluates_both_models(scene, predictions, fake_nclaw, monkeypatch):
    upstream = os.environ.get("NCLAW_BASELINE_TEST_DIR")
    if not upstream:
        pytest.skip("Set NCLAW_BASELINE_TEST_DIR to a prepared pinned checkout")
    calls, metrics = fake_nclaw
    monkeypatch.setattr(sys, "argv", ["script.py", "--gt"])
    runpy.run_path(
        str(Path(upstream) / f"experiments/scripts/eval/{scene}.py"), run_name="__main__"
    )
    evaluations = [
        dict(part.split("=", 1) for part in call[2:])
        for call in calls
        if Path(call[1]).name == "eval.py"
    ]
    learned = [args for args in evaluations if "env.blob.material.ckpt" in args]
    assert len(learned) == predictions
    assert len(metrics) == predictions
    assert len(evaluations) - len(learned) == (12 if scene == "vel" else 4)
    assert all(skip == 5 for _, _, skip in metrics)
    for material, shape in report.SHAPES.items():
        for model in report.MODELS.values():
            for epoch in ("0000", "0300"):
                selected = [
                    args
                    for args in learned
                    if args["env.blob.material.ckpt"] == f"{material}/train/{model}/ckpt/{epoch}.pt"
                ]
                assert len(selected) == (3 if scene == "vel" else 1)
                for args in selected:
                    assert args["env"] == (shape if scene == "shape" else material)
                    assert args["name"].endswith(epoch)
                    assert ("env/blob/material/elasticity" in args) == (model != "sysid")
                    assert ("env/blob/material/plasticity" in args) == (model != "sysid")
                    if scene == "shape":
                        assert args["sim"] == "high"
                        assert args["sim.num_steps"] == "2000"
                    if scene == "time":
                        assert args["sim.num_steps"] == ("5000" if material == "water" else "2000")
    for src, tar, _ in metrics:
        # Score against this scene's own truth, not a learned or reconstruction trajectory.
        assert "/train/" in str(src)
        assert "/train/" not in str(tar)


def populate_metrics(tmp_path):
    for model in report.MODELS.values():
        for material, shape in report.SHAPES.items():
            root = tmp_path / material / "train" / model
            values = {
                "time": 6.0,
                f"shape/{shape}": 9.0,
                "vel/0001": 1.0,
                "vel/0007": 2.0,
                "vel/0008": 3.0,
            }
            for scene, mse in values.items():
                path = root / scene / "0300/info.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"mse": mse}))
    return tmp_path


def test_report_weights_scenarios_not_individual_velocity_runs(tmp_path):
    result = report.collect(populate_metrics(tmp_path))
    assert len(result["rows"]) == 8
    assert all(row["mean_loss"] == pytest.approx(17 / 3) for row in result["rows"])
    assert all(row["timing"] is None for row in result["rows"])
    assert "not recorded" in report.markdown(result)


def test_report_rejects_missing_nan_and_resumed_results(tmp_path):
    populate_metrics(tmp_path)
    root = tmp_path / "jelly/train/sysid"
    metric = root / "time/0300/info.json"
    metric.unlink()
    with pytest.raises(FileNotFoundError):
        report.collect(tmp_path)
    metric.write_text('{"mse": NaN}')
    with pytest.raises(ValueError, match="Invalid MSE"):
        report.collect(tmp_path)
    metric.write_text('{"mse": 1}')
    (root / "baseline_timing.json").write_text('{"epochs": 300, "resumed": true}')
    with pytest.raises(ValueError, match="fresh"):
        report.collect(tmp_path)


def test_prepare_is_idempotent_and_preserves_conflicts(tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "file.txt").write_text("before\n")
    prepare.git(repo, "add", "file.txt")
    prepare.git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "fixture",
    )
    pin = prepare.git(repo, "rev-parse", "HEAD").stdout.strip()
    assets = tmp_path / "integration"
    assets.mkdir()
    (assets / "upstream.patch").write_text(
        "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-before\n+after\n"
    )
    (assets / "sysid.py").write_text("# launcher\n")
    (repo / "experiments/scripts/train").mkdir(parents=True)
    monkeypatch.setattr(prepare, "HERE", assets)
    monkeypatch.setattr(prepare, "PIN", {"commit": pin})
    prepare.prepare(repo)
    prepare.prepare(repo)
    assert (repo / "file.txt").read_text() == "after\n"
    (repo / "file.txt").write_text("user edit\n")
    with pytest.raises(RuntimeError, match="conflict"):
        prepare.prepare(repo)
    assert (repo / "file.txt").read_text() == "user edit\n"

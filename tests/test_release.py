"""Portable release contracts: data integrity and frozen experiment settings."""
import json
from pathlib import Path
import tarfile
import io

import numpy as np
import pytest

from reproduce import data
from reproduce import run


def test_bundle_install_rejects_unlisted_files(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    root.mkdir()
    monkeypatch.setattr(data, 'ROOT', root)
    manifest = root / 'manifest.json'
    monkeypatch.setattr(data, 'MANIFEST', manifest)
    manifest.write_text(json.dumps({'files': []}))
    archive = tmp_path / 'unexpected.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo('../outside')
        member.size = 1
        stream.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unexpected archive member'):
        data.unpack([archive])
    assert not (tmp_path / 'outside').exists()


def test_bundle_round_trip_and_internal_mount(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    path = root / 'data/artifacts/experiment/measurements.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'original measured values')
    manifest = root / 'data/manifest.json'
    manifest.write_text(json.dumps({'files': [dict(path=str(path.relative_to(root)),
        mount='out/experiment/measurements.bin', bundle='test', bytes=path.stat().st_size,
        sha256=data.digest(path))]}))
    monkeypatch.setattr(data, 'ROOT', root)
    monkeypatch.setattr(data, 'MANIFEST', manifest)
    archives = tmp_path / 'archives'
    data.pack(archives)
    path.unlink()
    data.unpack([archives / 'form-test.tar.gz'])
    data.verify()
    assert (root / 'out/experiment/measurements.bin').read_bytes() == b'original measured values'
    assert (root / 'out/experiment').resolve().is_relative_to(root)
    path.write_bytes(b'modified')
    with pytest.raises(FileExistsError, match='Conflicting local data'):
        data.unpack([archives / 'form-test.tar.gz'])


@pytest.mark.skipif(not (run.ROOT / 'out/press_paper_update_20260913/execution80').exists(),
                    reason='Requires shaping_sim dataset')
def test_simulated_shaping_uses_frozen_commands(tmp_path, monkeypatch):
    from experiments.robotics import x_motion_shaping as core
    calls = []

    def simulate(law, command, phases, initial, device, grid, dt):
        assert grid == 80 and dt == .00004
        assert len(command['gaps_mm']) == 6
        assert len(initial['initial']) == len(initial['vol0'])
        calls.append((law, command['gaps_mm'].copy()))
        return {'x': initial['initial'][:1]}

    monkeypatch.setattr(core, 'simulate', simulate)
    run.shaping(tmp_path / 'results', 'cpu', False)
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[0][1], calls[2][1])
    np.testing.assert_array_equal(calls[1][1], calls[3][1])
    assert calls[0][0] == calls[1][0]
    assert calls[2][0] == calls[3][0]


@pytest.mark.skipif(not (run.ROOT / 'out/putting_swing_20260913').exists(),
                    reason='Requires elastic dataset')
def test_putting_retains_five_second_execution(tmp_path, monkeypatch):
    from experiments.elastic import putting_study
    calls = []

    def prepare(destination, source):
        destination.mkdir()

    def execute(root, material, planned, device, *, video):
        # The original executor uses video=True to select the paper's 5 s horizon.
        assert video is True
        assert (root / f'plan_{planned}/plan.json').exists()
        assert (root / 'franka' / f'plan_plan_{planned}.npz').exists()
        calls.append((material, planned))

    monkeypatch.setattr(putting_study, 'prepare', prepare)
    monkeypatch.setattr(putting_study, 'execute', execute)
    run.putting(tmp_path / 'putting', 'cpu', False)
    assert calls == [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]

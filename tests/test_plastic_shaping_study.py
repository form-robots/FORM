"""Study checks for collider orientation, metric units, and held-out targets."""
import numpy as np
import pytest
from experiments.robotics.plastic_shaping_study import (
    ACTION_GRID,
    TARGET_ACTIONS,
    cached,
    chamfer_mm,
    grip_boxes,
)


@pytest.mark.parametrize("axis,index", [("x", 0), ("y", 1)])
def test_fingers_face_each_other_with_requested_gap(axis, index):
    centers, half = grip_boxes(axis, 0.071, 0.01875)
    lower_inner = centers[0, index] + half[index]
    upper_inner = centers[1, index] - half[index]
    assert upper_inner - lower_inner == pytest.approx(0.071)
    # The old demo kept x-oriented thin fingers when moving them along y.
    assert half[index] < half[1-index]
    np.testing.assert_allclose(centers[:, 1-index], 0.15)


def test_chamfer_is_symmetric_unsquared_and_in_millimetres():
    x = np.array([[0., 0., 0.], [.020, 0., 0.]])
    target = x + np.array([0., .003, .004])
    assert chamfer_mm(x, target) == pytest.approx(5.)
    assert chamfer_mm(target, x) == pytest.approx(5.)
    assert chamfer_mm(x, x) == 0.


def test_targets_are_not_exact_actions_available_to_planner():
    assert ACTION_GRID.shape == (121, 2)
    for target in TARGET_ACTIONS:
        assert not np.any(np.all(np.isclose(ACTION_GRID, target), axis=1))
        assert np.all(target > ACTION_GRID.min(0))
        assert np.all(target < ACTION_GRID.max(0))


def test_interrupted_trajectory_file_is_not_reused(tmp_path):
    path = tmp_path / "interrupted.npz"
    path.write_bytes(b"PK\x03\x04incomplete archive")
    assert not cached(path)

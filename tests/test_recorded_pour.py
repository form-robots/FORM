"""Recorded pours retain the same clock when cached trajectories replace pose moves."""
import json

import numpy as np
import pytest
from examples.pour_recorded_twin import load_episode, recorded_pour_actions
from experiments.pour.pour_perception import fit_level


@pytest.mark.parametrize("pour_type", ["go_to_pose", "execute_trajectory", "angle_only"])
def test_recorded_motion_and_dwell_window(tmp_path, pour_type):
    upright = [0.5, 0.5, -0.5, 0.5]
    actions = [
        dict(type="execute_trajectory", target_quat_xyzw=upright, t_send=1, t_ack=3),
        dict(type=pour_type, target_quat_xyzw=[0.6830127, 0.1830127, -0.1830127, 0.6830127],
             t_send=10, t_ack=12, dwell_s=2),
        dict(type="go_to_pose", target_quat_xyzw=upright, t_send=14, t_ack=16),
        dict(type="go_to_pose", target_quat_xyzw=upright, t_send=20, t_ack=22),
    ]
    if pour_type == "angle_only":
        actions[1].update(type="execute_trajectory", pour_angle_deg=60)
        del actions[1]["target_quat_xyzw"]
    (tmp_path / "actions.jsonl").write_text("\n".join(json.dumps(a) for a in actions))
    states = [dict(t=float(t), state=dict(joint_position=[0.0] * 7, gripper_width=0.005))
              for t in np.arange(8.0, 21.0, 0.05)]
    (tmp_path / "states.jsonl").write_text("\n".join(json.dumps(s) for s in states))
    (tmp_path / "meta.json").write_text("{}")

    episode = load_episode(tmp_path, pre_roll=1.0, hold=10.0)
    assert episode["t0"] == 9.0
    assert episode["t_pour"] == 1.0
    assert episode["t_hold"] == 3.0
    assert episode["t_return_start"] == 5.0  # the recorded dwell remains in the track
    assert episode["t_return_done"] == 7.0
    assert episode["duration"] == pytest.approx(10.95)  # stop before the unload lift


def test_upright_quaternion_sign_does_not_start_pour(tmp_path):
    actions = [dict(type="execute_trajectory", target_quat_xyzw=[-0.5, -0.5, 0.5, -0.5],
                    t_send=1, t_ack=2),
               dict(type="go_to_pose", target_quat_xyzw=[0.5, 0.5, -0.5, 0.5],
                    t_send=3, t_ack=4)]
    (tmp_path / "actions.jsonl").write_text("\n".join(json.dumps(a) for a in actions))
    with pytest.raises(ValueError, match="No pour followed by a return"):
        recorded_pour_actions(tmp_path)


def test_stream_occlusion_does_not_change_inferred_pool_level():
    # A known horizontal pool seen through rays with linearly increasing turn-on
    # heights. A wide opaque/amber stream hides its center, including dry pixels.
    levels = np.linspace(0.0, 0.10, 101)[:, None] * np.ones((1, 40))
    chord = np.full_like(levels, 0.05)
    pool = levels < 0.045
    excluded = np.zeros_like(pool)
    excluded[:, 4:36] = True
    reference, _, _ = fit_level(levels, chord, pool)
    for stream_is_amber in (False, True):
        observed = np.where(excluded, stream_is_amber, pool)
        inferred, iou, prediction = fit_level(levels, chord, observed, excluded=excluded)
        assert inferred == pytest.approx(reference, abs=0.0005)
        assert iou == pytest.approx(1.0)
        assert not np.any(prediction[excluded])

"""Angle planning preserves the measured template and physical motion joins."""
import numpy as np
import pytest
from experiments.pour.pour_angle_plan import bracket_candidate
from experiments.pour.pour_angle_sweep import AngleMotion, interp_columns, invert_angles


@pytest.fixture
def motion():
    t = np.linspace(-1.5, 12.5, 701)
    progress = np.interp(t, [0, 6.3, 8.3, 10.7], [0, 1, 1, 0])
    p = np.linspace(0, 1, 101)
    # Different nonlinear forward/return paths expose incorrect phase joins.
    forward = np.array([p, p**2, np.sin(p), p*0, p*0, p*0, p*0]).T
    returning = forward + 0.1 * (p * (1 - p))[:, None]
    q = interp_columns(progress, p, forward)
    back = t > 8.3
    q[back] = interp_columns(progress[back], p, returning)
    return AngleMotion(t, q, np.full_like(t, 0.01),
                       np.array([0, 6.3]), np.array([0, 1]),
                       np.array([8.3, 10.7]), np.array([1, 0]),
                       (p, forward), (p, returning), 60, 6, 6.3, 8.3, 10.7, 12)


def test_reference_identity_across_all_phases(motion):
    t = np.linspace(-1.2, 12, 1201)
    np.testing.assert_allclose(motion.reference_clock(t, 60), t, atol=1e-14)
    np.testing.assert_allclose(motion.joints(t, 60),
                               interp_columns(t, motion.reference_t, motion.reference_q),
                               atol=1e-14)
    sampled = motion.sample_times(60)
    reconstructed = interp_columns(t, sampled, motion.joints(sampled, 60))
    np.testing.assert_allclose(reconstructed, motion.joints(t, 60), atol=1e-10)


@pytest.mark.parametrize("angle", [45, 50, 63, 65])
def test_dwell_return_duration_and_continuity(motion, angle):
    ack, ret, done = motion.timing(angle)
    assert ack == pytest.approx(angle / 10 + 0.3)
    assert ret - ack == pytest.approx(2.0)
    assert done - ret == pytest.approx(2.4)
    for boundary in [0, ack, ret, done]:
        before, after = motion.joints([boundary - 1e-8, boundary + 1e-8], angle)
        np.testing.assert_allclose(before, after, atol=2e-8)
    np.testing.assert_allclose(motion.joints([-1, done + 0.5], angle),
                               motion.joints([-1, motion.return_ack_s + 0.5], 60),
                               atol=1e-12)


def test_inversion_and_invalid_curves():
    np.testing.assert_allclose(invert_angles([40, 50, 60], [0, 100, 200], [50, 170]),
                               [45, 57])
    with pytest.raises(ValueError, match="outside"):
        invert_angles([45, 60], [40, 159], [170])
    with pytest.raises(ValueError, match="increase"):
        invert_angles([45, 50, 60], [40, 160, 150], [100])
    with pytest.raises(ValueError, match="increase"):
        invert_angles([45, 50, 60], [40, 40, 150], [100])


def test_reject_unsupported_angle(motion):
    for angle in [float("nan"), 30, 70]:
        with pytest.raises(ValueError, match="between"):
            motion.timing(angle)


def test_local_refinement_keeps_decreasing_samples():
    # A locally decreasing finite-grid response must remain visible; select the
    # observed increasing crossing without sorting or modifying the volumes.
    angle = bracket_candidate([45, 46, 50], [42, 41, 80], 50)
    assert angle == pytest.approx(46.92)
    with pytest.raises(ValueError, match="No increasing"):
        bracket_candidate([45, 46, 50], [42, 41, 80], 100)
    with pytest.raises(ValueError, match="resolution"):
        bracket_candidate([46.91, 46.92], [41, 80], 50)

"""Small per-step motion must accumulate even below float32 state resolution."""
import numpy as np
import pytest
import warp as wp
from experiments.robotics.plate_observable_precision_audit import run
from warpmpm import Solver, GridConfig


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_manufactured_slow_motion_and_strain(device):
    if device.startswith('cuda') and not wp.get_cuda_device_count():
        pytest.skip('requires CUDA')
    for result in run(device).values():
        assert result['max_position_error_um'] < .03
        assert result['max_F_error'] < 2e-6


def test_state_import_discards_previous_integration_residual():
    import torch
    x = np.array([[.1, .1, .1]], dtype='float32')
    s = Solver(GridConfig(16, .2), device='cpu').load_particles(x, np.array([1e-9], 'float32'))
    st = s._sim.mpm_state
    st.particle_x_roundoff.fill_(wp.vec3(1e-9))
    st.particle_F_roundoff.fill_(wp.mat33(1e-8))
    s._sim.import_particle_x_from_torch(torch.from_numpy(x.copy()), device='cpu')
    s._sim.import_particle_F_from_torch(torch.eye(3)[None], device='cpu')
    np.testing.assert_array_equal(st.particle_x_roundoff.numpy(), 0)
    np.testing.assert_array_equal(st.particle_F_roundoff.numpy(), 0)


def test_constitutive_projection_resets_only_replaced_elastic_state():
    from warpmpm.materials import vonmises
    from warpmpm.kernels.mpm_utils import compute_stress_from_F_trial
    x = np.array([[.09, .1, .1], [.11, .1, .1]], dtype='float32')
    s = Solver(GridConfig(16, .2), device='cpu').load_particles(x, np.full(2, 1e-9, 'float32'))
    s.set_material(vonmises(E=80000, nu=.3, yield_stress=1000, density=1000))
    st = s._sim.mpm_state
    trial = np.array([np.diag([1.0001, 1., 1.]), np.diag([1.1, .95, .95])], dtype='float32')
    wp.copy(st.particle_F_trial, wp.array(trial, dtype=wp.mat33, device='cpu'))
    residual = np.full((2, 3, 3), 1e-8, 'float32')
    wp.copy(st.particle_F_roundoff, wp.array(residual, dtype=wp.mat33, device='cpu'))
    wp.launch(compute_stress_from_F_trial, dim=2, inputs=[st, s._sim.mpm_model, 1e-5], device='cpu')
    np.testing.assert_array_equal(st.particle_F.numpy()[0], trial[0])
    assert not np.array_equal(st.particle_F.numpy()[1], trial[1])
    np.testing.assert_array_equal(st.particle_F_roundoff.numpy()[0], residual[0])
    np.testing.assert_array_equal(st.particle_F_roundoff.numpy()[1], 0)

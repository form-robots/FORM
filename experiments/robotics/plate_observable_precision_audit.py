"""Manufactured grid-to-particle motion check; no material fitting or truth tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpmpm import Solver, GridConfig
from warpmpm.kernels.mpm_utils import g2p_particle
from warpmpm.kernels.warp_utils import MPMStateStruct, MPMModelStruct


@wp.kernel
def advance(st: MPMStateStruct, model: MPMModelStruct, dt: float, steps: int):
    p = wp.tid()
    for _ in range(steps):
        g2p_particle(st, model, dt, p)
        st.particle_F[p] = st.particle_F_trial[p]


def run(device='cpu'):
    dt = 1e-5
    steps = 100000
    x = np.array([[.06, .06, .06], [.10, .10, .10], [.14, .14, .14]], dtype='float32')
    results = {}
    for name, rate, speed in [('translation', 0., .0001), ('slow_strain', .005, 0.),
                              ('faster_strain', .02, 0.)]:
        s = Solver(GridConfig(16, .2), device=device).load_particles(x, np.full(3, 1e-9, 'float32'))
        st, model = s._sim.mpm_state, s._sim.mpm_model
        axes = np.arange(16) * (.2 / 16)
        nodes = np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), axis=-1)
        velocity = (rate * (nodes - .1) + speed).astype('float32')
        wp.copy(st.grid_v_out, wp.array(velocity, dtype=wp.vec3, device=device))
        wp.launch(advance, dim=3, inputs=[st, model, dt, steps], device=device)
        factor = (1. + float(np.float32(dt)) * rate) ** steps
        expected_x = (x.astype(float) - .1) * factor + .1 if rate else x.astype(float) + steps * float(np.float32(dt)) * speed
        actual_x, actual_F = s.x(), s.F()
        expected_F = np.eye(3) * factor
        results[name] = dict(dt=dt, steps=steps, rate=rate, speed=speed,
                            initial_x=x.tolist(), final_x=actual_x.tolist(), expected_x=expected_x.tolist(),
                            max_position_error_um=float(np.max(abs(actual_x-expected_x))*1e6),
                            F_diagonal=np.diagonal(actual_F, axis1=-2, axis2=-1).tolist(),
                            expected_F_diagonal=[factor]*3,
                            max_F_error=float(np.max(abs(actual_F-expected_F))))
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    result = run(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

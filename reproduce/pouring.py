"""Replay the frozen pouring model and invert its simulated volume curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from reproduce.run import ROOT, read


def replay(output, device, angle=None):
    import numpy as np
    from experiments.pour import pour_transfer_recalibration_reference as reference
    from experiments.pour.pour_measured_timing import MeasuredTimingMotion, characterize

    output.mkdir(parents=True, exist_ok=False)
    timing = read(ROOT / 'out/pour_navier_calibration/measured_timing_20260907/motion_characterization.json')
    original_factory = reference.planned_motion
    original_project = reference.twin.project_out_of_solid
    original_output = reference.OUT
    gate = {}

    def timed_motion(max_angle):
        motion, episode = original_factory(max_angle)
        assert characterize(motion) == timing
        return MeasuredTimingMotion(motion, timing), episode

    def check_settling(x, v, *args, **kwargs):
        if not gate:
            speed = np.linalg.norm(v, axis=1)
            gate['mean_speed_m_s'] = float(speed.mean())
            if not np.isfinite(speed).all() or speed.mean() >= reference.twin.SETTLE_SPEED:
                raise RuntimeError('Initial liquid settling failed')
        return original_project(x, v, *args, **kwargs)

    reference.OUT = output
    if angle is not None:
        reference.planned_motion = timed_motion
    reference.twin.project_out_of_solid = check_settling
    try:
        reference.run(SimpleNamespace(source_friction=.272, grid=160, phase=0.,
            wall='original-separable', slip_mm=None, angle=angle, max_angle=70.,
            dt_scale=1., device=device))
    finally:
        reference.OUT = original_output
        reference.planned_motion = original_factory
        reference.twin.project_out_of_solid = original_project
    paths = list(output.rglob('result.json'))
    assert len(paths) == 1
    result = read(paths[0])
    assert result['particle_count'] == 229280
    (output / 'settling.json').write_text(json.dumps(gate, indent=2) + '\n')
    return result


def plan(output, device):
    from experiments.pour.pour_transfer_target_plan import linear_proposal

    output.mkdir(parents=True, exist_ok=False)
    points = {}
    selected = {}
    targets = [60, 80, 100, 120, 140, 160]

    def simulate(angle):
        if angle in points:
            return
        result = replay(output / f'angle_{angle:.2f}', device, angle)
        if result['outside_ml'] > 3 or result['tail_variation_ml'] > .5:
            raise RuntimeError('Pouring ledger or terminal-state check failed')
        points[angle] = result['receiver_ml']
        (output / 'curve.json').write_text(json.dumps(sorted(points.items()), indent=2) + '\n')

    for angle in [48.37, 53.04]:
        simulate(angle)
    while len(selected) < len(targets):
        pairs = sorted(points.items())
        if any(b[1] <= a[1] for a, b in zip(pairs, pairs[1:])):
            raise RuntimeError('Nonmonotone simulated volume curve')
        proposals = []
        for target in targets:
            if target in selected:
                continue
            best = min(points, key=lambda a: abs(points[a] - target))
            if abs(points[best] - target) <= 1.:
                selected[target] = dict(angle_deg=best, predicted_ml=points[best])
            else:
                angle = linear_proposal(pairs, target)
                if angle in points:
                    raise RuntimeError('Repeated inverse proposal outside the 1 mL tolerance')
                proposals.append(angle)
        for angle in dict.fromkeys(proposals):
            if len(points) >= 22:
                raise RuntimeError('Target simulation budget exhausted')
            simulate(angle)
    (output / 'selected.json').write_text(json.dumps(selected, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--angle', type=float)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.angle is not None and not 45 <= args.angle <= 68:
        parser.error('--angle must lie between 45 and 68 degrees')
    if args.plan:
        plan(args.out, args.device)
    else:
        replay(args.out, args.device, args.angle)

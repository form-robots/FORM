"""Run paper computations in this checkout, with fresh outputs and frozen inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def read(path):
    return json.loads(Path(path).read_text())


def launch(module, *args, cwd=ROOT):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(cwd / 'src'), str(cwd)]))
    subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=cwd, env=env, check=True)


def runtime(profile):
    """Cache a source snapshot without changing any running profile."""
    import hashlib
    names = ['src', 'experiments', 'examples', 'fe-weights', 'reproduce', 'assets']
    files = sorted(p for name in names for p in (ROOT / name).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    fingerprint = hashlib.sha256()
    for path in files:
        fingerprint.update(str(path.relative_to(ROOT)).encode())
        fingerprint.update(path.read_bytes())
    revision = fingerprint.hexdigest()
    dest = ROOT / 'out/runtime' / f'{profile}-{revision[:16]}'
    marker = dest / 'runtime.json'
    if marker.exists():
        assert read(marker) == dict(profile=profile, source_sha256=revision)
        return dest
    dest.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copytree(ROOT / name, dest / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['out', 'data', 'pouring_real_data', 'press_real_data', 'shaping_real_data']:
        source, link = ROOT / name, dest / name
        if source.exists():
            link.symlink_to(os.path.relpath(source, link.parent), target_is_directory=True)
    patch = ROOT / 'reproduce/profiles' / f'{profile}.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=dest, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=dest, check=True)
    marker.write_text(json.dumps(dict(profile=profile, source_sha256=revision), indent=2) + '\n')
    return dest


def identify(kind, output):
    output.mkdir(parents=True, exist_ok=False)
    if kind == 'elastic':
        from experiments.elastic.strip_camera_identify import identify as fit
        base = ROOT / 'out/strip_texture_20260912'
        for material in 'AB':
            dest = output / material
            dest.mkdir()
            shutil.copy2(base / f'fit_{material}/tracks.npz', dest / 'tracks.npz')
            fit(base / f'inputs_{material}', dest)
    elif kind == 'plastic':
        from experiments.robotics.plate_separated_identify import fit
        base = ROOT / 'out/press_separated_20260913/monotonic320'
        for material in 'AB':
            fit(base, material, output / material)
    elif kind == 'hardware':
        from experiments.robotics.press_hardware_linear_identify import fit
        base = ROOT / 'out/press_weakform_restart_20260914/identification_inputs'
        for material, episode in [('play_doh', 'ep0003'), ('butter_slime', 'ep0007'), ('plasticine', 'ep0011')]:
            fit(base, episode, output / material)
    else:
        import numpy as np
        from experiments.pour.pour_weakform_recovery import fit
        base = ROOT / 'out/pour_three_video_identification/09-04-60-2s'
        reference = read(base / 'identify.json')
        observations = dict(np.load(base / 'observations.npz'))
        eta, se, keep, dv, cumulative, forcing, rms = fit(observations, tuple(reference['fit_window_s']))
        (output / 'identification.json').write_text(json.dumps(
            dict(eta=eta, eta_se=se, rms_mL=rms, t_fit=reference['fit_window_s']), indent=2) + '\n')
        np.savez_compressed(output / 'weak_system.npz', keep=keep,
                            delta_volume_ml=dv, cumulative_forcing=cumulative)
        print(f'Effective viscosity: {eta:.12g} Pa s')


def insertion(output, device, replan):
    from experiments.elastic.rod_insertion_study import read, plan, simulate
    from experiments.elastic.rod_insertion_franka import compile_motion
    source = ROOT / 'out/rod_insertion_franka_20260912'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / 'protocol.json', output / 'protocol.json')
    shutil.copytree(source / 'franka', output / 'franka')
    for material in 'AB':
        if replan:
            plan(output, material, device)
            compile_motion(output, material)
        else:
            shutil.copytree(source / f'plan_{material}', output / f'plan_{material}')
    protocol = read(output / 'protocol.json')
    for material in 'AB':
        for planned in 'AB':
            selected = read(output / f'plan_{planned}/plan.json')
            simulate(protocol, protocol['true_E_pa'][material], selected['controls'],
                     protocol['fine_grid'], device, physical_wall=True,
                     output=output / f'fine_{material}_plan_{planned}',
                     motion=output / f'franka/plan_{planned}.npz')


def putting(output, device, replan):
    from experiments.elastic.putting_study import prepare, plan, execute
    source = ROOT / 'out/putting_swing_20260913'
    prepare(output, source)
    if not replan:
        shutil.copytree(source / 'franka', output / 'franka', dirs_exist_ok=True)
    for material in 'AB':
        if replan:
            plan(output, material, device)
        else:
            shutil.copytree(source / f'plan_{material}', output / f'plan_{material}')
    for material in 'AB':
        for planned in 'AB':
            execute(output, material, planned, device, video=True)


def shaping(output, device, hardware):
    import numpy as np
    from experiments.robotics import x_motion_shaping as core
    from experiments.robotics import x_motion_search as search
    if hardware:
        from experiments.shaping_real.frozen_rollout import run
        from reproduce.planning import prepare
        source = ROOT / 'out/press_iou_independent_20260915/study'
        prepare(source, output)
        for material in ['play_doh', 'butter_slime', 'plasticine']:
            selected = read(source / f'plans/{material}/FINAL.json')
            parameters = selected.get('parameters', selected.get('selected_parameters'))
            if parameters is None:
                parameters = selected['selected']['parameters']
            run(output, material, parameters, material, device)
    else:
        output.mkdir(parents=True, exist_ok=False)
        source = ROOT / 'out/press_paper_update_20260913/execution80'
        protocol = read(source / 'protocol.json')
        core.geometry.CONFIG.update(protocol['scene'])
        models = read(source / 'inputs/models.json')
        initial = dict(np.load(source / 'inputs/specimen.npz'))
        frozen = read(source / 'execution_plan.json')
        for material in 'AB':
            for planned in 'AB':
                row = frozen['selected'][planned]
                stored = core.arrays(source / row['file'])
                command = {k: stored[k] for k in ['time', 'tool_centers', 'phase_id', 'pinch_id', 'start_pose', 'gaps_mm']}
                phases = read((source / row['file']).with_suffix('.phases.json'))
                result = core.simulate(models['true_' + material], command, phases, initial,
                                       device, protocol['grid'], protocol['dt'])
                np.savez_compressed(output / f'{material}_plan_{planned}.npz', **result)


def predict_pressing(output, device):
    from experiments.robotics.press_hardware_refine_forward import run
    source = ROOT / 'out/press_weakform_restart_20260914'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / 'protocol.json', output / 'protocol.json')
    for material, episode in [('play_doh', 'ep0002'), ('butter_slime', 'ep0006'), ('plasticine', 'ep0010')]:
        shutil.copytree(source / 'prediction_inputs' / episode, output / episode)
        (output / material).mkdir()
        shutil.copy2(source / material / 'fit.json', output / material / 'fit.json')
        settings = read(source / material / 'prediction/completion.json')
        run(output, material, grid=settings['grid'], friction=settings['friction_assumed'],
            duration=15., tick=settings['tick'], device=device, tag='prediction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['identify-elastic', 'identify-plastic', 'identify-hardware',
                        'identify-fluid', 'predict-pressing', 'insertion', 'putting', 'shaping-sim', 'shaping-real',
                        'benchmark', 'benchmark-fe', 'benchmark-ingest', 'pouring', 'plan-pouring'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replan', action='store_true')
    parser.add_argument('--material', choices=['jelly', 'sand', 'plasticine', 'water'], default='jelly')
    parser.add_argument('--tier', choices=['full', 'no-stress', 'positions-only'], default='full')
    parser.add_argument('--nclaw-root', type=Path)
    parser.add_argument('--angle', type=float)
    args = parser.parse_args()
    if args.task.startswith('benchmark'):
        cwd = runtime('benchmark')
        if args.task == 'benchmark-ingest':
            if args.nclaw_root is None:
                parser.error('--nclaw-root is required for ingestion')
            nclaw = args.nclaw_root.resolve()
            launch('experiments.nclaw.cross_generalize', '--log-root', nclaw / 'experiments/log',
                   '--nclaw-dir', nclaw, '--material', args.material, cwd=cwd)
        elif args.task == 'benchmark-fe':
            launch('experiments.fe_ls.cross', args.material, f'--device={args.device}', cwd=cwd)
        else:
            extra = [] if args.tier == 'full' else ['--' + args.tier]
            launch('experiments.nclaw.compare', args.material, f'--device={args.device}', *extra, cwd=cwd)
        return
    output = (args.out or ROOT / 'out/reproduced' / args.task).resolve()
    if args.task in ['pouring', 'plan-pouring']:
        cwd = runtime('pouring')
        extra = ['--plan'] if args.task == 'plan-pouring' else ([] if args.angle is None else ['--angle', args.angle])
        launch('reproduce.pouring', '--out', output, '--device', args.device, *extra, cwd=cwd)
    elif args.replan and args.task in ['shaping-sim', 'shaping-real']:
        from reproduce.planning import simulated, hardware
        (hardware if args.task == 'shaping-real' else simulated)(output, args.device)
    elif args.task.startswith('identify-'):
        identify(args.task.removeprefix('identify-'), output)
    elif args.task == 'predict-pressing':
        predict_pressing(output, args.device)
    elif args.task == 'insertion':
        insertion(output, args.device, args.replan)
    elif args.task == 'putting':
        putting(output, args.device, args.replan)
    else:
        shaping(output, args.device, args.task == 'shaping-real')


if __name__ == '__main__':
    main()

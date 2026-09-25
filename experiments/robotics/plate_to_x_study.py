"""Transfer frozen image/force plate estimates to the existing X-shaping task.

The materials, target, contact and six-action family are unchanged. Each
model receives the same additional planning budget. Cross-results never
select a plan. Prior shaping development supplies explicitly recorded starts.
"""
from pathlib import Path
import argparse
import shutil

from experiments.robotics import x_motion_search as search
from experiments.robotics import x_motion_shaping as core
from experiments.robotics.plastic_shaping_study import save_json, save_npz

PLATE = core.ROOT / 'out/plate_observable_physical_preload_20260912'
PRIOR = core.ROOT / 'out/x_motion_refinement_20260911/six_placement'


def prepare(folder):
    spec = core.read(PRIOR / 'protocol.json')['spec']
    spec.update(name='plate_to_x', title='X shaping from image/force identification',
                start={k: core.read(PRIOR / f'plans/{k}/selected.json')['parameters'] for k in 'AB'},
                start_rule='Previous six-pinch own-model selections; no new cross-outcomes used',
                warm_start_source=str(PRIOR), cumulative_prior_evaluations_each=152)
    spec_path = folder.parent / (folder.name + '_spec.json')
    save_json(spec_path, spec)
    search.prepare(core.ROOT / 'out/x_motion_shaping_20260911', spec_path, folder)
    models = core.read(folder / 'inputs/models.json')
    evidence = {}
    for k in 'AB':
        fit_path = PLATE / f'divfree_{k}/identification.json'
        fit = core.read(fit_path)
        models['identified_' + k] = dict(E=fit['E_pa'], nu=.3, yield_stress=fit['yield_pa'])
        copy = folder / f'inputs/plate_fit_{k}.json'
        shutil.copy2(fit_path, copy)
        evidence[k] = dict(source=str(fit_path), sha256=core.digest(fit_path))
    save_json(folder / 'inputs/models.json', models)
    p = core.read(folder / 'protocol.json')
    # Freeze numerical sources; presentation-only files can evolve while runs
    # execute. All original source copies remain in the snapshot for recovery.
    dependencies = {'x_motion_search.py', 'x_motion_shaping.py', 'x_shaping.py',
                    'plastic_shaping_study.py', 'plastic_shaping_figure.py',
                    'hex_shaping_surface.py', 'hex_shaping_figure.py', 'plate_to_x_study.py'}
    p['source_sha256'] = {rel: sha for rel, sha in p['source_sha256'].items()
                          if rel.startswith('src/') or Path(rel).name in dependencies}
    p.update(input_sha256={str(x.relative_to(folder)): core.digest(x) for x in (folder/'inputs').iterdir()},
             plate_estimates=evidence,
             identification='Synthetic textured stereo and plate force; known-family nonlinear plastic-history fit',
             geometry_transfer='30x30x25 mm identification specimen to fresh 60x60x25 mm shaping specimen',
             solver='Current compensated position/strain accumulation; all displayed outcomes rerun',
             evaluation_policy='48 further own-model evaluations each, then all four cases at grids 64 and 80; no cross-outcome retuning')
    save_json(folder / 'protocol.json', p)
    search.check(folder)


def finer(parent, folder):
    p = search.check(parent)
    frozen = core.read(parent / 'execution_plan.json')
    folder.mkdir(exist_ok=False)
    shutil.copytree(parent / 'inputs', folder / 'inputs')
    shutil.copytree(parent / 'source_snapshot', folder / 'source_snapshot')
    xyz, vol = core.geometry.specimen(80)
    save_npz(folder / 'inputs/specimen.npz', dict(initial=xyz, vol0=vol))
    for row in frozen['selected'].values():
        src, dst = parent / row['file'], folder / row['file']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        shutil.copy2(src.with_suffix('.phases.json'), dst.with_suffix('.phases.json'))
        assert core.digest(dst) == row['data_sha256']
    p.update(grid=80, dt=.00004, parent=str(parent), planning_grid=64,
             purpose='Unchanged coarse-grid plans executed at finer resolution',
             input_sha256={str(x.relative_to(folder)): core.digest(x) for x in (folder/'inputs').iterdir()})
    save_json(folder/'protocol.json', p)
    frozen.update(protocol_sha256=core.digest(folder/'protocol.json'),
                  parent_execution_plan_sha256=core.digest(parent/'execution_plan.json'))
    save_json(folder/'execution_plan.json', frozen)


def execute(folder, material, device):
    p = search.check(folder)
    frozen = core.read(folder/'execution_plan.json')
    assert frozen['protocol_sha256'] == core.digest(folder/'protocol.json')
    law = core.read(folder/'inputs/models.json')['true_' + material]
    for planned, row in frozen['selected'].items():
        stored = core.arrays(folder/row['file'])
        command = {k: stored[k] for k in ['time', 'tool_centers', 'phase_id', 'pinch_id', 'start_pose', 'gaps_mm']}
        phases = core.read((folder/row['file']).with_suffix('.phases.json'))
        record = search.rollout(folder, folder/f'{material}_plan_{planned}.npz', law,
            command, phases, device, dict(material=material, planned_for=planned,
                selected_data_sha256=row['data_sha256'], execution_plan_sha256=core.digest(folder/'execution_plan.json')),
            grid=p['grid'], dt=p['dt'])
        print(record, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'finer', 'execute'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--material', choices=['A', 'B'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.out)
    elif args.stage == 'finer': finer(args.parent, args.out)
    else: execute(args.out, args.material, args.device)

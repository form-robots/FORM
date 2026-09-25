"""Portable planning entry points using the paper's frozen search protocols."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from reproduce.data import digest
from reproduce.run import ROOT, read


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(source, output, *, snapshot=False):
    output.mkdir(parents=True, exist_ok=False)
    protocol = read(source / 'protocol.json')
    save(output / 'reference_protocol.json', protocol)
    sources = [p for directory in ['src', 'experiments', 'reproduce']
               for p in (ROOT / directory).rglob('*.py')]
    protocol['source_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if snapshot:
        for p in sources:
            target = output / 'source_snapshot' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    if (source / 'inputs').exists():
        shutil.copytree(source / 'inputs', output / 'inputs')
        protocol['input_sha256'] = {str(p.relative_to(output)): digest(p)
                                   for p in (output / 'inputs').rglob('*') if p.is_file()}
    if (source / 'target.vtp').exists():
        shutil.copy2(source / 'target.vtp', output / 'target.vtp')
    protocol['release_note'] = 'Numerical settings retained; source hashes refer to this portable release. Original protocol retained alongside.'
    save(output / 'protocol.json', protocol)
    return protocol


def simulated(output, device):
    from experiments.robotics import x_motion_search as search, x_motion_shaping as core
    from experiments.robotics.plate_to_x_study import finer, execute

    planning = output / 'planning'
    protocol = prepare(ROOT / 'out/press_paper_update_20260913/planning', planning, snapshot=True)
    core.geometry.CONFIG.update(protocol['scene'])
    for material in 'AB':
        search.plan(planning, material, device)
    search.freeze(planning)
    fine = output / 'execution80'
    finer(planning, fine)
    for material in 'AB':
        execute(fine, material, device)


def hardware(output, device):
    import numpy as np
    from scipy.optimize import minimize
    from experiments.shaping_real.frozen_rollout import run

    protocol = prepare(ROOT / 'out/press_iou_independent_20260915/study', output)
    spec = protocol['spec']

    def feasible(row):
        return (row['components'] == 1 and row['inversion_count'] == 0
                and row['surface_genus_sum'] == 0 and row['surface_open_edges'] == 0)

    for material in ['play_doh', 'butter_slime', 'plasticine']:
        candidates = []
        for index, start in enumerate(spec['starts']):
            cache, evaluations = {}, []
            x = np.array(start, float)
            bounds = np.array(spec['bounds'])
            simplex = np.tile(x, (len(x) + 1, 1))
            for i, step in enumerate(spec['simplex_steps']):
                simplex[i + 1, i] += -step if x[i] - step >= bounds[i, 0] else step
            destination = Path('plans') / material / f'start_{index}'

            def objective(parameters):
                parameters = np.round(parameters, 4)
                key = tuple(parameters)
                if key not in cache:
                    cache[key] = run(output, material, parameters,
                                     destination / f'trial_{len(cache):03d}', device, 64)
                row = cache[key]
                cost = 1 - row['footprint_iou'] + (0 if feasible(row) else 100)
                evaluations.append(dict(parameters=row['parameters'], objective=cost,
                                        iou=row['footprint_iou'], valid=feasible(row)))
                save(output / destination / 'evaluations.json', evaluations)
                return cost

            minimize(objective, x, method='Nelder-Mead', bounds=bounds,
                     options=dict(maxfev=spec['evaluations_per_start'], initial_simplex=simplex,
                                  xatol=0., fatol=0.))
            assert len(evaluations) == spec['evaluations_per_start']
            candidates.extend(row for row in cache.values() if feasible(row))
        finalists, seen = [], set()
        for row in sorted(candidates, key=lambda row: (-row['footprint_iou'], row['path'])):
            parameters = tuple((np.round(np.array(row['parameters']) * 2) / 2).tolist())
            if parameters not in seen:
                seen.add(parameters)
                finalists.append(parameters)
            if len(finalists) == 3:
                break
        if len(finalists) != 3:
            raise RuntimeError('Fewer than three feasible finalists')
        save(output / 'plans' / material / 'FINALISTS.json', finalists)
        results = [run(output, material, parameters,
                       Path('rounded_coarse') / f'{material}_{i:02d}', device, 64)
                   for i, parameters in enumerate(finalists)]
        valid = [row for row in results if feasible(row)]
        if not valid:
            raise RuntimeError('No feasible rounded command')
        save(output / 'plans' / material / 'FINAL.json',
             dict(selected=max(valid, key=lambda row: row['footprint_iou']),
                  selection_grid=64, rounded_results=results))

"""Direct elastic/plastic weak fits for a designed monotonic compression probe.

This reduced estimator assumes an initially elastic interval and a later
fully yielded interval with approximately proportional, coaxial deformation.
It does NOT reconstruct an arbitrary plastic history. The test field removes
isotropic stress; the observed spatial Hencky deviator supplies both columns.
Two disjoint scalar fits are one block-diagonal linear least-squares problem.
No yield-ratio search, return mapping, or forward dynamics occurs in fitting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from experiments.robotics.plate_observable_field import reconstruct, volume_quadrature
from experiments.robotics.plate_observable_interior import weak

DEFAULT = dict(elastic_centers=[.40,.60], plastic_centers=[1.60,1.78], order=8)


def hencky_columns(F, nu):
    """Unit-E elastic deviator and unit-Y saturated proportional deviator.

The yield parameter is the Frobenius cap of the Kirchhoff stress deviator,
matching this repository's Hencky/von-Mises implementation.
"""
    U, s, _ = np.linalg.svd(F)
    H = np.einsum('...ik,...k,...jk->...ij', U, np.log(s), U)
    H -= np.eye(3) * np.trace(H, axis1=-2, axis2=-1)[..., None, None] / 3
    norm = np.linalg.norm(H, axis=(-2,-1))
    return H / (1 + nu), H / np.maximum(norm[...,None,None], 1e-12)


def solve_columns(elastic, plastic, b, centers, config):
    estimates = {}; diagnostics = {}
    for name, a, bounds in [('E_pa', elastic, config['elastic_centers']),
                             ('yield_pa', plastic, config['plastic_centers'])]:
        use = (centers >= bounds[0]-1e-10) & (centers <= bounds[1]+1e-10)
        if use.sum() < 6 or float(a[use]@a[use]) <= 1e-20:
            raise ValueError(f'Uninformative {name} interval')
        value = float(a[use] @ b[use] / (a[use] @ a[use]))
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f'Nonpositive {name} estimate')
        estimates[name] = value
        diagnostics[name] = dict(rows=int(use.sum()), center_range_s=[float(centers[use].min()),float(centers[use].max())],
            weak_residual_relative_l2=float(np.linalg.norm(a[use]*value-b[use])/np.linalg.norm(b[use])),
            per_window_estimates=[float(a[use][j:j+3]@b[use][j:j+3]/(a[use][j:j+3]@a[use][j:j+3])) for j in range(0,int(use.sum()),3)])
    return estimates, diagnostics


def assemble(root, material, order=8):
    p = json.loads((root/f'inputs_{material}/known.json').read_text())
    selected = json.loads((root/'field_selection.json').read_text())['selected']
    p['cross_section_degree'] = selected.get('cross_section_degree',3)
    tracks = np.load(root/f'fit_{material}/tracks.npz')
    q, weights = volume_quadrature(p, selected['spacing'], order)
    X, V, F, _, errors = reconstruct(tracks,p,selected['spacing'],quadrature=q,smoothing=selected.get('smoothing',0))
    sensor = np.genfromtxt(root/f'inputs_{material}/force.csv',delimiter=',',names=True)
    W, b, g = weak(X,V,tracks['time'],p,sensor,weights)
    stress_E, stress_Y = hencky_columns(F,p['nu'])
    def column(stress):
        return (W@np.einsum('tnij,tmnij,n->tm',stress,g,weights)).ravel()*np.prod(p['size'])
    centers = np.repeat(np.arange(.20,p['fit_end']-.159,.06)+.08,3)
    data = dict(elastic=column(stress_E),plastic=column(stress_Y),b=b,centers=centers)
    info = dict(surface_fit_rmse_mm=float(np.sqrt(np.mean(errors**2))*1000),
                min_reconstructed_J=float(np.linalg.det(F).min()),known_nu=p['nu'],quadrature_order=order)
    return data,info


def fit(root, material, dest, config=None):
    config = DEFAULT if config is None else config
    start=time.perf_counter(); data,info=assemble(root,material,config['order']); assembled=time.perf_counter()
    estimates,diagnostics=solve_columns(**data,config=config); solved=time.perf_counter()
    result=dict(**estimates,**info,diagnostics=diagnostics,config=config,
                estimator='Separated proportional elastic/plastic weak balances; one block-diagonal linear least-squares problem',
                limitations=['Prescribed elastic and saturated proportional loading intervals',
                             'Not a general nonproportional plastic-history estimator',
                             'Known nu, density, initial state and frictionless symmetric geometry'],
                forward_simulation_calls=0,ratio_search_evaluations=0,constitutive_history_replays=0,
                assembly_wall_s=assembled-start,coefficient_solve_wall_s=solved-assembled,
                simulator_states_used=False)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'identification.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(dest/'weak_system.npz',**data)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--material',required=True,choices=['A','B']);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--config',type=Path)
    args=ap.parse_args();fit(args.root,args.material,args.out,json.loads(args.config.read_text()) if args.config else None)

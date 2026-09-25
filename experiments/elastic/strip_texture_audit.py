"""Evaluation-only accuracy checks and an observation-isolated repeat."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from experiments.elastic.strip_camera_identify import reconstruct_tracks,save
from experiments.elastic.strip_camera_reconstruction import fit_balance
from ident.weakform.elastic_grid import corotated_cauchy_columns


def evaluate(root):
    protocol=json.loads((root/'protocol.json').read_text()); result={}
    for k in ['A','B']:
        p=json.loads((root/f'inputs_{k}/known.json').read_text()); d=np.load(root/f'fit_{k}/tracks.npz')
        x=np.load(root/f'truth_{k}/x.npy',mmap_mode='r'); axes=[np.unique(x[0,:,i]) for i in range(3)]
        shape=tuple(len(a) for a in axes)+(3,)
        # Initial coordinates come from triangulation; only evaluate the implied
        # material-attached surface locations after the camera fit is frozen.
        ref=d['reference'].copy(); ref[:,1]=p['surface_reference_y']
        true=np.array([RegularGridInterpolator(axes,x[round(t/protocol['tick'])].reshape(shape),
                         bounds_error=False,fill_value=None)(ref) for t in d['time']])
        valid=d['valid']; motion=(d['world']-d['world'][0])-(true-true[0])
        X,V,F,vol,_=reconstruct_tracks(dict(world=true,reference=true[0],valid=valid,time=d['time']),p)
        sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',delimiter=',',names=True)
        ideal,_,_=fit_balance(X,V,F,vol,d['time'],sensor['time'],sensor['material_on_pusher_Fx'],p)
        result[k]=dict(surface_motion_rmse_mm=float(np.sqrt(np.mean(np.sum(motion[valid]**2,axis=1)))*1000),
                       ideal_surface_diagnostic=ideal)
        np.save(root/f'evaluation_surface_{k}.npy',true)
    save(root/'observation_audit.json',result)
    return result


def stress_audit(root,previous):
    out={}
    for path in [previous,root]:
        p=json.loads((path/'protocol.json').read_text()); out[path.name]={}
        for k in ['A','B']:
            F=np.load(path/f'truth_{k}/F.npy',mmap_mode='r'); x=np.load(path/f'truth_{k}/x.npy',mmap_mode='r')
            t=np.load(path/f'truth_{k}/time.npy'); a=b=0.
            for i in np.flatnonzero((t>=.32)&(t<=.88))[::5]:
                use=(x[i,:,2]>.112)&(x[i,:,2]<.150)
                sm,sl=corotated_cauchy_columns(F[i,use].astype(float));nu=p['known_nu']
                stress=sm/(2*(1+nu))+sl*nu/((1+nu)*(1-2*nu))
                a+=np.sum(stress[:,1,1]**2); b+=np.sum(stress**2)
            out[path.name][k]=dict(normal_stress_to_full_stress_rms_ratio=float(np.sqrt(a/b)))
    save(root/'plane_stress_audit.json',out)
    return out


def isolated(root):
    repo=Path(__file__).resolve().parents[2]; work=root/'isolated_check';work.mkdir(exist_ok=False)
    result={}
    for k in ['A','B']:
        bundle=work/f'inputs_{k}';bundle.mkdir(); dest=work/f'repeat_{k}';dest.mkdir()
        for name in ['camera_0.npy','camera_1.npy','time.npy','known.json','calibration.json','force.csv']:
            os.link(root/f'inputs_{k}'/name,bundle/name)
        script='''import sys
from pathlib import Path
repo,bundle,dest=map(Path,sys.argv[1:])
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)): return
    p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
    if p.is_relative_to(repo/'out') and not (p.is_relative_to(bundle) or p.is_relative_to(dest)):
        raise PermissionError('Forbidden experiment input: '+str(p))
sys.addaudithook(guard)
try:
    open(repo/'out/forbidden_simulator_state.npy','rb')
    raise AssertionError('Guard failed')
except PermissionError: pass
from experiments.elastic.strip_texture_identify import track
from experiments.elastic.strip_texture_dic import refine
from experiments.elastic.strip_camera_identify import identify
track(bundle,dest/'flow')
refine(bundle,dest/'flow',dest/'fit')
identify(bundle,dest/'fit')
'''
        run=subprocess.run([sys.executable,'-c',script,str(repo),str(bundle.resolve()),str(dest.resolve())],
                            cwd=repo,text=True,capture_output=True,check=True)
        (work/f'{k}.log').write_text(run.stdout+run.stderr)
        original=np.load(root/f'fit_{k}/tracks.npz'); repeated=np.load(dest/'fit/tracks.npz')
        np.testing.assert_allclose(original['world'],repeated['world'],rtol=0,atol=0,equal_nan=True)
        a=json.loads((root/f'fit_{k}/identification.json').read_text())['E_pa']
        b=json.loads((dest/'fit/identification.json').read_text())['E_pa']
        np.testing.assert_allclose(a,b,rtol=1e-12)
        result[k]=dict(exact_tracks_reproduced=True,E_pa=b,forbidden_data_guard_active=True)
    save(root/'observation_isolation.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['evaluate','isolate','stress'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--previous',type=Path,default=Path('out/strip_bending_20260912_clearance'))
    a=ap.parse_args(); print(evaluate(a.out) if a.stage=='evaluate' else isolated(a.out) if a.stage=='isolate' else stress_audit(a.out,a.previous))

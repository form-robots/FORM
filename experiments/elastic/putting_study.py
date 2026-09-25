"""Finite-budget forward planning and frozen matched/swapped putting executions.

Planning scores only the identified model. True stiffness is used only when
executing a frozen plan. Every trial and numerical refinement is retained.
"""
from pathlib import Path
import argparse
import json
import shutil
import fcntl
import numpy as np
from scipy.optimize import minimize
from experiments.elastic.putting import protocol,save,digest,simulate,ROOT
from experiments.elastic.putting_franka import compile_motion,load_motion
from experiments.elastic.rod_insertion_franka import snapshot

def read(path):return json.loads(Path(path).read_text())

def prepare(root,source=None):
    root.mkdir(parents=True,exist_ok=False)
    if source is None:
        p=protocol();p.update(development='Arcing swing replaces the archived straight stroke. Same material estimates, club geometry, target, ball and floor. A fixed 28 degree backswing and -22 degree follow-through; only forward duration and aim are optimized. All pilot trials are retained.')
    else:
        source=Path(source);p=read(source/'protocol.json')
        (root/'franka').mkdir()
        shutil.copytree(source/'franka/panda_model_snapshot',root/'franka/panda_model_snapshot')
    save(root/'protocol.json',p)
    scene_path=Path(source)/'render_scene.json' if source is not None else None
    if scene_path is not None and scene_path.exists():shutil.copy2(scene_path,root/'render_scene.json')
    else:save(root/'render_scene.json',dict(table_bounds=[-.03,3.,-.18,3.,-.005,.02],green='#386b50',key_light=[-.4,-.6,1.5],key_intensity=.65,fill_intensity=.35,ambient_occlusion=True,scope='Widened display green; original floor height and contact region unchanged.'))
    snapshot(root)
    (root/'sources').mkdir()
    files=list((ROOT/'experiments/elastic').glob('putting*.py'))+[ROOT/'tests/test_putting.py']
    for file in files:shutil.copy2(file,root/'sources'/file.name)
    save(root/'source_hashes.json',{str(f.relative_to(ROOT)):digest(f) for f in files})
    for k in 'AB':
        ident=ROOT/f'out/strip_texture_20260912/fit_{k}/identification.json' if source is None else source/f'identification_{k}.json'
        assert digest(ident)==p['identification_sha256'][k]
        shutil.copy2(ident,root/f'identification_{k}.json')

def plan(root,k,device,stage='plan',grid=None,maxfev=None,initial_override=None):
    p=read(root/'protocol.json');dest=root/f'{stage}_{k}';dest.mkdir(exist_ok=False)
    if grid:p['grid']=grid;p['dt']=.0000125*160/grid
    initial=initial_override if initial_override is not None else (p['initial'] if stage=='plan' else read(root/f'plan_{k}/plan.json')['controls'])
    broad=initial_override is not None or stage=='plan'
    simplex=np.array([initial,np.array(initial)+[.015 if broad else .005,0.],np.array(initial)+[0.,1.5 if broad else .5]])
    history=[];budget=p['maxfev'] if maxfev is None else maxfev
    def objective(u):
        index=len(history);trial=dest/f'trial_{index:03d}'
        r=simulate(p,p['E_identified'][k],u,device,trial,record_frames=False)
        score=r['error_mm'] if r['stopped'] else 1e6+r['error_mm']
        history.append(dict(index=index,controls=list(map(float,u)),score_mm=score,result=r))
        save(dest/'history.json',history);print(k,index,list(u),score,flush=True)
        return score
    def callback(_):
        if min(h['score_mm'] for h in history)<=p['early_stop_error_mm']:raise StopIteration
    opt=minimize(objective,initial,method='Nelder-Mead',bounds=p['bounds'],callback=callback,
        options=dict(maxfev=budget,initial_simplex=simplex,xatol=.0001,fatol=.3))
    best=min(history,key=lambda h:h['score_mm'])
    result=dict(material=k,E_pa=p['E_identified'][k],controls=best['controls'],predicted_error_mm=best['score_mm'],
        best_trial=best['index'],evaluations=len(history),optimizer_message=str(opt.message),
        protocol_sha256=digest(root/'protocol.json'),planning_grid=p['grid'],planning_dt=p['dt'],
        initial=list(initial),initial_simplex=simplex.tolist(),objective=p['objective'],optimizer='Bounded Nelder-Mead; best evaluated candidate, finite budget, no global-optimality claim.')
    save(dest/'plan.json',result);print(result,flush=True)

def execute(root,m,k,device,grid=None,stage='plan',prefix='execution',coupling=None,video=True):
    p=read(root/'protocol.json');pl=read(root/f'{stage}_{k}/plan.json');u=pl['controls']
    p['grid']=grid or pl['planning_grid'];p['dt']=.0000125*160/p['grid'];p['end']=5. if video else p['end'];p['video_dt']=.01
    if coupling:p['coupling_dt']=coupling
    label=f'{stage}_{k}';motion_path=root/'franka'/f'plan_{label}.npz'
    (root/'franka').mkdir(exist_ok=True)
    with (root/'franka/compile.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not motion_path.exists():compile_motion(root,label,dict(p,end=5.),u)
    motion=load_motion(motion_path,tick=p['coupling_dt'])
    dest=root/f'{prefix}_{m}_plan_{k}'
    r=simulate(p,p['E_true'][m],u,device,dest,motion=motion,record_frames=video)
    save(dest/'execution_provenance.json',dict(plan_sha256=digest(root/f'{stage}_{k}/plan.json'),plan_path=str((root/f'{stage}_{k}/plan.json').resolve()),
        motion_sha256=digest(motion_path),motion=str(motion_path.resolve()),plan_material=k,executed_material=m,
        interpolation='Linear interpolation of frozen joint waypoints; Panda FK evaluated at every MPM/ball coupling time.'))
    print(r,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','plan','execute','check'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',default='A');ap.add_argument('--plan',default='A')
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--grid',type=int);ap.add_argument('--maxfev',type=int)
    ap.add_argument('--stage',default='plan');ap.add_argument('--prefix',default='execution');ap.add_argument('--coupling',type=float)
    ap.add_argument('--initial',type=float,nargs=2)
    ap.add_argument('--source',type=Path)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.out,a.source)
    elif a.command=='plan':plan(a.out,a.material,a.device,a.stage,a.grid,a.maxfev,a.initial)
    else:execute(a.out,a.material,a.plan,a.device,a.grid,a.stage,a.prefix,a.coupling,a.command=='execute')

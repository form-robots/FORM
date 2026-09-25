"""Two-variable, open-loop insertion using frozen texture-derived stiffness.

Planning simulates free motion and penalizes intersection with the fixed wall.
Execution adds physical wall contact; no tip/force feedback or replanning occurs.
All initial states are undeformed specimens, already held at the planned pose.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import minimize
from experiments.elastic.block_drop_study import ROOT, save, sha


def read(path): return json.loads(Path(path).read_text())


def initialize(root, source):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists(): raise FileExistsError(root/'protocol.json')
    p=dict(domain=.32, grid=256, fine_grid=320, dt=.00002, fine_dt=.000015,
           tick=.01, size=[.060,.012,.004], grip_length=.015,
           grip_x=.120, center_y=.160, nu=.45, density=1000.,
           align_start=.3, align_end=3.3, settle_end=3.8, advance_end=6.8, end=7.3, advance=.054,
           wall_x=.198, wall_thickness=.006, opening_z=.150,
           opening_height=.040, opening_width=.020, wall_half_width=.040,
           wall_half_height=.070, target_depth=.008, required_depth=.005,
           clearance_margin=.002, contact_force_threshold=1.e-5,
           sdf_cell=.001, contact_band=.00025,
           target_z_offset=-.008,
           bounds=[[.140,.220],[-30.,80.]], initial_guess=[.170,20.],
           maxfev=48, true_E_pa={'A':80000.,'B':240000.}, identified_E_pa={},
           identification_source=str(source.resolve()),
           controller='Start hanging, rotate smoothly to planned tilt over 3 s, wait 0.5 s, translate forward 54 mm over 3 s, hold 0.5 s. Grip height constant. No feedback.',
           objective='Squared terminal tip-centroid distance plus squared maximum geometric wall intersection (free-space planning).',
           selection='Same frozen task, bounds, start, and budget for both identified models. No swapped-execution score enters planning.',
           assumptions='3D fixed-corotated elasticity, known nu/rho/geometry; ideal bonded 15 mm grip, frictionless rigid wall; no added damping. Fresh undeformed vertically hanging specimen at planned grip height. No arm IK or hardware simulation.',
           task_development='Edgewise 100 mm strip twisted. Flat 100 mm strip, including deeper grasp, hung too steeply for a compact vertical aperture. Use a 60 mm specimen with unchanged cross-section/materials/15 mm grip and a slow rotation from hanging. Dimensions and target below aperture center accommodate the sagging upstream segment; frozen before the final searches and all cross-executions.')
    for k in ['A','B']:
        src=source/f'fit_{k}/identification.json'
        dst=root/'inputs'/f'identification_{k}.json'; dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src,dst); p['identified_E_pa'][k]=read(dst)['E_pa']
    save(root/'protocol.json',p)
    save(root/'input_hashes.json',{f.name:sha(f) for f in (root/'inputs').glob('*')})


def rotation(deg):
    a=np.radians(deg);c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,-s],[0,1,0],[s,0,c]])


def translation(t,p):
    u=np.clip((t-p['settle_end'])/(p['advance_end']-p['settle_end']),0.,1.)
    return p['advance']*u*u*(3-2*u)


def tilt(t, final, p):
    if 'align_end' not in p: return final
    u=np.clip((t-p['align_start'])/(p['align_end']-p['align_start']),0.,1.)
    return -90.+(final+90.)*u*u*(3.-2.*u)


def particles(p, grid, controls):
    size=np.array(p['size']); n=np.rint(size/(p['domain']/grid/2)).astype(int)
    axes=[(np.arange(k)+.5)*s/k for k,s in zip(n,size)]
    axes[0]-=p['grip_length']/2;axes[1]-=size[1]/2;axes[2]-=size[2]/2
    ref=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    xyz=ref@rotation(controls[1]).T+np.array([p['grip_x'],p['center_y'],controls[0]])
    return xyz.astype(np.float32),np.full(len(xyz),size.prod()/len(xyz),np.float32),ref,axes


@lru_cache(maxsize=12)
def box_sdf(half_tuple, cell):
    from warpmpm.geometry import SDFData
    half=np.array(half_tuple);extent=np.ceil((half.max()+.003)/cell)*cell
    axis=np.linspace(-extent,extent,round(2*extent/cell)+1)
    q=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    d=np.abs(q)-half
    vals=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)
    grads=np.stack(np.gradient(vals,cell,edge_order=2),axis=-1)
    return SDFData(vals.astype(np.float32),grads.astype(np.float32),np.full(3,-extent),cell,float(vals.max()))


def wall_boxes(p):
    x=p['wall_x'];y=p['center_y'];z=p['opening_z']
    hx=p['wall_thickness']/2;wy=p['wall_half_width'];hz=p['wall_half_height']
    gy=p['opening_width']/2;gz=p['opening_height']/2
    return [([x,y,z+sgn*(hz+gz)/2],[hx,wy,(hz-gz)/2]) for sgn in [-1,1]]+[
        ([x,y+sgn*(wy+gy)/2,z],[hx,(wy-gy)/2,gz]) for sgn in [-1,1]]


def wall_violation(x,p,pad):
    """Aperture clearance violation within the inflated wall slab, metres.

    In-plane distance supplies a useful penalty even for a point fully inside
    the thin wall, where ordinary signed penetration saturates at half thickness.
    """
    inside=np.abs(x[:,0]-p['wall_x'])<=p['wall_thickness']/2+pad
    if not inside.any():return 0.
    q=x[inside]
    return float(max(0.,(np.abs(q[:,1]-p['center_y'])-p['opening_width']/2+pad).max(),
                     (np.abs(q[:,2]-p['opening_z'])-p['opening_height']/2+pad).max()))


def simulate(p,E,controls,grid,device,physical_wall=False,duration=None,output=None):
    from warpmpm.core.solver import Solver,GridConfig
    from warpmpm.materials import elastic
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); x,vol,ref,axes=particles(p,grid,[controls[0],tilt(0,controls[1],p)])
    sim=Solver(GridConfig(grid,p['domain']),device=device,inversion_policy='raise').load_particles(x,vol)
    sim.set_material(elastic(E=float(E),nu=p['nu'],density=p['density']),rpic_damping=0.,grid_v_damping_scale=1.)
    c0=np.array([p['grip_x'],p['center_y'],controls[0]])
    angle=np.radians(tilt(0,controls[1],p)); quat=[0,-np.sin(angle/2),0,np.cos(angle/2)]
    grip=sim.add_sdf_collider(box_sdf((p['grip_length']/2,.008,.010),p['sdf_cell']),
            center=c0,quat=quat,band=p['contact_band'],surface='sticky',friction=0.)
    walls=[]
    if physical_wall:
        for center,half in wall_boxes(p):
            walls.append(sim.add_sdf_collider(box_sdf(tuple(half),p['sdf_cell']),center=center,
                band=p['contact_band'],surface='separable',friction=0.))
    end=p['end'] if duration is None else duration
    times=np.arange(round(end/p['tick'])+1)*p['tick']
    dt=p['fine_dt'] if grid==p['fine_grid'] else p['dt']; substeps=int(np.ceil(p['tick']/dt));dt=p['tick']/substeps
    tip=ref[:,0]>=ref[:,0].max()-.002
    # One layer nearest each undeformed boundary, retained for geometric audits.
    bound=np.zeros(len(ref),bool)
    for j in range(3): bound|=(ref[:,j]==ref[:,j].min())|(ref[:,j]==ref[:,j].max())
    if output:
        X=np.lib.format.open_memmap(output/'x.npy',mode='w+',dtype='float32',shape=(len(times),len(x),3))
        np.save(output/'reference.npy',ref);np.save(output/'vol0.npy',vol);np.save(output/'time.npy',times)
        np.save(output/'tip_mask.npy',tip)
        save(output/'config.json',dict(E_pa=E,controls=list(controls),grid=grid,dt=dt,physical_wall=physical_wall,protocol=p))
    data=[];max_violation=0.;min_J=1.;peak_force=0.;contact_impulse=0.;max_speed=0.
    pad=p['domain']/grid/4+p['clearance_margin']
    for i,t in enumerate(times):
        force=np.zeros(3); force_norm=0.
        if i:
            c=c0+np.array([translation(times[i-1],p),0,0])
            v=np.array([(translation(t,p)-translation(times[i-1],p))/p['tick'],0,0])
            a0=np.radians(tilt(times[i-1],controls[1],p));a1=np.radians(tilt(t,controls[1],p))
            sim.set_sdf_pose(grip,center=c,velocity=v,quat=[0,-np.sin(a0/2),0,np.cos(a0/2)],
                             omega=[0,-(a1-a0)/p['tick'],0])
            for w in walls:sim.reset_sdf_force(w)
            sim.step(dt,substeps=substeps)
            for w in walls:
                f=sim.sdf_wrench(w,p['tick'])['force'];force+=f;force_norm+=float(np.linalg.norm(f))
        xf=sim.x();vf=sim.v()
        if not np.isfinite(xf).all() or not np.isfinite(vf).all():raise RuntimeError('Nonfinite state')
        if i%10==0 or i==len(times)-1:
            j=np.linalg.det(sim.F());min_J=min(min_J,float(j.min()))
            if j.min()<=0:raise RuntimeError('Inverted state')
        max_speed=max(max_speed,float(np.linalg.norm(vf,axis=1).max()))
        violation=wall_violation(xf[bound],p,pad);max_violation=max(max_violation,violation)
        peak_force=max(peak_force,force_norm);contact_impulse+=force_norm*p['tick']
        tip_x=xf[tip];centroid=tip_x.mean(0)
        data.append([t,*centroid,float(np.linalg.norm(vf[tip],axis=1).mean()),force_norm,violation,
                     float(tip_x[:,0].min()-p['wall_x']-p['wall_thickness']/2),*force])
        if output:X[i]=xf
    target=np.array([p['wall_x']+p['wall_thickness']/2+p['target_depth'],p['center_y'],p['opening_z']+p.get('target_z_offset',0.)])
    # Score only the terminal hold, including persistent motion rather than a chosen lucky frame.
    d=np.asarray(data); use=d[:,0]>=min(p['advance_end'],end)
    target_error=np.linalg.norm(d[use,1:4]-target,axis=1)
    cost=float(np.mean((target_error*1000)**2)+25*(max_violation*1000)**2)
    depth=float(d[use,7].min())
    result=dict(E_pa=float(E),controls=list(map(float,controls)),grid=grid,wall_s=time.monotonic()-start,
        cost=cost,tip_error_mm=float(np.sqrt(np.mean(target_error**2))*1000),
        final_tip_mm=(d[-1,1:4]*1000).tolist(),min_tip_depth_hold_mm=depth*1000,
        max_geometric_violation_mm=max_violation*1000,peak_wall_force_N=peak_force,wall_impulse_Ns=contact_impulse,
        hold_tip_mean_speed_mm_s=float(d[use,4].mean()*1000),min_J=min_J,max_particle_speed_m_s=max_speed,
        inverted_count=sim.inverted_count(),all_frames_completed=True,
        success=bool(physical_wall and depth>=p['required_depth'] and peak_force<=p['contact_force_threshold']))
    if output:
        X.flush();np.savetxt(output/'signals.csv',d,delimiter=',',comments='',header='time,tip_x,tip_y,tip_z,tip_mean_speed,wall_force_norm,wall_violation,min_tip_depth,wall_fx,wall_fy,wall_fz')
        save(output/'result.json',result)
    return result


def plan(root,k,device):
    p=read(root/'protocol.json'); history=[];E=p['identified_E_pa'][k]
    out=root/f'plan_{k}';out.mkdir(exist_ok=False)
    # Work in mm/degrees to give both coordinates meaningful simplex scales.
    bounds=[(a*1000,b*1000) if i==0 else (a,b) for i,(a,b) in enumerate(p['bounds'])]
    guess=np.array([p['initial_guess'][0]*1000,p['initial_guess'][1]])
    def objective(q):
        controls=[q[0]/1000,q[1]]
        r=simulate(p,E,controls,p['grid'],device)
        history.append(r);save(out/'history.json',history)
        print(k,len(history),json.dumps(r),flush=True)
        return r['cost']
    opt=minimize(objective,guess,method='Nelder-Mead',bounds=bounds,
        options=dict(maxfev=p['maxfev'],xatol=.15,fatol=.02,
                     initial_simplex=[guess,guess+[5,0],guess+[0,8]]))
    best=min(history,key=lambda x:x['cost'])
    save(out/'plan.json',dict(material=k,E_pa=E,controls=best['controls'],predicted=best,
        evaluations=len(history),optimizer_message=str(opt.message),protocol_sha256=sha(root/'protocol.json')))
    simulate(p,E,best['controls'],p['grid'],device,physical_wall=True,output=out/'predicted_execution')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['init','pilot','plan','execute'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source',type=Path,default=Path('out/strip_texture_20260912'))
    ap.add_argument('--material',choices=['A','B'],default='A');ap.add_argument('--plan',choices=['A','B'],default='A')
    ap.add_argument('--device',default='cuda:1');ap.add_argument('--height',type=float,default=.150);ap.add_argument('--angle',type=float,default=0)
    ap.add_argument('--fine',action='store_true')
    a=ap.parse_args()
    if a.stage=='init':initialize(a.out,a.source);return
    p=read(a.out/'protocol.json')
    if a.stage=='pilot':
        r=simulate(p,p['identified_E_pa'][a.material],[a.height,a.angle],p['grid'],a.device,duration=p['settle_end'],output=a.out/'development'/f'pilot_{a.material}_{a.angle:g}')
        print(json.dumps(r),flush=True)
    elif a.stage=='plan':plan(a.out,a.material,a.device)
    else:
        pl=read(a.out/f'plan_{a.plan}/plan.json');assert pl['protocol_sha256']==sha(a.out/'protocol.json')
        r=simulate(p,p['true_E_pa'][a.material],pl['controls'],p['fine_grid'] if a.fine else p['grid'],a.device,
            physical_wall=True,output=a.out/f"{'fine' if a.fine else 'execution'}_{a.material}_plan_{a.plan}")
        print(json.dumps(r),flush=True)


if __name__=='__main__':main()

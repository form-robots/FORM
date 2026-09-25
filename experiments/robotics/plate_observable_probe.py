"""Observable plastic identification: common top-plate loading and unloading."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.elastic.strip_camera_observe import save

PROTOCOL=dict(E_pa={'A':80000.,'B':80000.},yield_pa={'A':1000.,'B':10000.},nu=.30,density=1000.,
 domain=.20,grid=128,fine_grid=160,dt=.00002,tick=.002,camera_dt=.01,
 size=[.030,.030,.025],center=[.100,.100,.0625],floor=.05,
 plate_half=[.032,.032,.003],band=.00025,sdf_cell=.0005,friction=0.,
 knots=[[0.,.026],[.20,.026],[1.00,.016],[1.40,.027],[1.50,.027],[2.30,.013],[2.70,.030],[2.90,.030]],
 fit_end=1.4,known=['geometry','density','nu','isotropic Hencky elasticity and associative perfect von Mises plasticity','initial stress-free state','frictionless plates'],
 scope='First loading/unloading identifies E and yield. Reload and final release are withheld predictions. No simulated kinematics enter fitting.')


def opening(t,p):
    for (a,x),(b,y) in zip(p['knots'][:-1],p['knots'][1:],strict=True):
        if t<=b:
            u=np.clip((t-a)/(b-a),0,1);return x+(y-x)*u*u*(3-2*u)
    return p['knots'][-1][1]


def plate_sdf(p):
    from warpmpm.geometry import SDFData
    half=np.asarray(p['plate_half']);cell=p['sdf_cell']
    # The collider API stores one resolution and requires a cubic voxel array.
    # The solid is a thin rectangular plate inside that cubic storage domain.
    extent=float(max(half))+.003
    axes=[np.arange(-extent,extent+cell/2,cell) for _ in range(3)]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1);d=abs(xyz)-half
    values=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)
    grad=np.stack(np.gradient(values,cell,edge_order=2),axis=-1)
    return SDFData(values.astype('float32'),grad.astype('float32'),np.array([a[0] for a in axes]),cell,float(values.max()))


def record(root,label,kind='truth',device='cuda:0',friction=None):
    from warpmpm import GridConfig,Solver
    from warpmpm.materials import vonmises
    p=json.loads((root/'protocol.json').read_text());grid=p['fine_grid'] if 'fine' in kind else p['grid']
    E,Y=p['E_pa'][label],p['yield_pa'][label]
    if 'prediction' in kind:
        fit=json.loads((root/f'divfree_{label}/identification.json').read_text());E,Y=fit['E_pa'],fit['yield_pa']
    friction=p['friction'] if friction is None else friction
    dest=root/f'{kind}_{label}';dest.mkdir(exist_ok=False)
    size=np.asarray(p['size']);n=np.rint(size/(p['domain']/grid/2)).astype(int)
    axes=[c+s*((np.arange(k)+.5)/k-.5) for c,s,k in zip(p['center'],size,n,strict=True)]
    x=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3).astype('float32');vol=np.full(len(x),size.prod()/len(x),'float32')
    sim=Solver(GridConfig(grid,p['domain']),device=device,inversion_policy='raise').load_particles(x,vol)
    sim.set_material(vonmises(E=E,nu=p['nu'],yield_stress=Y,density=p['density']),rpic_damping=0.,grid_v_damping_scale=1.)
    sim.add_plane((0,0,p['floor']),(0,0,1),surface='separable',friction=friction)
    def center(t):return [p['center'][0],p['center'][1],p['floor']+opening(t,p)+p['plate_half'][2]]
    plate=sim.add_sdf_collider(plate_sdf(p),center=center(0),band=p['band'],surface='separable',friction=friction)
    times=np.arange(round(p['knots'][-1][0]/p['tick'])+1)*p['tick'];stride=round(p['camera_dt']/p['tick'])
    shape=(len(times[::stride]),len(x),3)
    X=np.lib.format.open_memmap(dest/'x.npy',mode='w+',dtype='float32',shape=shape)
    F=np.lib.format.open_memmap(dest/'Fe.npy',mode='w+',dtype='float32',shape=(*shape[:2],3,3)) if kind=='truth' else None
    rows=[];minj=1.
    for i,t in enumerate(times):
        if i:
            a=np.asarray(center(t-p['tick']));b=np.asarray(center(t))
            sim.set_sdf_pose(plate,center=a,velocity=(b-a)/p['tick']);sim.reset_sdf_force(plate)
            sim.step(p['dt'],substeps=round(p['tick']/p['dt']))
            force=sim.sdf_wrench(plate,p['tick'])['force']
        else:force=np.zeros(3)
        if i%stride==0:
            xx=sim.x();ff=sim.F();assert np.isfinite(xx).all() and np.isfinite(ff).all()
            X[i//stride]=xx;minj=min(minj,float(np.linalg.det(ff).min()))
            if F is not None:F[i//stride]=ff
        rows.append([t,opening(t,p),*force])
        if i%250==0:print(kind,label,t,force[2],flush=True)
    X.flush()
    if F is not None:F.flush()
    np.save(dest/'time.npy',times[::stride]);np.save(dest/'vol0.npy',vol)
    np.savetxt(dest/'force.csv',rows,delimiter=',',header='time,opening,Fx,Fy,Fz',comments='')
    save(dest/'completion.json',dict(E_pa=E,yield_pa=Y,friction=friction,grid=grid,min_J=minj,particles=len(x),
         complete=True,inversions=sim.inverted_count()))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['init','truth','fine','prediction','fine_prediction','friction_check'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],default='A');ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args()
    if a.stage=='init':a.out.mkdir(parents=True,exist_ok=False);save(a.out/'protocol.json',PROTOCOL)
    else:record(a.out,a.material,a.stage,a.device,friction=.15 if a.stage=='friction_check' else None)

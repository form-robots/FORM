"""Hardware X planning from frozen weak estimates. No material fitting."""
import argparse,contextlib,hashlib,io,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyvista as pv
from scipy.optimize import minimize
from experiments.robotics import x_motion_shaping as core
from experiments.robotics import x_motion_search as search
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.shaping_real.outline_metric import measure

BASE=Path(__file__).resolve().parent
IDENT=ROOT/'out/press_weakform_restart_20260914'

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup(p):
 core.geometry.CONFIG.update(p['scene'])


def run(out,material,parameters,name,device='cuda:0',grid=None,law=None):
 p=read(out/'protocol.json');setup(p);grid=grid or p['grid']
 if device=='adaptive_yellow':device='cuda:1' if (out/'plans/plasticine/selected.json').exists() else 'cuda:0'
 for rel,digest in p['source_sha256'].items():
  assert sha(ROOT/rel)==digest, 'Numerical source changed: '+rel
 for m,digest in p['identification_hashes'].items():
  assert sha(IDENT/m/'fit.json')==digest, 'Frozen identification changed: '+m
 dt=p['dt'] if grid==64 else .00002
 acts=search.actions(p['spec'],parameters)
 cmd,phases=core.motion(acts)
 initial=dict(zip(['initial','vol0'],core.geometry.specimen(grid)))
 model=law or p['models'][material]
 dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
 assert not dest.with_suffix('.json').exists()
 save(dest.with_suffix('.config.json'),dict(material=material,parameters=list(parameters),law=model,grid=grid,dt=dt,device=device,scene=p['scene']))
 start=time.monotonic()
 data=core.simulate(model,cmd,phases,initial,device,grid,dt)
 mesh=surface(data['x_after_1s'],data['vol0'],h=.00125)
 target=pv.read(out/'target.vtp');error=mesh_distance_mm(mesh,target);outline=measure(mesh,target)
 topo=mesh.triangulate().clean();euler=topo.n_points-topo.extract_all_edges().n_cells+topo.n_cells
 force=np.linalg.norm(data['reaction_force'],axis=-1)
 # Closing-axis contact force per physical finger, not sum of both fingers.
 peak=float(force.max());phase=[]
 for i,ph in enumerate(phases):
  mask=cmd['phase_id']==i
  phase.append(dict(name=ph['name'],peak_resultant_per_finger_N=float(force[mask].max()),duration_s=ph['duration_s']))
 comps=mesh.connectivity(extraction_mode='all');ncomp=int(comps.cell_data['RegionId'].max())+1
 row=dict(material=material,parameters=list(parameters),actions=acts,law=model,grid=grid,dt=dt,surface_mm=error,
          **outline, peak_resultant_per_finger_N=peak,phase_forces=phase,components=ncomp,surface_genus_sum=float((2*ncomp-euler)/2),surface_open_edges=int(topo.n_open_edges),
          max_floor_penetration_mm=float(max(0,p['scene']['floor']-data['extent_samples'][:,3].min())*1000),
          final_extent_mm=(np.ptp(data['x_after_1s'],axis=0)*1000).tolist(),
          released_speed_rms_mm_s=float(np.sqrt(np.mean(np.sum(data['v_after_1s']**2,axis=1)))*1000),
          inversion_count=int(data['inverted_count']),duration_s=float(cmd['time'][-1]),wall_s=time.monotonic()-start,
          path=str(dest.with_suffix('.npz')),force_scope='Predicted peak resultant on one finger; not independently validated hardware force')
 for rel,digest in p['source_sha256'].items():
  assert sha(ROOT/rel)==digest, 'Numerical source changed during run: '+rel
 np.savez_compressed(dest.with_suffix('.npz'),**data)
 mesh.save(dest.with_suffix('.vtp'))
 save(dest.with_suffix('.phases.json'),phases);save(dest.with_suffix('.json'),row)
 print(json.dumps({k:row[k] for k in ['material','parameters','outline_mm','surface_mm','peak_resultant_per_finger_N','wall_s','grid']}),flush=True)
 return row

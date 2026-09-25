"""Independent checks of a saved prescribed-motion shaping rollout."""
from pathlib import Path
import json,numpy as np,pyvista as pv
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.shaping_real.outline_metric import measure

def audit(run,protocol,target):
 run=Path(run);p=json.loads(Path(protocol).read_text());r=json.loads(run.with_suffix('.json').read_text());d=np.load(run.with_suffix('.npz'));phases=json.loads(run.with_suffix('.phases.json').read_text());c=p['scene']
 assert all(np.isfinite(d[k]).all() for k in d.files)
 assert int(d['inverted_count'])==0
 assert np.isclose(np.sum(d['vol0'],dtype=float),np.prod(c['size']),rtol=1e-6)
 reconstructed=surface(d['x_after_1s'],d['vol0'],h=.00125)
 error=mesh_distance_mm(reconstructed,pv.read(target));assert np.isclose(error,r['surface_mm'],rtol=1e-10)
 outline=measure(reconstructed,pv.read(target));assert np.isclose(outline['outline_mm'],r['outline_mm'],rtol=1e-10)
 topo=reconstructed.triangulate().clean();euler=topo.n_points-topo.extract_all_edges().n_cells+topo.n_cells
 components=int(topo.connectivity(extraction_mode='all').cell_data['RegionId'].max())+1
 genus=(2*components-euler)/2
 pose=d['tool_centers'];gaps=(np.linalg.norm(pose[:,1]-pose[:,0],axis=-1)-2*c['radius'])*1000
 previous=np.concatenate([d['start_pose'][None],pose[:-1]],axis=0)
 velocity=(pose-previous)/c['tick'];assert np.allclose(velocity,d['tool_velocity'],rtol=0,atol=1e-12)
 off=[i for i,ph in enumerate(phases) if ph['name'].endswith(':reposition') or ph['name']=='final:wait']
 assert np.max(np.abs(d['reaction_force'][np.isin(d['phase_id'],off)]))==0
 checks=[]
 for i,a in enumerate(r['actions']):
  ix=np.flatnonzero(d['pinch_id']==i);last=ix[-1]
  assert np.isclose(gaps[last],a['gap_mm'],atol=1e-7)
  assert np.max(np.linalg.norm(velocity[ix],axis=-1))<=.020000001
  center=pose[last].mean(0)
  assert np.allclose(center[:2],c['domain']/2+np.array(a['offset_xy_mm'])/1000,atol=1e-12)
  normal=(pose[last,1]-pose[last,0])/np.linalg.norm(pose[last,1]-pose[last,0]);angle=np.deg2rad(a['angle_deg'])
  assert np.allclose(normal,[np.cos(angle),np.sin(angle),0],atol=1e-12)
  assert np.isclose(center[2]-c['finger_height']/2-c['floor'],c['finger_bottom'],atol=1e-12)
  phase_idx=int(d['phase_id'][last]);assert phases[phase_idx]['name']==f'{i}:close'
  checks.append({'pinch':i+1,'gap_mm':float(gaps[last]),'max_per_finger_speed_mm_s':float(np.linalg.norm(velocity[ix],axis=-1).max()*1000)})
 return {'run':str(run),'finite':True,'no_inversions':True,'volume_input_exact':True,'saved_motion_identity':True,'raised_reposition_and_final_wait_contact_zero':True,'surface_metric_reproduced':True,'outline_metric_reproduced':True,**outline,'surface_components':components,'surface_genus_sum':genus,'surface_open_edges':topo.n_open_edges,'surface_mm':error,'closing_phases':checks,'floor_penetration_mm':r['max_floor_penetration_mm'],'force_scope':'Predicted resultant per finger; not a commanded force target'}

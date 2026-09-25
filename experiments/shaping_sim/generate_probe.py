"""Generate the paper's monotonic pressing probe and stereo observations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,time,shutil
from pathlib import Path
from experiments.robotics.plate_observable_probe import record
from experiments.robotics.plate_observable_camera import render
from experiments.robotics.plate_observable_identify import track
from experiments.robotics.plate_observable_field import select
BASE=Path('out/plate_observable_physical_preload_20260912');OUT=Path('out/reproduced/press_archive');RAM=Path('out/reproduced/press_observations')
def init(tag,grid):
 root=(RAM/tag).resolve()
 if root.exists():return root
 root.mkdir(parents=True)
 p=json.loads((BASE/'protocol.json').read_text())
 p.update(grid=grid,fine_grid=grid,knots=[[0,.026],[.2,.026],[1.,.024],[1.8,.020],[2.,.020]],fit_end=2.,
   reconstruction_cv_times=[.4,.5,.6,.8,1.,1.2,1.4,1.6,1.8,2.],scope='One monotonic two-stage press, no unloading. Fresh specimen. Identification only; no planning.',validation_start=None)
 cfg=json.loads((BASE/'identification_protocol.json').read_text());cfg.pop('balanced_identification_intervals');cfg.update(height_span_counts=[6],cross_section_degrees=[3],smoothing_values=[0.])
 (root/'protocol.json').write_text(json.dumps(p,indent=2));(root/'identification_protocol.json').write_text(json.dumps(cfg,indent=2))
 permanent=OUT/tag;permanent.mkdir(parents=True)
 shutil.copy2(root/'protocol.json',permanent/'protocol.json');shutil.copy2(root/'identification_protocol.json',permanent/'identification_protocol.json')
 return root
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--grid',type=int,required=True);ap.add_argument('--material',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--stage',default='record');a=ap.parse_args()
 root=init(a.tag,a.grid);k=a.material
 if a.stage=='record':record(root,k,'candidate',a.device)
 elif a.stage=='observe':
  render(root,k,kind='candidate');(root/f'inputs_{k}').symlink_to(root/f'candidate_inputs_{k}')
  cfg=json.loads((root/'identification_protocol.json').read_text());track(root/f'inputs_{k}',root/f'fit_{k}',**cfg['tracking'])
  permanent=OUT/a.tag
  shutil.copytree(root/f'fit_{k}',permanent/f'fit_{k}')
  dst=permanent/f'inputs_{k}';dst.mkdir()
  for name in ['known.json','calibration.json','time.npy','force.csv','preview.png']:shutil.copy2(root/f'inputs_{k}'/name,dst/name)
  shutil.copy2(root/f'candidate_{k}/completion.json',permanent/f'completion_{k}.json')
 elif a.stage=='select':
  select(root,quadrature_order=8,validate_volume_folds=True)
  for name in ['field_selection.json','field_candidates.json']:shutil.copy2(root/name,OUT/a.tag/name)

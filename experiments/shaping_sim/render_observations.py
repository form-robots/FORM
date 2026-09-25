"""Regenerate synthetic camera observations from the durable surface archive.

No MPM simulation is required. The original float64 rendering surfaces were
archived as float32, so tiny rasterization differences are possible and recorded.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,json
from pathlib import Path
import numpy as np
import cv2
import pyvista as pv
from experiments.elastic.strip_texture_observe import meshes
from experiments.robotics.plate_observable_camera import texture
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--material',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--frames',default='');a=ap.parse_args()
p=json.loads((a.root/'protocol.json').read_text());k=a.material;camera=p['camera'];archive=np.load(a.root/f'observation_surfaces_{k}.npz');s=archive['surface'];offsets=archive['offsets'];times=archive['time'];force=np.genfromtxt(a.root/f'generated_force_{k}.csv',delimiter=',',names=True)
selected=set(map(int,a.frames.split(','))) if a.frames else set(range(len(times)));a.out.mkdir(parents=True,exist_ok=False)
(front,ref),others=meshes(p);allmeshes=[front]+[m for m,q in others];pl=pv.Plotter(off_screen=True,window_size=(1280,1280));pl.set_background('#f5f6f5');pl.enable_anti_aliasing('ssaa');pl.add_mesh(front,texture=pv.numpy_to_texture(texture()),lighting=False)
for m,_ in others:pl.add_mesh(m,color='#9eaaa9',smooth_shading=True)
cx,cy,_=p['center'];floor=p['floor'];hx,hy,hz=p['plate_half'];pl.add_mesh(pv.Box(bounds=(cx-.038,cx+.038,cy-.038,cy+.038,floor-.006,floor)),color='#acb5bb')
rng=np.random.default_rng(camera['seed'])
for f,t in enumerate(times):
 if f in selected:
  for j,m in enumerate(allmeshes):m.points=s[f,offsets[j]:offsets[j+1]].astype('float64')
  h=np.interp(t,force['time'],force['opening']);pl.add_mesh(pv.Box(bounds=(cx-hx,cx+hx,cy-hy,cy+hy,floor+h,floor+h+2*hz)),name='plate',color='#acb5bb')
 for ci in range(2):
  noise=rng.normal(0,.5,(1280,1280))
  if f not in selected:continue
  pl.camera.position=camera['centers'][ci];pl.camera.focal_point=camera['target'];pl.camera.up=(0,0,1);pl.camera.view_angle=float(np.degrees(2*np.arctan(1280/(2*camera['focal_px']))));pl.reset_camera_clipping_range();pl.render()
  raw=cv2.cvtColor(pl.screenshot(return_img=True),cv2.COLOR_RGB2GRAY);im=np.clip(np.rint(raw.astype(float)+noise),0,255).astype('uint8');cv2.imwrite(str(a.out/f'camera_{ci}_frame_{f:03d}.png'),im)
pl.close()

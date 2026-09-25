"""Synthetic stereo observation export; simulator states stay outside fit inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from experiments.elastic.strip_camera_observe import calibration,save
from experiments.elastic.strip_texture_observe import meshes

CAMERA=dict(width=1280,height=1280,focal_px=4800.,fps=100,
 centers=[[.01,-.15,.082],[.19,-.15,.082]],target=[.1,.1,.067],noise_sd_gray=.5,seed=912)


def texture():
    noise=gaussian_filter(np.random.default_rng(923).normal(size=(1000,1200)),4.)
    gray=np.clip(175+48*np.tanh(noise/noise.std()),95,230).astype('uint8')
    return np.repeat(gray[:,:,None],3,axis=2)


def render(root,k,kind='truth'):
    import pyvista as pv
    p=json.loads((root/'protocol.json').read_text());src=root/f'{kind}_{k}';camera=p.get('camera',CAMERA)
    x=np.load(src/'x.npy',mmap_mode='r');times=np.load(src/'time.npy');ids=times<=p['fit_end']+1e-9
    media=root/'media';media.mkdir(exist_ok=True)
    video=np.lib.format.open_memmap(media/f'{kind}_{k}_frames.npy',mode='w+',dtype='uint8',shape=(len(times),720,1080))
    inputs=root/f'inputs_{k}' if kind=='truth' else root/f'{kind}_inputs_{k}'
    inputs.mkdir(exist_ok=False)
    keys=['size','center','floor','density','nu','fit_end','knots','friction']
    known={name:p[name] for name in keys};known.update(surface_y=p['center'][1]-p['size'][1]/2,
      camera=camera,constitutive_family='isotropic Hencky elasticity with associative perfect von Mises plasticity',
      initial_state='stress-free at first image',reconstruction='Observed surface motion; interior reconstruction is specified and validated separately',
      estimated=['E','yield_stress'],known_nu=p['nu'],force_tick=p['tick'])
    if 'reconstruction_cv_times' in p:known['reconstruction_cv_times']=p['reconstruction_cv_times']
    save(inputs/'known.json',known);save(inputs/'calibration.json',calibration(camera));np.save(inputs/'time.npy',times[ids])
    sensor=np.genfromtxt(src/'force.csv',delimiter=',',names=True)[1:]
    use=sensor['time']-.5*p['tick']<=p['fit_end']
    np.savetxt(inputs/'force.csv',np.column_stack([sensor['time'][use]-.5*p['tick'],sensor['Fz'][use],sensor['opening'][use]]),
        delimiter=',',header='time,Fz,opening',comments='')
    images=[np.lib.format.open_memmap(inputs/f'camera_{i}.npy',mode='w+',dtype='uint8',shape=(int(sum(ids)),1280,1280)) for i in range(2)]
    axes=[np.unique(x[0,:,i]) for i in range(3)];shape=tuple(len(a) for a in axes)+(3,)
    (face,ref),others=meshes(p);pl=pv.Plotter(off_screen=True,window_size=(1280,1280))
    pl.set_background('#f5f6f5');pl.enable_anti_aliasing('ssaa');pl.add_mesh(face,texture=pv.numpy_to_texture(texture()),lighting=False)
    for m,_ in others:pl.add_mesh(m,color='#9eaaa9',smooth_shading=True)
    cx,cy,_=p['center'];floor=p['floor'];hx,hy,hz=p['plate_half']
    pl.add_mesh(pv.Box(bounds=(cx-.038,cx+.038,cy-.038,cy+.038,floor-.006,floor)),color='#acb5bb')
    force=np.genfromtxt(src/'force.csv',delimiter=',',names=True)
    rng=np.random.default_rng(camera['seed'])
    for f,t in enumerate(times):
        deform=RegularGridInterpolator(axes,x[f].reshape(shape),bounds_error=False,fill_value=None)
        face.points=deform(ref)
        for m,q in others:m.points=deform(q)
        h=np.interp(t,force['time'],force['opening'])
        pl.add_mesh(pv.Box(bounds=(cx-hx,cx+hx,cy-hy,cy+hy,floor+h,floor+h+2*hz)),name='plate',color='#acb5bb')
        for ci in range(2 if ids[f] else 1):
            pl.camera.position=camera['centers'][ci];pl.camera.focal_point=camera['target'];pl.camera.up=(0,0,1)
            pl.camera.view_angle=float(np.degrees(2*np.arctan(1280/(2*camera['focal_px']))));pl.reset_camera_clipping_range();pl.render()
            raw=cv2.cvtColor(pl.screenshot(return_img=True),cv2.COLOR_RGB2GRAY)
            if ci==0:video[f]=raw[280:1000,100:1180]
            if ids[f]:images[ci][f]=np.clip(np.rint(raw.astype(float)+rng.normal(0,.5,raw.shape)),0,255).astype('uint8')
        if f%50==0:print(kind,k,'render',f,flush=True)
    pl.close();video.flush()
    for im in images:im.flush()
    cv2.imwrite(str(inputs/'preview.png'),images[0][100])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--material',choices=['A','B'],required=True);ap.add_argument('--kind',default='truth')
    a=ap.parse_args();render(a.out,a.material,a.kind)

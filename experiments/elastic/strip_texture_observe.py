"""Render an opaque, material-textured strip into calibrated stereo images.

Only this observation generator reads simulated positions. No particle state,
texture UV identity, true modulus, or marker location enters the input bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from experiments.elastic.strip_camera_observe import CAMERA, calibration, save


def texture():
    rng=np.random.default_rng(912)
    noise=gaussian_filter(rng.normal(size=(4000,480)),3.)
    noise/=noise.std()
    # Irregular, continuous matte texture, approximately 0.2 mm features.
    gray=np.clip(174+49*np.tanh(noise*1.2),90,230).astype(np.uint8)
    return np.repeat(gray[:,:,None],3,axis=2)


def meshes(p):
    import pyvista as pv
    lo=np.asarray(p['center'])-np.asarray(p['size'])/2
    hi=lo+np.asarray(p['size'])
    def face(normal,value,a,b):
        aa=np.linspace(lo[a],hi[a],max(9,round(p['size'][a]/.0005)+1))
        bb=np.linspace(lo[b],hi[b],max(9,round(p['size'][b]/.0005)+1))
        aa,bb=np.meshgrid(aa,bb,indexing='ij')
        q=np.empty((aa.size,3)); q[:,normal]=value; q[:,a]=aa.ravel(); q[:,b]=bb.ravel()
        faces=[]; na,nb=aa.shape
        for i in range(na-1):
            for j in range(nb-1):
                x=i*nb+j; y=x+nb
                faces.extend([3,x,y,y+1,3,x,y+1,x+1])
        return pv.PolyData(q,np.array(faces)),q
    front,q=face(1,lo[1],0,2)
    front.active_texture_coordinates=np.column_stack([(q[:,0]-lo[0])/p['size'][0],(q[:,2]-lo[2])/p['size'][2]])
    others=[face(n,v,a,b) for n,v,a,b in [(0,lo[0],1,2),(0,hi[0],1,2),
              (1,hi[1],0,2),(2,lo[2],0,1),(2,hi[2],0,1)]]
    return (front,q),others


def render(root,label):
    import pyvista as pv
    from experiments.elastic.strip_bending_study import pusher_position
    p=json.loads((root/'protocol.json').read_text())
    case=root/f'inputs_{label}'; case.mkdir(exist_ok=False)
    keys=['known_nu','density','size','center','domain','grid','grip_lower_z','fit_interval',
          'test_transitions','settle_end','bend_end','hold_end','withdraw_end','end','pusher_radius','pusher_length']
    known={k:p[k] for k in keys}
    known.update(camera=CAMERA,law='fixed-corotated elasticity',
        surface_reference_y=p['center'][1]-p['size'][1]/2,
        assumptions='Known initially flat face; calibrated stereo; planar in-plane motion uniform across thickness; plane-stress closure with known nu.',
        image_end_s=p['bend_end'],window_duration=.25,window_stride_s=.10,
        spline_degree_x=2,spline_spacing_z=.008,
        texture='Material-attached random matte pattern, no supplied feature identities.',
        quadrature_counts=[12,4,100])
    save(case/'known.json',known); save(case/'calibration.json',calibration(CAMERA))
    X=np.load(root/f'truth_{label}/x.npy',mmap_mode='r')
    axes=[np.unique(X[0,:,i]) for i in range(3)]; shape=tuple(len(x) for x in axes)+(3,)
    ids=np.arange(0,round(p['bend_end']/p['tick'])+1,round(1/CAMERA['fps']/p['tick']))
    times=ids*p['tick']; np.save(case/'time.npy',times)
    sensor=np.genfromtxt(root/f'truth_{label}/signals.csv',delimiter=',',names=True)[1:]
    use=sensor['time']-.5*p['tick']<=p['bend_end']
    np.savetxt(case/'force.csv',np.column_stack([sensor['time'][use]-.5*p['tick'],sensor['reaction_x'][use]]),
        delimiter=',',header='time,material_on_pusher_Fx',comments='')
    images=[np.lib.format.open_memmap(case/f'camera_{i}.npy',mode='w+',dtype='uint8',
        shape=(len(times),CAMERA['height'],CAMERA['width'])) for i in range(2)]
    (front,ref),others=meshes(p)
    pl=pv.Plotter(off_screen=True,window_size=(CAMERA['width'],CAMERA['height']))
    pl.set_background('#f5f6f5'); pl.enable_anti_aliasing('ssaa')
    pl.add_mesh(front,texture=pv.numpy_to_texture(texture()),lighting=False)
    for mesh,_ in others: pl.add_mesh(mesh,color='#8a999e',smooth_shading=True)
    yf=known['surface_reference_y']; yb=p['center'][1]+p['size'][1]/2
    for bounds in [(.101,.139,yf-.012,yf,.165,.193),(.101,.139,yb,yb+.012,.165,.193),
                   (.101,.139,yf-.012,yb+.012,.193,.204)]:
        pl.add_mesh(pv.Box(bounds=bounds),color='#aab4bc')
    rng=np.random.default_rng(CAMERA['seed'])
    for fi,i in enumerate(ids):
        deform=RegularGridInterpolator(axes,X[i].reshape(shape),bounds_error=False,fill_value=None)
        front.points=deform(ref)
        for mesh,q in others: mesh.points=deform(q)
        pl.add_mesh(pv.Cylinder(center=pusher_position(times[fi],p),direction=(0,1,0),
            radius=p['pusher_radius'],height=p['pusher_length'],resolution=64),name='pusher',color='#aab4bc')
        for ci,eye in enumerate(CAMERA['centers']):
            pl.camera.position=eye; pl.camera.focal_point=CAMERA['target']; pl.camera.up=(0,0,1)
            pl.camera.view_angle=float(np.degrees(2*np.arctan(CAMERA['height']/(2*CAMERA['focal_px']))))
            pl.camera.parallel_projection=False; pl.reset_camera_clipping_range(); pl.render()
            im=cv2.cvtColor(pl.screenshot(return_img=True),cv2.COLOR_RGB2GRAY).astype(float)
            images[ci][fi]=np.clip(np.rint(im+rng.normal(0,CAMERA['noise_sd_gray'],im.shape)),0,255).astype('uint8')
        if fi%30==0: print(label,'render',fi,flush=True)
    pl.close()
    for im in images: im.flush()
    cv2.imwrite(str(case/'preview.png'),images[0][-1])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--material',choices=['A','B'],required=True)
    a=ap.parse_args(); render(a.out,a.material)

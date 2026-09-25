"""Generate calibrated stereo images and force recordings from the strip pilot.

This is the only observation-stage module allowed to read simulator positions.
It never exports hidden states into the estimator's input bundle. The private
evaluation directory records image-generation truth for later tracking audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator


CAMERA=dict(width=1280,height=1280,focal_px=3800.,fps=100,
            centers=[[.120-.140,-.300,.140],[.120+.140,-.300,.140]],
            target=[.120,.120,.140],noise_sd_gray=0.5,seed=912)
MARKER_X=np.linspace(.1146,.1254,7)
MARKER_Z=np.arange(.082,.1621,.004)


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def calibration(c):
    result=[]
    for eye in c['centers']:
        eye=np.array(eye); forward=np.array(c['target'])-eye
        forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]); right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        R=np.stack([right,-up,forward]); t=-R@eye
        K=np.array([[c['focal_px'],0,(c['width']-1)/2],
                    [0,c['focal_px'],(c['height']-1)/2],[0,0,1.]])
        result.append(dict(K=K.tolist(),R=R.tolist(),t=t.tolist(),
                           P=(K@np.column_stack([R,t])).tolist()))
    return result


def project(x,P):
    q=np.column_stack([x,np.ones(len(x))])@np.asarray(P).T
    return q[:,:2]/q[:,2,None]


def texture():
    # Fixed, material-attached black dots on a light printed face. No marker
    # positions are supplied to tracking after the initial calibrated pattern.
    w,h=600,5000
    tex=np.full((h,w,3),[225,229,227],np.uint8)
    for x in MARKER_X:
        for z in MARKER_Z:
            u=round((x-.114)/.012*(w-1)); v=round((.180-z)/.100*(h-1))
            cv2.circle(tex,(u,v),round(.00035/.012*w),(18,21,20),-1,cv2.LINE_AA)
    return tex


def front_mesh():
    import pyvista as pv
    ax=np.linspace(.114,.126,25); az=np.linspace(.080,.180,201)
    q=np.stack(np.meshgrid(ax,[.110],az,indexing='ij'),axis=-1).reshape(-1,3)
    faces=[]
    for i in range(len(ax)-1):
        for j in range(len(az)-1):
            a=i*len(az)+j; b=(i+1)*len(az)+j
            faces.extend([3,a,b,b+1,3,a,b+1,a+1])
    mesh=pv.PolyData(q,np.array(faces))
    mesh.active_texture_coordinates=np.column_stack([(q[:,0]-.114)/.012,(q[:,2]-.080)/.100])
    return mesh,q


def other_faces():
    import pyvista as pv
    all_q=[]; all_faces=[]
    specs=[(0,.114,1,2),(0,.126,1,2),(1,.130,0,2),(2,.080,0,1),(2,.180,0,1)]
    bounds=[(.114,.126),(.110,.130),(.080,.180)]
    for normal,value,a,b in specs:
        aa=np.linspace(*bounds[a],17 if a!=2 else 101)
        bb=np.linspace(*bounds[b],17 if b!=2 else 101)
        aa,bb=np.meshgrid(aa,bb,indexing='ij')
        q=np.empty((aa.size,3)); q[:,normal]=value
        q[:,a]=aa.ravel(); q[:,b]=bb.ravel()
        off=sum(len(x) for x in all_q); na,nb=aa.shape
        for i in range(na-1):
            for j in range(nb-1):
                x=off+i*nb+j; y=x+nb
                all_faces.extend([3,x,y,y+1,3,x,y+1,x+1])
        all_q.append(q)
    q=np.concatenate(all_q)
    return pv.PolyData(q,np.array(all_faces)),q


def make_observations(source,out,label):
    import pyvista as pv
    from experiments.elastic.strip_bending_study import pusher_position
    source=Path(source); out=Path(out)
    case=out/f'inputs_{label}'; case.mkdir(parents=True,exist_ok=False)
    evaluation=out/f'evaluation_{label}'; evaluation.mkdir(exist_ok=False)
    p=json.loads((source/'protocol.json').read_text())
    allowed=['known_nu','density','size','center','domain','grid','grip_lower_z',
             'fit_interval','test_transitions','settle_end','bend_end','hold_end',
             'withdraw_end','end','pusher_radius','pusher_length']
    known={k:p[k] for k in allowed}
    known.update(camera=CAMERA,law='fixed-corotated elasticity',
                 surface_reference_y=.110,marker_x=MARKER_X.tolist(),marker_z=MARKER_Z.tolist(),
                 assumptions='Known initially flat marked face; calibrated stereo; symmetric planar bending.',
                 image_end_s=.90,window_duration=.25,window_stride_s=.10,
                 spline_degree_x=2,spline_spacing_z=.008)
    save(case/'known.json',known)
    cams=calibration(CAMERA); save(case/'calibration.json',cams)
    X=np.load(source/f'truth_{label}/x.npy',mmap_mode='r')
    axes=[np.unique(X[0,:,i]) for i in range(3)]
    shape=tuple(len(x) for x in axes)+(3,)
    frame_ids=np.arange(0,round(.9/p['tick'])+1,round(1/CAMERA['fps']/p['tick']))
    times=frame_ids*p['tick']; np.save(case/'time.npy',times)
    signal=np.genfromtxt(source/f'truth_{label}/signals.csv',delimiter=',',names=True)
    np.savetxt(case/'force.csv',np.column_stack([signal['time'][1:]-.5*p['tick'],
                signal['reaction_x'][1:]]),delimiter=',',header='time,material_on_pusher_Fx',comments='')
    # Do not export motion/force observations after loading to the identifier.
    sensor=np.genfromtxt(case/'force.csv',delimiter=',',names=True)
    sensor=sensor[sensor['time']<=.90]
    np.savetxt(case/'force.csv',np.column_stack([sensor[k] for k in sensor.dtype.names]),
               delimiter=',',header=','.join(sensor.dtype.names),comments='')
    material_points=np.stack(np.meshgrid(MARKER_X,[.110],MARKER_Z,indexing='ij'),axis=-1).reshape(-1,3)
    true_markers=[]
    images=[np.lib.format.open_memmap(case/f'camera_{i}.npy',mode='w+',dtype='uint8',
             shape=(len(times),CAMERA['height'],CAMERA['width'])) for i in range(2)]
    face,ref_face=front_mesh(); body,ref_body=other_faces()
    tex=pv.numpy_to_texture(texture())
    pl=pv.Plotter(off_screen=True,window_size=(CAMERA['width'],CAMERA['height']))
    pl.set_background('#f5f6f5'); pl.enable_anti_aliasing('ssaa')
    pl.add_mesh(body,color='#5f9ab4' if label=='A' else '#d48c63',smooth_shading=True)
    pl.add_mesh(face,texture=tex,lighting=False)
    for bounds in [(.101,.139,.099,.111,.165,.193),(.101,.139,.129,.141,.165,.193),
                   (.101,.139,.099,.141,.193,.204)]:
        pl.add_mesh(pv.Box(bounds=bounds),color='#aab4bc')
    rng=np.random.default_rng(CAMERA['seed'])
    for fi,i in enumerate(frame_ids):
        deformation=RegularGridInterpolator(axes,X[i].reshape(shape),bounds_error=False,fill_value=None)
        face.points=deformation(ref_face); body.points=deformation(ref_body)
        true_markers.append(deformation(material_points))
        pose=pusher_position(times[fi],p)
        pl.add_mesh(pv.Cylinder(center=pose,direction=(0,1,0),radius=p['pusher_radius'],
                  height=p['pusher_length'],resolution=64),name='pusher',color='#aab4bc')
        for ci,eye in enumerate(CAMERA['centers']):
            pl.camera.position=eye; pl.camera.focal_point=CAMERA['target']; pl.camera.up=(0,0,1)
            pl.camera.view_angle=float(np.degrees(2*np.arctan(CAMERA['height']/(2*CAMERA['focal_px']))))
            pl.camera.parallel_projection=False; pl.reset_camera_clipping_range()
            pl.render()
            im=pl.screenshot(return_img=True)
            gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(float)
            gray=np.clip(np.rint(gray+rng.normal(0,CAMERA['noise_sd_gray'],gray.shape)),0,255).astype(np.uint8)
            images[ci][fi]=gray
            if fi in [0,30,60,90]: cv2.imwrite(str(case/f'camera_{ci}_frame_{fi:03d}.png'),gray)
        if fi%20==0: print(label,'render',fi,'/',len(frame_ids),flush=True)
    pl.close()
    for im in images: im.flush()
    np.save(evaluation/'true_marker_positions.npy',np.asarray(true_markers))
    save(evaluation/'source.json',dict(source=str(source.resolve()),label=label,
                                      rendering_uses_positions_only=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--material',choices=['A','B'],required=True)
    args=ap.parse_args(); make_observations(args.source,args.out,args.material)

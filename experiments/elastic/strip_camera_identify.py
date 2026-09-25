"""Identify strip stiffness from the camera/force bundle only.

Inputs are grayscale stereo images, timestamps, calibration, a known marker
pattern/specimen geometry and measured pusher force. No simulator state or true
modulus is loaded. Surface marker motion is measured from image pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from experiments.elastic.strip_camera_reconstruction import quadrature,fit_balance


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def projected(x,P):
    q=np.column_stack([x,np.ones(len(x))])@np.asarray(P).T
    return q[:,:2]/q[:,2,None]


def dots(im):
    n,labels,stats,_=cv2.connectedComponentsWithStats((im<100).astype(np.uint8),8)
    candidates=[]
    for i in range(1,n):
        if not 8<=stats[i,cv2.CC_STAT_AREA]<=100:
            continue
        x,y,w,h=stats[i,:4]
        if min(w,h)<2 or max(w,h)>15:
            continue
        mask=labels[y:y+h,x:x+w]==i
        weight=mask*np.maximum(180-im[y:y+h,x:x+w].astype(float),0)
        yy,xx=np.indices(weight.shape)
        candidates.append([x+np.sum(xx*weight)/weight.sum(),y+np.sum(yy*weight)/weight.sum()])
    return np.asarray(candidates).reshape(-1,2)


def associate(queries,candidates,radius):
    out=np.full_like(queries,np.nan,dtype=float); good=np.zeros(len(queries),bool)
    if not len(candidates): return out,good
    distance=np.linalg.norm(queries[:,None,:]-candidates[None,:,:],axis=2)
    rows,cols=linear_sum_assignment(np.nan_to_num(distance,nan=1e6))
    use=distance[rows,cols]<radius
    out[rows[use]]=candidates[cols[use]]; good[rows[use]]=True
    return out,good


def track(bundle,dest):
    p=json.loads((bundle/'known.json').read_text())
    cams=json.loads((bundle/'calibration.json').read_text())
    nominal=np.stack(np.meshgrid(p['marker_x'],[p['surface_reference_y']],p['marker_z'],indexing='ij'),axis=-1).reshape(-1,3)
    n=len(nominal); pixels=[]; status=[]
    for ci,cam in enumerate(cams):
        seq=np.load(bundle/f'camera_{ci}.npy',mmap_mode='r')
        xy=np.full((len(seq),n,2),np.nan); valid=np.zeros((len(seq),n),bool)
        xy[0],valid[0]=associate(projected(nominal,cam['P']),dots(seq[0]),5.)
        for f in range(1,len(seq)):
            active=np.flatnonzero(valid[f-1])
            if not len(active): break
            prev=xy[f-1,active].astype(np.float32).reshape(-1,1,2)
            nxt,ok,_=cv2.calcOpticalFlowPyrLK(seq[f-1],seq[f],prev,None,
                winSize=(13,13),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
            back,okb,_=cv2.calcOpticalFlowPyrLK(seq[f],seq[f-1],nxt,None,
                winSize=(13,13),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
            cand,matched=associate(nxt[:,0],dots(seq[f]),2.)
            use=ok[:,0].astype(bool)&okb[:,0].astype(bool)&matched
            use&=np.linalg.norm(back[:,0]-prev[:,0],axis=1)<.5
            xy[f,active[use]]=cand[use]; valid[f,active[use]]=True
        pixels.append(xy); status.append(valid)
    pixels=np.array(pixels); valid=status[0]&status[1]
    world=np.full((*valid.shape,3),np.nan); residual=np.full(valid.shape,np.nan)
    for f in range(len(world)):
        ids=np.flatnonzero(valid[f])
        if not len(ids):
            raise RuntimeError(f'No valid stereo correspondences at frame {f}; check calibration and tracking')
        h=cv2.triangulatePoints(np.asarray(cams[0]['P']),np.asarray(cams[1]['P']),
                               pixels[0,f,ids].T,pixels[1,f,ids].T)
        xyz=(h[:3]/h[3]).T
        error=np.maximum(np.linalg.norm(projected(xyz,cams[0]['P'])-pixels[0,f,ids],axis=1),
                         np.linalg.norm(projected(xyz,cams[1]['P'])-pixels[1,f,ids],axis=1))
        keep=error<.35
        valid[f,ids[~keep]]=False
        world[f,ids[keep]]=xyz[keep]; residual[f,ids]=error
    dest.mkdir(parents=True,exist_ok=False)
    np.savez(dest/'tracks.npz',world=world,valid=valid,pixels=pixels,
             reprojection_error_px=residual,time=np.load(bundle/'time.npy'),reference=world[0])
    result=dict(markers=n,initial_stereo_tracks=int(valid[0].sum()),
                final_stereo_tracks=int(valid[-1].sum()),
                min_stereo_tracks=int(valid.sum(1).min()),
                rms_reprojection_px=float(np.sqrt(np.nanmean(residual[valid]**2))),
                tracker='Pyramidal Lucas-Kanade with forward/backward check and image-dot centroid refinement',
                no_ground_truth_tracks=True)
    save(dest/'tracking.json',result); print(result,flush=True)


def spline_basis(points,p):
    """Tensor displacement basis; resolution selected using marker prediction."""
    x0=p['center'][0]-p['size'][0]/2; x1=x0+p['size'][0]
    z0=p['center'][2]-p['size'][2]/2; z1=z0+p['size'][2]
    degree=p.get('spline_degree_x',2); spacing=p.get('spline_spacing_z',.01)
    tx=np.r_[[x0]*(degree+1),[x1]*(degree+1)]
    tz=np.r_[[z0]*4,np.arange(z0+spacing,z1-.001,spacing),[z1]*4]
    bx=BSpline(tx,np.eye(len(tx)-degree-1),degree)
    bz=BSpline(tz,np.eye(len(tz)-4),3)
    x,z=points[:,0],points[:,2]
    out=[]
    for ix,iz in [(0,0),(1,0),(0,1)]:
        xx=bx(x,nu=ix); zz=bz(z,nu=iz)
        out.append((xx[:,:,None]*zz[:,None,:]).reshape(len(points),-1))
    return out


def reconstruct_tracks(tracks,p):
    reference=tracks['reference']; world=tracks['world']; valid=tracks['valid']
    t=tracks['time']; dt=float(t[1]-t[0]); q,vol=quadrature(p)
    reference=reference.copy()
    # Missing initial marks remain excluded; their coordinates are never filled
    # from image-generation truth or subsequent material states.
    reference[~valid[0]]=p['center']
    design=spline_basis(reference,p)[0]
    gx=np.linspace(p['center'][0]-p['size'][0]/2,p['center'][0]+p['size'][0]/2,5)
    gz=np.linspace(p['grip_lower_z']+.006,p['center'][2]+p['size'][2]/2,3)
    grip=np.stack(np.meshgrid(gx,[p['surface_reference_y']],gz,indexing='ij'),axis=-1).reshape(-1,3)
    constraints=spline_basis(grip,p)[0]
    coeff=[]; fit_errors=[]
    for f in range(len(t)):
        use=valid[f]&valid[0]
        M=np.vstack([design[use],constraints*3])
        target=np.vstack([(world[f,use]-reference[use])[:,[0,2]],np.zeros((len(grip),2))])
        c,_,rank,_=np.linalg.lstsq(M,target,rcond=1e-9)
        if rank<M.shape[1]:
            raise RuntimeError(f'Insufficient marker geometry at frame {f}: rank {rank}/{M.shape[1]}')
        coeff.append(c)
        fit_errors.append(float(np.sqrt(np.mean((design[use]@c-target[:use.sum()])**2))))
    coeff=np.asarray(coeff)
    position_c=savgol_filter(coeff,7,3,axis=0)
    velocity_c=savgol_filter(coeff,7,3,deriv=1,delta=dt,axis=0)
    N,Nx,Nz=spline_basis(q,p)
    X=[]; V=[]; F=[]
    for c,vc in zip(position_c,velocity_c,strict=True):
        x=q.copy(); v=np.zeros_like(q); f=np.tile(np.eye(3),(len(q),1,1))
        x[:,[0,2]]+=N@c; v[:,[0,2]]=N@vc
        f[:,[0,2],0]+=(Nx@c)
        f[:,[0,2],2]+=(Nz@c)
        g=f[:,[0,2],:][:,:,[0,2]]; d=np.linalg.det(g)
        ratio=p['known_nu']/(1-2*p['known_nu'])
        # Plane stress closure for fixed-corotated elasticity, sigma_yy=0.
        # Its E factor cancels. This is a constitutive/geometric approximation,
        # not a camera measurement of the transverse stretch or interior.
        f[:,1,1]=(1+ratio*d)/(1+ratio*d*d)
        X.append(x); V.append(v); F.append(f)
    diagnostics=dict(spatial_fit_rms_mm=float(np.sqrt(np.mean(np.square(fit_errors))))*1000,
                     max_frame_fit_rms_mm=max(fit_errors)*1000,
                     temporal_filter='Savitzky-Golay, 7 frames, cubic, loading images only',
                     volume_assumption='Planar motion uniform through width; transverse stretch from plane stress and known nu',
                     simulator_states_used=False)
    return np.asarray(X),np.asarray(V),np.asarray(F),vol,diagnostics


def identify(bundle,dest):
    p=json.loads((bundle/'known.json').read_text())
    t=np.load(dest/'tracks.npz')
    X,V,F,vol,diag=reconstruct_tracks(t,p)
    sensor=np.genfromtxt(bundle/'force.csv',delimiter=',',names=True)
    result,A,b=fit_balance(X,V,F,vol,t['time'],sensor['time'],sensor['material_on_pusher_Fx'],p)
    if not result['E_pa']>0 or result['min_reconstructed_J']<=0:
        raise RuntimeError(f'Nonphysical reconstruction/fit: {result}')
    result.update(diag,known_nu=p['known_nu'],estimated=['E'],
                  inputs=['camera pixels','camera calibration','marker pattern','timestamps','pusher force','known geometry and density'],
                  true_E_used=False,simulator_F_used=False,simulator_velocity_used=False,
                  recovery_observations_used=False)
    np.savez(dest/'weak_system.npz',a_E=A,b=b)
    # These are explicitly reconstructed quadrature states, not MPM particles.
    np.savez(dest/'reconstructed.npz',x=X,v=V,F=F,vol0=vol,time=t['time'])
    save(dest/'identification.json',result); print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fit-only',action='store_true')
    args=ap.parse_args()
    if not args.fit_only: track(args.inputs,args.out)
    identify(args.inputs,args.out)

"""Stiffness from textured stereo pixels and force, with no simulator inputs.

The known initial plane rectifies images but does not specify feature identities.
Corners are detected in the image, matched between cameras, tracked temporally,
and triangulated. Missing correspondences are excluded, never filled from truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.elastic.strip_camera_identify import identify, projected, save, spline_basis, reconstruct_tracks

RECT=dict(xmin=.075,zmax=.185,scale=10000.,width=700,height=1100)


def rectification(cam,p):
    s=RECT['scale']
    world=np.array([[1/s,0,RECT['xmin']],[0,0,p['surface_reference_y']],
                    [0,-1/s,RECT['zmax']],[0,0,1.]])
    return np.asarray(cam['P'])@world


def warp_points(x,H):
    h=np.column_stack([x,np.ones(len(x))])@H.T
    return h[:,:2]/h[:,2,None]


def flow(a,b,x,guess=None):
    src=x.astype('float32').reshape(-1,1,2)
    target=None if guess is None else guess.astype('float32').reshape(-1,1,2)
    # Rectified stereo is already locally aligned. Coarse pyramids mix the
    # narrow specimen with background and can destroy this valid initialization.
    levels=1 if guess is None else 0
    nxt,ok,err=cv2.calcOpticalFlowPyrLK(a,b,src,target,winSize=(21,21),maxLevel=levels,
        flags=0 if guess is None else cv2.OPTFLOW_USE_INITIAL_FLOW,
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,.002))
    back,okb,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,src.copy(),winSize=(21,21),maxLevel=levels,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,.002))
    good=ok[:,0].astype(bool)&okb[:,0].astype(bool)
    good&=np.linalg.norm(back[:,0]-src[:,0],axis=1)<.25
    good&=err[:,0]<18
    return nxt[:,0],good


def track(bundle,dest):
    cv2.setNumThreads(1)
    p=json.loads((bundle/'known.json').read_text())
    cams=json.loads((bundle/'calibration.json').read_text())
    homographies=[rectification(c,p) for c in cams]
    sequences=[np.load(bundle/f'camera_{i}.npy',mmap_mode='r') for i in range(2)]
    def rect(i,f):
        return cv2.warpPerspective(sequences[i][f],np.linalg.inv(homographies[i]),
                  (RECT['width'],RECT['height']),flags=cv2.INTER_LINEAR,borderValue=245)
    initial=[rect(i,0) for i in range(2)]
    mask=np.zeros_like(initial[0]); s=RECT['scale']
    lo=np.asarray(p['center'])-np.asarray(p['size'])/2; hi=lo+np.asarray(p['size'])
    x0=round((lo[0]+.00065-RECT['xmin'])*s); x1=round((hi[0]-.00065-RECT['xmin'])*s)
    y0=round((RECT['zmax']-(p['grip_lower_z']-.002))*s); y1=round((RECT['zmax']-(lo[2]+.001))*s)
    mask[y0:y1,x0:x1]=255
    queries=cv2.goodFeaturesToTrack(initial[0],maxCorners=650,qualityLevel=.02,
                                   minDistance=9,mask=mask,blockSize=7)[:,0]
    right,ok=flow(initial[0],initial[1],queries,queries.copy())
    ok&=np.linalg.norm(right-queries,axis=1)<1.0
    queries,right=queries[ok],right[ok]
    n=len(queries); T=len(sequences[0])
    xy=np.full((2,T,n,2),np.nan); valid=np.zeros((T,n),bool)
    xy[0,0]=queries; xy[1,0]=right; valid[0]=True
    previous=initial
    for f in range(1,T):
        current=[rect(i,f) for i in range(2)]; ids=np.flatnonzero(valid[f-1])
        if len(ids)<50: raise RuntimeError(f'Insufficient surviving tracks at frame {f}: {len(ids)}')
        left,okl=flow(previous[0],current[0],xy[0,f-1,ids])
        right,okr=flow(previous[1],current[1],xy[1,f-1,ids])
        # Independent temporal paths must agree with the current stereo match.
        matched,oks=flow(current[0],current[1],left,right)
        keep=okl&okr&oks&(np.linalg.norm(matched-right,axis=1)<.5)
        xy[0,f,ids[keep]]=left[keep]; xy[1,f,ids[keep]]=matched[keep]
        valid[f,ids[keep]]=True; previous=current
        if f%30==0: print('tracking',f,int(keep.sum()),flush=True)
    pixels=np.full_like(xy,np.nan)
    for ci,H in enumerate(homographies):
        pixels[ci]=warp_points(xy[ci].reshape(-1,2),H).reshape(T,n,2)
    world=np.full((T,n,3),np.nan); residual=np.full((T,n),np.nan)
    for f in range(T):
        ids=np.flatnonzero(valid[f])
        h=cv2.triangulatePoints(np.asarray(cams[0]['P']),np.asarray(cams[1]['P']),
                               pixels[0,f,ids].T,pixels[1,f,ids].T)
        xyz=(h[:3]/h[3]).T
        err=np.maximum(*[np.linalg.norm(projected(xyz,c['P'])-pixels[i,f,ids],axis=1)
                         for i,c in enumerate(cams)])
        keep=err<.35; valid[f,ids[~keep]]=False
        world[f,ids[keep]]=xyz[keep]; residual[f,ids]=err
    dest.mkdir(parents=True,exist_ok=False)
    np.savez(dest/'tracks.npz',world=world,valid=valid,pixels=pixels,
             reference=world[0],time=np.load(bundle/'time.npy'),reprojection_error_px=residual)
    result=dict(features=n,initial_stereo_tracks=int(valid[0].sum()),final_stereo_tracks=int(valid[-1].sum()),
        min_stereo_tracks=int(valid.sum(1).min()),rms_reprojection_px=float(np.sqrt(np.nanmean(residual[valid]**2))),
        tracker='Image corners; pyramidal Lucas-Kanade; forward/backward and stereo consistency checks',
        no_ground_truth_tracks=True,no_supplied_texture_correspondences=True)
    save(dest/'tracking.json',result); print(result,flush=True)


def select_reconstruction(root):
    """Choose shared resolution using held-image-feature motion, never stiffness."""
    scores=[]
    for degree in [1,2,3]:
        for spacing in [.008,.010,.012,.016,.020]:
            errors=[]; rank_ok=True; min_j=1.
            for k in ['A','B']:
                p=json.loads((root/f'inputs_{k}/known.json').read_text())
                p.update(spline_degree_x=degree,spline_spacing_z=spacing)
                d=np.load(root/f'fit_{k}/tracks.npz'); reference=d['reference'].copy()
                reference[~d['valid'][0]]=p['center']; M=spline_basis(reference,p)[0]
                lo=np.asarray(p['center'])-np.asarray(p['size'])/2; hi=lo+np.asarray(p['size'])
                grip=np.stack(np.meshgrid(np.linspace(lo[0],hi[0],5),[p['surface_reference_y']],
                              np.linspace(p['grip_lower_z']+.006,hi[2],3),indexing='ij'),axis=-1).reshape(-1,3)
                C=spline_basis(grip,p)[0]*3
                for f in [30,40,50,60,70,80]:
                    valid=d['valid'][f]&d['valid'][0]; target=(d['world'][f]-reference)[:,[0,2]]
                    for fold in range(4):
                        hold=(np.arange(len(reference))%4==fold)&valid; use=valid&~hold
                        mat=np.vstack([M[use],C]); y=np.vstack([target[use],np.zeros((len(C),2))])
                        coeff,_,rank,_=np.linalg.lstsq(mat,y,rcond=1e-9)
                        rank_ok&=rank==mat.shape[1]; errors.extend((M[hold]@coeff-target[hold]).ravel())
                _,_,F,_,_=reconstruct_tracks(d,p)
                min_j=min(min_j,float(np.linalg.det(F).min()))
            scores.append(dict(spline_degree_x=degree,spline_spacing_z=spacing,
                cv_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))*1000),full_rank=bool(rank_ok),min_J=min_j))
    good=[s for s in scores if s['full_rank'] and s['min_J']>0]
    if not good: raise RuntimeError('No physically admissible observed-motion reconstruction')
    selected=min(good,key=lambda s:s['cv_rmse_mm'])
    save(root/'reconstruction_selection.json',dict(candidates=scores,selected=selected,
        criterion='Minimum four-fold held-feature displacement error pooled across A/B, subject to full rank and positive reconstructed volume. No true E or simulator states.'))
    for k in ['A','B']:
        path=root/f'inputs_{k}/known.json'; p=json.loads(path.read_text())
        p.update({n:selected[n] for n in ['spline_degree_x','spline_spacing_z']}); save(path,p)
    print(selected)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fit-only',action='store_true')
    a=ap.parse_args()
    if not a.fit_only: track(a.inputs,a.out)
    identify(a.inputs,a.out)
    path=a.out/'identification.json'; result=json.loads(path.read_text())
    result['inputs']=['textured stereo pixels','camera calibration','timestamps','pusher force','known initial geometry, density, Poisson ratio and elastic family']
    save(path,result)

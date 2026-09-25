"""Reference-image affine subset correlation to remove incremental flow drift.

All patches and correspondences come from the two recorded image sequences.
Local affine warps permit rotation and strain; no stiffness or beam kinematics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from experiments.elastic.strip_texture_identify import RECT,rectification,warp_points
from experiments.elastic.strip_camera_identify import projected,save


def sample(im,xy):
    return cv2.remap(im.astype('float32'),xy[...,0].astype('float32'),xy[...,1].astype('float32'),
                     cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=245)


def correlate(reference,current,centers,guess,affine=None,radius=9):
    """Inverse-compositional affine DIC with a fixed reference for each subset."""
    offsets=np.stack(np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1)),axis=-1).reshape(-1,2)
    coords=centers[:,None,:]+offsets[None,:,:]
    template=sample(reference,coords); template-=template.mean(1,keepdims=True)
    gx=cv2.Sobel(reference.astype('float32'),cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(reference.astype('float32'),cv2.CV_32F,0,1,ksize=3)/8
    grad=np.stack([sample(gx,coords),sample(gy,coords)],axis=-1)
    J=np.stack([grad[:,:,0]*offsets[:,0],grad[:,:,0]*offsets[:,1],grad[:,:,0],
                grad[:,:,1]*offsets[:,0],grad[:,:,1]*offsets[:,1],grad[:,:,1]],axis=-1)
    J-=J.mean(1,keepdims=True)
    h=np.einsum('npi,npj->nij',J,J)
    inv=np.linalg.pinv(h,rcond=1e-9)
    mat=np.tile(np.eye(2),(len(centers),1,1)) if affine is None else affine.copy()
    shift=guess.copy()
    for _ in range(35):
        coords=np.einsum('nij,pj->npi',mat,offsets)+shift[:,None,:]
        observed=sample(current,coords); observed-=observed.mean(1,keepdims=True)
        # Remove an intensity gain as well as the offset; no geometric correction.
        gain=np.sum(observed*template,axis=1)/(np.sum(observed**2,axis=1)+1e-9)
        error=observed*gain[:,None]-template
        delta=np.einsum('nij,nj->ni',inv,np.einsum('npi,np->ni',J,error))
        delta=np.clip(delta,[-.04,-.04,-.6,-.04,-.04,-.6],[.04,.04,.6,.04,.04,.6])
        dm=delta.reshape(-1,2,3); dm[:,:,:2]+=np.eye(2)
        updated=mat@np.linalg.inv(dm[:,:,:2])
        shift-=np.einsum('nij,nj->ni',updated,dm[:,:,2]); mat=updated
    error_rms=np.sqrt(np.mean(error**2,axis=1))
    return shift,mat,error_rms


def refine(bundle,source,dest):
    cv2.setNumThreads(1)
    p=json.loads((bundle/'known.json').read_text()); cams=json.loads((bundle/'calibration.json').read_text())
    tracks=np.load(source/'tracks.npz'); valid=tracks['valid'].copy(); pixels=tracks['pixels'].copy()
    T,N=valid.shape; residuals=np.full((2,T,N),np.nan)
    for ci,cam in enumerate(cams):
        H=rectification(cam,p); seq=np.load(bundle/f'camera_{ci}.npy',mmap_mode='r')
        def rect(f): return cv2.warpPerspective(seq[f],np.linalg.inv(H),(RECT['width'],RECT['height']),borderValue=245)
        reference=rect(0); centers=warp_points(pixels[ci,0],np.linalg.inv(H))
        _,neighbors=cKDTree(centers).query(centers,k=16)
        for f in range(1,T):
            ids=np.flatnonzero(valid[f]); guesses=warp_points(tracks['pixels'][ci,f],np.linalg.inv(H))
            mats=[]
            for j in ids:
                near=neighbors[j]; use=near[tracks['valid'][f,near]]
                M=np.column_stack([centers[use]-centers[j],np.ones(len(use))])
                coef=np.linalg.lstsq(M,guesses[use],rcond=None)[0]
                mats.append(coef[:2].T if len(use)>=6 else np.eye(2))
            measured,affine,err=correlate(reference,rect(f),centers[ids],guesses[ids],np.asarray(mats))
            # Pixel-only rejection of occluded/ambiguous subsets.
            good=(err<9)&(np.linalg.norm(measured-guesses[ids],axis=1)<4)
            good&=(np.linalg.det(affine)>.75)&(np.linalg.det(affine)<1.3)
            valid[f,ids[~good]]=False; pixels[ci,f,ids]=warp_points(measured,H); residuals[ci,f,ids]=err
            if f%30==0: print('DIC camera',ci,'frame',f,'retained',int(good.sum()),flush=True)
    world=np.full((T,N,3),np.nan); reprojection=np.full((T,N),np.nan)
    for f in range(T):
        ids=np.flatnonzero(valid[f]); h=cv2.triangulatePoints(np.asarray(cams[0]['P']),np.asarray(cams[1]['P']),pixels[0,f,ids].T,pixels[1,f,ids].T)
        xyz=(h[:3]/h[3]).T
        err=np.maximum(*[np.linalg.norm(projected(xyz,c['P'])-pixels[i,f,ids],axis=1) for i,c in enumerate(cams)])
        good=err<.35; valid[f,ids[~good]]=False; world[f,ids[good]]=xyz[good]; reprojection[f,ids]=err
    dest.mkdir(exist_ok=False)
    np.savez(dest/'tracks.npz',world=world,reference=world[0],valid=valid,pixels=pixels,
        time=tracks['time'],reprojection_error_px=reprojection,dic_residual_gray=residuals)
    save(dest/'tracking.json',dict(features=N,initial_stereo_tracks=int(valid[0].sum()),
        final_stereo_tracks=int(valid[-1].sum()),min_stereo_tracks=int(valid.sum(1).min()),
        rms_reprojection_px=float(np.sqrt(np.nanmean(reprojection[valid]**2))),
        tracker='Image corners and Lucas-Kanade initialization; reference-image affine DIC (19x19 px), intensity offset/gain normalization, stereo triangulation',
        no_ground_truth_tracks=True,no_supplied_texture_correspondences=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['inputs','source','out']: ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args(); refine(a.inputs,a.source,a.out)

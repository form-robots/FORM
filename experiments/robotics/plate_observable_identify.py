"""Image/force-only plastic fit with locally updated constitutive histories.

There are no forward MPM rollouts in this estimator. A scalar search over Y/E
updates a homogeneous material-point history along the measured total stretches;
E is solved by bounded linear least squares at each ratio. The full fit is not
one convex solve. Unknown nu/hardening are not claimed to be identified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from experiments.elastic.strip_texture_dic import correlate
from experiments.elastic.strip_texture_identify import flow,warp_points
from experiments.elastic.strip_camera_identify import projected,save

RECT=dict(xmin=.062,zmax=.080,scale=18000.,width=1368,height=720)


def rectangle(p):
    # Fixed physical crop relative to the known specimen, independent of world
    # translation or the numerical simulation domain.
    r=dict(RECT);r.update(xmin=p['center'][0]-.038,zmax=p['floor']+p['size'][2]+.005,
                         height=round((p['size'][2]+.015)*RECT['scale']))
    return r


def homography(cam,p):
    r=rectangle(p);s=r['scale'];plane=np.array([[1/s,0,r['xmin']],[0,0,p['surface_y']],
        [0,-1/s,r['zmax']],[0,0,1.]])
    return np.asarray(cam['P'])@plane


def track(bundle,dest,features=280,margin_mm=1.5,min_distance=17):
    cv2.setNumThreads(1);p=json.loads((bundle/'known.json').read_text());cams=json.loads((bundle/'calibration.json').read_text())
    seq=[np.load(bundle/f'camera_{i}.npy',mmap_mode='r') for i in range(2)];H=[homography(c,p) for c in cams];r=rectangle(p)
    def rect(ci,f):return cv2.warpPerspective(seq[ci][f],np.linalg.inv(H[ci]),(r['width'],r['height']),borderValue=245)
    first=[rect(i,0) for i in range(2)];mask=np.zeros_like(first[0]);s=r['scale']
    lo=np.asarray(p['center'])-np.asarray(p['size'])/2;hi=lo+np.asarray(p['size'])
    margin=margin_mm/1000
    mask[round((r['zmax']-hi[2]+margin)*s):round((r['zmax']-lo[2]-margin)*s),
         round((lo[0]+margin-r['xmin'])*s):round((hi[0]-margin-r['xmin'])*s)]=255
    q=cv2.goodFeaturesToTrack(first[0],maxCorners=features,qualityLevel=.03,minDistance=min_distance,mask=mask,blockSize=7)[:,0]
    right,ok=flow(first[0],first[1],q,q.copy());ok&=np.linalg.norm(right-q,axis=1)<1
    initial=[q[ok],right[ok]];N=len(initial[0]);T=len(seq[0]);print('Initial features',N,flush=True)
    raw=np.full((2,T,N,2),np.nan);refined=raw.copy();valid=np.zeros((T,N),bool);valid[0]=True
    for ci in [0,1]:raw[ci,0]=refined[ci,0]=initial[ci]
    previous=first;resid=np.full((2,T,N),np.nan)
    neighbors=cKDTree(initial[0]).query(initial[0],k=14)[1]
    for f in range(1,T):
        ids=np.flatnonzero(valid[f-1]);current=[rect(i,f) for i in range(2)]
        if len(ids)<40:raise RuntimeError(f'Insufficient tracks: {len(ids)} at {f}')
        good=np.ones(len(ids),bool)
        for ci in [0,1]:
            guess,ok=flow(previous[ci],current[ci],raw[ci,f-1,ids])
            raw[ci,f,ids]=guess
            mats=[]
            for j in ids:
                near=neighbors[j];use=near[valid[f-1,near]]
                M=np.column_stack([initial[ci][use]-initial[ci][j],np.ones(len(use))])
                coeff=np.linalg.lstsq(M,raw[ci,f,use],rcond=None)[0]
                mats.append(coeff[:2].T if len(use)>=6 else np.eye(2))
            refined[ci,f,ids],a,e=correlate(first[ci],current[ci],initial[ci][ids],guess,np.array(mats))
            resid[ci,f,ids]=e
            good&=ok&(e<10)&(np.linalg.det(a)>.35)&(np.linalg.det(a)<1.5)
            good&=np.linalg.norm(refined[ci,f,ids]-guess,axis=1)<5
        valid[f,ids[good]]=True;previous=current
        if f%30==0:print('Frame',f,'valid',int(good.sum()),flush=True)
    pixels=np.array([warp_points(refined[i].reshape(-1,2),H[i]).reshape(T,N,2) for i in [0,1]])
    world=np.full((T,N,3),np.nan);reprojection=np.full((T,N),np.nan)
    for f in range(T):
        ids=np.flatnonzero(valid[f]);h=cv2.triangulatePoints(np.asarray(cams[0]['P']),np.asarray(cams[1]['P']),pixels[0,f,ids].T,pixels[1,f,ids].T)
        xyz=(h[:3]/h[3]).T
        error=np.maximum(*[np.linalg.norm(projected(xyz,c['P'])-pixels[i,f,ids],axis=1) for i,c in enumerate(cams)])
        good=error<.4;valid[f,ids[~good]]=False;world[f,ids[good]]=xyz[good];reprojection[f,ids]=error
    dest.mkdir(exist_ok=False)
    np.savez(dest/'tracks.npz',world=world,reference=world[0],time=np.load(bundle/'time.npy'),valid=valid,pixels=pixels)
    save(dest/'tracking.json',dict(initial_features=N,final_features=int(valid[-1].sum()),min_features=int(valid.sum(1).min()),
        reprojection_rms_px=float(np.sqrt(np.nanmean(reprojection[valid]**2))),method='Stereo corners, Lucas-Kanade initialization, reference-image affine DIC',simulator_states_used=False,
        configuration=dict(features=features,margin_mm=margin_mm,min_distance=min_distance)))


def reconstruct(d,p):
    ref=d['reference'];center=np.array([p['center'][0],p['center'][1],p['floor']]);q=ref-center
    stretches=[];translations=[];errors=[]
    for f,world in enumerate(d['world']):
        use=d['valid'][f]&d['valid'][0];w=world[use]-center
        a=np.zeros(3);b=np.zeros(3)
        for j in [0,2]:
            a[j],b[j]=np.linalg.lstsq(np.column_stack([q[use,j],np.ones(use.sum())]),w[:,j],rcond=None)[0]
        a[1]=float(q[use,1]@w[:,1]/(q[use,1]@q[use,1]))
        residual=w-q[use]*a-b
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))))
        stretches.append(a);translations.append(b)
    stretches=savgol_filter(stretches,7,3,axis=0);translations=savgol_filter(translations,7,3,axis=0)
    # Initial frame defines the stress-free reference; no deformed state reset.
    stretches[0]=1.;translations[0]=0.
    if np.min(stretches)<=0:raise RuntimeError('Nonpositive reconstructed stretch')
    return stretches,translations,np.array(errors)


def unit_stress(stretches,ratio,nu):
    """Diagonal multiplicative Hencky return map, stress in units of E."""
    mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu));eps=np.zeros(3);old=np.ones(3);tau=[]
    for stretch in stretches:
        trial=eps+np.log(stretch/old);mean=trial.mean();dev=trial-mean;norm=np.linalg.norm(dev)
        if 2*mu*norm>ratio:dev*=ratio/(2*mu*norm)
        eps=mean+dev;tau.append(2*mu*eps+lam*eps.sum());old=stretch
    return np.array(tau)


def weak_system(stretches,translations,t,p,sensor):
    dt=float(t[1]-t[0]);size=np.array(p['size']);mass=size.prod()*p['density']
    # Moments of uniform reference volume, centered laterally and above support.
    mean=np.array([0.,0.,size[2]/2]);second=size**2/12+mean**2
    v=savgol_filter(stretches,7,3,deriv=1,delta=dt,axis=0)
    vb=savgol_filter(translations,7,3,deriv=1,delta=dt,axis=0)
    momentum=mass*(v*stretches*second+(v*translations+vb*stretches)*mean+vb*translations)/size
    conv=mass*(v*v*second+2*v*vb*mean+vb*vb)/size
    conv[:,2]-=mass*9.81*(stretches[:,2]*mean[2]+translations[:,2])/size[2]
    W=[];loads=[]
    # Fixed time windows, shared between materials. Continuous force integration.
    for begin in np.arange(.20,p['fit_end']-.159,.06):
        end=begin+.16;u=(t-begin)/(end-begin);use=(u>=0)&(u<=1)
        chi=np.where(use,np.sin(np.pi*np.clip(u,0,1))**4,0)
        dchi=np.where(use,4*np.pi/(end-begin)*np.sin(np.pi*np.clip(u,0,1))**3*np.cos(np.pi*np.clip(u,0,1)),0)
        load=np.sum((chi[:,None]*conv+dchi[:,None]*momentum),axis=0)*dt
        su=(sensor['time']-begin)/(end-begin);keep=(su>=0)&(su<=1)
        load[2]-=np.sum(np.sin(np.pi*su[keep])**4*sensor['Fz'][keep]*sensor['opening'][keep]/size[2])*p['force_tick']
        W.append(chi*dt);loads.append(load)
    return np.array(W),np.array(loads).ravel()


def fit(bundle,dest):
    p=json.loads((bundle/'known.json').read_text());d=np.load(dest/'tracks.npz');sensor=np.genfromtxt(bundle/'force.csv',delimiter=',',names=True)
    stretch,translation,error=reconstruct(d,p);W,b=weak_system(stretch,translation,d['time'],p,sensor)
    size=np.array(p['size']);tries=[];start=time.monotonic()
    def score(log_ratio,record=True):
        ratio=np.exp(log_ratio);tau=unit_stress(stretch,ratio,p['nu'])
        A=((W@tau)*size.prod()/size).ravel();E=float(np.clip(A@b/(A@A),5000,500000))
        loss=float(np.sum((E*A-b)**2)/np.sum(b*b))
        if record:tries.append(dict(E_pa=E,yield_pa=ratio*E,ratio=ratio,loss=loss))
        return loss
    grid=np.linspace(np.log(.001),np.log(.5),65);losses=[score(v) for v in grid]
    j=int(np.argmin(losses));opt=minimize_scalar(score,bounds=(grid[max(0,j-1)],grid[min(64,j+1)]),method='bounded',options={'xatol':1e-7})
    score(opt.x);best=min(tries,key=lambda v:v['loss']);ratio=best['ratio'];tau=unit_stress(stretch,ratio,p['nu'])*best['E_pa']
    best.update(residual_relative_l2=float(np.sqrt(best.pop('loss'))),fit_wall_s=time.monotonic()-start,evaluations=len(tries),
        uniform_surface_residual_rms_mm=float(np.sqrt(np.mean(error**2)))*1000,
        uniform_surface_residual_max_mm=float(error.max()*1000),known_nu=p['nu'],
        estimator='Profile scalar search in Y/E with conditional bounded linear E solve; local constitutive updates along observed stretch history',
        forward_simulation_calls=0,simulator_states_used=False,estimated=['E','yield_stress'])
    np.savez(dest/'reconstructed.npz',time=d['time'],stretch=stretch,translation=translation,surface_residual_mm=error*1000,tau=tau,W=W,b=b)
    save(dest/'identification.json',best);save(dest/'search.json',tries);print(best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fit-only',action='store_true')
    a=ap.parse_args()
    if not a.fit_only:track(a.inputs,a.out)
    fit(a.inputs,a.out)

"""Divergence-free test fields flat near the contact layers.

The bottom virtual displacement is zero and the top is a unit vertical
translation. Thus a measured net normal force supplies exact boundary work,
while the stress integral samples the observable interior. Flat test fields
are not an assumption that the material deformation is flat or uniform.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from experiments.robotics.plate_observable_field import reconstruct,unit_stress,volume_quadrature
from experiments.elastic.strip_camera_observe import save

SPANS=((.1,.7),(.2,.8),(.3,.9))


def virtual(X,p):
    h=min(gap for t,gap in p['knots'] if t<=p['fit_end']+1e-9)
    r=X-np.array([p['center'][0],p['center'][1],p['floor']]);fields=[];grads=[]
    for lo,hi in SPANS:
        L=(hi-lo)*h;s=np.clip((r[...,2]-lo*h)/L,0,1)
        f=s**3*(10-15*s+6*s*s);df=30*s*s*(1-s)**2/L;ddf=60*s*(1-s)*(1-2*s)/L**2
        w=np.stack([-r[...,0]*df/2,-r[...,1]*df/2,f],axis=-1)
        g=np.zeros((*X.shape[:-1],3,3));g[...,0,0]=g[...,1,1]=-df/2;g[...,2,2]=df
        g[...,0,2]=-r[...,0]*ddf/2;g[...,1,2]=-r[...,1]*ddf/2
        fields.append(w);grads.append(g)
    return np.stack(fields,axis=1),np.stack(grads,axis=1)


def weak(X,V,t,p,sensor,weights):
    w,g=virtual(X,p);mass=np.prod(p['size'])*p['density']
    momentum=mass*np.einsum('tmn,n->tm',np.einsum('tni,tmni->tmn',V,w),weights)
    load=mass*np.einsum('tmn,n->tm',np.einsum('tni,tmnij,tnj->tmn',V,g,V)-9.81*w[...,2],weights)
    W=[];rhs=[];dt=t[1]-t[0]
    for begin in np.arange(.20,p['fit_end']-.159,.06):
        u=np.clip((t-begin)/.16,0,1);chi=np.sin(np.pi*u)**4;dchi=4*np.pi/.16*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        b=np.sum(chi[:,None]*load+dchi[:,None]*momentum,axis=0)*dt
        u=(sensor['time']-begin)/.16;use=(u>=0)&(u<=1)
        b-=np.sum(np.sin(np.pi*u[use])**4*sensor['Fz'][use])*p['force_tick']
        W.append(chi*dt);rhs.append(b)
    return np.array(W),np.array(rhs).ravel(),g


def fit(root,k,quadrature_order=8,tag='interior'):
    order=quadrature_order
    p=json.loads((root/f'inputs_{k}/known.json').read_text());d=np.load(root/f'fit_{k}/tracks.npz')
    selected=json.loads((root/'field_selection.json').read_text())['selected'];spacing=selected['spacing']
    p['cross_section_degree']=selected.get('cross_section_degree',3)
    q,weights=volume_quadrature(p,spacing,order);X,V,F,q,error=reconstruct(d,p,spacing,quadrature=q,smoothing=selected.get('smoothing',0.))
    sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',delimiter=',',names=True);W,b,g=weak(X,V,d['time'],p,sensor,weights)
    config=json.loads((root/'identification_protocol.json').read_text()) if (root/'identification_protocol.json').exists() else {}
    row_scale=np.ones_like(b)
    intervals=config.get('balanced_identification_intervals')
    if intervals:
        # Equal relative weak-balance loss for each prescribed probe cycle.
        # Otherwise the large plastic-press forces swamp the preliminary
        # stiffness probe. Normalization uses measured loads, never true E/Y;
        # the full plastic history is retained, with no elastic-window shortcut.
        centers=np.arange(.20,p['fit_end']-.159,.06)+.08
        assigned=np.zeros(len(centers),dtype=int)
        for lo,hi in intervals:
            use=(centers>=lo)&(centers<hi);rows=np.repeat(use,len(SPANS))
            if not rows.any() or np.linalg.norm(b[rows])<=1e-12:raise ValueError('Uninformative identification interval')
            row_scale[rows]=1/np.linalg.norm(b[rows]);assigned[use]+=1
        if not np.all(assigned==1):raise ValueError('Identification intervals must partition all weak windows')
    target=b*row_scale
    tries=[];start=time.monotonic()
    def score(lr):
        ratio=np.exp(lr);stress=unit_stress(F,ratio,p['nu'],True)
        A=(W@np.einsum('tnij,tmnij,n->tm',stress,g,weights)).ravel()*np.prod(p['size'])
        design=A*row_scale
        E=float(np.clip(design@target/(design@design),5000,500000));loss=float(np.sum((E*design-target)**2)/np.sum(target*target))
        tries.append(dict(E_pa=E,yield_pa=E*ratio,ratio=ratio,loss=loss));return loss
    grid=np.linspace(np.log(.001),np.log(.5),33);losses=[score(v) for v in grid];j=int(np.argmin(losses))
    opt=minimize_scalar(score,bounds=(grid[max(0,j-1)],grid[min(32,j+1)]),method='bounded',options={'xatol':1e-6});score(opt.x)
    best=dict(min(tries,key=lambda x:x['loss']));best['residual_relative_l2']=float(np.sqrt(best.pop('loss')))
    best.update(fit_wall_s=time.monotonic()-start,evaluations=len(tries),quadrature_order=order,
        surface_fit_rmse_mm=float(np.sqrt(np.mean(error**2))*1000),min_reconstructed_J=float(np.linalg.det(F).min()),
        known_nu=p['nu'],estimated=['E','yield_stress'],test_field_spans=SPANS,
        estimator='Nonlinear Y/E search with conditional E solve and locally inferred plastic history',
        reconstruction='Square-symmetric low-order interior extension of tracked surface displacement',
        pressure_eliminated=True,constant_contact_virtual_displacement=True,forward_simulation_calls=0,simulator_states_used=False)
    best['balanced_identification_intervals']=intervals
    dest=root/f'{tag}_{k}';dest.mkdir(exist_ok=False);save(dest/'identification.json',best);save(dest/'search.json',tries);print(k,best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],required=True)
    ap.add_argument('--order',type=int,default=8);ap.add_argument('--tag',default='interior');a=ap.parse_args();fit(a.out,a.material,a.order,a.tag)

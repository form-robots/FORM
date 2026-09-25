"""Strip kinematics and weak balance from observed surface marker trajectories.

No simulator imports or true material parameters. The width reconstruction is
an explicit symmetry approximation, not a measurement of the interior.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.signal import savgol_filter

from ident.weakform.elastic_grid import corotated_cauchy_columns, _temporal_window
from ident.weakform.grid_assembly import _bspline_weights_1d


def virtual_fields(x, grid, p):
    """Known nodal bending fields, with unit contact trace and zero grip trace."""
    dx=p['domain']/grid
    z=x[:,2]/dx; base=np.floor(z-.5).astype(int)
    weights,dweights=_bspline_weights_1d(z-base)
    nodes=(base[:,None]+np.arange(3))*dx
    fields=[]; offset=x[:,0]-p['center'][0]
    for lo,hi in p['test_transitions']:
        u=np.clip((nodes-lo)/(hi-lo),0,1)
        s=1-10*u**3+15*u**4-6*u**5
        q=(-30*u**2+60*u**3-30*u**4)/(hi-lo)
        si=(weights*s).sum(axis=1); qi=(weights*q).sum(axis=1)
        ds=(dweights*s).sum(axis=1)/dx; dq=(dweights*q).sum(axis=1)/dx
        w=np.zeros_like(x,dtype=float); grad=np.zeros((len(x),3,3))
        w[:,0]=si; w[:,2]=-offset*qi
        grad[:,0,2]=ds; grad[:,2,0]=-qi; grad[:,2,2]=-offset*dq
        fields.append((w,grad))
    return fields


def quadrature(p):
    """Cell centers determined by the measured dimensions, not simulator IDs."""
    size=np.asarray(p['size']); center=np.asarray(p['center'])
    counts=np.asarray(p.get('quadrature_counts',[8,10,80]))
    axes=[c+s*((np.arange(n)+.5)/n-.5) for c,s,n in zip(center,size,counts,strict=True)]
    x=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    return x,np.full(len(x),size.prod()/len(x))


def reconstruct(surface,reference_x,reference_z,times,p):
    """Surface shape (T, nx, nz, 3) on the initially flat front side.

    x,z are uniform across width; y varies linearly about the known midplane.
    This permits observed lateral contraction and uses no fitted stiffness.
    """
    dt=float(times[1]-times[0])
    smooth=savgol_filter(surface,7,3,axis=0)
    velocity=savgol_filter(surface,7,3,deriv=1,delta=dt,axis=0)
    q,vol=quadrature(p)
    center=np.asarray(p['center']); half=p['size'][1]/2
    ratio=(q[:,1]-center[1])/(-half)
    all_x=[]; all_v=[]; all_F=[]
    for sf,sv in zip(smooth,velocity,strict=True):
        x=np.empty_like(q); v=np.empty_like(q); F=np.zeros((len(q),3,3))
        for d in range(3):
            # Fit displacement to preserve the exact undeformed affine map.
            ref=np.full(sf.shape[:2],center[d])
            if d==0: ref[:]=reference_x[:,None]
            if d==1: ref[:]=center[1]-half
            if d==2: ref[:]=reference_z[None,:]
            spl=RectBivariateSpline(reference_x,reference_z,sf[:,:,d]-ref,kx=3,ky=3)
            vspl=RectBivariateSpline(reference_x,reference_z,sv[:,:,d],kx=3,ky=3)
            disp=spl.ev(q[:,0],q[:,2])
            dx=spl.ev(q[:,0],q[:,2],dx=1)
            dz=spl.ev(q[:,0],q[:,2],dy=1)
            vel=vspl.ev(q[:,0],q[:,2])
            if d==1:
                x[:,1]=q[:,1]+ratio*disp
                v[:,1]=ratio*vel
                F[:,1,0]=ratio*dx; F[:,1,2]=ratio*dz
                F[:,1,1]=1-disp/half
            else:
                x[:,d]=q[:,d]+disp; v[:,d]=vel
                F[:,d,0]=dx; F[:,d,2]=dz; F[:,d,d]+=1
        all_x.append(x); all_v.append(v); all_F.append(F)
    return np.asarray(all_x),np.asarray(all_v),np.asarray(all_F),vol


def fit_balance(X,V,F,vol,times,force_t,force_x,p):
    """Fit E with known nu from reconstructed volume kinematics and net force."""
    dt=float(times[1]-times[0]); nu=p['known_nu']; mass=vol*p['density']
    frames=np.flatnonzero((times>=p['fit_interval'][0])&(times<=p['fit_interval'][1]))
    # Integrate the faster force sensor at its own rate. Sampling its contact
    # impulses only at camera timestamps would alias the boundary load.
    force_dt=float(np.median(np.diff(force_t)))
    a=[]; load=[]; mom=[]
    for i in frames:
        x,v,f=X[i],V[i],F[i]
        sm,sl=corotated_cauchy_columns(f)
        tau=(np.linalg.det(f)*vol)[:,None,None]*(sm/(2*(1+nu))+sl*nu/((1+nu)*(1-2*nu)))
        ar=[]; br=[]; mr=[]
        for w,grad in virtual_fields(x,p['grid'],p):
            ar.append(np.einsum('pij,pij->',tau,grad))
            br.append(float(np.sum(mass*(-9.81*w[:,2]+np.einsum('pi,pij,pj->p',v,grad,v)))))
            mr.append(float(np.sum(mass*np.sum(v*w,axis=1))))
        a.append(ar); load.append(br); mom.append(mr)
    a,load,mom=map(np.asarray,(a,load,mom))
    nw=round(p.get('window_duration',.25)/dt)+1
    stride=round(p.get('window_stride_s',.10)/dt)
    chi,dchi=_temporal_window(nw,dt,2)
    AA=[]; bb=[]
    for j in range(0,len(frames)-nw+1,stride):
        s=slice(j,j+nw)
        AA.extend((chi[:,None]*a[s]).sum(0)*dt)
        begin=times[frames[j]]; end=times[frames[j+nw-1]]
        sensor=(force_t>=begin)&(force_t<=end)
        phase=(force_t[sensor]-begin)/(end-begin)
        boundary=-float(np.sum(np.sin(np.pi*phase)**4*force_x[sensor])*force_dt)
        bb.extend((chi[:,None]*load[s]+dchi[:,None]*mom[s]).sum(0)*dt+boundary)
    A,b=np.asarray(AA),np.asarray(bb)
    E=float(A@b/(A@A))
    return dict(E_pa=E,residual_relative_l2=float(np.linalg.norm(A*E-b)/np.linalg.norm(b)),
                per_field_E_pa=[float(A[j::3]@b[j::3]/(A[j::3]@A[j::3])) for j in range(3)],
                rows=len(A),min_reconstructed_J=float(np.linalg.det(F[frames]).min())),A,b

"""Nonuniform square-symmetric volume reconstruction from side-surface tracks.

The centered square specimen and vertical isotropic contacts admit reflection
and quarter-turn symmetry. We use the lowest cross-sectional polynomial family
that independently represents x/y expansion and quadratic axial barreling:
ux=a*(A(z)+B(z)*a²+C(z)*b²), uy=b*(A(z)+B(z)*b²+C(z)*a²),
uz=D(z)+H(z)*(a²+b²), with normalized a,b in [-1,1].
Coefficients come from measured surface displacement, not candidate material
parameters. Interior interpolation remains an explicit approximation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from experiments.elastic.strip_camera_observe import save


def height_edges(p,spacing):
    height=p['size'][2];n=max(1,round(height/spacing))
    return np.linspace(p['floor'],p['floor']+height,n+1)


def basis(q,p,spacing=.005,z_derivative=0):
    size=np.asarray(p['size']);half=size[0]/2;z0=p['floor'];z1=z0+size[2]
    knots=np.r_[[z0]*4,height_edges(p,spacing)[1:-1],[z1]*4]
    n=len(knots)-4;spline=BSpline(knots,np.eye(n),3)
    Z=spline(q[:,2],nu=z_derivative);dZ=spline(q[:,2],nu=z_derivative+1)
    a=(q[:,0]-p['center'][0])/half;b=(q[:,1]-p['center'][1])/half
    degree=p.get('cross_section_degree',3)
    if degree not in [3,5,7]:raise ValueError('Supported cross-section degrees: 3, 5, 7')
    m=(degree-1)//2;nxy=1+2*m;groups=3*m+2
    M=np.zeros((len(q),3,groups*n));G=np.zeros((len(q),3,3,groups*n))
    terms=[(0,0,a,1/half+0*a,0*a),(1,0,b,0*b,1/half+0*b),
           (2,nxy,np.ones_like(a),0*a,0*a)]
    # Enrich the same square-symmetric extension with side-observable powers.
    # Do not fit mixed higher powers that add modes vanishing on every side:
    # those interior coefficients cannot be inferred from these observations.
    for j in range(1,m+1):
        power=2*j;own=2*j-1;other=2*j
        terms.extend([(0,own,a**(power+1),(power+1)*a**power/half,0*a),
                      (0,other,a*b**power,b**power/half,power*a*b**(power-1)/half),
                      (1,own,b**(power+1),0*b,(power+1)*b**power/half),
                      (1,other,b*a**power,power*b*a**(power-1)/half,a**power/half),
                      (2,nxy+j,a**power+b**power,power*a**(power-1)/half,power*b**(power-1)/half)])
    for dim,group,f,dx,dy in terms:
        sl=slice(group*n,(group+1)*n);M[:,dim,sl]=f[:,None]*Z
        G[:,dim,0,sl]=dx[:,None]*Z;G[:,dim,1,sl]=dy[:,None]*Z;G[:,dim,2,sl]=f[:,None]*dZ
    return M,G


def height_penalty(p,spacing):
    """Mean squared surface curvature in normalized height, not a strain constraint."""
    u,w=np.polynomial.legendre.leggauss(5);v,vw=np.polynomial.legendre.leggauss(4)
    edges=height_edges(p,spacing);height=p['size'][2]
    z=np.concatenate([(a+b)/2+(b-a)*v/2 for a,b in zip(edges[:-1],edges[1:],strict=True)])
    zw=np.concatenate([vw*(b-a)/(2*height) for a,b in zip(edges[:-1],edges[1:],strict=True)])
    a,zz=np.meshgrid(u,z,indexing='ij')
    q=np.column_stack([p['center'][0]+a.ravel()*p['size'][0]/2,
                       np.full(a.size,p['surface_y']),zz.ravel()])
    weights=np.outer(w/2,zw).ravel()
    M,_=basis(q,p,spacing,z_derivative=2)
    return (M*height**2*np.sqrt(weights[:,None,None]/3)).reshape(-1,M.shape[-1])


def coefficients(d,p,spacing,holdout=False,validation_quadrature=None,smoothing=0.):
    ref=d['reference'].copy()
    # The measured initial surface is planar. Its normal coordinate is known
    # from initial geometry; no later surface coordinates are prescribed.
    ref[:,1]=p['surface_y'];M,_=basis(ref,p,spacing)
    coeff=[];errors=[];cv=[]
    cv_times=p.get('reconstruction_cv_times',[.4,.6,.8,1.,1.1,1.2])
    cv_frames={int(np.argmin(abs(d['time']-t))) for t in cv_times}
    validation_G=None if validation_quadrature is None else basis(validation_quadrature,p,spacing)[1]
    penalty=height_penalty(p,spacing) if smoothing else None
    solvers={}
    def solve(use,target):
        # Visibility changes only occasionally. Reuse each observation operator;
        # it depends on geometry/validity, never force or material parameters.
        key=use.tobytes()
        if key not in solvers:
            A=M[use].reshape(-1,M.shape[-1]);normalizer=np.sqrt(len(A)) if smoothing else 1.
            aug=np.vstack([A/normalizer,np.sqrt(smoothing)*penalty]) if smoothing else A
            U,s,Vh=np.linalg.svd(aug,full_matrices=False);rank=int(np.sum(s>s[0]*1e-8))
            if rank<A.shape[1]:raise RuntimeError(f'Insufficient surface coverage: {rank}/{A.shape[1]}')
            solvers[key]=((Vh.T/s)@U[:len(A)].T/normalizer,A)
        operator,A=solvers[key];b=target[use].ravel()
        return operator@b,A,b
    for i,w in enumerate(d['world']):
        valid=d['valid'][i]&d['valid'][0];target=w-d['world'][0]
        c,A,b=solve(valid,target)
        coeff.append(c);errors.append(np.sqrt(np.mean((A@c-b)**2)))
        if holdout and i in cv_frames:
            for fold in range(4):
                hold=valid&(np.arange(len(ref))%4==fold);use=valid&~hold
                c,_,_=solve(use,target)
                if validation_G is not None:
                    J=np.linalg.det(np.eye(3)+np.einsum('ndkj,j->ndk',validation_G,c))
                    if J.min()<=0:raise RuntimeError(f'Nonpositive validation-fold volume at frame {i}, fold {fold}: {J.min():.6g}')
                cv.extend((np.einsum('ndj,j->nd',M[hold],c)-target[hold]).ravel())
    return np.array(coeff),np.array(errors),np.array(cv)


def volume_quadrature(p,spacing,order=6):
    """Tensor Gauss rule, split at every height-spline knot; weights sum to one."""
    s=np.array(p['size']);c=np.array(p['center']);u,w=np.polynomial.legendre.leggauss(order)
    ends=height_edges(p,spacing)
    zz=np.concatenate([(a+b)/2+(b-a)*u/2 for a,b in zip(ends[:-1],ends[1:],strict=True)])
    wz=np.concatenate([w*(b-a)/(2*s[2]) for a,b in zip(ends[:-1],ends[1:],strict=True)])
    q=np.stack(np.meshgrid(c[0]+s[0]*u/2,c[1]+s[1]*u/2,zz,indexing='ij'),axis=-1).reshape(-1,3)
    weights=np.einsum('i,j,k->ijk',w/2,w/2,wz).ravel()
    return q,weights


def reconstruct(d,p,spacing,counts=(8,8,14),quadrature=None,smoothing=0.):
    coeff,error,_=coefficients(d,p,spacing,smoothing=smoothing);dt=float(d['time'][1]-d['time'][0])
    c=savgol_filter(coeff,7,3,axis=0);v=savgol_filter(coeff,7,3,deriv=1,delta=dt,axis=0);c[0]=0
    size=np.array(p['size']);axes=[center+s*((np.arange(n)+.5)/n-.5) for center,s,n in zip(p['center'],size,counts,strict=True)]
    q=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3) if quadrature is None else quadrature
    M,G=basis(q,p,spacing)
    X=q+np.einsum('ndj,tj->tnd',M,c);V=np.einsum('ndj,tj->tnd',M,v)
    F=np.eye(3)+np.einsum('ndkj,tj->tndk',G,c)
    if np.linalg.det(F).min()<=0:raise RuntimeError('Nonpositive reconstructed volume')
    return X,V,F,q,error


def select(root,quadrature_order=None,validate_volume_folds=False):
    candidates=[]
    height=json.loads((root/'inputs_A/known.json').read_text())['size'][2]
    config=json.loads((root/'identification_protocol.json').read_text()) if (root/'identification_protocol.json').exists() else {}
    options=[(height/n,smoothing,degree) for degree in config.get('cross_section_degrees',[3]) for n in config.get('height_span_counts',[6,5,4,3,2]) for smoothing in config.get('smoothing_values',[0.])]
    for spacing,smoothing,degree in options:
        residual=[];minj=1.;valid=True;reason=None
        try:
            for k in ['A','B']:
                p=json.loads((root/f'inputs_{k}/known.json').read_text());d=np.load(root/f'fit_{k}/tracks.npz')
                p['cross_section_degree']=degree
                q=None if quadrature_order is None else volume_quadrature(p,spacing,quadrature_order)[0]
                _,_,e=coefficients(d,p,spacing,True,validation_quadrature=q if validate_volume_folds else None,smoothing=smoothing);residual.extend(e)
                _,_,F,_,_=reconstruct(d,p,spacing,counts=(4,4,10),quadrature=q,smoothing=smoothing);minj=min(minj,float(np.linalg.det(F).min()))
        except RuntimeError as error:valid=False;reason=str(error)
        candidates.append(dict(spacing=spacing,smoothing=smoothing,cross_section_degree=degree,cv_rmse_mm=float(np.sqrt(np.mean(np.square(residual)))*1000) if len(residual) else None,admissible=valid,min_J=minj,rejection_reason=reason))
        save(root/'field_candidates.json',candidates)
    best=min((x for x in candidates if x['admissible']),key=lambda v:v['cv_rmse_mm'])
    save(root/'field_selection.json',dict(candidates=candidates,selected=best,admissibility_quadrature_order=quadrature_order,validate_volume_folds=validate_volume_folds,criterion='Shared four-fold held-surface-feature displacement prediction, positive reconstructed volume; no true parameters.'))
    print(best)


def unit_stress(F,ratio,nu,full=False):
    mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu));Fe=np.tile(np.eye(3),(F.shape[1],1,1));old=Fe.copy();tau=[]
    for f in F:
        trial=f@np.linalg.inv(old)@Fe;U,s,Vh=np.linalg.svd(trial);eps=np.log(s);mean=eps.mean(1);dev=eps-mean[:,None]
        norm=np.linalg.norm(dev,axis=1);scale=np.minimum(1.,ratio/(2*mu*np.maximum(norm,1e-20)));eps=mean[:,None]+dev*scale[:,None]
        Fe=(U*np.exp(eps)[:,None,:])@Vh
        principal=2*mu*eps+lam*eps.sum(1)[:,None]
        tau.append(np.einsum('nik,nk,njk->nij',U,principal,U) if full else np.einsum('nik,nk,nik->ni',U,principal,U));old=f
    return np.array(tau)


def weak(X,V,t,p,sensor):
    size=np.array(p['size']);mass=size.prod()*p['density'];center=np.array([p['center'][0],p['center'][1],p['floor']])
    w=(X-center)/size
    momentum=mass*np.mean(V*w,axis=1)
    load=mass*np.mean(V*V/size,axis=1);load[:,2]-=mass*9.81*np.mean(w[:,:,2],axis=1)
    W=[];rhs=[];dt=t[1]-t[0]
    for begin in np.arange(.20,p['fit_end']-.159,.06):
        end=begin+.16;u=np.clip((t-begin)/(end-begin),0,1)
        chi=np.sin(np.pi*u)**4;dchi=4*np.pi/(end-begin)*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        b=np.sum(chi[:,None]*load+dchi[:,None]*momentum,axis=0)*dt
        u=(sensor['time']-begin)/(end-begin);use=(u>=0)&(u<=1)
        b[2]-=np.sum(np.sin(np.pi*u[use])**4*sensor['Fz'][use]*sensor['opening'][use]/size[2])*p['force_tick']
        W.append(chi*dt);rhs.append(b)
    return np.array(W),np.array(rhs).ravel()


def fit(root,k):
    p=json.loads((root/f'inputs_{k}/known.json').read_text());d=np.load(root/f'fit_{k}/tracks.npz')
    selected=json.loads((root/'field_selection.json').read_text())['selected'];spacing=selected['spacing']
    p['cross_section_degree']=selected.get('cross_section_degree',3)
    X,V,F,q,error=reconstruct(d,p,spacing,smoothing=selected.get('smoothing',0.));sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',delimiter=',',names=True)
    W,b=weak(X,V,d['time'],p,sensor);size=np.array(p['size']);tries=[];start=time.monotonic()
    def score(logratio):
        ratio=np.exp(logratio);stress=unit_stress(F,ratio,p['nu']).mean(1)
        A=((W@stress)*size.prod()/size).ravel();E=float(np.clip(A@b/(A@A),5000,500000));loss=float(np.sum((E*A-b)**2)/np.sum(b*b))
        tries.append(dict(E_pa=E,yield_pa=ratio*E,ratio=ratio,loss=loss));return loss
    grid=np.linspace(np.log(.001),np.log(.5),33);losses=[score(v) for v in grid];j=int(np.argmin(losses))
    opt=minimize_scalar(score,bounds=(grid[max(0,j-1)],grid[min(32,j+1)]),method='bounded',options={'xatol':1e-6});score(opt.x)
    best=dict(min(tries,key=lambda v:v['loss']));best['residual_relative_l2']=float(np.sqrt(best.pop('loss')))
    best.update(fit_wall_s=time.monotonic()-start,evaluations=len(tries),surface_fit_rmse_mm=float(np.sqrt(np.mean(error**2))*1000),
        min_reconstructed_J=float(np.linalg.det(F).min()),known_nu=p['nu'],estimated=['E','yield_stress'],
        estimator='Profile search Y/E; conditional linear E fit; local multiplicative Hencky/von-Mises updates along measured nonuniform deformation histories',
        reconstruction='Low-order square-symmetric 3D displacement field fitted to one observed side; no uniform compression or plane-stress assumption',
        forward_simulation_calls=0,simulator_states_used=False)
    dest=root/f'field_{k}';dest.mkdir(exist_ok=False)
    np.savez(dest/'reconstructed.npz',x=X,v=V,F=F,q=q,time=d['time'],W=W,b=b)
    save(dest/'identification.json',best);save(dest/'search.json',tries);print(best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['select','fit']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],default='A')
    a=ap.parse_args();select(a.out) if a.stage=='select' else fit(a.out,a.material)

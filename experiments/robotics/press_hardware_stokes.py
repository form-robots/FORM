"""Minimum-dissipation axisymmetric interior as an explicit kinematic prior.

Uses a divergence-free streamfunction enrichment of cumulative-layer transport.
Observed boundary shape is prescribed, not material-point correspondence. The
Newtonian dissipation principle is a reconstruction prior, not the fitted law.
"""
from pathlib import Path
import argparse
import shutil
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import comb
from experiments.robotics.press_hardware_refine import poly
from experiments.robotics.press_hardware_moving_tests import fields
from experiments.robotics.press_hardware_observe import read,save

S=Polynomial([0,1]);ONE=Polynomial([1,-1])
FNS=[S*ONE*comb(3,j)*S**j*ONE**(3-j) for j in range(4)]
GNS=[S**k*(1-S) for k in range(3)]


def velocity(r,z,h,hd,c,cd,volume):
    """Base/enrichment velocity and gradient at azimuth zero; exact div=0."""
    s=z/h;a=poly(c);ad=poly(cd);A=a.antiderivative();Ad=ad.antiderivative()
    av=a(s);ap=a.derivative()(s);app=a.derivative(2)(s);avd=ad(s);apd=ad.derivative()(s)
    total=float(np.mean(c));td=float(np.mean(cd));C=(A(s)-A(0))/total
    B=Ad(s)-Ad(0)-C*td;Bp=avd-av/total*td;Bpp=apd-ap/total*td
    wz=hd/h-Bp/av+B*ap/av**2
    wzz=(-Bpp/av+2*Bp*ap/av**2+B*app/av**2-2*B*ap**2/av**3)/h
    w=hd*s-h*B/av
    v=np.c_[-r*wz/2,np.zeros(len(r)),w]
    L=np.zeros((len(r),3,3));L[:,0,0]=L[:,1,1]=-wz/2;L[:,2,2]=wz;L[:,0,2]=-r*wzz/2
    radius2=volume/(np.pi*h*total)*av;q=r*r/radius2;b=ap/av;bp=app/av-b*b
    vs=[];Ls=[]
    for fp in FNS:
        f=fp(s);fs=fp.deriv()(s);fss=fp.deriv(2)(s)
        for gp in GNS:
            g=gp(q);gq=gp.deriv()(q);gqq=gp.deriv(2)(q)
            D=fs*g-f*q*b*gq
            vr=-r*D;vz=2*h*f*(g+q*gq)
            grad=np.zeros_like(L)
            grad[:,0,0]=-D-2*q*(fs*gq-f*b*(gq+q*gqq))
            grad[:,1,1]=-D
            grad[:,2,2]=2*(fs*(g+q*gq)-f*q*b*(2*gq+q*gqq))
            grad[:,2,0]=4*h*f*r/radius2*(2*gq+q*gqq)
            grad[:,0,2]=-r/h*(fss*g-2*fs*q*b*gq+f*q*(b*b-bp)*gq+f*q*q*b*b*gqq)
            vs.append(np.c_[vr,np.zeros(len(r)),vz]);Ls.append(grad)
    return v,L,np.stack(vs,-1),np.stack(Ls,-1)


def dissipation_coefficients(h,hd,c,cd,volume,slip):
    zz,wz=np.polynomial.legendre.leggauss(24);rr,wr=np.polynomial.legendre.leggauss(5)
    s,u=np.meshgrid((zz+1)/2,(rr+1)/2,indexing='ij');s=s.ravel();u=u.ravel()
    radius=np.sqrt(volume/(np.pi*h*np.mean(c))*poly(c)(s))
    weights=((wz[:,None]/2)*(poly(c)(((zz+1)/2))[:,None]/np.mean(c))*(wr[None,:]*((rr+1)/2))).ravel()
    v,L,V,G=velocity(u*radius,h*s,h,hd,c,cd,volume)
    D=(L+L.swapaxes(1,2))/2;DG=(G+G.swapaxes(1,2))/2
    A=(DG*np.sqrt(weights)[:,None,None,None]).reshape(-1,12);target=-(D*np.sqrt(weights)[:,None,None]).ravel()
    if np.isfinite(slip):
        s=np.repeat([0.,1.],len(rr));u=np.tile((rr+1)/2,2)
        radius=np.sqrt(volume/(np.pi*h*np.mean(c))*poly(c)(s))
        vb,_,VB,_=velocity(u*radius,h*s,h,hd,c,cd,volume)
        # Slip penalty integrates contact radial speed squared over disk area.
        weight=np.tile(wr*((rr+1)/2),2)*np.pi*radius**2/(volume*slip)
        A=np.vstack([A,VB[:,0,:]*np.sqrt(weight)[:,None]])
        target=np.r_[target,-vb[:,0]*np.sqrt(weight)]
    lhs=A.T@A;regularization=1e-8*np.trace(lhs)/12
    coef=np.linalg.solve(lhs+regularization*np.eye(12),A.T@target)
    ratio=float(np.linalg.norm(A@coef-target)/max(np.linalg.norm(target),1e-12))
    return coef,ratio


def flow(q,s,h,hd,c,cd,volume,co):
    radius=np.sqrt(volume/(np.pi*h*np.mean(c))*poly(c)(s));r=q*radius
    v,L,V,G=velocity(r,h*s,h,hd,c,cd,volume)
    dv=V@co;v+=dv;L+=G@co
    # Dimensionless coordinates keep the moving boundaries explicit.
    a=poly(c);ad=poly(cd);A=a.antiderivative();Ad=ad.antiderivative()
    C=(A(s)-A(0))/np.mean(c);base_sd=-(Ad(s)-Ad(0)-C*np.mean(cd))/a(s)
    sd=base_sd+dv[:,2]/h
    qd=(dv[:,0]-q*radius*a.derivative()(s)/(2*a(s))*dv[:,2]/h)/radius
    return qd,sd,v,L,np.c_[r,np.zeros(len(r)),h*s]


def reconstruct(source,out,episode,slip=.001,substeps=10):
    d=dict(np.load(source/episode/'motion.npz'));geo=read(source/episode/'geometry.json');p=read(source/'protocol.json')
    t=d['time'];h=d['height'];cs=d['profile_coefficients'];hd=np.gradient(h,t,edge_order=2);cd=np.gradient(cs,t,axis=0,edge_order=2)
    volume=geo['volume_m3'];c0=np.array(geo['profile_coefficients0'])
    coeff=[];ratios=[]
    for hh,hh_d,c,c_d in zip(h,hd,cs,cd):
        co,ratio=dissipation_coefficients(hh,hh_d,c,c_d,volume,slip);coeff.append(co);ratios.append(ratio)
    coeff=np.array(coeff)
    zz,wz=np.polynomial.legendre.leggauss(24);rr,wr=np.polynomial.legendre.leggauss(4)
    s,q=np.meshgrid((zz+1)/2,(rr+1)/2,indexing='ij')
    weights=((wz[:,None]/2)*poly(c0)(s)/np.mean(c0)*(wr[None,:]*q)).ravel()
    s=s.ravel();q=q.ravel();F=np.tile(np.eye(3),(len(q),1,1));Xs=[];Fs=[];Vs=[];max_j=0.
    def at(i,alpha):
        return [(1-alpha)*a[i]+alpha*a[i+1] for a in [h,hd,cs,cd,coeff]]
    for i in range(len(t)):
        _,_,v,_,x=flow(q,s,h[i],hd[i],cs[i],cd[i],volume,coeff[i]);Xs.append(x);Fs.append(F.copy());Vs.append(v)
        if i==len(t)-1:break
        dt=(t[i+1]-t[i])/substeps
        for j in range(substeps):
            hh,hh_d,c,c_d,co=at(i,j/substeps)
            qd,sd,_,_,_=flow(q,s,hh,hh_d,c,c_d,volume,co)
            qm=q+dt/2*qd;sm=s+dt/2*sd
            hh,hh_d,c,c_d,co=at(i,(j+.5)/substeps)
            qd,sd,_,L,_=flow(qm,sm,hh,hh_d,c,c_d,volume,co)
            q+=dt*qd;s+=dt*sd
            if not (q.min()>0 and q.max()<1.001 and s.min()>0 and s.max()<1):
                raise RuntimeError((episode,i,j,'particle left reconstructed domain',q.min(),q.max(),s.min(),s.max()))
            # Second-order Cayley update; report its tiny J drift before enforcing
            # the exact incompressibility prior on the numerical deformation map.
            inc=np.linalg.solve(np.eye(3)-dt*L/2,np.eye(3)+dt*L/2)
            raw_J=np.linalg.det(inc);max_j=max(max_j,float(abs(raw_J-1).max()))
            inc/=np.cbrt(raw_J)[:,None,None];F=inc@F
    x=np.array(Xs);F=np.array(Fs);v=np.array(Vs)
    w,g,wt=fields(x,h,hd);rho=p['density_kg_m3']
    M=volume*rho*np.einsum('tni,tmni,n->tm',v,w,weights)
    C=volume*rho*(np.einsum('tni,tmnij,tnj,n->tm',v,g,v,weights)+np.einsum('tni,tmni,n->tm',v,wt,weights))
    body=-9.81*volume*rho*np.einsum('tmn,n->tm',w[...,2],weights)
    target=[]
    for begin in d['window_begin']:
        u=np.clip((t-begin)/.6,0,1);chi=np.sin(np.pi*u)**4;dc=4*np.pi/.6*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        target.append(np.sum(chi[:,None]*(C+body-d['force'][:,None])+dc[:,None]*M,axis=0)*p['export_dt_s'])
    d.update(x=x,F=F,velocity=v,weights=weights,test_gradient=g,target=np.array(target),stokes_coefficients=coeff)
    dest=out/episode;dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'motion.npz',**d)
    for name in ['geometry.json','load_only.npz']:shutil.copy2(source/episode/name,dest/name)
    save(dest/'stokes_audit.json',dict(slip_length_m=slip,substeps=substeps,basis_count=12,
        max_increment_J_error_before_projection=max_j,final_J_error=float(abs(np.linalg.det(F)-1).max()),
        mean_dissipation_norm_ratio=float(np.mean(ratios)),min_q=float(q.min()),max_q=float(q.max()),min_s=float(s.min()),max_s=float(s.max()),
        scope='Minimum-dissipation kinematic reconstruction with assumed contact slip penalty; not observed particle identities or identified viscosity'))
    print(episode,'Stokes complete','max step J drift',max_j,'mean dissipation ratio',np.mean(ratios),flush=True)


def check(out):
    c=np.array([.06,.7,1.1,.8,.04])*.02**2;cd=np.array([.1,-.03,.02,-.02,.07])*.02**2
    h=.04;hd=-.002;vol=np.pi*h*np.mean(c);r=np.array([.003,.008,.01]);z=np.array([.006,.017,.032]);eps=1e-7
    v,L,V,G=velocity(r,z,h,hd,c,cd,vol)
    errors=[]
    for dim in [0,2]:
        vp,_,Vp,_=velocity(r+(eps if dim==0 else 0),z+(eps if dim==2 else 0),h,hd,c,cd,vol)
        vm,_,Vm,_=velocity(r-(eps if dim==0 else 0),z-(eps if dim==2 else 0),h,hd,c,cd,vol)
        errors.extend([float(abs((vp-vm)/(2*eps)-L[:,:,dim]).max()),float(abs((Vp-Vm)/(2*eps)-G[:,:,dim,:]).max())])
    div=float(abs(np.trace(G,axis1=1,axis2=2)).max());assert div<1e-12 and max(errors)<1e-6
    s=np.linspace(.01,.99,30);R=np.sqrt(vol/(np.pi*h*np.mean(c))*poly(c)(s))
    _,_,VV,_=velocity(R,h*s,h,hd,c,cd,vol)
    normal=float(abs(VV[:,0,:]-R[:,None]*poly(c).derivative()(s)[:,None]/(2*poly(c)(s)[:,None]*h)*VV[:,2,:]).max())
    assert normal<1e-12
    save(out/'stokes_checks.json',dict(status='passed',gradient_errors=errors,enrichment_divergence=div,surface_normal_perturbation=normal))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--episode');a.add_argument('--slip',type=float,default=.001);a.add_argument('--substeps',type=int,default=10)
    args=a.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    p=read(args.source/'protocol.json');p.update(interior='Axisymmetric minimum-dissipation streamfunction reconstruction',stokes_slip_length_m=args.slip)
    save(args.out/'protocol.json',p);check(args.out)
    for ep in ([args.episode] if args.episode else [f'ep{i:04}' for i in range(12)]):reconstruct(args.source,args.out,ep,args.slip,args.substeps)

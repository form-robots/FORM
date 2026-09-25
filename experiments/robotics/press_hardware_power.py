"""Known-family nonlinear-overstress candidate fitted along fixed motion.

Plastic log-strain rate norm = max(||dev(tau)||-Y,0)^m / ((2G)^m*T).
T is a strain-normalized flow-time parameter, not a constant relaxation time
unless m=1. E remains a conditional linear solve for each Y/E,T,m candidate.
"""
from pathlib import Path
import argparse
import numpy as np
import warp as wp
from scipy.optimize import minimize
from experiments.robotics.press_hardware_observe import read,save


@wp.kernel
def relax_power(Ft:wp.array(dtype=wp.mat33),Fe:wp.array(dtype=wp.mat33),
                rounding:wp.array(dtype=wp.mat33),threshold:float,dt_over_T:float,power:float):
    i=wp.tid();U=wp.mat33(0.0);V=wp.mat33(0.0);s=wp.vec3(0.0)
    wp.svd3(Ft[i],U,s,V)
    e=wp.vec3(wp.log(wp.max(s[0],1e-8)),wp.log(wp.max(s[1],1e-8)),wp.log(wp.max(s[2],1e-8)))
    mean=(e[0]+e[1]+e[2])/3.;dev=e-wp.vec3(mean,mean,mean);norm=wp.length(dev)
    if norm>threshold:
        excess=norm-threshold;b=dt_over_T*wp.pow(excess,power-1.)
        lo=float(0.);hi=float(1.);z=wp.min(1.,wp.pow(wp.max(b,1e-30),-1./power))
        for k in range(16):
            f=z+b*wp.pow(z,power)-1.
            if f<0.:lo=z
            if f>0.:hi=z
            proposal=z-f/(1.+b*power*wp.pow(wp.max(z,1e-30),power-1.))
            if proposal>=lo and proposal<=hi:z=proposal
            else:z=(lo+hi)/2.
        de=-(excess-excess*z)/norm*dev;ds=wp.vec3(0.)
        for j in range(3):
            a=de[j]
            if wp.abs(a)<.01:ds[j]=s[j]*(a+a*a/2.+a*a*a/6.+a*a*a*a/24.)
            else:ds[j]=s[j]*(wp.exp(a)-1.)
        correction=U*wp.mat33(ds[0],0.,0.,0.,ds[1],0.,0.,0.,ds[2])*wp.transpose(V)
        old=Ft[i];increment=correction-rounding[i];result=old+increment
        rounding[i]=(result-old)-increment;Ft[i]=result;Fe[i]=result


def remaining_excess(excess,dt_over_T,power):
    excess=np.asarray(excess);b=dt_over_T*np.maximum(excess,1e-30)**(power-1)
    lo=np.zeros_like(b);hi=np.ones_like(b);z=np.minimum(1,np.maximum(b,1e-300)**(-1/power))
    for _ in range(16):
        f=z+b*z**power-1;lo=np.where(f<0,z,lo);hi=np.where(f>0,z,hi)
        proposal=z-f/(1+b*power*np.maximum(z,1e-300)**(power-1))
        z=np.where((proposal>=lo)&(proposal<=hi),proposal,(lo+hi)/2)
    return excess*z


def stress_history(F,ratio,T,power,nu,dt):
    mu=1/(2*(1+nu));Fe=np.tile(np.eye(3),(F.shape[1],1,1));old=Fe.copy();out=[]
    for f in F:
        trial=f@np.linalg.inv(old)@Fe;U,s,Vh=np.linalg.svd(trial)
        eps=np.log(s);mean=eps.mean(1);dev=eps-mean[:,None];norm=np.linalg.norm(dev,axis=1)
        excess=np.maximum(norm-ratio/(2*mu),0);delta=excess-remaining_excess(excess,dt/T,power)
        dev*= (1-delta/np.maximum(norm,1e-20))[:,None]
        Fe=(U*np.exp(mean[:,None]+dev)[:,None,:])@Vh
        out.append((U*(2*mu*dev)[:,None,:])@U.transpose(0,2,1));old=f
    return np.array(out)


def fit(out,material):
    p=read(out/'protocol.json');data=[];records=[]
    for ep in p['train'][material]:
        d=dict(np.load(out/ep/'motion.npz'));d['volume']=read(out/ep/'geometry.json')['volume_m3'];data.append(d)
    def evaluate(v):
        ratio,T,power=np.exp(v);AA=[];bb=[]
        for d in data:
            stress=stress_history(d['F'],ratio,T,power,p['nu'],p['export_dt_s'])
            A=d['windows']@(d['volume']*np.einsum('tnij,tmnij,n->tm',stress,d['test_gradient'],d['weights']))
            b=d['target'];keep=(d['window_begin']<10)|(d['window_begin']>=10.8);scale=np.linalg.norm(b[keep])
            AA.append(A[keep].ravel()/scale);bb.append(b[keep].ravel()/scale)
        A=np.concatenate(AA);b=np.concatenate(bb);E=float(np.clip(A@b/(A@A),1000,2e6));loss=float(np.linalg.norm(E*A-b)**2/len(data))
        records.append(dict(E_Pa=E,Y_Pa=E*ratio,ratio=ratio,tau_s=T,flow_power=power,
            law='Hencky/von Mises power overstress',relative_weak_residual=np.sqrt(loss),training_episodes=p['train'][material],
            flow_time_scope='T in plastic log-strain rate=(elastic overstress strain)^m/T; not constant relaxation time for m!=1'))
        return loss
    initial=read(out/material/'fit.json')['candidates'][1]
    candidates=[]
    for power in [.6,1.,2.]:
        start=np.log([initial['ratio'],initial['tau_s']*.2**(power-1),power])
        opt=minimize(evaluate,start,method='Nelder-Mead',bounds=[(np.log(1e-5),0.),(np.log(.002),np.log(300)),(np.log(.35),np.log(3.))],options=dict(maxiter=120,xatol=.01,fatol=1e-7))
        candidates.append(opt)
    evaluate(min(candidates,key=lambda a:a.fun).x)
    save(out/material/'power_fit.json',dict(candidate=records[-1],scope='Training-only nonlinear rate-law hypothesis; no automatic adoption',search=records))
    print(material,records[-1],flush=True)


def check(out):
    from experiments.robotics.press_hardware_model import stress_history as linear
    e=np.logspace(-12,1,40);residual=[]
    for power in [.35,.6,1.,2.,3.]:
        for a in [.0001,.1,25.]:
            q=remaining_excess(e,a,power);residual.append(float(np.max(abs(q+a*q**power-e)/e)))
    assert max(residual)<1e-7,max(residual)
    F=np.array([np.diag(np.exp(np.array([-.5,-.5,1])*v)) for v in [0.,-.1,-.2,-.21,-.15]])[:,None]
    error=float(abs(stress_history(F,.02,2.,1.,.45,.05)-linear(F,.02,2.,.45,.05)).max());assert error<1e-12
    save(out/'power_checks.json',dict(status='passed',implicit_root_max_relative_residual=max(residual),linear_limit_stress_error=error))


def check_kernel(out):
    wp.init();records=[]
    for device in ['cpu','cuda:0']:
        for power in [.45,1.,2.]:
            dev=np.array([.2,-.1,-.1]);norm=np.linalg.norm(dev);threshold=.03;factor=.05/8
            new_norm=threshold+remaining_excess(np.array([norm-threshold]),factor,power)[0]
            expected=np.diag(np.exp(dev*new_norm/norm));f0=np.diag(np.exp(dev))[None].astype('float32')
            a=wp.array(f0,dtype=wp.mat33,device=device);b=wp.array(f0,dtype=wp.mat33,device=device);r=wp.zeros(1,dtype=wp.mat33,device=device)
            wp.launch(relax_power,dim=1,inputs=[a,b,r,threshold,factor,power],device=device)
            error=float(abs(a.numpy()[0]-expected).max());assert error<2e-6
            records.append(dict(device=device,power=power,error=error))
    save(out/'power_kernel_checks.json',dict(status='passed',checks=records));print(records,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,required=True);a.add_argument('--material',default='butter_slime');a.add_argument('--check-kernel',action='store_true');args=a.parse_args();check(args.out)
    if args.check_kernel:check_kernel(args.out)
    else:fit(args.out,args.material)

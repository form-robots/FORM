"""Spacetime weak tests spanning the changing specimen height.

These prescribed test fields are divergence free and contact-flat at each time.
Their explicit time derivative must be included in the weak momentum equation.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np
from experiments.robotics.press_hardware_observe import read,save


def fields(x,h,hdot):
    h=np.asarray(h)[:,None];hdot=np.asarray(hdot)[:,None]
    ws=[];gs=[];wts=[]
    for lo,hi in [(.12,.70),(.2,.8),(.3,.88)]:
        width=(hi-lo)*h;u=np.clip((x[...,2]-lo*h)/width,0,1)
        f=u**3*(10-15*u+6*u*u);fu=30*u*u*(1-u)**2;fuu=60*u*(1-u)*(1-2*u)
        df=fu/width;ddf=fuu/width**2
        w=np.stack([-x[...,0]*df/2,-x[...,1]*df/2,f],-1)
        g=np.zeros((*x.shape[:-1],3,3));g[...,0,0]=g[...,1,1]=-df/2;g[...,2,2]=df
        g[...,0,2]=-x[...,0]*ddf/2;g[...,1,2]=-x[...,1]*ddf/2
        ut=-x[...,2]*hdot/(width*h)
        dft=fuu*ut/width-df*hdot/h
        wt=np.stack([-x[...,0]*dft/2,-x[...,1]*dft/2,fu*ut],-1)
        ws.append(w);gs.append(g);wts.append(wt)
    return np.stack(ws,1),np.stack(gs,1),np.stack(wts,1)


def prepare(source,out):
    out.mkdir(exist_ok=False,parents=True);p=read(source/'protocol.json')
    p['weak_test_scope']='Divergence-free fields adapt to reconstructed height; explicit partial-time term included; spatial gradient covers current body rather than fixed bottom region'
    p['reconstruction_source']=str(source.resolve());save(out/'protocol.json',p)
    for i in range(12):
        ep=f'ep{i:04}';dest=out/ep;dest.mkdir()
        for name in ['geometry.json','load_only.npz']:
            shutil.copy2(source/ep/name,dest/name)
        d=dict(np.load(source/ep/'motion.npz'));geo=read(dest/'geometry.json')
        t=d['time'];x=d['x'];v=d['velocity'];h=d['height'];volume=geo['volume_m3'];rho=p['density_kg_m3'];weights=d['weights']
        w,g,wt=fields(x,h,np.gradient(h,t,edge_order=2))
        M=volume*rho*np.einsum('tni,tmni,n->tm',v,w,weights)
        C=volume*rho*(np.einsum('tni,tmnij,tnj,n->tm',v,g,v,weights)+np.einsum('tni,tmni,n->tm',v,wt,weights))
        body=-9.81*volume*rho*np.einsum('tmn,n->tm',w[...,2],weights)
        target=[]
        for begin in d['window_begin']:
            u=np.clip((t-begin)/.6,0,1);chi=np.sin(np.pi*u)**4;dc=4*np.pi/.6*np.sin(np.pi*u)**3*np.cos(np.pi*u)
            target.append(np.sum(chi[:,None]*(C+body-d['force'][:,None])+dc[:,None]*M,axis=0)*p['export_dt_s'])
        d.update(test_gradient=g,target=np.asarray(target))
        np.savez_compressed(dest/'motion.npz',**d)
    check(out)


def check(out):
    x=np.array([[[.012,.007,.013],[.007,.003,0],[.01,.008,.04]]]);h=np.array([.04]);hd=np.array([-.004]);eps=1e-6
    w,g,wt=fields(x,h,hd);wp=fields(x,h+eps*hd,hd)[0];wm=fields(x,h-eps*hd,hd)[0]
    error=float(abs((wp-wm)/(2*eps)-wt).max());assert error<1e-8
    divergence=float(abs(np.trace(g,axis1=-2,axis2=-1)).max());assert divergence<1e-12
    assert np.allclose(w[:,:,1],0) and np.allclose(w[:,:,2],[0,0,1])
    # Material derivative under homogeneous incompressible compression:
    # compare d/dt w(x(t),t) with partial_t w + grad(w) velocity.
    v=x*np.array([-.5*hd[0]/h[0],-.5*hd[0]/h[0],hd[0]/h[0]])
    plus=fields(x+eps*v,h+eps*hd,hd)[0];minus=fields(x-eps*v,h-eps*hd,hd)[0]
    material_error=float(abs((plus-minus)/(2*eps)-wt-np.einsum('tmnij,tnj->tmni',g,v)).max());assert material_error<1e-8
    save(out/'moving_test_checks.json',dict(status='passed',partial_time_derivative_error=error,material_derivative_error=material_error,divergence=divergence))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();prepare(args.source,args.out)

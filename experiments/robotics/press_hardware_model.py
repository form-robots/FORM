"""Observable-only axisymmetric reconstruction and known-family weak-form fits.

The interior closure transports equal cumulative-volume layers and scales each
cross-section radially. It assumes axisymmetry and constant volume, but does not
claim observed material identities or impose no slip. No forward dynamics in fit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from experiments.robotics.press_hardware_observe import read, save, segment

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'out/press_observation_assessment_20260913'
RAW = ROOT / 'press_real_data'
MATERIALS = ['play_doh', 'butter_slime', 'plasticine']


def protocol(out):
    out.mkdir(exist_ok=False, parents=True)
    p = dict(materials=MATERIALS, train={m:[f'ep{4*i+j:04}' for j in [0,1,3]] for i,m in enumerate(MATERIALS)},
        heldout={m:f'ep{4*i+2:04}' for i,m in enumerate(MATERIALS)},
        source=str(RAW), observations=str(BASE), density_kg_m3=1000., nu=.45,
        reference_ep='ep0001', reference_height_m=.047, hand_platform_pixel_y=267.,
        shape_family='R(s)^2 = Rmid^2 * (1 - k*(2*s-1)^2), 0<=s<=1; k from visible RGB widths',
        interior='Equal cumulative-volume layer transport plus radial section scaling; axisymmetric, volume preserving; correspondence assumption',
        reference_state='First force-baseline state treated as elastically stress-free; unknown real preload/history remains a sensitivity',
        fit_window_s=[.4,13.4], export_dt_s=.05, physical_preload_N=0.,
        comparison_duration_s=13.4,primary_prediction_tag='validation',bulk_modulus_assumed_Pa=1e6,
        bulk_scope='Separate numerical near-incompressibility assumption; pressure-free weak fit cannot identify bulk modulus. Keep fitted G,Y,tau fixed in bulk checks.',
        fit_phase_balance='Equal episode relative residual with one force scale across loading and unloading. Low-load rows are not divided by their own small force.',
        diagnostic_screen=dict(gap_rmse_mm=2.,width_rmse_mm=3.,late_hold_force_relative_l2=.05,
            scope='Provisional engineering screen, not statistical proof of model identification; geometry/contact uncertainty remains'),
        law_candidates=['Hencky/perfect von Mises','Hencky/von Mises with linear overstress relaxation'],
        yield_convention='Y is the Frobenius norm of deviatoric Kirchhoff stress at yield, matching the repository metal law',
        selection='Training episodes only; final 21 N predictions excluded from parameter/model selection',
        geometry_status='Conditional metric reconstruction; no unvalidated stereo correction applied',
        contact='Slight stickiness user-confirmed; adhesion/friction unmeasured. Weak test fields eliminate tangential contact work.',
        assumptions=['47 mm initial reference height based on supplied approximate size, not independent caliper measurement',
                     'Same tool/platform offset across episodes; per-episode height variation from initial EEF z',
                     'Density and Poisson ratio fixed assumptions, not identified',
                     'No constitutive-family discovery or jointly convex plastic-history solve'])
    save(out/'protocol.json',p)


def cumulative(s,k):
    return (s-k*(4*s**3/3-2*s*s+s))/(1-k/3)


def geometry(h,k,volume,s0,k0,h0,rho):
    """Current positions and exact F at reference quadrature points (azimuth=0)."""
    q=cumulative(s0,k0)
    s=q.copy()
    for _ in range(12):
        dq=(1-k*(2*s-1)**2)/(1-k/3)
        s=np.clip(s-(cumulative(s,k)-q)/dq,1e-8,1-1e-8)
    Rm=np.sqrt(volume/(np.pi*h*(1-k/3)))
    Rm0=np.sqrt(volume/(np.pi*h0*(1-k0/3)))
    g=np.sqrt(1-k*(2*s-1)**2);g0=np.sqrt(1-k0*(2*s0-1)**2)
    R=Rm*g;R0=Rm0*g0
    dR=-2*Rm*k*(2*s-1)/g;dR0=-2*Rm0*k0*(2*s0-1)/g0
    ds=(g0*g0/(1-k0/3))/(g*g/(1-k/3))
    stretch=R/R0
    F=np.zeros((len(s0),3,3));F[:,0,0]=F[:,1,1]=stretch
    F[:,0,2]=rho*(dR*ds-stretch*dR0)/h0
    F[:,2,2]=h/h0*ds
    x=np.column_stack([rho*R,np.zeros(len(s)),h*s])
    return x,F


def quadrature(k0, nz=24,nr=4):
    z,wz=np.polynomial.legendre.leggauss(nz);r,wr=np.polynomial.legendre.leggauss(nr)
    s,rho=np.meshgrid((z+1)/2,(r+1)/2,indexing='ij')
    weight=(wz[:,None]/2)*(1-k0*(2*s-1)**2)/(1-k0/3)*(wr[None,:]*rho)
    return s.ravel(),rho.ravel(),weight.ravel()


def virtual(x,h_min):
    fields=[];grads=[]
    for lo,hi in [(.12,.70),(.2,.8),(.3,.88)]:
        width=(hi-lo)*h_min;s=np.clip((x[...,2]-lo*h_min)/width,0,1)
        f=s**3*(10-15*s+6*s*s);df=30*s*s*(1-s)**2/width;ddf=60*s*(1-s)*(1-2*s)/width**2
        w=np.stack([-x[...,0]*df/2,-x[...,1]*df/2,f],-1)
        g=np.zeros((*x.shape[:-1],3,3));g[...,0,0]=g[...,1,1]=-df/2;g[...,2,2]=df
        g[...,0,2]=-x[...,0]*ddf/2;g[...,1,2]=-x[...,1]*ddf/2
        fields.append(w);grads.append(g)
    return np.stack(fields,1),np.stack(grads,1)


def prepare(out,episode):
    p=read(out/'protocol.json');a=read(BASE/episode/'assessment.json');meta=read(RAW/episode/'meta.json')
    ref=read(BASE/p['reference_ep']/'assessment.json');camera=meta['cameras']['hand'];intr=camera['intrinsics']
    h0=p['reference_height_m']+a['robot_z_baseline_m']-ref['robot_z_baseline_m']
    obs=np.load(BASE/episode/'hand_observations.npz')
    center=np.array(a['cameras']['hand']['initial_sphere']['center_base_m'])
    T=np.array(meta['extrinsics']['cameras']['hand']['T_base_cam']['matrix']);depth=((center-T[:3,3])@T[:3,:3])[2]
    scale=depth/intr['fx']
    cap=cv2.VideoCapture(str(RAW/episode/'hand_rgb.mp4'));rows=[]
    for idx,tt,plate in zip(obs['frame_idx'],obs['time'],obs['plate_white_edge_y_px']):
        if not -.15<=tt<=p['fit_window_s'][1]+.15:continue
        cap.set(1,int(idx));ok,image=cap.read();assert ok
        dep=cv2.imread(str(RAW/episode/f'hand_depth/{idx:06}.png'),-1)
        if a['material']=='plasticine':
            # Remove the nearby background stripe using the initial body depth.
            # This is a segmentation cue, not a material-point correspondence.
            dep=dep.copy();dep[dep*camera['depth_scale']>depth+.006]=0
        mask,_=segment(image,dep,'hand',a['material'])
        if a['material']=='plasticine':
            yy0=int((plate+267)/2)
            yy,xx=np.nonzero(mask[max(0,yy0-5):yy0+6])
            if len(xx):
                cx=float(np.median(xx));rx=max(25.,float(np.percentile(xx,95)-np.percentile(xx,5))/2+5)
                cy=(plate+10+267)/2;ry=max(8.,(267-plate-10)/2)
                gy,gx=np.mgrid[:480,:848];ellipse=((gx-cx)/rx)**2+((gy-cy)/ry)**2
                labels=np.where(ellipse<1.4,cv2.GC_PR_FGD,cv2.GC_BGD).astype('uint8')
                labels[ellipse<.4]=cv2.GC_FGD
                labels[(gy<plate+10)|(gy>267)]=cv2.GC_BGD
                cv2.grabCut(image,labels,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
                mask=np.uint8((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD))
        floor=p['hand_platform_pixel_y']; height_px=floor-plate
        yy=np.arange(mask.shape[0]); s=(floor-yy)/max(height_px,1)
        choose=(s>.15)&(s<.85)&(mask.sum(1)>12)
        widths=[];coords=[]
        for y in yy[choose]:
            xx=np.flatnonzero(mask[y]);widths.append((xx[-1]-xx[0])*scale/2);coords.append(s[y])
        if len(widths)>=7:
            B=np.column_stack([np.ones(len(coords)),-(2*np.array(coords)-1)**2])
            coef=np.linalg.lstsq(B,np.square(widths),rcond=None)[0]
            k=float(np.clip(coef[1]/max(coef[0],1e-9),-.5,.995))
            rm=float(np.sqrt(max(coef[0],1e-8)))
            err=float(np.sqrt(np.mean((np.sqrt(np.maximum(B@np.array([rm*rm,rm*rm*k]),1e-12))-widths)**2)))
            rows.append([tt,k,rm,err,len(widths)])
    cap.release();rows=np.array(rows)
    assemble(out,episode,rows,h0,scale,depth)


def assemble(out,episode,rows,h0,scale,depth):
    p=read(out/'protocol.json');a=read(BASE/episode/'assessment.json')
    robot=np.load(BASE/episode/'robot_force.npz')
    times=np.arange(0,p['fit_window_s'][1]+.0001,p['export_dt_s'])
    shape_k=savgol_filter(np.interp(times,rows[:,0],rows[:,1]),21,3)
    shape_k=np.clip(shape_k,-.5,.995)
    # The initial state uses a short baseline average, not the contact transient.
    k0=float(np.median(rows[(rows[:,0]>=-.15)&(rows[:,0]<=.2),1]));shape_k[times<=.15]=k0
    rm0=float(np.median(rows[(rows[:,0]>=-.15)&(rows[:,0]<=.2),2]))
    volume=np.pi*rm0**2*h0*(1-k0/3)
    raw_h=h0+np.interp(times,robot['time'],robot['ee_pos'][:,2])-a['robot_z_baseline_m']
    h=savgol_filter(raw_h,9,3);h[times<=.15]=h0
    assert h.min()>.003,(episode,h.min())
    ss,rr,weights=quadrature(k0)
    XF=[geometry(hh,kk,volume,ss,k0,h0,rr) for hh,kk in zip(h,shape_k)]
    x=np.array([v[0] for v in XF]);F=np.array([v[1] for v in XF]);assert abs(np.linalg.det(F)-1).max()<1e-7
    vel=np.gradient(x,times,axis=0,edge_order=2)
    force=np.interp(times,robot['time'],robot['incremental_normal_force'])+p['physical_preload_N']
    w,g=virtual(x,float(h.min()))
    momentum=volume*p['density_kg_m3']*np.einsum('tni,tmni,n->tm',vel,w,weights)
    convective=volume*p['density_kg_m3']*np.einsum('tni,tmnij,tnj,n->tm',vel,g,vel,weights)
    body=-9.81*volume*p['density_kg_m3']*np.einsum('tmn,n->tm',w[...,2],weights)
    windows=[];target=[]
    begins=np.arange(.4,p['fit_window_s'][1]-.59,.2)
    for begin in begins:
        u=np.clip((times-begin)/.6,0,1);chi=np.sin(np.pi*u)**4;dc=4*np.pi/.6*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        dt=p['export_dt_s'];windows.append(chi*dt)
        target.append(np.sum(chi[:,None]*(convective+body-force[:,None])+dc[:,None]*momentum,axis=0)*dt)
    dest=out/episode;dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'motion.npz',time=times,height=h,raw_height=raw_h,shape_k=shape_k,F=F,x=x,velocity=vel,
        weights=weights,force=force,test_gradient=g,windows=windows,target=target,window_begin=begins,raw_shape_samples=rows)
    np.savez_compressed(dest/'load_only.npz',time=times,force=force)
    info=dict(episode=episode,material=a['material'],h0_m=h0,volume_m3=volume,radius_mid0_m=rm0,k0=k0,
        camera_depth_m=float(depth),pixel_scale_m=float(scale),valid_shape_frames=len(rows),
        shape_profile_fit_rms_mm=float(np.sqrt(np.mean(rows[:,3]**2))*1000),min_J=float(np.linalg.det(F).min()),
        max_J=float(np.linalg.det(F).max()),initial_geometry_time_window_s=[-.15,.2],shape_scope='Low-order fit to visible middle silhouette; uncertain contacts extrapolated; constant-volume layer motion assumed',
        height_scope='47 mm reference-height assumption plus recorded EEF displacement; independent RGB agreement evaluated separately')
    save(dest/'geometry.json',info);print(info,flush=True)


def stress_history(F,ratio,tau,nu,dt):
    """Unit-E deviatoric Hencky stress; backward-Euler linear overstress flow.

    e_new = e_trial - max(0,e_trial-Y/(2mu))*dt/(tau+dt).
    tau=0 is the repository perfect-plastic limit. eta=2mu*tau.
    """
    mu=1/(2*(1+nu));Fe=np.tile(np.eye(3),(F.shape[1],1,1));old=Fe.copy();out=[]
    for f in F:
        trial=f@np.linalg.inv(old)@Fe;U,s,Vh=np.linalg.svd(trial);eps=np.log(s);mean=eps.mean(1);dev=eps-mean[:,None]
        norm=np.linalg.norm(dev,axis=1);relax=np.maximum(0,norm-ratio/(2*mu))*dt/(tau+dt)
        dev*= (1-relax/np.maximum(norm,1e-20))[:,None]
        Fe=(U*np.exp(mean[:,None]+dev)[:,None,:])@Vh
        out.append((U*(2*mu*dev)[:,None,:])@U.transpose(0,2,1));old=f
    return np.array(out)


def fit(out,material):
    p=read(out/'protocol.json');names=p['train'][material];data=[]
    for name in names:
        d=dict(np.load(out/name/'motion.npz'));d['volume']=read(out/name/'geometry.json')['volume_m3'];data.append(d)
    global_scale=np.sqrt(sum(np.linalg.norm(d['target'][(d['window_begin']<10.)|(d['window_begin']>=10.8)])**2 for d in data)/len(data))
    trials=[];start=time.monotonic()
    def evaluate(lr,lt):
        ratio=np.exp(lr);tau=0. if lt is None else np.exp(lt);As=[];bs=[]
        for d in data:
            stress=stress_history(d['F'],ratio,tau,p['nu'],p['export_dt_s'])
            instant=d['volume']*np.einsum('tnij,tmnij,n->tm',stress,d['test_gradient'],d['weights'])
            A=(d['windows']@instant);b=d['target'];begins=d['window_begin']
            select=(begins<10.)|(begins>=10.8)
            row_weight=np.ones(np.count_nonzero(select))
            if p.get('balance_loading_unloading',False):
                if p.get('fit_weighting')=='absolute_force':raise ValueError('Combined absolute/phase weighting has not been defined')
                unloading=begins[select]>=10.8
                row_weight[unloading]=np.sqrt(np.count_nonzero(~unloading)/np.count_nonzero(unloading))
            # Robot force uncertainty is in N, not relative to the instantaneous
            # load. Use one scale per episode so uncertain low-load unloading is
            # not amplified by dividing by its small force magnitude.
            scale=global_scale if p.get('fit_weighting')=='absolute_force' else np.linalg.norm(b[select]*row_weight[:,None])
            As.append((A[select]*row_weight[:,None]).ravel()/scale);bs.append((b[select]*row_weight[:,None]).ravel()/scale)
        A=np.concatenate(As);b=np.concatenate(bs);E=float(np.clip(A@b/(A@A),1000,2e6))
        loss=float(np.mean((E*A-b)**2)*len(b)/len(As))
        rec=dict(E_Pa=E,Y_Pa=E*ratio,ratio=ratio,tau_s=tau,eta_Pa_s=E/(1+p['nu'])*tau,
            relative_weak_residual=float(np.sqrt(loss)),training_episodes=names)
        trials.append(rec);return loss
    selected=[]
    ratios=np.logspace(-4,-.1,11)
    for viscous in [False,True]:
        tau_grid=np.logspace(-2,2,7) if viscous else [None]
        candidates=[]
        for ratio in ratios:
            for tau in tau_grid:
                score=evaluate(np.log(ratio),None if tau is None else np.log(tau))
                candidates.append((score,np.log(ratio),None if tau is None else np.log(tau)))
        _,lr,lt=min(candidates)
        if viscous:
            opt=minimize(lambda v:evaluate(v[0],v[1]),[lr,lt],method='Nelder-Mead',
                bounds=[(np.log(1e-5),np.log(1)),(np.log(.002),np.log(300))],options=dict(maxiter=70,xatol=.005,fatol=1e-7))
            evaluate(*opt.x)
        else:
            opt=minimize(lambda v:evaluate(v[0],None),[lr],method='Nelder-Mead',bounds=[(np.log(1e-5),np.log(1))],options=dict(maxiter=45,xatol=.005))
            evaluate(opt.x[0],None)
        rec=trials[-1].copy();rec['law']='Hencky/von Mises linear overstress' if viscous else 'Hencky/perfect von Mises';selected.append(rec)
        print(material,rec,flush=True)
    # Extra rate parameter must substantially improve the training weak balance.
    best=selected[1] if selected[1]['relative_weak_residual']<.85*selected[0]['relative_weak_residual'] else selected[0]
    dest=out/material;dest.mkdir(exist_ok=True)
    save(dest/'fit.json',dict(selected=best,candidates=selected,fit_wall_s=time.monotonic()-start,
        fit_weighting=p.get('fit_weighting','equal_relative_episode'),
        balance_loading_unloading=p.get('balance_loading_unloading',False),
        forward_calls_in_identification=0,nu_assumed=p['nu'],density_assumed=p['density_kg_m3'],
        fitted_variables=['E','Y']+(['relaxation_time_tau'] if best['tau_s'] else []),
        parameter_scope='Optimization variables, not proof of identifiability; assess sensitivity and independent predictions separately',
        estimator='Known-family ratio and relaxation-time search; conditional linear E; fixed observed/reconstructed motion; no forward dynamics',
        status='Conditional estimate pending withheld forward validation'))
    save(dest/'fit_search.json',trials)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['init','prepare','fit'])
    a.add_argument('--out',type=Path,required=True);a.add_argument('--episode');a.add_argument('--material',choices=MATERIALS);args=a.parse_args()
    cv2.setNumThreads(1)
    if args.stage=='init':protocol(args.out)
    elif args.stage=='prepare':
        for name in ([args.episode] if args.episode else [f'ep{i:04}' for i in range(12)]):prepare(args.out,name)
    else:fit(args.out,args.material)

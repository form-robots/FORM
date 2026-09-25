"""Dimension-anchored contour reconstruction without imposing constant volume.

The interior map transports normalized cumulative-volume layers. Its Jacobian
is the reconstructed global volume ratio, an explicit uniform-dilatation prior.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.optimize import lsq_linear, nnls, minimize
from scipy.signal import savgol_filter
from experiments.robotics.press_hardware_observe import read,save,segment
from experiments.robotics.press_hardware_geometry import fit_camera,initial_mask,BALL_DIAMETER,PAD_DIAMETER,ROOT,RAW,OBS
from experiments.robotics.press_hardware_refine import basis,poly
from experiments.robotics.press_hardware_moving_tests import fields
from experiments.robotics.press_hardware_model import stress_history


def init(out):
    out.mkdir(parents=True,exist_ok=False)
    p=read(ROOT/'out/press_hardware_validation_20260913/protocol.json')
    p.update(reference_height_m=BALL_DIAMETER,ball_diameter_m=BALL_DIAMETER,plate_radius_m=PAD_DIAMETER/2,
        shape_family='Measured initial 45 mm sphere; positive quartic visible profile during pressing',
        interior='Normalized cumulative-volume layer transport; J equals measured-profile global volume ratio, uniform dilatation prior',
        selection='Exploratory, 21 N excluded from material fitting; known-family weak reconstruction fit',
        assumptions=['Nominal supplied diameter used as initial tangent-contact gap; initial preload/contact flattening unresolved',
                     'Axisymmetric contour interpretation and uniform local volume ratio; camera material labels are not observed',
                     'No measured adhesion; density 1000 kg/m3 retained'],
        video_header='DIMENSION-CORRECTED EXPLORATORY PRESSES',diagnostic_screen=dict(gap_rmse_mm=2,width_rmse_mm=3,unloading_gap_rmse_mm=2))
    save(out/'protocol.json',p)


def prepare(out,ep):
    camera,rgb0=fit_camera(ep);dest=out/ep;dest.mkdir(exist_ok=False);save(dest/'camera.json',camera)
    a=read(OBS/ep/'assessment.json');meta=read(RAW/ep/'meta.json');obs=np.load(OBS/ep/'hand_observations.npz');robot=np.load(OBS/ep/'robot_force.npz')
    R=np.array(meta['extrinsics']['cameras']['hand']['T_base_cam']['matrix'])[:3,:3]
    origin=np.array(camera['camera_position_m']);center=np.array([.09,.09,.05+BALL_DIAMETER/2]);normal=center-origin;normal[2]=0;normal/=np.linalg.norm(normal)
    intr=camera['intrinsics'];cx,cy,fx,fy=[intr[k] for k in ['ppx','ppy','fx','fy']]
    center_px=camera['circle_center_px'][0];cap=cv2.VideoCapture(str(RAW/ep/'hand_rgb.mp4'))
    records=[];last=-100;D=np.diff(np.eye(5),n=2,axis=0);examples=[]
    for idx,t,edge in zip(obs['frame_idx'],obs['time'],obs['plate_white_edge_y_px']):
        if not -.15<=t<=13.55 or t-last<.085:continue
        last=t;cap.set(1,int(idx));ok,rgb=cap.read()
        if not ok:raise RuntimeError((ep,idx))
        depth=cv2.imread(str(RAW/ep/f'hand_depth/{idx:06}.png'),-1)
        mask=initial_mask(rgb,depth,edge,a['material'],center_px) if a['material']=='plasticine' else segment(rgb,depth,'hand',a['material'])[0]
        h=BALL_DIAMETER+np.interp(t,robot['time'],robot['ee_pos'][:,2])-a['robot_z_baseline_m']
        if h<=.003:raise RuntimeError((ep,t,h))
        uv=[]
        for y in range(int(edge)+8,273):
            xx=np.flatnonzero(mask[y]);xx=xx[(xx>center_px-90)&(xx<center_px+90)]
            if len(xx)>14:uv.extend([[xx[0],y],[xx[-1],y]])
        if len(uv)<14:continue
        uv=np.array(uv);rays=np.c_[(uv[:,0]-cx)/fx,(uv[:,1]-cy)/fy,np.ones(len(uv))]@R.T
        distance=(center-origin)@normal/(rays@normal);xyz=origin+distance[:,None]*rays
        z=xyz[:,2].reshape(-1,2).mean(1)-.05;radius=np.linalg.norm(np.diff(xyz.reshape(-1,2,3),axis=1)[:,0,:2],axis=1)/2
        s=z/h;sel=(s>.035)&(s<.965);s=s[sel];radius=radius[sel]
        if len(radius)<7:continue
        sc=max(radius)**2;B=basis(s)
        fit=lsq_linear(np.vstack([B,.03*D]),np.r_[radius**2/sc,np.zeros(3)],bounds=(0,3))
        coeff=fit.x*sc;error=np.sqrt(np.mean((np.sqrt(B@coeff)-radius)**2))
        records.append([t,h,float(np.percentile(radius,95)*2),error,*coeff])
        if any(abs(t-v)<.06 for v in [0,3,10,13]):
            cv2.polylines(rgb,[uv.reshape(-1,2,2)[:,0].astype(np.int32)],False,(0,255,0),1)
            examples.append(cv2.resize(rgb[130:285,340:580],(480,310)))
    cap.release();samples=np.array(records);np.savez_compressed(dest/'profile_observations.npz',samples=samples)
    if examples:cv2.imwrite(str(dest/'profiles.jpg'),np.concatenate(examples[:8],axis=1))
    assemble(out,ep,samples,camera)


def assemble(out,ep,samples,camera,baseline_correction=False):
    """Build motion after rejecting isolated, gross contour-width failures."""
    dest=out/ep;a=read(OBS/ep/'assessment.json');robot=np.load(OBS/ep/'robot_force.npz');fx=camera['intrinsics']['fx']
    keep=np.ones(len(samples),dtype=bool)
    rejected=[]
    for i in range(1,len(samples)-1):
        neighbors=samples[[i-1,i+1],2]
        # A disappearing/reattaching segmentation is not specimen deformation.
        # Require two agreeing neighboring widths, and preserve raw samples.
        if np.ptp(neighbors)<.003 and abs(samples[i,2]-np.mean(neighbors))>.004:
            keep[i]=False
            rejected.append(dict(time_s=float(samples[i,0]),width_mm=float(samples[i,2]*1000),neighbor_widths_mm=(neighbors*1000).tolist()))
    save(dest/'profile_filter.json',dict(rule='Reject isolated width discrepancy >4 mm when adjacent 0.1 s samples agree within 3 mm; interpolate between retained observations',rejected=rejected,raw_samples_preserved=True))
    samples=samples[keep]
    t=np.arange(0,13.40001,.05);h=BALL_DIAMETER+np.interp(t,robot['time'],robot['ee_pos'][:,2])-a['robot_z_baseline_m'];raw_h=h.copy();h=savgol_filter(h,9,3);h[t<=.15]=BALL_DIAMETER
    r=BALL_DIAMETER/2;c0=np.array([0,r*r,4*r*r/3,r*r,0]);cs=np.column_stack([np.interp(t,samples[:,0],samples[:,4+j]) for j in range(5)])
    cs=savgol_filter(cs,11,2,axis=0);cs=np.maximum(cs,0)
    if baseline_correction:
        # Sensitivity: use profile changes relative to unforced baseline, so
        # initial silhouette/template mismatch does not become plastic strain.
        # Absolute observed width remains unchanged for forward evaluation.
        baseline=np.median(samples[(samples[:,0]>=-.15)&(samples[:,0]<=.25),4:],axis=0)
        ss=np.linspace(.002,.998,101);BB=basis(ss)
        radial_offset=np.sqrt(BB@c0)-np.sqrt(BB@baseline)
        corrected=[]
        for cc in cs:
            radii=np.maximum(0,np.sqrt(BB@cc)+radial_offset)
            corrected.append(lsq_linear(BB,radii*radii,bounds=(0,np.inf),tol=1e-12).x)
        cs=np.array(corrected)
        save(dest/'baseline_profile_correction.json',dict(reference_coefficients=baseline.tolist(),rms_radial_correction_mm=float(np.sqrt(np.mean(radial_offset**2))*1000),max_radial_correction_mm=float(abs(radial_offset).max()*1000),scope='Kinematic sensitivity only: additive radius correction versus normalized height from unforced baseline to nominal sphere. Raw observed diameters remain the evaluation reference.'))
    cs[t<=.2]=c0
    zz,wz=np.polynomial.legendre.leggauss(24);rr,wr=np.polynomial.legendre.leggauss(4);s0,rho=np.meshgrid((zz+1)/2,(rr+1)/2,indexing='ij')
    weights=((wz[:,None]/2)*poly(c0)(s0)/np.mean(c0)*(wr[None,:]*rho)).ravel();s0=s0.ravel();rho=rho.ravel();x=[];F=[]
    for hh,c in zip(h,cs):
        aa=poly(c);aa0=poly(c0);A=aa.antiderivative();A0=aa0.antiderivative();q=(A0(s0)-A0(0))/np.mean(c0)
        lo=np.zeros_like(q);hi=np.ones_like(q)
        for _ in range(40):
            ss=(lo+hi)/2;err=(A(ss)-A(0))/np.mean(c)-q;lo=np.where(err<0,ss,lo);hi=np.where(err>=0,ss,hi)
        s=(lo+hi)/2;ds=aa0(s0)/np.mean(c0)/(aa(s)/np.mean(c));rad=np.sqrt(aa(s));rad0=np.sqrt(aa0(s0));stretch=rad/rad0
        f=np.zeros((len(s0),3,3));f[:,0,0]=f[:,1,1]=stretch;f[:,2,2]=hh/BALL_DIAMETER*ds
        f[:,0,2]=rho*(rad*aa.derivative()(s)/(2*aa(s))*ds-stretch*rad0*aa0.derivative()(s0)/(2*aa0(s0)))/BALL_DIAMETER
        F.append(f);x.append(np.c_[rho*rad,np.zeros(len(s)),hh*s])
    x=np.array(x);F=np.array(F);v=np.gradient(x,t,axis=0,edge_order=2);volume=4*np.pi*r**3/3;J=h*np.mean(cs,axis=1)/(BALL_DIAMETER*np.mean(c0));assert np.max(abs(np.linalg.det(F)-J[:,None]))<1e-8
    w,g,wt=fields(x,h,np.gradient(h,t,edge_order=2));axw=w.copy();axw[...,:2]=0;axg=g.copy();axg[...,:2,:]=0;axwt=wt.copy();axwt[...,:2]=0
    w=np.concatenate([w,axw],axis=1);g=np.concatenate([g,axg],axis=1);wt=np.concatenate([wt,axwt],axis=1)
    momentum=volume*1000*np.einsum('tni,tmni,n->tm',v,w,weights)
    convection=volume*1000*(np.einsum('tni,tmnij,tnj,n->tm',v,g,v,weights)+np.einsum('tni,tmni,n->tm',v,wt,weights))
    body=-9.81*volume*1000*np.einsum('tmn,n->tm',w[...,2],weights);force=np.interp(t,robot['time'],robot['incremental_normal_force']);begins=np.arange(.4,12.81,.2);windows=[];target=[]
    for begin in begins:
        u=np.clip((t-begin)/.6,0,1);chi=np.sin(np.pi*u)**4;dc=4*np.pi/.6*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        windows.append(chi*.05);target.append(np.sum(chi[:,None]*(convection+body-force[:,None])+dc[:,None]*momentum,axis=0)*.05)
    np.savez_compressed(dest/'motion.npz',time=t,height=h,raw_height=raw_h,F=F,x=x,velocity=v,weights=weights,force=force,test_gradient=g,windows=windows,target=target,window_begin=begins,profile_coefficients=cs,volume_ratio=J,observed_max_width=np.interp(t,samples[:,0],samples[:,2]))
    np.savez_compressed(dest/'load_only.npz',time=t,force=force)
    save(dest/'geometry.json',dict(h0_m=BALL_DIAMETER,volume_m3=volume,radius_mid0_m=r,k0=1.,profile_coefficients0=c0.tolist(),pixel_scale_m=camera['camera_center_of_ball_m'][2]/fx,camera_depth_m=camera['camera_center_of_ball_m'][2],camera=camera,scope='User-supplied sphere dimensions, tangent-contact initial geometry candidate'))
    save(dest/'reconstruction_audit.json',dict(profile_radial_rms_mm=float(np.sqrt(np.mean(samples[:,3]**2))*1000),J_at_10s=float(np.interp(10,t,J)),J_at_13s=float(np.interp(13,t,J)),min_J=float(J.min()),max_J=float(J.max()),max_J_map_error=float(np.max(abs(np.linalg.det(F)-J[:,None]))),scope='Geometry-derived volume ratio, including uncertain hidden contacts, nominal initial gap and axisymmetric interpretation; not direct density measurement'))
    print(ep,'J10/13',np.interp(10,t,J),np.interp(13,t,J),'profileerrmm',np.sqrt(np.mean(samples[:,3]**2))*1000,flush=True)


def fit(out,material,viscous=False,tests='all'):
    p=read(out/'protocol.json');data=[]
    for ep in p['train'][material]:
        d=dict(np.load(out/ep/'motion.npz'));d['volume']=read(out/ep/'geometry.json')['volume_m3'];data.append(d)
    trials=[]
    def evaluate(v):
        ratio=np.exp(v[0]);tau=np.exp(v[1]) if viscous else 0;As=[];bs=[]
        for d in data:
            # nu=-.5 makes the existing unit-E history a unit-G history.
            dev=stress_history(d['F'],ratio,tau,-.5,.05);bulk=np.log(np.linalg.det(d['F']))[...,None,None]*np.eye(3)
            A=[]
            for stress in [dev,bulk]:
                instant=d['volume']*np.einsum('tnij,tmnij,n->tm',stress,d['test_gradient'],d['weights']);A.append(d['windows']@instant)
            A=np.stack(A,axis=-1);b=d['target'];sel=(d['window_begin']<10)|(d['window_begin']>=10.8)
            cols=slice(None) if tests=='all' else (slice(0,3) if tests=='divfree' else slice(3,None))
            A=A[sel,cols];b=b[sel,cols];scale=np.linalg.norm(b);As.append(A.reshape(-1,2)/scale);bs.append(b.ravel()/scale)
        A=np.concatenate(As);b=np.concatenate(bs);values,res=nnls(A,b);G,K=values;loss=res/np.sqrt(len(data));rec=dict(G_Pa=float(G),K_Pa=float(K),Y_Pa=float(G*ratio),ratio_Y_G=float(ratio),tau_s=float(tau),relative_weak_residual=float(loss));trials.append(rec);return loss
    candidates=[]
    for r in np.logspace(-3,1,14):
        for tau in ([.1,1,10,100] if viscous else [None]):
            v=[np.log(r)]+([np.log(tau)] if viscous else []);candidates.append((evaluate(v),v))
    _,v=min(candidates,key=lambda v:v[0]);opt=minimize(evaluate,v,method='Nelder-Mead',bounds=[(np.log(1e-5),np.log(100))]+([(np.log(.002),np.log(300))] if viscous else []),options=dict(maxiter=90,xatol=.003,fatol=1e-6));evaluate(opt.x);best=trials[-1]
    G=best['G_Pa'];best.update(E_Pa=2*(1+p['nu'])*G,eta_Pa_s=2*G*best['tau_s'],training_episodes=p['train'][material],law='Compressible Hencky/von Mises '+('linear overstress' if viscous else 'perfect plasticity'))
    if tests=='divfree':
        best.pop('K_Pa');best['bulk_scope']='Not identifiable from divergence-free tests; forward bulk remains a declared assumption'
    dest=out/material;dest.mkdir(exist_ok=True);tag=('viscous' if viscous else 'perfect')+'_'+tests;save(dest/(tag+'_fit.json'),dict(selected=best,search=trials,scope='Joint nonnegative linear G,K solve at each Y/G and optional tau; fixed reconstructed motion, no forward dynamics; tests='+tests));print(material,tag,best,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['init','prepare','fit']);a.add_argument('--out',type=Path,required=True);a.add_argument('--episode');a.add_argument('--material');a.add_argument('--viscous',action='store_true');a.add_argument('--tests',choices=['all','axial','divfree'],default='all');args=a.parse_args()
    if args.stage=='init':init(args.out)
    elif args.stage=='prepare':
        for ep in ([args.episode] if args.episode else [f'ep{i:04}' for i in range(12)]):prepare(args.out,ep)
    else:fit(args.out,args.material,args.viscous,args.tests)

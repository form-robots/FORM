"""Exploratory hardware refinement: asymmetric silhouette profiles and audit.

The profile is observed; volume conservation and cumulative-layer material
correspondence remain explicit reconstruction assumptions. No forward solve is
used here. All original study outputs remain unchanged.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.optimize import lsq_linear, least_squares
from scipy.special import comb
from scipy.interpolate import BPoly
from scipy.signal import savgol_filter
from experiments.robotics.press_hardware_model import virtual
from experiments.robotics.press_hardware_observe import read, save, segment

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'out/press_hardware_validation_20260913'
OBS=ROOT/'out/press_observation_assessment_20260913'
RAW=ROOT/'press_real_data'
DEGREE=4


def basis(s):
    s=np.asarray(s)
    return np.stack([comb(DEGREE,j)*s**j*(1-s)**(DEGREE-j) for j in range(DEGREE+1)],-1)


def poly(c):
    return BPoly(np.asarray(c)[:,None],[0.,1.])


def curve_geometry(h,c,c0,h0,s0,rho,volume):
    """Exact incompressible layer map for positive Bernstein area profiles."""
    a=poly(c);a0=poly(c0);A=a.antiderivative();A0=a0.antiderivative()
    total=float(A(1)-A(0));total0=float(A0(1)-A0(0))
    q=(A0(s0)-A0(0))/total0
    s=s0.copy()
    lo=np.zeros_like(s);hi=np.ones_like(s)
    for _ in range(40):
        error=(A(s)-A(0))/total-q
        lo=np.where(error<0,s,lo);hi=np.where(error>=0,s,hi)
        proposal=s-error*total/a(s)
        s=np.where((proposal>lo)&(proposal<hi),proposal,(lo+hi)/2)
    ds=a0(s0)/total0/(a(s)/total)
    r=np.sqrt(volume/(np.pi*h*total)*a(s));r0=np.sqrt(volume/(np.pi*h0*total0)*a0(s0))
    dr=r*a.derivative()(s)/(2*a(s));dr0=r0*a0.derivative()(s0)/(2*a0(s0))
    stretch=r/r0
    F=np.zeros((len(s0),3,3));F[:,0,0]=F[:,1,1]=stretch
    F[:,0,2]=rho*(dr*ds-stretch*dr0)/h0;F[:,2,2]=h/h0*ds
    x=np.column_stack([rho*r,np.zeros_like(r),h*s])
    return x,F


def init(out):
    out.mkdir(parents=True,exist_ok=False)
    p=read(OLD/'protocol.json')
    p.update(shape_family='Positive quartic Bernstein radius-squared profile, asymmetric about mid-height',
        selection='Exploratory refinement; 21 N excluded from material fitting; previously inspected comparisons are not pristine confirmatory tests',
        primary_prediction_tag='validation',previous_study=str(OLD))
    p['diagnostic_screen'].update(unloading_gap_rmse_mm=2.)
    save(out/'protocol.json',p)


def extract(out,episode,start=-.15,end=13.55,depth_background=False,adaptive_center=False):
    """Fit visible row widths; graph-cut gray masks are independently seeded."""
    cv2.setNumThreads(1)
    a=read(OBS/episode/'assessment.json');g=read(OLD/episode/'geometry.json')
    obs=np.load(OBS/episode/'hand_observations.npz')
    center_x=float(np.nanmedian(obs['mask_centroid_px'][(obs['time']>=-.5)&(obs['time']<=.2),0])) if adaptive_center else 462.
    robot=np.load(OBS/episode/'robot_force.npz')
    selected=(obs['time']>=-.15)&(obs['time']<=13.55)
    z=np.interp(obs['time'],robot['time'],robot['ee_pos'][:,2])
    B=np.c_[np.ones(len(z)),(z-z[selected][0])*1000]
    edge_fit=least_squares(lambda v:(B@v-obs['plate_white_edge_y_px'])[selected],
        [obs['plate_white_edge_y_px'][selected][0],-2.],loss='soft_l1')
    bad_edge=abs(B@edge_fit.x-obs['plate_white_edge_y_px'])>8.
    # Reject gross edge-detector jumps incompatible with rigid robot motion.
    # Do not replace them with a desired silhouette or simulation prediction.
    cap=cv2.VideoCapture(str(RAW/episode/'hand_rgb.mp4'))
    records=[];examples=[];last_t=-100.
    for idx,t,plate,bad in zip(obs['frame_idx'],obs['time'],obs['plate_white_edge_y_px'],bad_edge):
        if not start<=t<=end or t-last_t<.085 or bad:continue
        last_t=t;cap.set(1,int(idx));ok,image=cap.read();assert ok
        dep=cv2.imread(str(RAW/episode/f'hand_depth/{idx:06}.png'),-1)
        mask,_=segment(image,dep,'hand',a['material'])
        if a['material']=='plasticine':
            # Small crop avoids unrelated objects. Only the central body core is
            # certain foreground. No preceding mask or material model is used.
            x0,x1,y0,y1=int(round(center_x))-90,int(round(center_x))+90,max(120,int(plate)+7),268
            roi=image[y0:y1,x0:x1];yy,xx=np.mgrid[y0:y1,x0:x1]
            cy=(y0+y1)/2;ry=max(4.,(y1-y0)/2)
            ell=((xx-center_x)/57)**2+((yy-cy)/ry)**2
            labels=np.where(ell<1.,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype('uint8')
            labels[ell>1.6]=cv2.GC_BGD
            labels[((xx-center_x)/13)**2+((yy-cy)/(ry*.45))**2<1]=cv2.GC_FGD
            if depth_background:
                # The second, unfrozen depth stream separates the dark body
                # from a similarly colored background. Missing depth stays
                # uncertain; it is not completed or treated as correspondence.
                scale=read(RAW/episode/'meta.json')['cameras']['hand']['depth_scale']
                depth_roi=dep[y0:y1,x0:x1]*scale
                far=(depth_roi>g['camera_depth_m']+.006)&(labels!=cv2.GC_FGD)
                labels[far]=cv2.GC_BGD
            cv2.setRNGSeed(int(idx))
            cv2.grabCut(roi,labels,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
            mask[:]=0;mask[y0:y1,x0:x1]=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD])
            n,lab,stats,centers=cv2.connectedComponentsWithStats(mask)
            ids=[j for j in range(1,n) if stats[j,4]>300 and abs(centers[j,0]-center_x)<25]
            if not ids:continue
            mask=np.uint8(lab==max(ids,key=lambda j:stats[j,4]))
        ss=[];rr=[];left=[];right=[]
        for y in range(max(0,int(plate)+8),267):
            xx=np.flatnonzero(mask[y])
            s=(267-y)/(267-plate)
            if len(xx)<14 or not .045<s<.94:continue
            ss.append(s);rr.append((xx[-1]-xx[0])*g['pixel_scale_m']/2)
            left.append(xx[0]);right.append(xx[-1])
        if len(rr)<7:continue
        ss=np.asarray(ss);rr=np.asarray(rr);B=basis(ss);scale=max(rr)**2
        # Mild curvature regularization controls extrapolation at hidden contacts.
        D=np.diff(np.eye(DEGREE+1),n=2,axis=0)
        fit=lsq_linear(np.vstack([B,.04*D]),np.r_[rr**2/scale,np.zeros(DEGREE-1)],bounds=(.004,3.))
        c=fit.x*scale;res=np.sqrt(np.mean((np.sqrt(B@c)-rr)**2))
        records.append([t,plate,res,len(rr),*c])
        if any(abs(t-target)<.065 for target in [0.,3.,10.,13.]):
            im=image.copy()
            y=np.arange(int(plate),268);s=(267-y)/(267-plate);r=np.sqrt(basis(s)@c)/g['pixel_scale_m']
            center=float(np.median((np.asarray(left)+np.asarray(right))/2))
            for sign in [-1,1]:
                pts=np.c_[center+sign*r,y].round().astype('int32');cv2.polylines(im,[pts],False,(0,255,0),1)
            cv2.line(im,(365,267),(560,267),(0,0,255),1)
            tile=cv2.resize(im[145:290,350:580],(460,290));cv2.putText(tile,f'{episode} {t:.2f}s',(5,22),0,.6,(255,255,255),1)
            examples.append(tile)
    cap.release();dest=out/episode;dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'profile_observations.npz',samples=records)
    save(dest/'plate_edge_audit.json',dict(rejected_times_s=obs['time'][selected&bad_edge].tolist(),
        threshold_pixel=8.,pixel_per_mm=float(-edge_fit.x[1]),segmentation_center_x=center_x,depth_background_cue=depth_background,
        scope='Robust rigid-motion regression used only to reject gross detector jumps. Remaining edge pixels are unchanged; profiles interpolate missing frames.'))
    if examples:
        cv2.imwrite(str(dest/'profiles.jpg'),np.concatenate(examples[:8],axis=1))
    print(episode,'profile frames',len(records),'median radius residual mm',np.median(np.array(records)[:,2])*1000,flush=True)


def assemble(out,episode,baseline_start=-.15,normalize_baseline=False):
    p=read(out/'protocol.json');g0=read(OLD/episode/'geometry.json');a=read(OBS/episode/'assessment.json')
    samples=np.load(out/episode/'profile_observations.npz')['samples'];times=np.arange(0,13.40001,.05)
    c0=np.median(samples[(samples[:,0]>=baseline_start)&(samples[:,0]<=.2),4:],axis=0)
    fit_samples=samples.copy()
    if normalize_baseline:fit_samples[fit_samples[:,0]<=.2,4:]=c0
    cs=np.column_stack([np.interp(times,fit_samples[:,0],fit_samples[:,4+j]) for j in range(DEGREE+1)])
    cs=savgol_filter(cs,15,2,axis=0);cs=np.maximum(cs,1e-7)
    cs[times<=.15]=c0
    h0=g0['h0_m'];robot=np.load(OBS/episode/'robot_force.npz')
    raw_h=h0+np.interp(times,robot['time'],robot['ee_pos'][:,2])-a['robot_z_baseline_m']
    h=savgol_filter(raw_h,9,3);h[times<=.15]=h0
    volume=np.pi*h0*np.mean(c0)
    z,wz=np.polynomial.legendre.leggauss(32);r,wr=np.polynomial.legendre.leggauss(4)
    ss,rr=np.meshgrid((z+1)/2,(r+1)/2,indexing='ij')
    weights=(wz[:,None]/2)*poly(c0)(ss)/np.mean(c0)*(wr[None,:]*rr)
    ss=ss.ravel();rr=rr.ravel();weights=weights.ravel()
    XF=[curve_geometry(hh,c,c0,h0,ss,rr,volume) for hh,c in zip(h,cs)]
    x=np.array([v[0] for v in XF]);F=np.array([v[1] for v in XF]);vel=np.gradient(x,times,axis=0,edge_order=2)
    force=np.interp(times,robot['time'],robot['incremental_normal_force'])+p['physical_preload_N']
    w,grad=virtual(x,float(h.min()));rho=p['density_kg_m3']
    momentum=volume*rho*np.einsum('tni,tmni,n->tm',vel,w,weights)
    convective=volume*rho*np.einsum('tni,tmnij,tnj,n->tm',vel,grad,vel,weights)
    body=-9.81*volume*rho*np.einsum('tmn,n->tm',w[...,2],weights)
    windows=[];target=[];begins=np.arange(.4,13.4-.59,.2)
    for begin in begins:
        u=np.clip((times-begin)/.6,0,1);chi=np.sin(np.pi*u)**4;dc=4*np.pi/.6*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        windows.append(chi*.05);target.append(np.sum(chi[:,None]*(convective+body-force[:,None])+dc[:,None]*momentum,axis=0)*.05)
    grid=np.linspace(0,1,129);radii=np.sqrt(np.stack([poly(c)(grid) for c in cs]))
    inferred_volume=np.pi*raw_h*np.mean(cs,axis=1)
    projection_scale=np.sqrt(volume/inferred_volume)
    dest=out/episode
    raw_shape=np.c_[samples[:,0],np.zeros(len(samples)),np.sqrt(samples[:,4:]@basis(np.array([.5])).T).ravel(),samples[:,2:4]]
    np.savez_compressed(dest/'motion.npz',time=times,height=h,raw_height=raw_h,F=F,x=x,velocity=vel,weights=weights,
        force=force,test_gradient=grad,windows=windows,target=target,window_begin=begins,raw_shape_samples=raw_shape,
        profile_coefficients=cs,profile_s=grid,observed_profile_radius=radii,volume_projection_scale=projection_scale,
        observed_max_width=2*radii.max(1),observed_volume=inferred_volume)
    np.savez_compressed(dest/'load_only.npz',time=times,force=force)
    info=dict(g0,volume_m3=volume,profile_s=grid.tolist(),profile_radius0_m=np.sqrt(poly(c0)(grid)).tolist(),
        initial_geometry_time_window_s=[baseline_start,.2],baseline_profile_normalized_before_smoothing=normalize_baseline,
        profile_coefficients0=c0.tolist(),radius_mid0_m=float(np.sqrt(poly(c0)(.5))),
        shape_scope='Asymmetric positive quartic area profile from visible row widths, contact extrapolation, constant-volume radial projection',
        profile_radius_rms_mm=float(np.sqrt(np.mean(samples[:,2]**2))*1000),
        raw_volume_ratio_range=[float(inferred_volume.min()/volume),float(inferred_volume.max()/volume)],
        volume_projection_radius_change_rms_mm=float(np.sqrt(np.mean((radii*(projection_scale[:,None]-1))**2))*1000),
        max_abs_J_error=float(abs(np.linalg.det(F)-1).max()),weight_sum=float(weights.sum()))
    save(dest/'geometry.json',info)
    print(episode,'volume mL',round(volume*1e6,2),'projection radius change mm',round(info['volume_projection_radius_change_rms_mm'],2),'J error',info['max_abs_J_error'],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['init','extract','assemble']);a.add_argument('--out',type=Path,required=True);a.add_argument('--episode');a.add_argument('--start',type=float,default=-.15);a.add_argument('--end',type=float,default=13.55);a.add_argument('--depth-background',action='store_true');a.add_argument('--normalize-baseline',action='store_true');a.add_argument('--adaptive-center',action='store_true')
    args=a.parse_args()
    if args.stage=='init':init(args.out)
    else:
        for ep in ([args.episode] if args.episode else [f'ep{i:04}' for i in range(12)]):
            if args.stage=='extract':extract(args.out,ep,args.start,args.end,args.depth_background,args.adaptive_center)
            else:assemble(args.out,ep,args.start,args.normalize_baseline)

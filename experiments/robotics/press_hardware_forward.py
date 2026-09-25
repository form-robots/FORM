"""Forward 3D MPM prediction under recorded normal loading.

The plate follows a force servo, not the recorded displacement. A local
operator-split overstress update extends the existing Hencky metal stress law
without editing engine sources. This update is checked against the identifier.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import warp as wp
from experiments.robotics.press_hardware_observe import read, save


@wp.kernel
def relax_overstress(Ft:wp.array(dtype=wp.mat33),Fe:wp.array(dtype=wp.mat33),
                     rounding:wp.array(dtype=wp.mat33),threshold:float,fraction:float):
    i=wp.tid()
    U=wp.mat33(0.0);V=wp.mat33(0.0);s=wp.vec3(0.0)
    wp.svd3(Ft[i],U,s,V)
    e=wp.vec3(wp.log(wp.max(s[0],1e-8)),wp.log(wp.max(s[1],1e-8)),wp.log(wp.max(s[2],1e-8)))
    mean=(e[0]+e[1]+e[2])/3.0
    dev=e-wp.vec3(mean,mean,mean);norm=wp.length(dev)
    if norm>threshold:
        de=-fraction*(norm-threshold)/norm*dev
        ds=wp.vec3(0.0)
        for j in range(3):
            a=de[j]
            if wp.abs(a)<0.01:
                ds[j]=s[j]*(a+a*a/2.0+a*a*a/6.0+a*a*a*a/24.0)
            else:
                ds[j]=s[j]*(wp.exp(a)-1.0)
        # Add only the plastic correction. Reconstructing the whole matrix from
        # an SVD each tick creates drift even when the physical correction is tiny.
        correction=U*wp.mat33(ds[0],0.0,0.0,0.0,ds[1],0.0,0.0,0.0,ds[2])*wp.transpose(V)
        old=Ft[i];increment=correction-rounding[i];result=old+increment
        rounding[i]=(result-old)-increment
        Ft[i]=result;Fe[i]=result


def plate_sdf(radius=.055,half=.003,cell=.0015):
    from warpmpm.geometry import SDFData
    extent=radius+.004;axis=np.arange(-extent,extent+cell/2,cell)
    xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
    q=np.stack([np.linalg.norm(xyz[...,:2],axis=-1)-radius,abs(xyz[...,2])-half],-1)
    distance=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
    gradient=np.stack(np.gradient(distance,cell,edge_order=2),-1)
    return SDFData(distance.astype('float32'),gradient.astype('float32'),np.full(3,axis[0]),cell,float(distance.max()))


def initial_particles(geometry,spacing=.0015,floor=.05):
    h=geometry['h0_m'];rm=geometry['radius_mid0_m'];k=geometry['k0']
    axis=np.arange(-rm*1.3,rm*1.3+spacing/2,spacing)
    z=(np.arange(max(3,round(h/spacing)))+.5)*h/max(3,round(h/spacing))
    x=np.stack(np.meshgrid(axis,axis,z,indexing='ij'),-1).reshape(-1,3)
    r=np.sqrt(geometry['volume_m3']/(np.pi*h*(1-k/3))*(1-k*(2*x[:,2]/h-1)**2))
    x=x[np.linalg.norm(x[:,:2],axis=1)<r]
    x+=np.array([.09,.09,floor])
    return x.astype('float32'),np.full(len(x),geometry['volume_m3']/len(x),np.float32)


def run(out,material,grid=80,friction=.15,device='cuda:0',duration=10.4,tag='prediction',tick=.002,bulk=None,summary_only=False):
    from warpmpm import Solver,GridConfig
    from warpmpm.materials import vonmises
    p=read(out/'protocol.json');fit=read(out/material/'fit.json')['selected'];episode=p['heldout'][material]
    geometry=read(out/episode/'geometry.json')
    # Force prediction mode uses only initial geometry and external load.
    # The explicitly separate displacement-calibration mode prescribes the
    # measured plate motion and predicts force and specimen shape.
    observed=np.load(out/episode/'load_only.npz');load_time=observed['time'];load=observed['force']
    displacement_control=p.get('control_mode','force')=='displacement_calibration'
    boundary_motion=np.load(out/episode/'motion.npz') if displacement_control else None
    withdrawal_start=p.get('withdrawal_start_s')
    withdrawal=np.load(out/episode/'withdrawal.npz') if withdrawal_start is not None else None
    withdrawal_anchor=None
    dest=out/material/tag;dest.mkdir(exist_ok=False)
    x,vol=initial_particles(geometry,spacing=.18/grid/1.5);floor=.05;half=.003
    sim=Solver(GridConfig(grid,.18),device=device,inversion_policy='raise').load_particles(x,vol)
    Y,tau=fit['Y_Pa'],fit['tau_s'];mu=fit.get('G_Pa',fit['E_Pa']/(2*(1+p['nu'])))
    bulk=fit.get('K_Pa',p.get('bulk_modulus_assumed_Pa',1e6)) if bulk is None else bulk
    if mu<=0 or bulk<=0:raise ValueError('Positive shear and bulk stiffness required')
    # Pressure-free identification determines shear response, not bulk stiffness.
    # A separately declared bulk penalty approximates the reconstruction's J=1.
    E=9*bulk*mu/(3*bulk+mu);nu=(3*bulk-2*mu)/(2*(3*bulk+mu))
    hardening=fit.get('H_Pa',0.)
    if hardening and tau:raise ValueError('Combined hardening/overstress forward update is not implemented')
    sim.set_material(vonmises(E=E,nu=nu,yield_stress=1e20 if tau else Y,hardening_xi=hardening/(2*mu),density=p['density_kg_m3']),
                     rpic_damping=0.,grid_v_damping_scale=1.)
    contact_band=p.get('contact_band_cells',0.)*(.18/grid)
    if contact_band:
        sim.add_sdf_collider(plate_sdf(radius=.075),center=[.09,.09,floor-half],
                            band=contact_band,surface='separable',friction=friction)
    else:
        sim.add_plane((0,0,floor),(0,0,1),surface='separable',friction=friction)
    normal=np.array(geometry.get('pad_normal0',[0.,0.,1.])) if p.get('use_measured_initial_tilt',False) else np.array([0.,0.,1.])
    normal=normal/np.linalg.norm(normal)
    quat=np.array([-normal[1],normal[0],0.,1+normal[2]]);quat/=np.linalg.norm(quat)
    h=geometry['h0_m']*(1+1/normal[2])/2
    initial_plate_gap=h
    handle=sim.add_sdf_collider(plate_sdf(radius=p.get('plate_radius_m',.055)),center=[.09,.09,floor+h+half/normal[2]],quat=quat.tolist(),
                    band=contact_band or .00025,surface='separable',friction=friction)
    lam=bulk-2*mu/3
    dt_max=.23*(.18/grid)/np.sqrt((lam+2*mu)/p['density_kg_m3'])
    substeps=max(1,int(np.ceil(tick/min(p.get('max_substep_s',.00006),dt_max))));dt=tick/substeps
    stride=round(.05/tick);total=round(duration/tick)
    chosen=np.linspace(0,len(x)-1,min(6000,len(x))).astype(int)
    frames=[x[chosen]];times=[0.];rows=[];shapes=[];previous=0.;filtered=0.;v=0.;start=time.monotonic()
    min_j=1.;maxspeed=0.;servo_gain=p.get('servo_gain_m_per_N_s',.012)
    for i in range(1,total+1):
        t=i*tick;target=float(np.interp(t,load_time,load))
        filtered+=(previous-filtered)*(1-np.exp(-tick/.004))
        if withdrawal_start is not None and t>withdrawal_start:
            # Preserve the independently predicted loaded gap. Only the
            # subsequent recorded tool displacement increment is prescribed;
            # the released specimen height and diameter remain predictions.
            if withdrawal_anchor is None:withdrawal_anchor=h
            delta=float(np.interp(t,withdrawal['time'],withdrawal['vertical_displacement_m']))
            delta0=float(np.interp(withdrawal_start,withdrawal['time'],withdrawal['vertical_displacement_m']))
            next_h=withdrawal_anchor+delta-delta0
            v=(h-next_h)/tick
        elif displacement_control:
            next_h=float(np.interp(t,boundary_motion['time'],boundary_motion['raw_height']))+initial_plate_gap-geometry['h0_m']
            v=(h-next_h)/tick
        else:
            desired=float(np.clip(servo_gain*(target-filtered),-.08,.15))
            # Positive servo speed is down. Fixed gain/limits unless the
            # numerical-control protocol declares a sensitivity value.
            v=float(np.clip(desired,v-3*tick,v+3*tick))
        maxspeed=max(maxspeed,abs(v))
        sim.set_sdf_pose(handle,center=[.09,.09,floor+h+half/normal[2]],velocity=[0,0,-v])
        sim.reset_sdf_force(handle);sim.step(dt,substeps=substeps)
        previous=float(sim.sdf_wrench(handle,tick)['force'][2]);h-=v*tick
        if tau:
            s=sim._sim.mpm_state
            if 'flow_power' in fit:
                from experiments.robotics.press_hardware_power import relax_power
                wp.launch(relax_power,dim=len(x),inputs=[s.particle_F_trial,s.particle_F,s.particle_F_roundoff,
                    Y/(2*mu),tick/tau,fit['flow_power']],device=sim.device)
            else:
                wp.launch(relax_overstress,dim=len(x),inputs=[s.particle_F_trial,s.particle_F,s.particle_F_roundoff,
                    Y/(2*mu),tick/(tau+tick)],device=sim.device)
        rows.append([t,target,previous,h,v])
        if i%stride==0:
            xx=sim.x();assert np.isfinite(xx).all()
            ff=sim.F();J=np.linalg.det(ff);min_j=min(min_j,float(J.min()))
            if not summary_only:frames.append(xx[chosen])
            times.append(t)
            radius=np.linalg.norm(xx[:,:2]-[.09,.09],axis=1)
            shapes.append([t,float(np.quantile(radius,.99)*2),float(np.quantile(xx[:,2],.995)-np.quantile(xx[:,2],.005)),float(np.mean(J)),
                float(np.ptp(xx[:,0])),float(np.ptp(xx[:,1])),float(np.ptp((xx[:,0]+xx[:,1])/np.sqrt(2))),float(np.ptp((xx[:,0]-xx[:,1])/np.sqrt(2))),
                float(((xx-[.09,.09,floor+h])@normal).max()),float(floor-xx[:,2].min())])
        if i%round(1/tick)==0:print(material,tag,'t',round(t,2),'target/reaction',round(target,2),round(previous,2),'gap mm',round(h*1000,2),'wall',round(time.monotonic()-start,1),flush=True)
        if not .004<h<.10:raise RuntimeError(f'Forward plate left admissible gap range: {h}')
    np.savez_compressed(dest/'trajectory.npz',time=times,x=np.array(frames) if not summary_only else np.empty((0,0,3)),particle_ids=chosen,force_log=rows,shape_log=shapes,
        initial_full=x,vol0=vol)
    save(dest/'completion.json',dict(complete=True,material=material,episode=episode,parameters=fit,grid=grid,dt=dt,tick=tick,
        particle_count=len(x),min_elastic_J=min_j,friction_assumed=friction,wall_s=time.monotonic()-start,
        shear_modulus_Pa=mu,bulk_modulus_assumed_Pa=bulk,bulk_scope='fitted' if 'K_Pa' in fit else 'assumed',equivalent_E_Pa=E,equivalent_nu=nu,
        plate_radius_m=p.get('plate_radius_m',.055),
        pad_normal=normal.tolist(),initial_plate_gap_m=initial_plate_gap,
        contact_band_m=contact_band or .00025,platform_contact='SDF with same numerical band' if contact_band else 'nodal plane',
        trajectory_saved=not summary_only,
        shape_log_columns=['time_s','radial_99pct_diameter_m','height_005_995_quantile_m','mean_J','x_extent_m','y_extent_m','diagonal_extent_m','other_diagonal_extent_m','top_particle_overrun_m','bottom_particle_overrun_m'],
        max_servo_speed_m_s=maxspeed,force_input=p.get('force_input_scope','Recorded episode normal force, interpolated in time; episode split and selection scope are in protocol.json'),
        servo=dict(gain_m_per_N_s=servo_gain,filter_s=.004,acceleration_limit_m_s2=3.,down_speed_limit_m_s=.15),
        recorded_displacement_used_as_control=displacement_control,recorded_deformed_shape_used=False,
        withdrawal_start_s=withdrawal_start,
        withdrawal_scope='After this time only recorded vertical tool displacement increments are prescribed, anchored to the predicted gap; specimen recovery remains a prediction. Adhesion is omitted.' if withdrawal_start is not None else None,
        control_mode='Measured plate displacement; force and lateral shape are predictions' if displacement_control else 'Measured force target; plate displacement and shape are predictions',
        initial_geometry=geometry,law_update='Engine Hencky stress; incremental linear isotropic hardening' if hardening else ('Engine Hencky stress; implicit power overstress update' if 'flow_power' in fit else 'Engine Hencky stress; perfect return or operator-split implicit linear overstress update'),
        hardening_modulus_Pa=hardening,
        contact='Separable plate/platform with Coulomb friction; unmeasured adhesive traction omitted',
        status='Forward result; quantitative evaluation performed separately'))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,required=True);a.add_argument('--material',required=True)
    a.add_argument('--grid',type=int,default=80);a.add_argument('--friction',type=float,default=.15);a.add_argument('--device',default='cuda:0')
    a.add_argument('--duration',type=float,default=10.4);a.add_argument('--tag',default='prediction');a.add_argument('--tick',type=float,default=.002)
    a.add_argument('--bulk',type=float)
    a.add_argument('--summary-only',action='store_true')
    args=a.parse_args();run(**vars(args))

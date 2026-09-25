"""MPM elastic putter coupled by reaction impulse to a rigid rolling ball.

The club receives prescribed grasp motion. Ball velocity is never prescribed:
club contact supplies its impulse, and the floor supplies normal/frictional
impulses and rolling resistance. Synthetic material estimates are frozen inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.elastic.rod_insertion_study import box_sdf

ROOT = Path(__file__).resolve().parents[2]

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def protocol():
    source = ROOT / 'out/strip_texture_20260912'
    estimates = {k: json.loads((source / f'fit_{k}/identification.json').read_text())['E_pa'] for k in 'AB'}
    return dict(E_true={'A': 80000., 'B': 240000.}, E_identified=estimates,
        identification_sha256={k: digest(source / f'fit_{k}/identification.json') for k in 'AB'},
        nu=.45, density=1000., domain=.24, grid=240, dt=.0000125*160/240, coupling_dt=.000125,
        floor=.02, ball_radius=.008, ball_mass=.002, ball_initial=[.118, .11, .028],
        floor_static_friction=.6, floor_sliding_friction=.45, rolling_resistance=.025,
        floor_restitution=0., gravity=9.81,
        shaft_width=.012, shaft_depth=.008, shaft_bottom=-.0675, shaft_top=.0075,
        head_size=[.012, .028, .016], head_center=[0.,0.,-.0675], grip_length=.015,
        grip_z=.098, address_offset=.018, pivot_radius=.10, settle=.15,
        backswing_angle_deg=28., followthrough_angle_deg=-22., backswing_duration=.55,
        backswing_pause=.08,
        lift_delay=.08, lift_duration=.30, lift_height=.05,
        end=2., video_dt=.02, sdf_cell=.0005, contact_band=.00015,
        initialization=dict(damping_rate_s=100.,min_time=.30,max_time=1.,speed_tolerance=2e-5,drift_tolerance=.0000003),
        target=[.718,.23], target_radius=.035,
        bounds=[[.35,.85],[0.,24.]], initial=[.55,11.31], maxfev=32, early_stop_error_mm=2.,
        table_bounds=[-.03,1.18,-.18,.36,-.005,.02],
        objective='Distance of the stopped ball center from fixed target center; no swapped score.',
        scope='Kinematic bonded grasp, homogeneous fixed-corotated club. Rigid solid sphere, Coulomb floor friction and rolling-resistance torque. Frictionless club-ball contact. No material damping, force or shape feedback.')

def smooth(t):
    t=np.clip(t,0.,1.); return t*t*t*(10+t*(-15+6*t))

def pose(t, controls, p):
    duration, yaw=controls
    # One prescribed arc, with only forward duration and horizontal aim optimized.
    # Positive pitch draws the head back; the low point is the address pose.
    start=p['settle']+p['backswing_duration']+p['backswing_pause']
    back=np.deg2rad(p['backswing_angle_deg'])
    finish=np.deg2rad(p['followthrough_angle_deg'])
    theta=back*smooth((t-p['settle'])/p['backswing_duration'])
    theta+=(finish-back)*smooth((t-start)/duration)
    a=np.deg2rad(yaw);c,s=np.cos(a),np.sin(a)
    yaw_R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    c,s=np.cos(theta),np.sin(theta)
    pitch_R=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    radius=p['pivot_radius']
    travel=-p['address_offset']-radius*s
    lift=p['lift_height']*smooth((t-start-duration-p['lift_delay'])/p['lift_duration'])
    center=np.r_[np.asarray(p['ball_initial'])[:2]+travel*yaw_R[:2,0],
                 p['grip_z']+radius*(1-c)+lift]
    return center,yaw_R@pitch_R

def particles(p, grid):
    h=p['domain']/grid/2;head=np.array(p['head_size']);hc=np.array(p['head_center'])
    # Partition the union into two disjoint boxes and integrate each exactly.
    # Clipping a bounding-box lattice changed mass and effective shaft width
    # across resolutions. Here every resolution has the same physical volume.
    shaftlo=np.array([-p['shaft_width']/2,-p['shaft_depth']/2,hc[2]+head[2]/2])
    shafthi=np.array([p['shaft_width']/2,p['shaft_depth']/2,p['shaft_top']])
    points=[];volumes=[]
    for lo,hi in [(hc-head/2,hc+head/2),(shaftlo,shafthi)]:
        n=np.ceil((hi-lo)/h).astype(int);spacing=(hi-lo)/n
        axes=[l+(np.arange(k)+.5)*d for l,k,d in zip(lo,n,spacing)]
        q=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        points.append(q);volumes.append(np.full(len(q),spacing.prod(),dtype=np.float32))
    return np.concatenate(points),np.concatenate(volumes)

def sphere_sdf(p):
    from warpmpm.geometry import SDFData
    h=p['sdf_cell']; extent=p['ball_radius']+.005
    axis=np.arange(-extent,extent+h/2,h)
    q=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
    norm=np.linalg.norm(q,axis=-1)
    values=norm-p['ball_radius'];grad=q/np.maximum(norm[...,None],1e-15)
    return SDFData(values.astype('float32'),grad.astype('float32'),np.full(3,-extent),h,float(values.max()))

class Ball:
    def __init__(self,p):
        self.p=p;self.x=np.array(p['ball_initial'],dtype=float);self.v=np.zeros(3);self.omega=np.zeros(3)
        self.quat=np.array([0.,0.,0.,1.])
        self.inertia=.4*p['ball_mass']*p['ball_radius']**2
        self.sliding_steps=0;self.rolling_steps=0;self.flight_steps=0

    def step(self, impulse, angular_impulse, dt):
        p=self.p;m=p['ball_mass'];r=p['ball_radius'];z0=p['floor']+r
        self.v+=impulse/m;self.v[2]-=p['gravity']*dt
        self.omega+=angular_impulse/self.inertia
        normal_impulse=0.
        if self.x[2]+self.v[2]*dt<=z0 and self.v[2]<=0.:
            normal_impulse=-(1+p['floor_restitution'])*m*self.v[2]
            self.v[2]=-p['floor_restitution']*self.v[2]
            # Resist horizontal spin. Floor friction transfers this torque into
            # translational deceleration; a rolling sphere has effective mass 7m/5.
            spin=np.linalg.norm(self.omega[:2])
            if spin:
                change=min(spin,p['rolling_resistance']*normal_impulse*r/self.inertia)
                self.omega[:2]*=1-change/spin
            arm=np.array([0.,0.,-r]);slip=(self.v+np.cross(self.omega,arm))[:2]
            needed=-slip/(1/m+r*r/self.inertia)
            if np.linalg.norm(needed)<=p['floor_static_friction']*normal_impulse:
                tangential=needed;self.rolling_steps+=1
            else:
                tangential=-p['floor_sliding_friction']*normal_impulse*slip/max(np.linalg.norm(slip),1e-30)
                self.sliding_steps+=1
            j=np.r_[tangential,0.];self.v+=j/m;self.omega+=np.cross(arm,j)/self.inertia
        else:self.flight_steps+=1
        self.x+=self.v*dt;self.x[2]=max(self.x[2],z0)
        # Orientation is passive for a sphere, but is recorded so a rendered
        # stripe displays its computed rotation rather than an invented spin.
        angle=float(np.linalg.norm(self.omega)*dt)
        if angle:
            v=self.omega*dt*(np.sin(angle/2)/angle);w=np.cos(angle/2)
            a=self.quat[:3];b=self.quat[3]
            self.quat=np.r_[w*a+b*v+np.cross(v,a),w*b-np.dot(v,a)]
            self.quat/=np.linalg.norm(self.quat)
        return normal_impulse

    def energy(self):
        return .5*self.p['ball_mass']*np.dot(self.v,self.v)+.5*self.inertia*np.dot(self.omega,self.omega)

def simulate(p, E, controls, device='cuda:0', out=None, end=None, motion=None, record_frames=True):
    from warpmpm import Solver,GridConfig
    from warpmpm.materials import elastic
    from scipy.spatial.transform import Rotation
    start=time.monotonic();end=p['end'] if end is None else end
    if out is not None:
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        save(out/'config.json',dict(protocol=p,E_pa=E,controls=list(controls),device=device))
    ref,vol=particles(p,p['grid']);center,R=pose(0.,controls,p)
    if motion is not None:center,R=motion(0.)
    sim=Solver(GridConfig(p['grid'],p['domain']),device=device,inversion_policy='raise',guard_interval=10).load_particles(ref@R.T+center,vol)
    sim.set_material(elastic(E=E,nu=p['nu'],density=p['density']),rpic_damping=0.,grid_v_damping_scale=1.)
    grip=sim.add_sdf_collider(box_sdf((p['shaft_width']/2+.001,p['shaft_depth']/2+.001,p['grip_length']/2),p['sdf_cell']),center=center,
        quat=Rotation.from_matrix(R).as_quat(),surface='sticky',friction=0.,band=p['contact_band'])
    sim.add_plane((0,0,p['floor']),(0,0,1),surface='separable',friction=.4)
    ball=Ball(p);sphere=sim.add_sdf_collider(sphere_sdf(p),center=ball.x,surface='separable',friction=0.,band=p['contact_band'])
    tick=p['coupling_dt'];substeps=int(np.ceil(tick/p['dt']));dt=tick/substeps
    # Numerical dynamic relaxation prepares the hanging static equilibrium.
    # It is not a fitted material damping coefficient. Remove it completely
    # before the prescribed stroke, and never reset the elastic deformation.
    init=p['initialization'];relax_dt=.02;previous=sim.x();relax_time=0.;relax_history=[]
    sim.set_material(elastic(E=E,nu=p['nu'],density=p['density']),rpic_damping=0.,
        grid_v_damping_scale=float(np.exp(-init['damping_rate_s']*dt)))
    while relax_time<init['max_time']-1e-10:
        steps=round(relax_dt/dt);sim.step(relax_dt/steps,substeps=steps);relax_time+=relax_dt
        current=sim.x();speed=float(np.linalg.norm(sim.v(),axis=1).max())
        drift=float(np.linalg.norm(current-previous,axis=1).max());previous=current
        relax_history.append([relax_time,speed,drift])
        if relax_time>=init['min_time'] and speed<init['speed_tolerance'] and drift<init['drift_tolerance']:break
    if speed>=init['speed_tolerance'] or drift>=init['drift_tolerance']:
        if out is not None:save(out/'initialization_failure.json',relax_history)
        raise RuntimeError(f'Equilibrium relaxation did not converge: speed={speed}, drift={drift}')
    sim.set_material(elastic(E=E,nu=p['nu'],density=p['density']),rpic_damping=0.,grid_v_damping_scale=1.)
    if out is not None:save(out/'initialization.json',dict(history=relax_history,scope='Static preparation only; damping disabled during the stroke. Elastic state retained; velocities not reset.'))
    times=np.arange(round(end/tick)+1)*tick;stride=round(p['video_dt']/tick)
    rows=[];snap=[];min_j=1.;peak_force=0.;total_impulse=np.zeros(3);max_overlap=0.;peak_speed=0.
    max_height=0.;contact_times=[];peak_torque=0.;club_floor_clear=np.inf
    head=ref[:,2]<p['head_center'][2]+p['head_size'][2]/2
    for i,t in enumerate(times):
        force=np.zeros(3)
        if i:
            c0,r0=pose(t-tick,controls,p);c1,r1=pose(t,controls,p)
            if motion is not None:
                c0,r0=motion(t-tick);c1,r1=motion(t)
            sim.set_sdf_pose(grip,center=c0,quat=Rotation.from_matrix(r0).as_quat(),velocity=(c1-c0)/tick,
                omega=Rotation.from_matrix(r1@r0.T).as_rotvec()/tick)
            sim.set_sdf_pose(sphere,center=ball.x,velocity=ball.v,omega=ball.omega)
            sim.reset_sdf_force(sphere);sim.step(dt,substeps=substeps)
            wrench=sim.sdf_wrench(sphere,tick);force=wrench['force'];torque=wrench['torque']
            ball.step(force*tick,torque*tick,tick)
            total_impulse+=force*tick;peak_force=max(peak_force,float(np.linalg.norm(force)))
            peak_torque=max(peak_torque,float(np.linalg.norm(torque)))
            if np.linalg.norm(force)>1e-5:contact_times.append(float(t))
            max_height=max(max_height,ball.x[2]-p['floor']-p['ball_radius'])
        if i%stride==0 or i==len(times)-1:
            x=sim.x();v=sim.v();j=np.linalg.det(sim.F())
            if not np.isfinite(x).all() or not np.isfinite(v).all() or j.min()<=0:raise RuntimeError('Invalid club state')
            min_j=min(min_j,float(j.min()));peak_speed=max(peak_speed,float(np.linalg.norm(v,axis=1).max()))
            max_overlap=max(max_overlap,float(p['ball_radius']-np.linalg.norm(x-ball.x,axis=1).min()))
            club_floor_clear=min(club_floor_clear,float(x[:,2].min()-p['floor']))
            rows.append([t,*ball.x,*ball.v,*ball.omega,*force,ball.energy(),*x[head].mean(0),*ball.quat])
            if out is not None and record_frames:snap.append(x.copy())
    # The simulation continues through the stroke and withdrawal. Extend only
    # free ball runout after the head is demonstrably lifted clear of the ball.
    can_runout=sim.x()[:,2].min()>p['floor']+2*p['ball_radius']+.005
    runout=[];elapsed=0.
    if can_runout:
        while (np.linalg.norm(ball.v)>1e-5 or np.linalg.norm(ball.omega[:2])>1e-3) and elapsed<12.:
            ball.step(np.zeros(3),np.zeros(3),tick);elapsed+=tick
            if round(elapsed/tick)%stride==0:runout.append([times[-1]+elapsed,*ball.x,*ball.v,*ball.omega])
    # Ideal point contact does not resist spin about the vertical normal. This
    # spin does not move the center and is excluded from the stopping criterion.
    stopped=np.linalg.norm(ball.v)<1e-5 and np.linalg.norm(ball.omega[:2])<1e-3
    error=float(np.linalg.norm(ball.x[:2]-p['target']))
    result=dict(E_pa=float(E),controls=list(map(float,controls)),final_ball_m=ball.x.tolist(),error_mm=1000*error,
        stopped=bool(stopped),success=bool(stopped and error+p['ball_radius']<=p['target_radius']),
        min_J=min_j,max_particle_ball_overlap_mm=max_overlap*1000,max_ball_lift_mm=max_height*1000,
        peak_club_speed_m_s=peak_speed,peak_contact_force_N=peak_force,peak_contact_torque_Nm=peak_torque,
        contact_time_span_s=None if not contact_times else [min(contact_times),max(contact_times)],
        contact_ticks=len(contact_times),total_club_ball_impulse_Ns=total_impulse.tolist(),
        club_floor_clearance_mm=club_floor_clear*1000,free_runout_allowed=bool(can_runout),
        runout_s=elapsed,final_speed_m_s=float(np.linalg.norm(ball.v)),final_omega_rad_s=ball.omega.tolist(),
        sliding_steps=ball.sliding_steps,rolling_steps=ball.rolling_steps,flight_steps=ball.flight_steps,
        particle_count=len(ref),actual_mpm_dt=dt,substeps_per_coupling=substeps,initialization_time_s=relax_time,initialization_max_speed_m_s=speed,initialization_max_drift_m=drift,wall_s=time.monotonic()-start)
    if out is not None:
        np.savez(out/'trajectory.npz',signals=np.asarray(rows),x=np.asarray(snap),final_x=sim.x(),reference=ref,vol0=vol,runout=np.asarray(runout))
        save(out/'result.json',result)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',default='A')
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--duration',type=float,default=.55);ap.add_argument('--angle',type=float,default=12.)
    ap.add_argument('--end',type=float);ap.add_argument('--grid',type=int);ap.add_argument('--coupling',type=float)
    a=ap.parse_args();p=protocol()
    if a.grid:p['grid']=a.grid;p['dt']=.0000125*160/a.grid
    if a.coupling:p['coupling_dt']=a.coupling
    print(json.dumps(simulate(p,p['E_true'][a.material],[a.duration,a.angle],a.device,a.out,a.end),indent=2),flush=True)

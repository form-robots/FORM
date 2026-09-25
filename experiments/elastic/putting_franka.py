"""Compile a frozen putting stroke to Panda joints; use its FK in MPM."""
from pathlib import Path
import argparse
import json
import itertools
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from experiments.elastic.putting import pose, save, digest
from experiments.elastic.rod_insertion_franka import Panda, snapshot, OFFSET

# The rod-frame grasp axis points down the club shaft. The stock pad gap is
# eight millimeters, matching the club's transverse thickness.
Q=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])

def compile_motion(root, label, p, controls):
    robot=Panda(snapshot(root));ts=np.arange(round(p['end']/.01)+1)*.01
    qs=[];cs=[];rs=[];errors=[];contacts=[]
    for t in ts:
        c,R=pose(t,controls,p);(cp,Rp),e=robot.solve(c,R@Q)
        qs.append(robot.q.copy());cs.append(cp);rs.append(Rp@Q.T)
        errors.append(e[:6]);contacts.append(robot.data.ncon)
    qs=np.asarray(qs);errors=np.asarray(errors)
    speed=np.max(abs(np.diff(qs,axis=0)/.01),axis=0);limits=np.array([2.175]*4+[2.61]*3)
    if np.any(speed>limits):raise RuntimeError(f'Joint speed limit exceeded: {speed}')
    if max(contacts):raise RuntimeError('Sampled robot self-contact')
    path=root/'franka'/f'plan_{label}.npz'
    np.savez(path,time=ts,q=qs,center=cs,rotation=rs)
    # Conservative AABB clearance of all collision geometry against tabletop.
    # Pedestal is outside the tabletop footprint; render the same box.
    bounds=p['table_bounds'];lo=np.array(bounds)[::2]+OFFSET;hi=np.array(bounds)[1::2]+OFFSET
    vertices={}
    for g in np.flatnonzero(robot.model.geom_group==3):
        model=robot.model
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            m=int(model.geom_dataid[g]);v=int(model.mesh_vertadr[m]);n=int(model.mesh_vertnum[m]);pts=model.mesh_vert[v:v+n]
        elif model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            pts=np.array(list(itertools.product([-1,1],repeat=3)))*model.geom_size[g]
        else:raise RuntimeError('Unsupported collision geometry')
        vertices[int(g)]=pts.copy()
    pedestal=[.55,.65,.02,.16,-OFFSET[2],bounds[4]]
    fixtures=[('table',lo,hi),('pedestal',np.array(pedestal)[::2]+OFFSET,np.array(pedestal)[1::2]+OFFSET)]
    clearance=np.inf;closest=None
    for i,q in enumerate(qs):
        robot.forward(q)
        for g,pts in vertices.items():
            world=pts@robot.data.geom_xmat[g].reshape(3,3).T+robot.data.geom_xpos[g]
            for name,a,b in fixtures:
                gap=np.maximum(np.maximum(a-world.max(0),world.min(0)-b),0)
                dist=float(np.linalg.norm(gap))
                if dist<clearance:clearance=dist;closest=[i,robot.model.body(int(robot.model.geom_bodyid[g])).name,name]
    audit=dict(trajectory_sha256=digest(path),controls=controls,peak_joint_speed_rad_s=speed.tolist(),
        joint_speed_limits_rad_s=limits.tolist(),max_position_error_mm=float(np.linalg.norm(errors[:,:3],axis=1).max()*1000),
        max_angle_error_rad=float(np.linalg.norm(errors[:,3:6],axis=1).max()/.1),
        sampled_self_contacts=max(contacts),table_aabb_clearance_mm=1000*clearance,closest_table_geometry=closest,
        scope='Kinematic Panda with bonded grasp; joint limits, speed, self-contact and table/pedestal AABB clearance sampled at 10 ms. No torque, grasp-slip or hardware validation.')
    save(root/'franka'/f'audit_{label}.json',audit)
    if clearance<=0:raise RuntimeError(f'Table clearance not established: {audit}')
    return path

def load_motion(path,tick=None):
    data=np.load(path);ts=data['time'];cs=data['center'];rs=data['rotation']
    if tick is not None:
        # The command is the saved joint trajectory. Interpolate joints, then
        # evaluate FK at every coupling time, rather than interpolating hand
        # positions independently of the arm's kinematics.
        robot=Panda(Path(path).parent/'panda_model_snapshot/panda.xml')
        dense=np.arange(round(ts[-1]/tick)+1)*tick
        qs=np.stack([np.interp(dense,ts,data['q'][:,j]) for j in range(7)],axis=1)
        cs=[];rs=[];contacts=[]
        for q in qs:
            c,R=robot.forward(q);cs.append(c);rs.append(R@Q.T);contacts.append(robot.data.ncon)
        if max(contacts):raise RuntimeError('Robot self-contact between stored waypoints')
        cs=np.asarray(cs);rs=np.asarray(rs)
        def dense_motion(t):
            i=int(round(t/tick))
            if abs(t-i*tick)>1e-8:raise ValueError('Motion requested off coupling grid')
            return cs[i],rs[i]
        return dense_motion
    rotation=Slerp(ts,Rotation.from_matrix(rs))
    def motion(t):
        t=float(np.clip(t,ts[0],ts[-1]));i=min(int(t/.01),len(ts)-2);a=(t-ts[i])/(ts[i+1]-ts[i])
        return (1-a)*cs[i]+a*cs[i+1],rotation(t).as_matrix()
    return motion

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--label',default='pilot')
    ap.add_argument('--duration',type=float,default=.55);ap.add_argument('--angle',type=float,default=12.)
    a=ap.parse_args();p=json.loads((a.out/'protocol.json').read_text())
    print(compile_motion(a.out,a.label,p,[a.duration,a.angle]))

"""Compile task-space rod motions to Panda joints and export the actual FK grasp.

The material has a bonded 15 mm grasp. The stock pad's inner faces are 8 mm
apart: each slide is 2.5 mm plus the pad's 1.5 mm inward-face offset.
This is kinematic simulation, not a torque-controlled or hardware experiment.
"""
from pathlib import Path
import argparse
import shutil
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from experiments.elastic.rod_insertion_study import read,rotation,tilt,translation
from experiments.elastic.block_drop_study import save,sha

OFFSET=np.array([.33,-.51,.30])
MOUNT=np.array([0.,0.,.1029])
BASIS=np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])
SLIDE=.0025

class Panda:
    def __init__(self,xml):
        self.model=mujoco.MjModel.from_xml_path(str(xml));self.data=mujoco.MjData(self.model)
        self.hand=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'hand')
        self.q=np.array([0.,-.5,0.,-2.2,0.,1.7,.8])
        self.initialized=False
    def forward(self,q):
        self.data.qpos[:7]=q;self.data.qpos[7:9]=SLIDE
        mujoco.mj_forward(self.model,self.data)
        R=self.data.xmat[self.hand].reshape(3,3).copy()
        c=self.data.xpos[self.hand]+R@MOUNT-OFFSET
        return c,R@BASIS.T
    def solve(self,c,R):
        ref=self.q.copy()
        def fun(q):
            cp,Rp=self.forward(q)
            return np.r_[cp-c,.1*Rotation.from_matrix(R@Rp.T).as_rotvec(),(1e-4 if self.initialized else 1e-6)*(q-ref)]
        opt=least_squares(fun,ref,bounds=(self.model.jnt_range[:7,0]+1e-5,self.model.jnt_range[:7,1]-1e-5),
                          max_nfev=150,xtol=1e-11,ftol=1e-11,gtol=1e-11)
        e=fun(opt.x);self.q=opt.x
        if np.linalg.norm(e[:3])>.0001 or np.linalg.norm(e[3:6])/.1>.001:
            raise RuntimeError(f'IK failed, c={c}, residual={e[:6]}')
        self.initialized=True
        return self.forward(opt.x),e
    def meshes(self,include_arm=True):
        import pyvista as pv
        out=[]
        for g in range(self.model.ngeom):
            if self.model.geom_group[g]!=2 or self.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            name=self.model.body(int(self.model.geom_bodyid[g])).name
            if not include_arm and name not in ['hand','left_finger','right_finger']:continue
            m=int(self.model.geom_dataid[g]);v=int(self.model.mesh_vertadr[m]);f=int(self.model.mesh_faceadr[m])
            pts=self.model.mesh_vert[v:v+self.model.mesh_vertnum[m]]
            faces=self.model.mesh_face[f:f+self.model.mesh_facenum[m]]
            pts=pts@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]-OFFSET
            mesh=pv.PolyData(pts,np.c_[np.full(len(faces),3),faces].ravel())
            mat=int(self.model.geom_matid[g]);color=self.model.geom_rgba[g,:3] if mat<0 else self.model.mat_rgba[mat,:3]
            out.append((g,mesh,color))
        return out

def snapshot(root):
    dest=root/'franka';dest.mkdir(exist_ok=True)
    model=dest/'panda_model_snapshot'
    if not model.exists():
        installed=Path(__file__).resolve().parents[2]/'assets/panda'
        if not installed.exists():
            import os
            os.environ.setdefault("ROBOT_DESCRIPTION_COMMIT", "feadf76d42f8a2162426f7d226a3b539556b3bf5")
            from robot_descriptions import panda_mj_description
            installed=Path(panda_mj_description.MJCF_PATH).parent
        shutil.copytree(installed,model)
    save(dest/'asset_hashes.json',{str(f.relative_to(model)):sha(f) for f in model.rglob('*') if f.is_file()})
    return model/'panda.xml'


def fixture_audit(root,k,trajectory):
    """Conservative sampled AABB separation of robot collision geometry and fixture.

    The whole wall box is treated as solid here. Thus a positive lower bound
    proves sampled robot clearance even without exploiting the circular opening.
    MPM specimen contact continues to use the exact circular-bore SDF.
    """
    p=read(root/'protocol.json');robot=Panda(root/'franka'/'panda_model_snapshot'/'panda.xml')
    model=robot.model;data=robot.data
    local={}
    for g in np.flatnonzero(model.geom_group==3):
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            m=int(model.geom_dataid[g]);v=int(model.mesh_vertadr[m]);n=int(model.mesh_vertnum[m]);pts=model.mesh_vert[v:v+n]
        elif model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            import itertools
            pts=np.array(list(itertools.product([-1,1],repeat=3)))*model.geom_size[g]
        else:raise RuntimeError(f'Unsupported collision shape {model.geom_type[g]}')
        local[int(g)]=pts.copy()
    wall_c=np.array([p['wall_x'],p['center_y'],p['opening_z']])+OFFSET
    wall_h=np.array([p['wall_thickness']/2,p['wall_half_width'],p['wall_half_height']])
    floor=-OFFSET[2];top=p['opening_z']-p['wall_half_height']
    stand_c=np.array([p['wall_x'],p['center_y'],(floor+top)/2])+OFFSET
    stand_h=np.array([.015,.042,(top-floor)/2])
    fixed=[('wall',wall_c-wall_h,wall_c+wall_h),('stand',stand_c-stand_h,stand_c+stand_h)]
    best=float('inf');closest=None
    for i,q in enumerate(trajectory['q']):
        robot.forward(q)
        for g,pts in local.items():
            world=pts@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
            lo,hi=world.min(0),world.max(0)
            for name,a,b in fixed:
                gap=np.maximum(np.maximum(a-hi,lo-b),0)
                lower=float(np.linalg.norm(gap))
                if lower<best:
                    best=lower;closest=[i,model.body(int(model.geom_bodyid[g])).name,name]
    result=dict(min_robot_fixture_clearance_lower_bound_mm=best*1000,closest=closest,sample_period_s=p['tick'],
                scope='Conservative AABB separation of all robot collision geometry from solid wall and stand, sampled every 10 ms. Not continuous-time collision certification or dynamics.')
    save(root/'franka'/f'fixture_audit_{k}.json',result)
    if best<=0:raise RuntimeError(f'Robot/fixture AABBs overlap, clearance not established: {result}')
    return result


def compile_motion(root,k,controls=None):
    p=read(root/'protocol.json');xml=snapshot(root);robot=Panda(xml)
    save(root/'execution_protocol.json',dict(planning_protocol_sha256=sha(root/'protocol.json'),
         robot='MuJoCo Menagerie Panda, kinematic joint trajectory driving the MPM grasp through FK',
         offset_m=OFFSET.tolist(),grasp_in_hand_m=MOUNT.tolist(),hand_basis_in_rod=BASIS.tolist(),
         finger_slide_m=SLIDE,bonded_grasp_length_m=p['grip_length'],
         scope='Adds robot kinematics to the frozen task-space planning protocol. No robot dynamics, torque or grasp-slip model.'))
    if controls is None:controls=read(root/f'plan_{k}/plan.json')['controls']
    time=np.arange(round(p['end']/p['tick'])+1)*p['tick']
    qs=[];cs=[];rs=[];errors=[];ncontacts=[]
    for t in time:
        c=np.array([p['grip_x']+translation(t,p),p['center_y'],controls[0]])
        R=rotation(tilt(t,controls[1],p))
        (cp,Rp),e=robot.solve(c,R)
        qs.append(robot.q.copy());cs.append(cp);rs.append(Rp);errors.append(e[:6])
        ncontacts.append(robot.data.ncon)
    qs=np.asarray(qs);errors=np.asarray(errors)
    speed=np.max(np.abs(np.diff(qs,axis=0)/p['tick']),axis=0)
    limits=np.array([2.175]*4+[2.61]*3)
    if np.any(speed>limits):raise RuntimeError(f'Joint speed limit: {speed}')
    if max(ncontacts)>0:raise RuntimeError(f'Robot self-contact at sampled poses: {max(ncontacts)}')
    out=root/'franka'/f'plan_{k}.npz'
    np.savez(out,time=time,q=qs,center=cs,rotation=rs,slide=SLIDE,offset=OFFSET,mount=MOUNT)
    audit=dict(controls=controls,trajectory_sha256=sha(out),samples=len(time),max_position_error_m=float(np.linalg.norm(errors[:,:3],axis=1).max()),
               max_angle_error_rad=float(np.linalg.norm(errors[:,3:6],axis=1).max()/.1),peak_joint_speed_rad_s=speed.tolist(),joint_speed_limits_rad_s=limits.tolist(),
               max_mujoco_self_contacts=int(max(ncontacts)),grip_pad_gap_mm=8.,bonded_grasp_mm=15.,
               scope='Joint position/speed and sampled IK feasibility. No robot dynamics, torque, or hardware validation. Fixture collision audit is separate.')
    audit['fixture']=fixture_audit(root,k,dict(q=qs))
    save(root/'franka'/f'audit_{k}.json',audit);print(audit,flush=True)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',default='A');ap.add_argument('--pilot',action='store_true')
    a=ap.parse_args();compile_motion(a.out,a.material,[.15,25.] if a.pilot else None)

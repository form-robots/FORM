"""Place the displayed robot on the green while preserving recorded grasp poses.

This retargets the robot visualization, not the MPM or ball simulation. Every
saved robot pose follows the original grasp; material trajectories are read only.
"""
from pathlib import Path
import json
import itertools
import numpy as np
import mujoco
from experiments.elastic.rod_insertion_franka import Panda,OFFSET
from experiments.elastic.putting_franka import Q,load_motion
from experiments.elastic.putting import save,digest

class ScenePanda(Panda):
    def __init__(self,xml,shift):
        self.shift=np.asarray(shift);super().__init__(xml)
    def forward(self,q):
        c,R=super().forward(q);return c+self.shift,R
    def meshes(self,include_arm=True):
        out=super().meshes(include_arm)
        for _,mesh,_ in out:mesh.points+=self.shift
        return out

def compile_scene(root):
    scene=json.loads((root/'render_scene.json').read_text());dest=root/'scene_robot';dest.mkdir(exist_ok=True)
    results={}
    for k in 'AB':
        prov=json.loads((root/f'execution_A_plan_{k}/execution_provenance.json').read_text());path=Path(prov['motion'])
        assert digest(path)==prov['motion_sha256']
        old=np.load(path);motion=load_motion(path,tick=.002)
        robot=ScenePanda(root/'franka/panda_model_snapshot/panda.xml',scene['robot_base_shift'])
        times=np.arange(round(old['time'][-1]/.002)+1)*.002;qs=[];errors=[];clear=np.inf;self_contacts=0;bounds=[]
        for t in times:
            c,R=motion(t);(_, _),e=robot.solve(c,R@Q);qs.append(robot.q.copy());errors.append(e[:6]);self_contacts=max(self_contacts,robot.data.ncon)
            # Moving collision meshes must stay above the green. Link 0 is the mounted base.
            for g in np.flatnonzero(robot.model.geom_group==3):
                model=robot.model;body=model.body(int(model.geom_bodyid[g])).name
                if body=='link0':continue
                if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                    m=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[m]);pts=model.mesh_vert[a:a+int(model.mesh_vertnum[m])]
                elif model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                    pts=np.array(list(itertools.product([-1,1],repeat=3)))*model.geom_size[g]
                else:raise RuntimeError('Unsupported collision mesh')
                world=pts@robot.data.geom_xmat[g].reshape(3,3).T+robot.data.geom_xpos[g]-OFFSET+robot.shift
                clear=min(clear,float(world[:,2].min()-scene['table_bounds'][5]))
            if round(t/.002)%50==0:
                for _,mesh,_ in robot.meshes():bounds.append([mesh.points.min(0),mesh.points.max(0)])
        qs=np.asarray(qs);errors=np.asarray(errors);speed=np.max(abs(np.diff(qs,axis=0))/.002,axis=0);limits=np.array([2.175]*4+[2.61]*3)
        assert self_contacts==0 and clear>0,(k,self_contacts,clear)
        assert np.all(speed<=limits),(k,speed)
        np.savez(dest/f'plan_{k}.npz',time=times,q=qs)
        results[k]=dict(original_motion_sha256=digest(path),scene_motion_sha256=digest(dest/f'plan_{k}.npz'),max_grasp_error_mm=float(np.linalg.norm(errors[:,:3],axis=1).max()*1000),max_rotation_error_rad=float(np.linalg.norm(errors[:,3:6],axis=1).max()/.1),peak_joint_speed_rad_s=speed.tolist(),joint_speed_limits_rad_s=limits.tolist(),min_moving_link_floor_clearance_mm=clear*1000,self_contacts=self_contacts,robot_bounds=[np.array(bounds)[:,0].min(0).tolist(),np.array(bounds)[:,1].max(0).tolist()])
        print(k,results[k],flush=True)
    save(dest/'audit.json',dict(plans=results,robot_base_shift=scene['robot_base_shift'],floor_z=scene['table_bounds'][5],scene_sha256=digest(root/'render_scene.json'),scope='Visualization retargeting at 2 ms. The original grasp trajectory is reproduced with a raised robot base. Moving-link floor clearance, sampled self-contact and joint speed are checked; the static mounted base is excluded from the floor-clearance test. No physics simulation was rerun.'))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();compile_scene(a.out)

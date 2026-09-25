"""Summarize every reported execution and preserve a reproducible experiment bundle."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import importlib.metadata
import numpy as np
from experiments.elastic.putting import ROOT,save,digest
from experiments.elastic.putting_franka import load_motion

def read(path):return json.loads(Path(path).read_text())

def audit_case(path):
    result=read(path/'result.json');config=read(path/'config.json');p=config['protocol']
    provenance=read(path/'execution_provenance.json');motion_path=Path(provenance['motion'])
    assert digest(motion_path)==provenance['motion_sha256']
    motion=load_motion(motion_path);data=np.load(path/'trajectory.npz');s=data['signals'];x=data['x'];ref=data['reference']
    assert np.isfinite(x).all() and np.isfinite(s).all()
    if len(x):assert result['runout_s']==0, 'Video must include the ball coming to rest, without freezing the club.'
    all_ball=np.vstack([s[:,:10],data['runout']]) if len(data['runout']) else s[:,:10]
    target=np.array(p['target']);error=np.linalg.norm(np.array(result['final_ball_m'])[:2]-target)*1000
    np.testing.assert_allclose(error,result['error_mm'],atol=1e-8)
    # A short numerical check can stop between saved 10 ms samples. During
    # free runout the speed decreases, so the last speed bounds this distance.
    lag=s[-1,0]+result['runout_s']-all_ball[-1,0]
    saved_gap=np.linalg.norm(all_ball[-1,1:4]-result['final_ball_m'])
    assert -.0000001<=lag<=p['video_dt']+.0000001
    assert saved_gap<=np.linalg.norm(all_ball[-1,4:7])*max(lag,0)+1e-8
    footprint=p['ball_radius'];b=p['table_bounds']
    assert (all_ball[:,1]>b[0]+footprint).all() and (all_ball[:,1]<b[1]-footprint).all()
    assert (all_ball[:,2]>b[2]+footprint).all() and (all_ball[:,2]<b[3]-footprint).all()
    core=(abs(ref[:,2])<.003)&(abs(ref[:,0])<.003)&(abs(ref[:,1])<.002)
    errors=[]
    for si,xi in zip(s,x):
        c,R=motion(si[0]);errors.append(np.linalg.norm(xi[core]-(ref[core]@R.T+c),axis=1).max())
    moving=np.flatnonzero(np.linalg.norm(all_ball[:,4:7],axis=1)>1e-5)
    result.update(max_grasp_core_pose_error_mm=float(max(errors)*1000) if errors else None,
        last_saved_ball_position_lag_mm=float(saved_gap*1000),
        stopping_time_s=float(all_ball[moving[-1],0]) if len(moving) else 0.,
        numerics=dict(grid=p['grid'],dt=p['coupling_dt']/int(np.ceil(p['coupling_dt']/p['dt'])),requested_max_dt=p['dt'],coupling_dt=p['coupling_dt']),
        case_sha256={name:digest(path/name) for name in ['config.json','result.json','execution_provenance.json']})
    return result

def audit_scene(root):
    """Check the enlarged display surface against the frozen robot commands."""
    import itertools
    import mujoco
    from experiments.elastic.rod_insertion_franka import Panda,OFFSET
    scene=read(root/'render_scene.json');p=read(root/'protocol.json')
    bounds=np.asarray(scene['table_bounds']);original=np.asarray(p['table_bounds'])
    assert np.all(bounds[[0,2]]<=original[[0,2]]) and np.all(bounds[[1,3]]>=original[[1,3]])
    np.testing.assert_array_equal(bounds[4:],original[4:])
    if 'robot_base_shift' in scene:
        audit=read(root/'scene_robot/audit.json')
        assert audit['robot_base_shift']==scene['robot_base_shift']
        assert audit['floor_z']==bounds[5]
        for k,r in audit['plans'].items():
            prov=read(root/f'execution_A_plan_{k}/execution_provenance.json')
            assert r['original_motion_sha256']==digest(prov['motion'])==prov['motion_sha256']
            assert r['scene_motion_sha256']==digest(root/f'scene_robot/plan_{k}.npz')
            assert r['max_grasp_error_mm']<.001 and r['max_rotation_error_rad']<1e-5
            assert r['min_moving_link_floor_clearance_mm']>0 and r['self_contacts']==0
            assert np.all(np.array(r['peak_joint_speed_rad_s'])<=r['joint_speed_limits_rad_s'])
        save(root/'scene_audit.json',dict(scene=scene,plans=audit['plans'],scope=audit['scope']))
        return audit['plans']
    lo,hi=bounds[::2]+OFFSET,bounds[1::2]+OFFSET
    robot=Panda(root/'franka/panda_model_snapshot/panda.xml');model=robot.model;vertices={}
    for g in np.flatnonzero(model.geom_group==3):
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            m=int(model.geom_dataid[g]);v=int(model.mesh_vertadr[m]);n=int(model.mesh_vertnum[m]);points=model.mesh_vert[v:v+n].copy()
        elif model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            points=np.array(list(itertools.product([-1,1],repeat=3)))*model.geom_size[g]
        else:raise RuntimeError('Unsupported fixture-audit geometry')
        vertices[int(g)]=points
    result={}
    for k in 'AB':
        provenance=read(root/f'execution_A_plan_{k}/execution_provenance.json')
        path=Path(provenance['motion']);assert digest(path)==provenance['motion_sha256']
        data=np.load(path);clearance=np.inf
        for q in data['q']:
            robot.forward(q)
            for g,pts in vertices.items():
                world=pts@robot.data.geom_xmat[g].reshape(3,3).T+robot.data.geom_xpos[g]
                gap=np.maximum(np.maximum(lo-world.max(0),world.min(0)-hi),0)
                clearance=min(clearance,float(np.linalg.norm(gap)))
        assert clearance>0,(k,clearance)
        result[k]=dict(min_fixture_clearance_mm=clearance*1000,motion_sha256=digest(path))
    save(root/'scene_audit.json',dict(scene=scene,plans=result,scope='Conservative collision-mesh AABB checks at every recorded 10 ms robot waypoint. The original physical floor is an infinite plane; the added green is display scenery outside the original contact region.'))
    return result

def summarize(root,prefix='execution'):
    reuse=all((root/f'{prefix}_{m}_plan_{k}').is_symlink() for m in 'AB' for k in 'AB')
    p=read(root/'protocol.json');audit_scene(root);report=dict(protocol=p,plans={},executions={},numerical_checks={})
    for k in 'AB':
        prov=read(root/f'{prefix}_A_plan_{k}/execution_provenance.json')
        report['plans'][k]=read(prov['plan_path'])
    for folder in sorted(root.glob('*_?_plan_?')):
        if not (folder/'execution_provenance.json').exists() or not (folder/'result.json').exists():continue
        m,k=folder.name.split('_')[-3],folder.name.split('_')[-1]
        cfg=read(folder/'config.json');prov=read(folder/'execution_provenance.json');plan=read(prov['plan_path'])
        assert digest(prov['plan_path'])==prov['plan_sha256']
        assert cfg['E_pa']==p['E_true'][m] and cfg['controls']==plan['controls']
        assert prov['plan_material']==k and prov['executed_material']==m
        for key,value in p.items():
            if key not in ['grid','dt','coupling_dt','end','video_dt']:assert cfg['protocol'][key]==value
        group='executions' if folder.name.startswith(prefix+'_') else 'numerical_checks'
        report[group][folder.name]=audit_case(folder)
    for k in 'AB':
        pa=read(root/f'{prefix}_A_plan_{k}/execution_provenance.json')
        pb=read(root/f'{prefix}_B_plan_{k}/execution_provenance.json')
        assert pa['motion_sha256']==pb['motion_sha256'] and pa['plan_sha256']==pb['plan_sha256']
    save(root/'report.json',report)
    dest=root/'final_sources';dest.mkdir(exist_ok=True)
    sources=list((ROOT/'experiments/elastic').glob('putting*.py'))+[ROOT/'tests/test_putting.py']
    for file in sources:shutil.copy2(file,dest/file.name)
    relevant=sources+[ROOT/f'experiments/elastic/{name}.py' for name in ['rod_insertion_franka','rod_insertion_study','block_drop_study','block_drop_report']]+list((ROOT/'src/warpmpm').rglob('*.py'))
    save(root/'reproduction_source_hashes.json',{str(f.relative_to(ROOT)):digest(f) for f in relevant})
    versions={}
    for package in ['numpy','scipy','warp-lang','mujoco','pyvista','vtk','imageio','imageio-ffmpeg','scikit-image','Pillow','torch']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='unknown'
    save(root/'environment.json',dict(python=sys.version,packages=versions,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gpu_and_driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        new_sources='New putting modules are saved in final_sources; the git revision alone does not contain this experiment.'))
    lines=['# Elastic-club putting: simulation report','',
        'The robot makes one prescribed stroke with a flexible club. The ball must come to rest entirely inside a marked circle, 612 mm from its starting center. The circle radius is 35 mm; the ball radius is 8 mm, so success requires a center error of at most 27 mm.','',
        '| True material | Plan using matched ID | Plan using swapped ID |',
        '|---|---:|---:|']
    for m in 'AB':
        a=report['executions'][f'{prefix}_{m}_plan_{m}'];other='B' if m=='A' else 'A';b=report['executions'][f'{prefix}_{m}_plan_{other}']
        def value(v):return f"{v['error_mm']:.1f} mm ({'inside' if v['success'] else 'outside'})"
        lines.append(f'| {m} | {value(a)} | {value(b)} |')
    fine_matches=[report['numerical_checks'].get(f'fine_{m}_plan_{m}') for m in 'AB']
    if all(r is not None for r in fine_matches):
        lines+=['',f"**Precision qualification.** On the finer grid, matched errors are {fine_matches[0]['error_mm']:.1f} mm (A) and {fine_matches[1]['error_mm']:.1f} mm (B). The exact stopping errors depend on numerical resolution; they are not a demonstrated continuum-accuracy claim. Further contact/grid convergence work is needed before claiming millimeter precision."]
    lines+=['','**Control.** Optimize two continuous parameters: stroke duration and aim angle. The robot addresses the ball, takes a 28° backswing over 0.55 s, pauses for 0.08 s, and swings through to −22° before withdrawal. The grasp follows a 100 mm-radius arc; the undeformed head center follows a 167.5 mm-radius arc. Both stages use smooth quintic time profiles. The complete Franka command is frozen and reused on both materials. There is no force, shape, or ball-position feedback.','',
        '| Plan | Identified stiffness | Stroke duration | Aim | Simulations in selected search |',
        '|---|---:|---:|---:|---:|']
    for k,pl in report['plans'].items():lines.append(f"| {k} | {pl['E_pa']/1000:.2f} kPa | {pl['controls'][0]:.4f} s | {pl['controls'][1]:.2f}° | {pl['evaluations']} |")
    lines+=['','**Identification inputs.** These are the existing textured-stereo bending estimates: A = 80.61 kPa and B = 248.11 kPa, against true stiffnesses of 80 and 240 kPa. They were not fitted to putting outcomes. The same homogeneous fixed-corotated material model is used in the new club geometry. The original bending reconstruction assumptions still apply.','',
        '**Numerical checks.**','',
        'The displayed executions use a 1.0 mm MPM grid and 125 μs coupling updates. The fine check uses a 0.75 mm grid; the contact-update check uses 62.5 μs updates on the 1.0 mm grid. The frozen commands are identical in every check. Integer substepping gives MPM steps of 8.33 μs in the main run, 6.25 μs on the finer grid, and 7.81 μs in the half-contact-interval check.','',
        '| Check | True material / plan | Center error | Change from displayed execution |',
        '|---|---|---:|---:|']
    for name,r in report['numerical_checks'].items():
        parts=name.split('_');m,k=parts[-3],parts[-1];base=report['executions'][f'{prefix}_{m}_plan_{k}']
        change=np.linalg.norm(np.array(r['final_ball_m'])-base['final_ball_m'])*1000
        lines.append(f"| {name.rsplit('_',3)[0]} | {m} / {k} | {r['error_mm']:.1f} mm | {change:.1f} mm |")
    if all(f'{pre}_{m}_plan_{k}' in report['numerical_checks'] for pre in ['fine','half_contact'] for m in 'AB' for k in 'AB'):
        consistent=all(report['numerical_checks'][f'{pre}_{m}_plan_{k}']['success']==(m==k) for pre in ['fine','half_contact'] for m in 'AB' for k in 'AB')
        lines+=['',f"The matched-success / swapped-failure pattern {'holds' if consistent else 'does not hold'} in both numerical checks. This is a task-outcome comparison, not a convergence claim for the exact stopping positions."]
    if 'repeat_fine_A_plan_A' in report['numerical_checks'] and 'fine_A_plan_A' in report['numerical_checks']:
        delta=np.linalg.norm(np.array(report['numerical_checks']['repeat_fine_A_plan_A']['final_ball_m'])-report['numerical_checks']['fine_A_plan_A']['final_ball_m'])*1000
        lines+=['',f'An independent repeat of the finer-grid A matched case changes the stopping point by {delta:.3f} mm.']
    lines+=['','The initial hanging state is prepared by numerical dynamic relaxation. That damping is disabled before the stroke, and elastic deformation is retained. Every resolution uses the same integrated club volume. The rigid-ball model passes independent rolling-distance and sliding-to-rolling checks.','',
        '**Scope.** This is a simulation experiment, with kinematic Franka motion and a bonded grasp. Ball mass, geometry, floor friction and rolling resistance are prescribed and known. Club–ball contact is frictionless; no material damping is identified. Robot torque limits, grasp slip, uncertain surface mechanics, and hardware transfer have not been validated.','',
        '**Development record.** The archived straight-stroke baseline is `out/putting_20260913/settled`. This study changes the motion to a backswing and forward arc while keeping the physical club geometry, materials, ball, floor and target unchanged. All planning evaluations are retained. No swapped-case error enters the planning objective. The selected motion is the earlier backswing, restored after a later prepared-strike variant. Wider scenery and revised lighting do not change the physical contact model.','',
        '**Videos.** [Synchronized stroke and outcome](media/paper_putting_demo.mp4), [Stroke sequence](media/swing_sequence.png), [Comparison](media/putting_comparison.mp4), [contact close-up](media/putting_contact_closeup.mp4), [full Franka view](media/franka_putting.mp4). The main and contact videos run at real time at 50 fps. A separate 2× contact clip is labeled. The green is wider and the lighting uses ambient occlusion; ball, club, robot and target states are the original recorded states. The enlarged surface passes the recorded robot-clearance checks. Surfaces are interpolated from saved MPM particles; the ball stripe follows its integrated angular velocity.','',
        '**Reproducibility.** The original physics outputs are preserved at `out/putting_swing_20260913`. The later prepared-strike code is archived at `archive/putting_prepared_strike_20260913`. Protocol, identification JSON files, every completed planning trial, frozen commands, robot assets, source snapshots, environment versions, and numerical checks are saved here. The manuscript and its PDF are unchanged.','']
    lines+=['The displayed executions are linked unchanged from the original study; no physics was rerun for this presentation.' if reuse else 'This directory contains newly executed simulation cases.', '', 'To render the saved states without running physics:', '', '```bash', f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.putting_report all --out {root.resolve()}', '```', '']
    lines+=['To repeat both searches, all four executions, numerical checks, and videos from the frozen inputs:', '',
        '```bash',f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.putting_reproduce --source {root.resolve()} --out out/putting_repeat',
        '```','', 'This command requires the recorded source version and two CUDA devices. GPU reductions can produce small floating-point differences; source hashes and the protocol are checked, but bitwise replay is not assumed.','']
    if 'robot_base_shift' in read(root/'render_scene.json'):
        for i,line in enumerate(lines):
            if line.startswith('**Videos.**'):
                lines[i]='**Video.** [Four-panel side view](media/putting_comparison.mp4). Rows show true materials A and B; columns show plans using matched and swapped IDs. All panels use the same camera and real-time 50 fps playback, with a one-second hold after ball rest. Material, ball, grasp and target trajectories are unchanged. The robot base is moved onto the green and its displayed joint poses are recomputed to follow the original grasp trajectory. This is visualization retargeting, not a new coupled physics run. At 2 ms samples, the new arm poses pass joint-speed, self-contact and moving-link floor-clearance checks; the largest grasp-position discrepancy is below 0.00002 mm. This does not establish continuous-time collision safety or torque feasibility. The green is display scenery for the existing infinite physical floor. Surfaces are interpolated from stored particles; the ball stripe follows its integrated angular velocity.'
    if read(root/'render_scene.json').get('target_flag'):
        lines+=['**Presentation.** This closer camera emphasizes the wrist, club and ball path. The flag stands behind the target circle and is a decorative landmark only; it is not a simulated obstacle. Any distant terrain and sky are also scenery. The physical floor, stopping-circle metric, ball and club states remain unchanged. The previous full-arm side-view video is preserved at `out/putting_sideview_20260913`.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prefix',default='execution')
    a=ap.parse_args();summarize(a.out,a.prefix)

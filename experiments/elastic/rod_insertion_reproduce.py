"""Reproduce frozen-image-estimate rod planning, Panda execution, and media."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiments.elastic.strip_insertion_reproduce import batch,provenance
from experiments.elastic.block_drop_study import ROOT,save,sha
from experiments.elastic.rod_insertion_study import read


def report(root):
    from experiments.elastic.rod_insertion_report import summarize
    summarize(root);r=read(root/'report.json');p=r['protocol']
    lines=['# Rod insertion with a kinematic Franka','',
        'The robot holds an elastic rod, rotates its wrist to compensate for gravitational bending, and advances through a circular opening. '
        'The identified stiffness selects grasp height and wrist tilt. The complete joint trajectory is frozen before execution; there is no force or shape feedback.','',
        '## Geometry and material transfer','',
        '- Rod: 60 mm long, 8 mm diameter; proximal 15 mm held by a bonded grasp. Circular bore: 12 mm diameter through a 6 mm wall. Nominal radial clearance is 2 mm.',
        '- The earlier identification used 100 × 12 × 4 mm strips, synthetic textured stereo and pusher force. We transfer the identified homogeneous isotropic stiffness to a different specimen geometry. We do not identify a new constitutive family or use hidden simulator states for identification.',
        '- Identified E: A 80.608894 kPa; B 248.112326 kPa. Execution E: A 80 kPa; B 240 kPa. Known ν = 0.45 and density = 1000 kg/m³. No material retuning.',
        '- The old ribbon required a large aperture for its bent shape during straight insertion. The circular section has similar area but about 3.14 times its weak-axis second moment of area. This motivated a new feasibility test, not a selection on swapped outcomes. The old experiment is retained.','',
        '## Control and evaluation','',
        '- Two optimized variables: constant grasp height and final wrist tilt. Begin hanging, rotate smoothly over 3 s, settle 0.5 s, advance 51 mm over 3 s, hold 0.5 s. Peak advance speed is 25.5 mm/s.',
        '- Each identified model uses the same bounded Nelder–Mead search and 48-rollout budget. Objective: final-hold tip position error, distal-axis alignment and a whole-rod bore-clearance penalty. The free-space planning rollouts use geometric wall checks; predicted replay and final execution use physical frictionless wall contact.',
        '- Panda IK compiles the selected motion into joints. Forward kinematics of those joints drives the MPM grasp. Identical complete commands are executed in both materials. True parameters and swapped outcomes never enter planning.',
        '- Success: all particles in the distal 2 mm band cross inside the circular bore, remain at least 10 mm beyond the rear wall throughout the final 0.5 s, and wall reaction remains ≤ 10 µN at each 10 ms sample. Depth/clearance use particle centers. Planning adds particle padding plus 1.4 mm clearance margin.',
        '- Tip error is RMS centroid-to-target distance over the final hold. It is secondary to collision-free insertion, not a selected-frame metric.','',
        '| Plan | Height relative to bore center (mm) | Wrist tilt (deg) | Simulations |', '|---|---:|---:|---:|']
    for k,v in r['plans'].items():lines.append(f"| {k} | {(v['controls'][0]-.15)*1000:+.3f} | {v['controls'][1]:.3f} | {v['evaluations']} |")
    lines+=['','## Results','', '| Grid | Material | Plan | Success | Minimum depth (mm) | Tip error (mm) | Peak wall force (N) |','|---|---|---|---|---:|---:|---:|']
    for n,v in r['executions'].items():
        m,k=n.split('_')[1],n.split('_')[3]
        lines.append(f"| {v['grid']} | {m} | {'Matched' if m==k else 'Swapped'} | {v['success']} | {v['min_tip_depth_hold_mm']:.3f} | {v['tip_error_mm']:.3f} | {v['peak_wall_force_N']:.6g} |")
    lines+=['','## Robot and limits of the evidence','',
        '- Stock Panda geometry and joint limits are taken from the archived MuJoCo Menagerie model. Finger slides are 2.5 mm each, giving 8 mm between pad inner faces. The simulated grasp is bonded, with no slip/contact-pressure prediction at the fingers.',
        '- Kinematic arm only: no joint torque, acceleration/jerk limits, grasp force limits, or hardware execution are validated. Joint positions, sampled speeds, pose errors and self-contact are audited separately from material insertion.',
        '- Full 3D fixed-corotated elasticity is used for planning/execution, with no added damping or planar constraint. Identification retains its previously documented camera/reconstruction assumptions.',
        '- Both grids use unchanged plans. A finer-grid check measures sensitivity; it does not establish continuum convergence. Any disagreement must be reported.',
        '- Videos use the finer-grid saved particle states, actual Panda FK and identical cameras/scales/timing. Material-coordinate affine interpolation draws the cylindrical surface; numerical metrics use raw particles. The wall is translucent for visibility.','',
        '## Reproduction','', '```bash',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.rod_insertion_reproduce --out out/rod_insertion_REPEAT --devices cuda:0,cuda:1',
        '```','',
        'The default frozen identification source is out/strip_texture_20260912. Use --source for another reproduction of the same identification. '
        'Source, input, plan, robot asset, joint trajectory and output hashes accompany this report. All development pilots remain archived. The manuscript is not modified by this command.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');(root/'REPRODUCE.md').write_text('\n'.join(lines[lines.index('## Reproduction'):])+'\n')


def run(root,devices,finish=False):
    base=[sys.executable,'-m','experiments.elastic.rod_insertion_study'];log=root/'logs'
    if not finish:
        batch([base+['plan','--out',str(root),'--material',k,'--device',d] for k,d in zip(['A','B'],devices)], [log/f'plan_{k}.log' for k in ['A','B']])
    from experiments.elastic.rod_insertion_franka import compile_motion
    for k in ['A','B']:compile_motion(root,k)
    for fine in [False,True]:
        for k in ['A','B']:
            batch([base+['execute','--out',str(root),'--material',m,'--plan',k,'--device',d]+(['--fine'] if fine else []) for m,d in zip(['A','B'],devices)],
                  [log/f"{'fine' if fine else 'execution'}_{m}_plan_{k}.log" for m in ['A','B']])
    report(root)
    vis=[sys.executable,'-m','experiments.elastic.rod_insertion_report']
    for k in ['A','B']:
        batch([vis+['render','--out',str(root),'--material',m,'--plan',k,'--fine'] for m in ['A','B']],
              [log/f'render_{m}_plan_{k}.log' for m in ['A','B']])
    subprocess.run(vis+['compose','--out',str(root),'--fine'],check=True,cwd=ROOT)
    batch([vis+['render','--out',str(root),'--material',m,'--plan',m,'--fine','--overview'] for m in ['A','B']],
          [log/f'overview_{m}.log' for m in ['A','B']])
    archive(root)


def archive(root):
    provenance(root)
    import shutil
    import importlib.metadata
    sources=read(root/'source_hashes.json')
    for relative in ['tests/test_rod_insertion.py','docs/reproduction.md']:
        f=ROOT/relative;dst=root/'source'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
        sources[relative]=sha(f)
    save(root/'source_hashes.json',sources)
    environment=read(root/'environment.json')
    environment['packages'].update({k:importlib.metadata.version(k) for k in ['mujoco','robot-descriptions']})
    save(root/'environment.json',environment)
    paths=[f for f in (root/'franka').rglob('*') if f.is_file()]+[root/'execution_protocol.json']
    hashes=read(root/'artifact_hashes.json')
    hashes.update({str(f.relative_to(root)):sha(f) for f in paths})
    save(root/'artifact_hashes.json',hashes)
    verify={str(f.relative_to(root)):sha(f)==hashes[str(f.relative_to(root))] for f in paths}
    core=read(root/'planning_source_audit.json') if (root/'planning_source_audit.json').exists() else {}
    core_unchanged=all(sha(ROOT/f)==v['sha256'] for f,v in core.items())
    r=read(root/'report.json')
    all_complete=all(v['all_frames_completed'] and v['inverted_count']==0 for v in r['executions'].values())
    save(root/'verification.json',dict(robot_and_execution_hashes_verified=all(verify.values()),core_unchanged_during_run=core_unchanged,
         executions=len(r['executions']),all_completed_without_detected_inversion=all_complete,
         note='Verification reports validity and provenance, not success of every manipulation.'))
    for f in [root/'verification.json',root/'source_hashes.json',root/'environment.json']:
        hashes[str(f.relative_to(root))]=sha(f)
    save(root/'artifact_hashes.json',hashes)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source',type=Path,default=Path('out/strip_texture_20260912'))
    ap.add_argument('--devices',default='cuda:0,cuda:1');ap.add_argument('--finish',action='store_true');ap.add_argument('--report-only',action='store_true');a=ap.parse_args()
    if a.report_only:report(a.out);archive(a.out)
    else:
        if not a.finish:
            from experiments.elastic.rod_insertion_study import initialize
            if a.out.exists():raise FileExistsError(a.out)
            initialize(a.out,a.source)
        dev=a.devices.split(',');run(a.out,(dev+dev)[:2],a.finish)

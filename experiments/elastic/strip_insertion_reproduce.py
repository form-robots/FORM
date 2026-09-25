"""Reproduce planning, crossed executions, resolution checks, and insertion media."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from experiments.elastic.block_drop_study import ROOT,save,sha
from experiments.elastic.strip_insertion_study import read


def batch(commands,logs):
    processes=[]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for cmd,log in zip(commands,logs,strict=True):
        log.parent.mkdir(parents=True,exist_ok=True)
        f=log.open('w');processes.append((subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT),f,cmd))
    failed=[]
    for proc,f,cmd in processes:
        ret=proc.wait();f.close()
        if ret:failed.append((ret,cmd))
    if failed:raise RuntimeError(failed)


def report(root):
    from experiments.elastic.strip_insertion_report import summarize
    r=summarize(root);p=r['protocol'];lines=[
        '# Elastic-strip insertion: identified models and open-loop planning','',
        'The robot holds a fresh strip, rotates its grip, and translates forward through a rectangular aperture. '
        'Only grip height and final tilt are optimized; the full motion is then frozen.','',
        '## Task and controls','',
        f"- Planning specimen: {p['size'][0]*1000:g} × {p['size'][1]*1000:g} × {p['size'][2]*1000:g} mm; {p['grip_length']*1000:g} mm held in an ideal bonded grip.",
        '- Identification used a 100 × 12 × 4 mm strip. The material pair, cross-section, known Poisson ratio and density are unchanged. This is transfer to a shorter specimen, not reidentification.',
        f"- Opening: {p['opening_width']*1000:g} mm wide × {p['opening_height']*1000:g} mm high in a {p['wall_thickness']*1000:g} mm wall.",
        f"- Target: tip centroid {p['target_depth']*1000:g} mm beyond the rear wall face and {-p['target_z_offset']*1000:g} mm below the aperture center. Its lower position allows clearance for the sagging upstream strip.",
        f"- Motion: hanging start, rotate over 3 s, wait 0.5 s, advance {p['advance']*1000:g} mm over 3 s, hold 0.5 s. Maximum forward grip speed is {1.5*p['advance']/3*1000:g} mm/s. No force or shape feedback.",
        '- Each bounded Nelder–Mead search starts from the same height/tilt with the same evaluation budget. It minimizes final-hold tip error plus a penalty for the whole strip obstructing the aperture. Planning uses free-space MPM with geometric collision checks; final predictions and all executions include physical wall contact.',
        '- Swapped ID exchanges the entire height/rotation/translation trajectory. True material parameters are used only by the execution simulator, never by the optimizer.','',
        '## Frozen plans','',
        '| Plan | Identified E (kPa) | Grip height above opening center (mm) | Tilt (deg) | Planning simulations |',
        '|---|---:|---:|---:|---:|']
    for k,v in r['plans'].items():lines.append(f"| {k} | {v['E_pa']/1000:.5f} | {(v['controls'][0]-p['opening_z'])*1000:+.3f} | {v['controls'][1]:.3f} | {v['evaluations']} |")
    lines+=['','## Execution results','',
        'Success requires every particle in the terminal 2 mm tip band to cross through the aperture, '
        f"remain at least {p['required_depth']*1000:g} mm past the rear wall face during the final hold, and have no resolved wall contact (summed wall-reaction magnitude ≤ {p['contact_force_threshold']:g} N at every 10 ms sample). "
        'Reported tip depth and crossing clearances use particle centers; geometry planning includes particle-size padding and a 2 mm clearance margin. '
        'Tip error is RMS distance of the tip centroid to the fixed target over the entire final 0.5 s hold, not a selected frame.','',
        '| Grid | Material | Plan | Outcome | Tip error (mm) | Min tip depth (mm) | Peak wall force (N) |',
        '|---|---|---|---|---:|---:|---:|']
    for name,v in r['executions'].items():
        parts=name.split('_');m,pl=parts[1],parts[3]
        label='Inserted, no contact' if v['success'] else ('Wall contact' if v['peak_wall_force_N']>p['contact_force_threshold'] else 'Insufficient insertion / aperture crossing')
        lines.append(f"| {v['grid']} | {m} | {'Matched' if m==pl else 'Swapped'} | {label} | {v['tip_error_mm']:.3f} | {v['min_tip_depth_hold_mm']:.3f} | {v['peak_wall_force_N']:.6g} |")
    lines+=['','The videos and preview show the finer-grid executions. Absolute tip accuracy is resolution-sensitive:','']
    for m in ['A','B']:
        coarse=r['executions'].get(f'execution_{m}_plan_{m}');fine=r['executions'].get(f'fine_{m}_plan_{m}')
        if coarse and fine:
            lines.append(f"- Matched {m}: {coarse['tip_error_mm']:.3f} mm on the planning grid versus {fine['tip_error_mm']:.3f} mm on the finer grid, with unchanged commands.")
    lines+=['','## What this does and does not establish','',
        '- This is a 3D MPM manipulation test using existing texture-and-force stiffness estimates: A 80.608894 kPa, B 248.112326 kPa. Reference execution stiffness is 80/240 kPa. Only E was identified; the elastic family, ν = 0.45 and density 1000 kg/m³ were known.',
        '- The earlier identification used synthetic calibrated stereo images and noiseless pusher force, with plane-stress / through-thickness reconstruction assumptions. This task does not turn that result into hardware validation or constitutive-family discovery.',
        '- Planning and execution use full 3D fixed-corotated elasticity. No planar kinematic constraint, force tracking, added damping, shape feedback, state reset, material adjustment, or swapped-score optimization is used.',
        '- The grip is a prescribed bonded boundary; the render illustrates a gripper fixture. There is no Franka inverse kinematics, robot-arm dynamics, grasp-slip model, or hardware demonstration.',
        f"- Planning grid: {p['grid']}³ ({p['domain']/p['grid']*1000:g} mm). Check grid: {p['fine_grid']}³ ({p['domain']/p['fine_grid']*1000:g} mm), with exactly the same plans. This is a resolution sensitivity check, not proof of continuum convergence.",
        '- The fixture is rendered translucent so the tip remains visible after entry. All cases share camera, scale and timing. Surfaces interpolate the saved particle lattice without changing the simulated states.','',
        '## Development and reproducibility','',
        'The initial 100 mm strip held on edge twisted strongly; an abruptly released flat strip also left the computational domain. A flat orientation with slow rotation from hanging was stable, '
        'but the long hanging section did not clear a compact vertical opening in these pilots. A deeper 40 mm grasp was also examined. '
        'The accepted task uses a 60 mm specimen and the original 15 mm grasp. Early pilots and the interrupted long-strip searches '
        'are retained in development/. The final aperture, target, motion, bounds and objective were frozen before the final two searches '
        'and before any true-material crossed execution. No trial was selected to worsen a swapped outcome.','',
        '```bash',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.strip_insertion_reproduce --out out/strip_insertion_REPEAT --devices cuda:0,cuda:1',
        '```','',
        'The command uses frozen estimates from out/strip_texture_20260912 by default; --source can point to a fresh reproduction of that identification experiment. '
        'It recreates both searches, all eight executions, audits, video and source/artifact hashes. The manuscript is unchanged.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'REPRODUCE.md').write_text('\n'.join(lines[lines.index('## Development and reproducibility'):])+'\n')


def provenance(root):
    paths=list((ROOT/'src/warpmpm').rglob('*.py'))+list((ROOT/'src/common').rglob('*.py'))
    paths+=list((ROOT/'experiments/elastic').glob('*.py'))+[ROOT/'tests/test_strip_insertion.py',ROOT/'experiments/robotics/plastic_shaping_figure.py']
    hashes={}
    for f in paths:
        dst=root/'source'/f.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
        hashes[str(f.relative_to(ROOT))]=sha(f)
    save(root/'source_hashes.json',hashes)
    save(root/'environment.json',dict(python=platform.python_version(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','warp-lang','pyvista','vtk','Pillow','imageio']},
         gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv'],text=True).strip()))
    paths=[root/'protocol.json',root/'report.json',root/'REPORT.md',root/'REPRODUCE.md']
    paths+=[f for f in [root/'verification.json',root/'planning_source_audit.json'] if f.exists()]
    for k in ['A','B']:
        paths+=list((root/f'plan_{k}').glob('*.json'))+list((root/f'plan_{k}').glob('*.csv'))
    paths+=list((root/'inputs').glob('*'))+list((root/'media').glob('*.mp4'))+list((root/'media').glob('*.png'))
    for case in list(root.glob('execution_*'))+list(root.glob('fine_*')):paths+=list(case.glob('*.json'))+list(case.glob('*.csv'))+list(case.glob('*.npy'))
    save(root/'artifact_hashes.json',{str(f.relative_to(root)):sha(f) for f in paths if f.is_file()})


def finish(root,dev):
    base=[sys.executable,'-m','experiments.elastic.strip_insertion_study']
    log=root/'logs'
    for fine in [False,True]:
        for plan in ['A','B']:
            batch([base+['execute','--out',str(root),'--material',m,'--plan',plan,'--device',d]+(['--fine'] if fine else []) for m,d in zip(['A','B'],dev)],
                  [log/f"{'fine' if fine else 'execution'}_{m}_plan_{plan}.log" for m in ['A','B']])
    report(root)
    vis=[sys.executable,'-m','experiments.elastic.strip_insertion_report']
    for m in ['A','B']:
        for pl in ['A','B']:subprocess.run(vis+['render','--out',str(root),'--material',m,'--plan',pl,'--fine'],check=True,cwd=ROOT)
    subprocess.run(vis+['compose','--out',str(root),'--fine'],check=True,cwd=ROOT)
    provenance(root)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source',type=Path,default=Path('out/strip_texture_20260912'));ap.add_argument('--devices',default='cuda:0,cuda:1');ap.add_argument('--report-only',action='store_true');a=ap.parse_args()
    if a.report_only:report(a.out);provenance(a.out);return
    if a.out.exists():raise FileExistsError(a.out)
    devices=a.devices.split(',');dev=(devices+devices)[:2]
    base=[sys.executable,'-m','experiments.elastic.strip_insertion_study']
    subprocess.run(base+['init','--out',str(a.out),'--source',str(a.source)],check=True,cwd=ROOT)
    log=a.out/'logs'
    batch([base+['plan','--out',str(a.out),'--material',k,'--device',d] for k,d in zip(['A','B'],dev)], [log/f'plan_{k}.log' for k in ['A','B']])
    finish(a.out,dev)


if __name__=='__main__':main()

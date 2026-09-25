"""Reproducible visual report for the thin, texture-observed elastic strip."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
from scipy.interpolate import RegularGridInterpolator

from experiments.elastic.block_drop_report import font
from experiments.elastic.strip_camera_observe import CAMERA,save
from experiments.elastic.strip_texture_observe import meshes,texture
from experiments.elastic.block_drop_study import sha

COLORS={'A':'#267aa5','B':'#be703e'}
INK='#24323b'


def read(p): return json.loads(Path(p).read_text())
def signals(p): return np.genfromtxt(p/'signals.csv',delimiter=',',names=True)


def summarize(root):
    p=read(root/'protocol.json'); audit=read(root/'observation_audit.json'); result={}
    for k in ['A','B']:
        fit=read(root/f'fit_{k}/identification.json'); truth=p['E_pa'][k]
        row=dict(true_E_pa=truth,identified_E_pa=fit['E_pa'],signed_error_percent=100*(fit['E_pa']/truth-1),
                 fit=fit,tracking=read(root/f'fit_{k}/tracking.json'),observation_audit=audit[k])
        for label,ref,pred in [('main','truth','prediction'),('fine','fine','fine_prediction')]:
            a=root/f'{ref}_{k}'; b=root/f'{pred}_{k}'
            xa=np.load(a/'x.npy',mmap_mode='r'); xb=np.load(b/'x.npy',mmap_mode='r')
            np.testing.assert_array_equal(xa[0],xb[0])
            t=np.load(a/'time.npy'); use=t>=p['withdraw_end']; sa,sb=signals(a),signals(b)
            rms=np.array([np.sqrt(np.mean(np.sum((x.astype(float)-y)**2,axis=1)))*1000 for x,y in zip(xa,xb,strict=True)])
            tip=np.column_stack([sa[n]-sb[n] for n in ['tip_x','tip_y','tip_z']])*1000
            force=np.column_stack([sb[n] for n in ['reaction_x','reaction_y','reaction_z']])
            complete=read(b/'completion.json'); assert complete['all_frames_completed'] and complete['inverted_count']==0
            assert read(b/'config.json')['E_pa']==fit['E_pa']
            assert np.max(np.linalg.norm(force[use],axis=1))==0
            row[label]=dict(recovery_particle_rmse_mm=float(np.sqrt(np.mean(rms[use]**2))),
                recovery_tip_rmse_mm=float(np.sqrt(np.mean(np.sum(tip[use]**2,axis=1)))),
                recovery_max_frame_rmse_mm=float(rms[use].max()),max_recovery_tool_force_n=0.,
                min_J=complete['min_J'])
            np.savetxt(root/f'recovery_{label}_{k}.csv',np.column_stack([t,rms]),delimiter=',',header='time,particle_rmse_mm',comments='')
        s=signals(root/f'truth_{k}'); row['peak_loading_force_mN']=float(np.max(abs(s['reaction_x'][s['time']<=p['bend_end']]))*1000)
        fine=signals(root/f'fine_{k}'); use=s['time']>=p['withdraw_end']
        row['main_fine_true_tip_rmse_mm']=float(np.sqrt(np.mean(np.sum(np.column_stack([s[n]-fine[n] for n in ['tip_x','tip_y','tip_z']])[use]**2,axis=1)))*1000)
        result[k]=row
    save(root/'report.json',result); return result


def views(root,k):
    """Reporting only: render full-episode truth and prediction with equal scale."""
    import pyvista as pv
    from experiments.elastic.strip_bending_study import pusher_position
    p=read(root/'protocol.json'); media=root/'media';media.mkdir(exist_ok=True)
    for kind in ['truth','prediction']:
        x=np.load(root/f'{kind}_{k}/x.npy',mmap_mode='r'); t=np.load(root/f'{kind}_{k}/time.npy')
        axes=[np.unique(x[0,:,i]) for i in range(3)]; shape=tuple(len(a) for a in axes)+(3,)
        ids=np.arange(0,len(t),5)
        frames=np.lib.format.open_memmap(media/f'{kind}_{k}_frames.npy',mode='w+',dtype='uint8',shape=(len(ids),890,660,3))
        (front,ref),others=meshes(p)
        pl=pv.Plotter(off_screen=True,window_size=(1280,1280));pl.set_background('#f5f6f5');pl.enable_anti_aliasing('ssaa')
        pl.add_mesh(front,texture=pv.numpy_to_texture(texture()),lighting=False)
        for mesh,_ in others: pl.add_mesh(mesh,color='#8a999e',smooth_shading=True)
        yf=p['center'][1]-p['size'][1]/2;yb=p['center'][1]+p['size'][1]/2
        for bounds in [(.101,.139,yf-.012,yf,.165,.193),(.101,.139,yb,yb+.012,.165,.193),(.101,.139,yf-.012,yb+.012,.193,.204)]:
            pl.add_mesh(pv.Box(bounds=bounds),color='#aab4bc')
        pl.camera.position=CAMERA['centers'][0];pl.camera.focal_point=CAMERA['target'];pl.camera.up=(0,0,1)
        pl.camera.view_angle=float(np.degrees(2*np.arctan(1280/(2*CAMERA['focal_px']))))
        for j,i in enumerate(ids):
            deform=RegularGridInterpolator(axes,x[i].reshape(shape),bounds_error=False,fill_value=None)
            front.points=deform(ref)
            for mesh,q in others: mesh.points=deform(q)
            pl.add_mesh(pv.Cylinder(center=pusher_position(t[i],p),direction=(0,1,0),radius=p['pusher_radius'],
                 height=p['pusher_length'],resolution=64),name='pusher',color='#aab4bc')
            pl.reset_camera_clipping_range();pl.render()
            frames[j]=pl.screenshot(return_img=True)[340:1230,300:960,:3]
            if j%60==0:print(kind,k,'report frame',j,flush=True)
        pl.close();frames.flush()
        if k=='A': np.save(media/'video_time.npy',t[ids])


def visuals(root):
    p=read(root/'protocol.json');result=read(root/'report.json');media=root/'media';media.mkdir(exist_ok=True)
    times=np.load(media/'video_time.npy')
    frames={(kind,k):np.load(media/f'{kind}_{k}_frames.npy',mmap_mode='r') for kind in ['truth','prediction'] for k in ['A','B']}
    sig={(kind,k):signals(root/f'{kind}_{k}') for kind in ['truth','prediction'] for k in ['A','B']}
    def frame(i):
        t=times[i];canvas=Image.new('RGB',(1600,1000),'white');d=ImageDraw.Draw(canvas)
        d.text((40,22),'Identify from texture and force. Predict the release.',font=font(35,True),fill=INK)
        d.text((40,77),'Calibrated stereo input · 12 × 4 × 100 mm strip · same prescribed push for both materials',font=font(21),fill='#65747e')
        for j,k in enumerate(['A','B']):
            base=35+j*800; r=result[k]
            d.text((base,130),f'Material {k}',font=font(28,True),fill=COLORS[k])
            d.text((base+170,137),f"E: {r['true_E_pa']/1000:.0f} → {r['identified_E_pa']/1000:.2f} kPa  ({r['signed_error_percent']:+.2f}%)",font=font(21),fill=INK)
            for col,kind in enumerate(['truth','prediction']):
                left=base+col*380
                d.text((left,180),'Reference simulation' if kind=='truth' else 'Identified model',font=font(21,True),fill=INK)
                im=Image.fromarray(frames[kind,k][i]).resize((360,486),Image.Resampling.LANCZOS)
                canvas.paste(im,(left,220))
                s=sig[kind,k];f=np.interp(t,s['time'],s['reaction_x'])*1000
                d.text((left,730),f'Pusher force  {abs(f):.1f} mN',font=font(20),fill=INK)
            d.text((base,789),f"Release prediction: {r['main']['recovery_particle_rmse_mm']:.2f} mm 3D RMSE",font=font(22,True),fill=COLORS[k])
        phase='Settling' if t<=p['settle_end'] else 'Bending: image observations' if t<=p['bend_end'] else 'Holding' if t<=p['hold_end'] else 'Pusher withdrawal' if t<=p['withdraw_end'] else 'Free recovery: excluded from identification'
        d.text((40,854),phase,font=font(25,True),fill=INK);d.text((1430,854),f'{t:.2f} s',font=font(25),fill=INK)
        d.rounded_rectangle((40,905,1560,917),radius=6,fill='#e7ecef')
        d.rounded_rectangle((40,905,40+max(12,1520*t/p['end']),917),radius=6,fill='#66859a')
        d.text((40,946),'4× slow motion · Equal camera scale · Prediction starts from the undeformed specimen; no state reset at release.',font=font(18),fill='#65747e')
        return np.asarray(canvas)
    with imageio.get_writer(media/'texture_bending.mp4',fps=25,codec='libx264',quality=8,macro_block_size=2,pixelformat='yuv420p') as writer:
        for _ in range(15): writer.append_data(frame(0))
        for i in range(len(times)):writer.append_data(frame(i))
        for _ in range(25):writer.append_data(frame(len(times)-1))
    Image.fromarray(frame(90)).save(media/'video_preview.png')
    # A compact report figure: camera observations large enough to see texture,
    # with quantitative prediction curves rather than selected best snapshots.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.035,.955,'Textured stereo identification and release prediction',fontsize=23,weight='bold',color=INK)
    fig.text(.035,.91,'Same material properties and motion; specimen thickness reduced from 20 to 4 mm.',fontsize=13,color='#63717b')
    for j,k in enumerate(['A','B']):
        r=result[k]; ax=fig.add_axes([.035+j*.225,.24,.205,.60])
        raw=np.load(root/f'inputs_{k}/camera_0.npy',mmap_mode='r')[-1]
        ax.imshow(raw[340:1230,300:960],cmap='gray',vmin=0,vmax=255);ax.axis('off')
        ax.set_title(f'Material {k} · end of bend',fontsize=14,weight='bold',color=COLORS[k],pad=12)
        fig.text(.045+j*.225,.205,f"E: {r['true_E_pa']/1000:.0f} → {r['identified_E_pa']/1000:.2f} kPa",fontsize=16,weight='bold',color=COLORS[k])
        fig.text(.045+j*.225,.167,f"Stiffness error  {r['signed_error_percent']:+.2f}%",fontsize=13,color=INK)
        ax=fig.add_axes([.555,.56-j*.34,.405,.23])
        for kind in ['truth','prediction']:
            s=sig[kind,k];use=s['time']>=p['withdraw_end']
            ax.plot(s['time'][use],(s['tip_x'][use]-p['center'][0])*1000,
                color='#303f49' if kind=='truth' else COLORS[k],ls='-' if kind=='truth' else '--',lw=2,
                label='Reference' if kind=='truth' else 'Identified model')
        ax.set_title(f"{k}: release prediction · {r['main']['recovery_particle_rmse_mm']:.2f} mm 3D RMSE",loc='left',fontsize=14,color=COLORS[k],weight='bold')
        ax.set_ylabel('Tip displacement (mm)');ax.grid(alpha=.18);ax.set_xlim(1.17,2)
        if j==0:fig.legend(*ax.get_legend_handles_labels(),frameon=False,ncol=2,fontsize=11,loc='upper left',bbox_to_anchor=(.55,.885))
        else:ax.set_xlabel('Time (s)')
    fig.text(.035,.075,'Inputs: stereo images + pusher force + known geometry, density, Poisson ratio and elastic family.',fontsize=12,color=INK)
    fig.text(.035,.037,'Synthetic camera test. Plane-stress and surface-to-interior assumptions remain. No simulator motion enters identification.',fontsize=11,color='#63717b')
    fig.savefig(media/'figure_preview.png',dpi=150);fig.savefig(media/'figure_preview.pdf');plt.close(fig)


def write_report(root):
    r=read(root/'report.json')
    lines=['# Textured stereo: thin elastic strip','',
      'A thinner specimen and reference-image texture correlation recover stiffness from synthetic stereo images and pusher force. No hidden simulator motion enters identification.','',
      '| Material | True E (kPa) | Identified E (kPa) | Signed error | Recovery 3D RMSE (mm) | Finer-grid RMSE (mm) |',
      '|---|---:|---:|---:|---:|---:|']
    for k,v in r.items():
        lines.append(f"| {k} | {v['true_E_pa']/1000:.0f} | {v['identified_E_pa']/1000:.6f} | {v['signed_error_percent']:+.6f}% | {v['main']['recovery_particle_rmse_mm']:.6f} | {v['fine']['recovery_particle_rmse_mm']:.6f} |")
    lines+=['','## What happens','',
      '- The upper 15 mm of a 12 × 4 × 100 mm strip is fixed. Both materials receive the same position-controlled cylindrical push: settle 0–0.30 s, bend 0.30–0.90 s, hold to 1.05 s, withdraw by 1.17 s, recover to 2.00 s. The pusher moves 24 mm and has radius 10 mm and length 16 mm.',
      '- The controller prescribes motion; force is measured, not regulated. Camera input stops at 0.90 s. Identification uses the 0.32–0.88 s interval, with measured force integrated at its own 500 Hz sampling rate.',
      '- Two calibrated cameras observe the same opaque textured face at 100 Hz. Image corners initialize correspondences. Local affine image-subset correlation against the first frame reduces drift and allows rotation and strain. There are no painted dots, supplied texture identities, or simulator point IDs.',
      '- A shared displacement spline is selected by four-fold held-feature displacement prediction, subject to full rank and positive volume. Final choice: quadratic across the face and cubic along its length with 12 mm knot spacing. No stiffness error enters that selection.',
      '- The time-weak momentum balance estimates only E, using the same three nodal bending virtual fields as the earlier experiment. The elastic family (fixed-corotated), Poisson ratio 0.45, density 1000 kg/m³, initial geometry and clamp are known.',
      '- Predicted episodes start from the undeformed specimen using frozen camera-derived E. They are never initialized from the true release state. Recovery is a withheld interval of the same episode, not an independent action or geometry-transfer test.','',
      '## Observation and mechanical assumptions','',
      'The camera images are synthetic: 1280 × 1280 pixels, exact calibration, opaque matte texture, no motion blur or lens distortion, and 0.5 gray-level Gaussian noise. Force is the net simulated pusher reaction with no added sensor noise. The camera-derived input bundle contains only images, timestamps, force and permitted metadata. The isolated repeat blocks access to all other experiment data and reproduces the final tracks and E exactly.',
      '',
      'The reconstruction assumes in-plane displacement uniform through the 4 mm thickness. The unmeasured transverse stretch is supplied by the plane-stress relation with known Poisson ratio. Thus it is a reduced surface-to-interior approximation, not a measurement of full internal deformation. It does not impose an unstretched centerline or rigid/plane beam cross-sections. Local affine DIC warps are image-patch interpolation, not a beam model.',
      '',
      'Thickness changed from 20 to 4 mm while in-plane geometry, E, nu, prescribed travel and timing remain unchanged; the cylinder was shortened to span the thinner strip. Grid resolution changed from 96/128 to 192/256, giving 1.25/0.9375 mm cells. The specimen spans only 3.2/4.27 cells through thickness, so the finer-grid check matters.',
      '',
      'A direct normal-stress audit in the loaded midsection (current z = 112–150 mm, t = 0.32–0.88 s) gives RMS sigma_yy / RMS full stress of 1.18% for A and 3.34% for B. The old thicker specimen gave 1.16% and 2.26% on its coarser grid. These results do not establish that thinning reduced this stress ratio. They support treating plane stress as an approximation, not asserting it became exact. The improved full-pipeline accuracy cannot be attributed solely to thickness: tracking, reconstruction selection and resolution also changed.','',
      '## Tracking and numerical checks','']
    for k,v in r.items():
        a=v['observation_audit'];tr=v['tracking']
        lines.append(f"- {k}: {tr['initial_stereo_tracks']} initial / {tr['final_stereo_tracks']} final stereo features; surface-motion RMSE {a['surface_motion_rmse_mm']:.6f} mm. Same reconstruction on ideal surface locations gives E = {a['ideal_surface_diagnostic']['E_pa']/1000:.6f} kPa (evaluation only). Peak loading force = {v['peak_loading_force_mN']:.3f} mN. Main/fine true recovery tip discrepancy = {v['main_fine_true_tip_rmse_mm']:.6f} mm.")
    lines+=['','Recovery error is the root mean square Euclidean distance between corresponding simulated particles over the full specimen and t = 1.17–2.00 s, with no alignment or rescaling. These simulator coordinates are evaluation data only. All four fitted-model main/fine runs complete without inversions and have zero pusher force throughout recovery.',
      '',
      'The loads are small (roughly 14–26 mN peak); transferring this setup to hardware requires a sensor capable of resolving these loads. No hardware accuracy claim is made.','',
      '## Development retained','',
      'The original dot-marker experiment remains at out/strip_camera_20260912. Within this experiment, the initial coarse-pyramid stereo matching failed before fitting. After fixing rectification-scale matching, incremental optical flow plus the old 8 mm spline produced an invalid free-tip extrapolation in A. Observation-only spline selection removed that inversion but gave E = 63.175/174.352 kPa. These poor fits and settings are retained in flow_A/B, development/, and logs. Reference-image affine DIC then reduced held-feature displacement error from 0.014834 to 0.004248 mm, selected the final reconstruction from the same candidate set, and produced the reported stiffness estimates. No true-E correction factor or material-parameter adjustment was used.','',
      '## Reproduce','',
      'Environment: repository .venv with opencv-python-headless==4.11.0.86, NumPy, SciPy, PyVista/VTK, Pillow, Matplotlib, imageio/ffmpeg, and Warp. Use a new output directory. The script runs sequentially on one GPU; the archived run used both GPUs for independent material cases.','',
      '```bash',
      'cd FORM',
      'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.strip_texture_reproduce --out out/strip_texture_repeat --device cuda:0',
      '```','',
      'Reproduction regenerates observations, image tracking, data-only spline selection, fits, main/fine predictions, an observation-isolated repeat, and the report/video. The optional old/new stress audit is:',
      '```bash',
      'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.strip_texture_audit stress --out out/strip_texture_repeat --previous out/strip_bending_20260912_clearance',
      '```','',
      'The paper and its existing figures have not been changed.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def provenance(root):
    repo=Path(__file__).resolve().parents[2]
    sources=list((repo/'experiments/elastic').glob('strip_*.py'))
    sources += [repo/'experiments/elastic/block_drop_study.py',repo/'experiments/elastic/block_drop_report.py']
    sources += list((repo/'src/ident').rglob('*.py'))+list((repo/'src/warpmpm').rglob('*.py'))+list((repo/'src/common').rglob('*.py'))
    sources += [repo/'tests/test_strip_texture.py',repo/'tests/test_strip_camera.py',repo/'tests/test_strip_bending_experiment.py']
    hashes={}
    for path in sources:
        dst=root/'source'/path.relative_to(repo);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);hashes[str(path.relative_to(repo))]=sha(path)
    save(root/'source_hashes.json',hashes)
    save(root/'environment.json',dict(python=platform.python_version(),
        packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','opencv-python-headless','pyvista','vtk','Pillow','matplotlib','imageio','warp-lang']},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))
    paths=[root/'protocol.json',root/'reconstruction_selection.json',root/'observation_isolation.json',root/'report.json']
    for k in ['A','B']:
        paths+=list((root/f'inputs_{k}').glob('*'))+list((root/f'fit_{k}').glob('*'))
    paths+=list((root/'media').glob('*.mp4'))+list((root/'media').glob('*.png'))+list((root/'media').glob('*.pdf'))
    save(root/'artifact_hashes.json',{str(f.relative_to(root)):sha(f) for f in paths if f.is_file()})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['summarize','views','visuals','report','provenance'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],default='A')
    a=ap.parse_args()
    if a.stage=='summarize': print(summarize(a.out))
    elif a.stage=='views':views(a.out,a.material)
    elif a.stage=='visuals':visuals(a.out)
    elif a.stage=='report':write_report(a.out)
    else:provenance(a.out)

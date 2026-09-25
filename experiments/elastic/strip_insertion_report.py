"""Scientific renders and audits of frozen elastic-strip insertion executions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from scipy.interpolate import RegularGridInterpolator
from experiments.elastic.block_drop_report import font
from experiments.elastic.block_drop_study import save,sha
from experiments.elastic.strip_insertion_study import read,rotation,tilt,translation,wall_boxes

INK='#26343d';MUTED='#627581';COLORS={'A':'#3184af','B':'#c37647'}


def surface_faces(ref,size):
    import pyvista as pv
    axes=[np.unique(ref[:,j]) for j in range(3)]
    lo=np.array([a.min() for a in axes]);hi=np.array([a.max() for a in axes])
    spacing=np.array([a[1]-a[0] for a in axes]);lo-=spacing/2;hi+=spacing/2
    faces=[]
    for n in range(3):
        a,b=[j for j in range(3) if j!=n]
        aa=np.linspace(lo[a],hi[a],max(3,int(np.ceil(size[a]/.001))+1))
        bb=np.linspace(lo[b],hi[b],max(3,int(np.ceil(size[b]/.001))+1))
        A,B=np.meshgrid(aa,bb,indexing='ij');na,nb=A.shape
        for side in [lo[n],hi[n]]:
            q=np.empty((A.size,3));q[:,n]=side;q[:,a]=A.ravel();q[:,b]=B.ravel()
            cells=[]
            for i in range(na-1):
                for j in range(nb-1):
                    k=i*nb+j;cells += [4,k,k+nb,k+nb+1,k+1]
            faces.append((pv.PolyData(q,np.array(cells)),q))
    return axes,faces


def render_case(root,material,plan,fine=False):
    import pyvista as pv
    name=f"{'fine' if fine else 'execution'}_{material}_plan_{plan}"
    case=root/name;p=read(case/'config.json')['protocol'];q=read(case/'config.json')['controls']
    x=np.load(case/'x.npy',mmap_mode='r');ref=np.load(case/'reference.npy');t=np.load(case/'time.npy')
    axes,faces=surface_faces(ref,p['size']);shape=tuple(len(a) for a in axes)+(3,)
    media=root/'media';media.mkdir(exist_ok=True)
    ids=np.arange(0,len(t),5) # 20 simulated frames/s, played at 20 fps.
    frames=np.lib.format.open_memmap(media/f'{name}_frames.npy',mode='w+',dtype='uint8',shape=(len(ids),520,760,3))
    pl=pv.Plotter(off_screen=True,window_size=(760,520));pl.set_background('#f2f5f6');pl.enable_anti_aliasing('ssaa')
    pl.enable_lightkit()
    for mesh,_ in faces:pl.add_mesh(mesh,color=COLORS[material],smooth_shading=True,ambient=.3,diffuse=.7,specular=.22,specular_power=24)
    # Translucent wall permits inspection of the tip after crossing. Aperture edges stay opaque.
    for center,half in wall_boxes(p):
        b=np.array([center,half]);bounds=np.column_stack([b[0]-b[1],b[0]+b[1]]).ravel()
        pl.add_mesh(pv.Box(bounds=bounds),color='#b0bec6',opacity=.24)
    wx=p['wall_x'];hx=p['wall_thickness']/2;y=p['center_y'];z=p['opening_z'];gy=p['opening_width']/2;gz=p['opening_height']/2
    for side in [-1,1]:
        vertices=np.array([[wx+side*hx,y-gy,z-gz],[wx+side*hx,y+gy,z-gz],
                           [wx+side*hx,y+gy,z+gz],[wx+side*hx,y-gy,z+gz],[wx+side*hx,y-gy,z-gz]])
        pl.add_mesh(pv.lines_from_points(vertices),color='#627783',line_width=3)
    for yy in [y-gy,y+gy]:
        for zz in [z-gz,z+gz]:pl.add_mesh(pv.Line((wx-hx,yy,zz),(wx+hx,yy,zz)),color='#627783',line_width=2)
    target=np.array([wx+hx+p['target_depth'],y,z+p.get('target_z_offset',0.)])
    pl.add_mesh(pv.Sphere(radius=.0017,center=target),color='#709786',lighting=False)
    pl.add_mesh(pv.Plane(center=(.16,y,.065),direction=(0,0,1),i_size=.23,j_size=.16),color='#e7edef',opacity=.7)
    pl.add_mesh(pv.Line((.07,.105,.066),(.09,.105,.066)),color='#71838d',line_width=3)
    focal=np.array([.161,y,.164]);pl.camera.position=focal+np.array([-.28,-.38,.16]);pl.camera.focal_point=focal;pl.camera.up=(0,0,1)
    pl.enable_parallel_projection();pl.camera.parallel_scale=.093
    for j,i in enumerate(ids):
        interp=RegularGridInterpolator(axes,x[i].reshape(shape),bounds_error=False,fill_value=None)
        for mesh,qr in faces:mesh.points=interp(qr)
        rot=rotation(tilt(t[i],q[1],p));c=np.array([p['grip_x']+translation(t[i],p),y,q[0]])
        # Two ideal bonded jaws plus a wrist connection. This is a kinematic fixture illustration.
        for side in [-1,1]:
            jaw_z=side*(p['size'][2]/2+.003)
            jaw=pv.Box(bounds=(-p['grip_length']/2,p['grip_length']/2,-.010,.010,jaw_z-.003,jaw_z+.003))
            jaw.points=jaw.points@rot.T+c
            pl.add_mesh(jaw,name=f'jaw{side}',color='#526772',smooth_shading=False)
        wrist=pv.Cylinder(center=c+rot@np.array([-.025,0,0]),direction=rot@np.array([1,0,0]),radius=.009,height=.030,resolution=32)
        pl.add_mesh(wrist,name='wrist',color='#91a3ae')
        pl.reset_camera_clipping_range();pl.render();frames[j]=pl.screenshot(return_img=True)[:,:,:3]
        if j%50==0:print(name,'frame',j,flush=True)
    frames.flush();pl.close();np.save(media/'video_time.npy',t[ids])


def audit_case(case,p):
    x=np.load(case/'x.npy',mmap_mode='r');tip=np.load(case/'tip_mask.npy');t=np.load(case/'time.npy')
    back=p['wall_x']+p['wall_thickness']/2;seen=np.zeros(int(tip.sum()),bool);clearance=np.full(seen.shape,np.nan)
    for i in range(1,len(t)):
        a=x[i-1,tip];b=x[i,tip];cross=(a[:,0]<back)&(b[:,0]>=back)&~seen
        if not cross.any():continue
        u=(back-a[cross,0])/(b[cross,0]-a[cross,0]);points=a[cross]+u[:,None]*(b[cross]-a[cross])
        clear=np.minimum(p['opening_width']/2-np.abs(points[:,1]-p['center_y']),p['opening_height']/2-np.abs(points[:,2]-p['opening_z']))
        clearance[cross]=clear;seen[cross]=True
    result=read(case/'result.json')
    result.update(all_tip_particles_crossed_aperture=bool(seen.all() and np.nanmin(clearance)>0),
                  min_tip_crossing_clearance_mm=float(np.nanmin(clearance)*1000) if seen.any() else None)
    result['success']=bool(result['success'] and result['all_tip_particles_crossed_aperture'])
    if (case/'reference.npy').exists():
        ref=np.load(case/'reference.npy');q=read(case/'config.json')['controls']
        core=(np.abs(ref[:,0])<p['grip_length']/2-.003)
        errors=[]
        for i,tt in enumerate(t):
            c=np.array([p['grip_x']+translation(tt,p),p['center_y'],q[0]])
            expected=ref[core]@rotation(tilt(tt,q[1],p)).T+c
            errors.append(float(np.max(np.linalg.norm(x[i,core]-expected,axis=1))))
        result['max_gripped_particle_pose_error_mm']=max(errors)*1000
    return result


def summarize(root):
    p=read(root/'protocol.json');r={'protocol':p,'plans':{},'executions':{}}
    for k in ['A','B']:
        r['plans'][k]=read(root/f'plan_{k}/plan.json')
        assert r['plans'][k]['protocol_sha256']==sha(root/'protocol.json')
        controls=r['plans'][k]['controls'];times=np.arange(round(p['end']/p['tick'])+1)*p['tick']
        centers=np.array([[p['grip_x']+translation(tt,p),p['center_y'],controls[0]] for tt in times])
        angles=np.array([tilt(tt,controls[1],p) for tt in times]);rad=np.radians(angles)
        quat=np.column_stack([np.zeros(len(times)),-np.sin(rad/2),np.zeros(len(times)),np.cos(rad/2)])
        velocity=np.vstack([np.diff(centers,axis=0)/p['tick'],np.zeros(3)])
        omega=np.r_[-np.diff(rad)/p['tick'],0.]
        np.savetxt(root/f'plan_{k}/commands.csv',np.column_stack([times,centers,quat,velocity,omega]),
                   delimiter=',',comments='',header='time,grip_x,grip_y,grip_z,qx,qy,qz,qw,vx,vy,vz,omega_y')
    for grid in ['execution','fine']:
        for mat in ['A','B']:
            for plan in ['A','B']:
                name=f'{grid}_{mat}_plan_{plan}';case=root/name
                if not (case/'result.json').exists():continue
                config=read(case/'config.json')
                assert config['controls']==r['plans'][plan]['controls']
                assert config['E_pa']==p['true_E_pa'][mat]
                assert config['protocol']==p
                result=audit_case(case,p)
                assert result['all_frames_completed'] and result['inverted_count']==0
                r['executions'][name]=result
    save(root/'report.json',r);return r


def compose(root,fine=False):
    media=root/'media';r=read(root/'report.json');p=r['protocol'];prefix='fine' if fine else 'execution';times=np.load(media/'video_time.npy')
    frames={(m,pl):np.load(media/f'{prefix}_{m}_plan_{pl}_frames.npy',mmap_mode='r') for m in ['A','B'] for pl in ['A','B']}
    def canvas(i,only=None):
        materials=['A','B'] if only is None else [only];height=1420 if only is None else 850
        out=Image.new('RGB',(1600,height),'white');d=ImageDraw.Draw(out);tt=times[i]
        d.text((34,20),'Identify stiffness. Plan the grip motion. Insert the tip.',font=font(32,True),fill=INK)
        d.text((34,69),'60 mm strip · 20 × 40 mm opening · Fixed motions, without force or visual feedback',font=font(21),fill=MUTED)
        d.text((145,116),'Plan using matched ID',font=font(25,True),fill=INK)
        d.text((945,116),'Plan using swapped ID',font=font(25,True),fill=INK)
        for row,m in enumerate(materials):
            top=165+row*570
            for col,pl in enumerate([m,'B' if m=='A' else 'A']):
                left=20+col*800;out.paste(Image.fromarray(frames[m,pl][i]),(left,top))
                d.text((left+15,top+12),f'Material {m}',font=font(23,True),fill=COLORS[m])
                q=r['plans'][pl]['controls']
                d.text((left+15,top+45),f'Grip: {(q[0]-p["opening_z"])*1000:+.1f} mm above opening center, {q[1]:+.1f}° tilt',font=font(17),fill=MUTED)
                result=r['executions'][f'{prefix}_{m}_plan_{pl}']
                label='INSERTED · no wall contact' if result['success'] else ('WALL CONTACT' if result['peak_wall_force_N']>p['contact_force_threshold'] else 'INSUFFICIENT INSERTION')
                if tt>=p['advance_end']:
                    d.text((left+15,top+478),label,font=font(20,True),fill='#387c65' if result['success'] else '#a85439')
                d.text((left+15,top+529),f"Final tip error: {result['tip_error_mm']:.2f} mm   |   Minimum tip depth: {result['min_tip_depth_hold_mm']:.1f} mm",font=font(18),fill=INK)
        phase='Hanging start' if tt<p['align_start'] else 'Rotate the grip' if tt<p['align_end'] else 'Hold before insertion' if tt<p['settle_end'] else 'Advance through the opening' if tt<p['advance_end'] else 'Final hold'
        base=height-86;d.text((34,base),phase,font=font(22,True),fill=INK);d.text((1430,base),f'{tt:.2f} s',font=font(22),fill=INK)
        d.text((34,base+38),'Real-time · Same scale · Green dot: tip target · Wall shown translucent · Kinematic grip, no arm dynamics',font=font(17),fill=MUTED)
        return np.asarray(out)
    for only,name in [(None,'insertion_comparison'),('A','insertion_A'),('B','insertion_B')]:
        with imageio.get_writer(media/f'{name}.mp4',fps=20,codec='libx264',quality=8,macro_block_size=2,pixelformat='yuv420p') as writer:
            for _ in range(10):writer.append_data(canvas(0,only))
            for i in range(len(times)):writer.append_data(canvas(i,only))
            for _ in range(35):writer.append_data(canvas(len(times)-1,only))
    Image.fromarray(canvas(len(times)-1)).save(media/'insertion_results.png')
    Image.fromarray(canvas(round(p['align_end']/.05))).save(media/'prepared_poses.png')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['render','summarize','compose']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--material',default='A');ap.add_argument('--plan',default='A');ap.add_argument('--fine',action='store_true');a=ap.parse_args()
    if a.stage=='render':render_case(a.out,a.material,a.plan,a.fine)
    elif a.stage=='summarize':summarize(a.out)
    else:compose(a.out,a.fine)

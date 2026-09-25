"""Render saved MPM states and Panda FK, with independently audited insertion metrics."""
from pathlib import Path
import argparse
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from experiments.elastic.block_drop_report import font
from experiments.elastic.block_drop_study import save,sha
from experiments.elastic.rod_insertion_study import read,wall_violation
from experiments.elastic.rod_insertion_franka import Panda,OFFSET
INK='#26343d';MUTED='#627581';COLORS={'A':'#3184af','B':'#c37647'}


def surface(ref,p):
    """Material-coordinate local affine interpolation, exclusively for rendering."""
    import pyvista as pv
    nx=81;nt=48
    xx=np.linspace(-p['grip_length']/2,p['size'][0]-p['grip_length']/2,nx)
    th=np.arange(nt)*2*np.pi/nt
    q=np.array([[x,p['radius']*np.cos(t),p['radius']*np.sin(t)] for x in xx for t in th])
    q=np.vstack([q,[xx[0],0,0],[xx[-1],0,0]])
    faces=[]
    for i in range(nx-1):
        for j in range(nt):
            a=i*nt+j;b=i*nt+(j+1)%nt
            faces.extend([4,a,b,b+nt,a+nt])
    for j in range(nt):
        faces.extend([3,nx*nt,(j+1)%nt,j])
        faces.extend([3,nx*nt+1,(nx-1)*nt+j,(nx-1)*nt+(j+1)%nt])
    d,idx=cKDTree(ref).query(q,k=32)
    local=ref[idx]-q[:,None,:];scale=d[:,-1,None,None]
    A=np.concatenate([np.ones((*idx.shape,1)),local/scale],axis=2)
    w=np.exp(-3*(d/d[:,-1,None])**2)
    G=np.einsum('nki,nk,nkj->nij',A,w,A)
    weights=np.einsum('ni,nki,nk->nk',np.linalg.pinv(G)[:,0,:],A,w)
    np.testing.assert_allclose(np.einsum('nk,nkj->nj',weights,ref[idx]),q,atol=1e-10)
    return pv.PolyData(q,np.array(faces)),idx,weights


def fixture(p):
    """Exact square plate with a circular bore, matching the contact SDF geometry."""
    import pyvista as pv
    nt=128;th=np.arange(nt)*2*np.pi/nt;c=np.cos(th);s=np.sin(th)
    outer=np.minimum(p['wall_half_width']/np.maximum(abs(c),1e-15),p['wall_half_height']/np.maximum(abs(s),1e-15))
    pts=[]
    for side in [-1,1]:
        for radius in [np.full(nt,p['opening_radius']),outer]:
            pts.extend(np.c_[np.full(nt,p['wall_x']+side*p['wall_thickness']/2),p['center_y']+radius*c,p['opening_z']+radius*s])
    faces=[]
    for j in range(nt):
        k=(j+1)%nt
        for a,b in [(0,nt),(2*nt,3*nt),(0,2*nt),(nt,3*nt)]:faces.extend([4,a+j,a+k,b+k,b+j])
    return pv.PolyData(np.array(pts),np.array(faces))


def audit_case(case,p):
    x=np.load(case/'x.npy',mmap_mode='r');tip=np.load(case/'tip_mask.npy');t=np.load(case/'time.npy')
    back=p['wall_x']+p['wall_thickness']/2;seen=np.zeros(int(tip.sum()),bool);clearance=np.full(seen.shape,np.nan)
    for i in range(1,len(t)):
        a=x[i-1,tip];b=x[i,tip];cross=(a[:,0]<back)&(b[:,0]>=back)&~seen
        if not cross.any():continue
        u=(back-a[cross,0])/(b[cross,0]-a[cross,0]);q=a[cross]+u[:,None]*(b[cross]-a[cross])
        clearance[cross]=p['opening_radius']-np.linalg.norm(q[:,1:]-[p['center_y'],p['opening_z']],axis=1);seen[cross]=True
    r=read(case/'result.json');r.update(all_tip_particles_crossed_aperture=bool(seen.all() and np.nanmin(clearance)>0),min_crossing_clearance_mm=float(np.nanmin(clearance)*1000) if seen.any() else None)
    r['success']=bool(r['success'] and r['all_tip_particles_crossed_aperture'])
    if (case/'config.json').exists():
        config=read(case/'config.json');motion=Path(config['motion']);assert sha(motion)==config['motion_sha256']
        overlap=max(wall_violation(xi,p,p['domain']/config['grid']/4) for xi in x)
        r['max_particle_wall_overlap_mm']=overlap*1000
        r['success']=bool(r['success'] and overlap<=1e-8)
        pad=p['domain']/config['grid']/4
        support_clear=all(not np.any((abs(xi[:,0]-p['wall_x'])<=.015+pad)&
            (abs(xi[:,1]-p['center_y'])<=.042+pad)&
            (xi[:,2]<=p['opening_z']-p['wall_half_height']+pad)) for xi in x)
        r['material_clear_of_rendered_support']=support_clear
        if not support_clear:raise RuntimeError(f'{case}: material reaches support; a physical support collider is required before reporting this trajectory')
        fk=np.load(motion);ref=np.load(case/'reference.npy');core=abs(ref[:,0])<p['grip_length']/2-.003
        err=[]
        for i in range(len(t)):
            expected=ref[core]@fk['rotation'][i].T+fk['center'][i]
            err.append(np.linalg.norm(x[i,core]-expected,axis=1).max())
        r['max_gripped_particle_pose_error_mm']=float(max(err)*1000)
    return r


def summarize(root):
    p=read(root/'protocol.json');report=dict(protocol=p,execution_protocol=read(root/'execution_protocol.json'),plans={},executions={},franka={})
    for k in ['A','B']:
        report['plans'][k]=read(root/f'plan_{k}/plan.json');report['franka'][k]=read(root/'franka'/f'audit_{k}.json')
        assert report['plans'][k]['protocol_sha256']==sha(root/'protocol.json')
    for pre in ['execution','fine']:
        for m in ['A','B']:
            for k in ['A','B']:
                name=f'{pre}_{m}_plan_{k}';case=root/name
                if not (case/'result.json').exists():continue
                cfg=read(case/'config.json')
                assert cfg['controls']==report['plans'][k]['controls'] and cfg['protocol']==p and cfg['E_pa']==p['true_E_pa'][m]
                report['executions'][name]=audit_case(case,p)
    save(root/'report.json',report)


def render_case(root,m,k,fine=False,overview=False):
    import pyvista as pv
    pre='fine' if fine else 'execution';case=root/f'{pre}_{m}_plan_{k}'
    p=read(case/'config.json')['protocol'];x=np.load(case/'x.npy',mmap_mode='r');ref=np.load(case/'reference.npy');t=np.load(case/'time.npy')
    fk=np.load(root/'franka'/f'plan_{k}.npz');robot=Panda(root/'franka'/'panda_model_snapshot'/'panda.xml')
    media=root/'media';media.mkdir(exist_ok=True);name=f'{pre}_{m}_plan_{k}'+('_overview' if overview else '')
    ids=np.arange(0,len(t),5);W,H=(960,720) if overview else (760,520)
    frames=np.lib.format.open_memmap(media/f'{name}_frames.npy',mode='w+',dtype='uint8',shape=(len(ids),H,W,3))
    pl=pv.Plotter(off_screen=True,window_size=(W,H));pl.set_background('#f1f4f5');pl.enable_anti_aliasing('ssaa');pl.enable_lightkit()
    mesh,idx,weights=surface(ref,p);pl.add_mesh(mesh,color=COLORS[m],smooth_shading=True,ambient=.28,diffuse=.72,specular=.3,specular_power=30)
    pl.add_mesh(fixture(p),color='#abb9c2',opacity=.40,smooth_shading=False)
    th=np.linspace(0,2*np.pi,129)
    for side in [-1,1]:
        q=np.c_[np.full(len(th),p['wall_x']+side*p['wall_thickness']/2),p['center_y']+p['opening_radius']*np.cos(th),p['opening_z']+p['opening_radius']*np.sin(th)]
        pl.add_mesh(pv.lines_from_points(q),color='#425e70',line_width=3)
    # The fixture is mounted on a support; its top terminates below all specimen motion.
    pl.add_mesh(pv.Box(bounds=(p['wall_x']-.015,p['wall_x']+.015,p['center_y']-.042,p['center_y']+.042,-OFFSET[2],p['opening_z']-p['wall_half_height'])),color='#c6cfd3')
    pl.add_mesh(pv.Plane(center=(.12,.16,-OFFSET[2]-.001),direction=(0,0,1),i_size=1.,j_size=1.),color='#e3e9ec')
    robot.forward(fk['q'][0]);geoms={}
    for g,body,color in robot.meshes(include_arm=overview):
        geoms[g]=body;pl.add_mesh(body,color=color,smooth_shading=True,ambient=.3,specular=.25)
    if overview:
        focal=np.array([.265,-.175,.45])-OFFSET;pl.camera.position=focal+[.75,-1.0,.40];pl.camera.focal_point=focal;pl.camera.up=(0,0,1);pl.enable_parallel_projection();pl.camera.parallel_scale=.54
    else:
        focal=np.array([.188,.16,.145]);pl.camera.position=focal+[-.18,-.26,.10];pl.camera.focal_point=focal;pl.camera.up=(0,0,1);pl.enable_parallel_projection();pl.camera.parallel_scale=.050
    for j,i in enumerate(ids):
        mesh.points=np.einsum('nk,nkj->nj',weights,x[i,idx])
        mesh.compute_normals(inplace=True)
        robot.forward(fk['q'][i])
        for g,body,_ in robot.meshes(include_arm=overview):geoms[g].points=body.points
        pl.reset_camera_clipping_range();pl.render();frames[j]=pl.screenshot(return_img=True)[:,:,:3]
        if j%50==0:print(name,j,flush=True)
    frames.flush();pl.close();np.save(media/'video_time.npy',t[ids])
    if overview:
        with imageio.get_writer(media/f'{name}.mp4',fps=20,codec='libx264',quality=8,macro_block_size=2) as writer:
            for j in range(len(ids)):
                im=Image.fromarray(frames[j]);d=ImageDraw.Draw(im)
                d.text((24,18),f'Franka motion · Material {m} · Plan for {k}',font=font(26,True),fill=INK)
                d.text((24,58),'Kinematic arm · Bonded grasp · No force or shape feedback',font=font(18),fill=MUTED)
                d.text((24,680),f'{t[ids[j]]:.2f} s',font=font(20),fill=INK);writer.append_data(np.asarray(im))
        Image.fromarray(frames[-1]).save(media/f'{name}.png')


def compose(root,fine=False):
    media=root/'media';r=read(root/'report.json');p=r['protocol'];pre='fine' if fine else 'execution';times=np.load(media/'video_time.npy')
    frames={(m,k):np.load(media/f'{pre}_{m}_plan_{k}_frames.npy',mmap_mode='r') for m in ['A','B'] for k in ['A','B']}
    def canvas(i,only=None):
        mats=['A','B'] if only is None else [only];H=1370 if only is None else 790
        im=Image.new('RGB',(1600,H),'white');d=ImageDraw.Draw(im);tt=times[i]
        d.text((28,16),'Elastic rod insertion with a Franka gripper',font=font(32,True),fill=INK)
        d.text((28,63),'8 mm rod · 12 mm circular opening · Same materials identified from strip bending',font=font(22),fill=MUTED)
        for col,title in enumerate(['Plan using matched ID','Plan using swapped ID']):d.text((30+800*col,110),title,font=font(27,True),fill=INK)
        for row,m in enumerate(mats):
            top=154+580*row
            for col,k in enumerate([m,'B' if m=='A' else 'A']):
                left=20+800*col;im.paste(Image.fromarray(frames[m,k][i]),(left,top))
                d.text((left+15,top+12),f'Material {m}',font=font(23,True),fill=COLORS[m])
                res=r['executions'][f'{pre}_{m}_plan_{k}']
                if tt>=p['advance_end']:
                    label='INSERTED · no wall contact' if res['success'] else ('WALL CONTACT' if res['peak_wall_force_N']>p['contact_force_threshold'] else 'INSUFFICIENT DEPTH')
                    d.text((left+15,top+477),label,font=font(22,True),fill='#327562' if res['success'] else '#a85439')
                d.text((left+15,top+528),f"Minimum depth: {res['min_tip_depth_hold_mm']:.1f} mm   |   Tip error: {res['tip_error_mm']:.2f} mm",font=font(20),fill=INK)
        phase='Hanging start' if tt<p['align_start'] else 'Rotate wrist' if tt<p['align_end'] else 'Settle' if tt<p['settle_end'] else 'Advance' if tt<p['advance_end'] else 'Final hold'
        d.text((28,H-55),f'{phase}  ·  {tt:.2f} s',font=font(21,True),fill=INK)
        d.text((500,H-55),'Real-time · Same scale · Wall translucent · Kinematic Franka',font=font(20),fill=MUTED)
        return np.asarray(im)
    for only,name in [(None,'rod_comparison'),('A','rod_A'),('B','rod_B')]:
        with imageio.get_writer(media/f'{name}.mp4',fps=20,codec='libx264',quality=8,macro_block_size=2,pixelformat='yuv420p') as w:
            for _ in range(10):w.append_data(canvas(0,only))
            for i in range(len(times)):w.append_data(canvas(i,only))
            for _ in range(30):w.append_data(canvas(len(times)-1,only))
    Image.fromarray(canvas(len(times)-1)).save(media/'rod_results.png')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['render','summarize','compose']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--material',default='A');ap.add_argument('--plan',default='A');ap.add_argument('--fine',action='store_true');ap.add_argument('--overview',action='store_true');a=ap.parse_args()
    if a.stage=='render':render_case(a.out,a.material,a.plan,a.fine,a.overview)
    elif a.stage=='summarize':summarize(a.out)
    else:compose(a.out,a.fine)

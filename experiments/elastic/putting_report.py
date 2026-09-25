"""Render stored particle trajectories and actual Panda poses, without retiming physics."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from experiments.elastic.block_drop_report import font
from experiments.elastic.putting import save,digest
from experiments.elastic.rod_insertion_franka import Panda,OFFSET

COLORS={'A':'#3184af','B':'#c37647'};INK='#233540';MUTED='#667982'

def read(path):return json.loads(Path(path).read_text())

def surface(ref,p):
    """Reference union surface, deformed by local affine particle interpolation.

    This interpolation is only for rendering. Metrics use the rigid ball state.
    """
    import pyvista as pv
    h=.0006;lo=np.array([-.008,-.016,-.078]);hi=np.array([.008,.016,.010])
    axes=[np.arange(a,b+h/2,h) for a,b in zip(lo,hi)]
    q=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    shaftlo=np.array([-p['shaft_width']/2,-p['shaft_depth']/2,p['shaft_bottom']]);shafthi=np.array([p['shaft_width']/2,p['shaft_depth']/2,p['shaft_top']])
    def box(c,half):
        d=abs(q-c)-half
        return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)
    sdf=np.minimum(box((shaftlo+shafthi)/2,(shafthi-shaftlo)/2),box(np.array(p['head_center']),np.array(p['head_size'])/2))
    pts,faces,_,_=marching_cubes(sdf,0,spacing=(h,h,h));pts+=lo
    d,idx=cKDTree(ref).query(pts,k=32);local=ref[idx]-pts[:,None,:]
    A=np.concatenate([np.ones((*idx.shape,1)),local/d[:,-1,None,None]],axis=2)
    w=np.exp(-3*(d/d[:,-1,None])**2);G=np.einsum('nki,nk,nkj->nij',A,w,A)
    weights=np.einsum('ni,nki,nk->nk',np.linalg.pinv(G)[:,0,:],A,w)
    np.testing.assert_allclose(np.einsum('nk,nkj->nj',weights,ref[idx]),pts,atol=1e-9)
    return pv.PolyData(pts,np.c_[np.full(len(faces),3),faces].ravel()),idx,weights

def last_video_frame(case, times):
    """Keep the recorded motion through ball rest, plus a brief viewing margin."""
    with np.load(case/'trajectory.npz') as data: signals=data['signals']
    moving=(np.linalg.norm(signals[:,4:7],axis=1)>1e-5)|(np.linalg.norm(signals[:,7:9],axis=1)>1e-3)
    stop=signals[np.flatnonzero(moving)[-1],0] if moving.any() else 0.
    return min(int(np.searchsorted(times,stop+.35)),len(times)-1)

def render_case(root,m,k,prefix='execution',view='course',preview=False):
    import pyvista as pv
    case=root/f'{prefix}_{m}_plan_{k}';cfg=read(case/'config.json');p=cfg['protocol'];scene=read(root/'render_scene.json')
    data=np.load(case/'trajectory.npz');signals=data['signals'];xs=data['x'];ref=data['reference']
    provenance=read(case/'execution_provenance.json');path=Path(provenance['motion'])
    assert digest(path)==provenance['motion_sha256'];fk=np.load(path)
    if 'robot_base_shift' in scene:
        from experiments.elastic.putting_scene import ScenePanda
        audit=read(root/'scene_robot/audit.json')
        assert audit['robot_base_shift']==scene['robot_base_shift']
        assert audit['floor_z']==scene['table_bounds'][5]
        assert audit['plans'][k]['original_motion_sha256']==provenance['motion_sha256']
        new_path=root/f'scene_robot/plan_{k}.npz'
        assert digest(new_path)==audit['plans'][k]['scene_motion_sha256']
        fk=np.load(new_path);robot=ScenePanda(root/'franka/panda_model_snapshot/panda.xml',scene['robot_base_shift'])
    else:robot=Panda(root/'franka/panda_model_snapshot/panda.xml')
    media=root/'media';media.mkdir(exist_ok=True);name=f'{prefix}_{m}_{k}_{view}'
    if view=='detail':
        ids=np.flatnonzero(signals[:,0]<=2.05)[::2];W,H=800,640
    else:ids=np.arange(0,len(signals),round((.04 if view=='overview' else .02)/(signals[1,0]-signals[0,0])));W,H=(1200,780) if view=='overview' else ((640,640) if view=='hero' else (960,470))
    if view=='course' and 'camera' in scene:W,H=1280,720
    if preview:name+='_preview'
    if preview:ids=np.array([0,int(np.argmin(abs(signals[:,0]-.73))),len(signals)-1])
    frames=np.lib.format.open_memmap(media/f'{name}_frames.npy',mode='w+',dtype='uint8',shape=(len(ids),H,W,3))
    pl=pv.Plotter(off_screen=True,window_size=(W,H));pl.set_background(scene.get('background','#f4f5f3'));pl.enable_anti_aliasing('ssaa');pl.enable_lightkit()
    if 'sky_top' in scene:pl.set_background(scene['background'],top=scene['sky_top'])
    # The widened scenery preserves the original floor height and contact region.
    scene=read(root/'render_scene.json')
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=scene['key_light'],focal_point=(.4,.2,0),intensity=scene['key_intensity'],light_type='scene light'))
    pl.add_light(pv.Light(position=(1.,.3,.8),focal_point=(.2,.1,0),intensity=scene['fill_intensity'],light_type='scene light'))
    if scene.get('shadows'):pl.enable_shadows()
    if scene['ambient_occlusion']:pl.enable_ssao(radius=.006,bias=.0003,kernel_size=64,blur=True)
    green_actor=pl.add_mesh(pv.Box(bounds=scene['table_bounds']),color=scene['green'],ambient=.28,diffuse=.8)
    if 'camera' in scene:green_actor.SetUseBounds(False)
    if 'robot_base_shift' not in scene:pl.add_mesh(pv.Box(bounds=(.55,.65,.02,.16,-OFFSET[2],p['table_bounds'][4])),color='#7b8789')
    if scene.get('distant_green'):
        # Decorative distant terrain, well outside every recorded ball path.
        gx,gy=np.meshgrid(np.linspace(-20,20,241),np.linspace(3,18,101),indexing='ij')
        gz=p['floor']+(.48+.18*np.sin(.5*gx)+.12*np.cos(.85*gx+.3))*np.exp(-((gy-8)/3.2)**2)
        hills=pv.StructuredGrid(gx,gy,gz)
        fade=(1-np.exp(-((gy-3)/3)**2)).ravel(order='F')[:,None]
        colors=np.rint(np.array([51,92,71])*(1-fade)+np.array([82,120,91])*fade).astype(np.uint8)
        actor=pl.add_mesh(hills,scalars=colors,rgb=True,ambient=1.,diffuse=0.,smooth_shading=True)
        actor.SetUseBounds(False)
    theta=np.linspace(0,2*np.pi,181);target=np.array(p['target']);r=p['target_radius']
    ring=np.c_[target[0]+r*np.cos(theta),target[1]+r*np.sin(theta),np.full(len(theta),p['floor']+.0004)]
    pl.add_mesh(pv.lines_from_points(ring).tube(radius=.0014),color='#f0e6c5')
    pl.add_mesh(pv.Disc(center=(*target,p['floor']+.0002),inner=0,outer=r,normal=(0,0,1),c_res=120),color='#47765c',opacity=1.)
    if 'target_flag' in scene:
        # Visual landmark outside the goal, with no collision or scoring role.
        flag=scene['target_flag'];xy=target+flag['offset'];z=p['floor'];height=flag['height']
        pl.add_mesh(pv.Cylinder(center=(*xy,z+height/2),direction=(0,0,1),radius=.0008,height=height),color='#e4ded0',ambient=.7,diffuse=.3)
        vertices=np.array([[*xy,z+height],[xy[0]+flag['width'],xy[1],z+height-.012],[*xy,z+height-.025]])
        pl.add_mesh(pv.PolyData(vertices,[3,0,1,2]),color=flag['color'],ambient=.7,diffuse=.3)
    # Small starting-location ring remains after the ball leaves.
    initial=np.array(p['ball_initial']);ring0=np.c_[initial[0]+.010*np.cos(theta),initial[1]+.010*np.sin(theta),np.full(len(theta),p['floor']+.0003)]
    pl.add_mesh(pv.lines_from_points(ring0).tube(radius=.0002),color='#a0b6a5')
    ground_z=p['floor']-.027 if 'robot_base_shift' in scene else -OFFSET[2]-.002
    ground_actor=pl.add_mesh(pv.Plane(center=(.35,.1,ground_z),direction=(0,0,1),i_size=20.,j_size=20.),color='#e6e9e7')
    if 'camera' in scene:ground_actor.SetUseBounds(False)
    club,idx,weights=surface(ref,p);pl.add_mesh(club,color=COLORS[m],smooth_shading=True,ambient=.40,diffuse=.75,specular=.22,specular_power=35)
    sphere=pv.Sphere(radius=p['ball_radius'],theta_resolution=40,phi_resolution=32)
    base=sphere.points.copy();sphere.points=base+initial
    if signals.shape[1]>=21:
        rgb=np.tile(np.array([255,253,243],dtype=np.uint8),(len(base),1));rgb[abs(base[:,0])<.0008]=[67,80,81]
        pl.add_mesh(sphere,scalars=rgb,rgb=True,smooth_shading=True,ambient=.4,specular=.35)
    else:pl.add_mesh(sphere,color='#fffdf3',smooth_shading=True,ambient=.4,specular=.35)
    robot.forward(fk['q'][0]);geoms={};local_normals={};include_arm=view not in ('detail','hero')
    for g,mesh,color in robot.meshes(include_arm=include_arm):
        mesh.compute_normals(cell_normals=False,inplace=True)
        local_normals[g]=np.asarray(mesh.point_data['Normals'])@robot.data.geom_xmat[g].reshape(3,3)
        geoms[g]=mesh;pl.add_mesh(mesh,color=color,smooth_shading=True,ambient=.32,specular=.2)
    if view=='overview':focal=np.array([.39,.18,.11]);eye=focal+[-.72,-1.2,.83];scale=.63
    elif view in ('detail','hero'):focal=np.array([.100,.118,.090]);eye=focal+[-.22,-.30,.065];scale=.120
    else:focal=np.array([.57,.14,.035]);eye=focal+[.06,-.85,.90];scale=.29
    if view=='course' and 'camera' in scene:
        focal=np.array(scene['camera']['focal']);eye=focal+scene['camera']['offset']
        pl.camera.view_angle=scene['camera']['view_angle']
    else:pl.enable_parallel_projection();pl.camera.parallel_scale=scale
    pl.camera.position=eye;pl.camera.focal_point=focal;pl.camera.up=(0,0,1)
    for j,i in enumerate(ids):
        club.points=np.einsum('nk,nkj->nj',weights,xs[i,idx])
        club.point_data.pop('Normals',None);club.cell_data.pop('Normals',None)
        club.compute_normals(inplace=True)
        if signals.shape[1]>=21:
            from scipy.spatial.transform import Rotation
            rotation=Rotation.from_quat(signals[i,17:21]).as_matrix()
            sphere.points=base@rotation.T+signals[i,1:4]
            sphere.point_data['Normals']=(base/np.linalg.norm(base,axis=1)[:,None])@rotation.T
        else:sphere.points=base+signals[i,1:4]
        ti=signals[i,0];qi=int(round(ti/(fk['time'][1]-fk['time'][0])));robot.forward(fk['q'][qi])
        for g,mesh,_ in robot.meshes(include_arm=include_arm):
            geoms[g].points=mesh.points
            geoms[g].point_data['Normals']=local_normals[g]@robot.data.geom_xmat[g].reshape(3,3).T
        pl.reset_camera_clipping_range()
        if view=='course' and 'camera' in scene:pl.camera.clipping_range=(.3,300.)
        pl.render();frames[j]=pl.screenshot(return_img=True)[:,:,:3]
        if preview:Image.fromarray(frames[j]).save(media/f'{name}_preview_{j}.png')
        if j%50==0:print(name,j,flush=True)
    frames.flush();pl.close();np.save(media/f'{name}_time.npy',signals[ids,0])
    return name

def compose(root,prefix='execution',view='course',slow=1):
    if slow not in (1,2) or (view=='course' and slow!=1):raise ValueError('Use real time, or 2x slow motion for the contact view.')
    media=root/'media';names={(m,k):f'{prefix}_{m}_{k}_{view}' for m in 'AB' for k in 'AB'}
    frames={key:np.load(media/f'{name}_frames.npy',mmap_mode='r') for key,name in names.items()}
    t=np.load(media/f'{names["A","A"]}_time.npy');n,h,w,_=frames['A','A'].shape
    fps=1/((t[1]-t[0])*slow)
    if view=='course':n=1+max(last_video_frame(root/f'{prefix}_{m}_plan_{k}',t) for m in 'AB' for k in 'AB')
    p=read(root/f'{prefix}_A_plan_A/config.json')['protocol']
    grid_label=''  # Numerical settings are documented in the accompanying report.
    def canvas(i):
        im=Image.new('RGB',(2*w+80,2*h+180),'white');d=ImageDraw.Draw(im)
        for col,title in enumerate(['Plan using matched ID','Plan using swapped ID']):
            d.text((65+col*(w+10),16),title,font=font(36 if w>=1200 else 28,True),fill=INK)
        for row,m in enumerate('AB'):
            y=65+row*(h+40)
            d.text((15,y+h//2-15),m,font=font(40 if w>=1200 else 30,True),fill=COLORS[m])
            for col,k in enumerate([m,'B' if m=='A' else 'A']):
                x=55+col*(w+10);im.paste(Image.fromarray(frames[m,k][i]),(x,y))
                if view=='course' and i>=n-2:
                    r=read(root/f'{prefix}_{m}_plan_{k}/result.json')
                    d.text((x+15,y+h+4),f"Final error {r['error_mm']:.1f} mm",font=font(28 if w>=1200 else 23),fill=INK)
        d.text((55,2*h+151),f"{'2× slow motion' if slow==2 else 'Real time'}{grid_label} · {t[i]:.2f} s",font=font(24 if w>=1200 else 20),fill=MUTED)
        return np.asarray(im)
    name='putting_comparison' if view=='course' else 'putting_contact_closeup'+('_2x' if slow==2 else '')
    with imageio.get_writer(media/f'{name}.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=2,pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
        for i in range(n):writer.append_data(canvas(i))
        for _ in range(round(fps)):writer.append_data(canvas(n-1))
    still=n-1 if view=='course' else int(np.argmin(abs(t-np.mean(read(root/f'{prefix}_A_plan_A/result.json')['contact_time_span_s']))))
    Image.fromarray(canvas(still)).save(media/f'{name}.png')

def paper_demo(root,prefix='execution'):
    """Synchronized close and wide views of one recorded execution, real time."""
    media=root/'media';close=np.load(media/f'{prefix}_A_A_hero_frames.npy',mmap_mode='r')
    wide=np.load(media/f'{prefix}_A_A_course_frames.npy',mmap_mode='r')
    ts=np.load(media/f'{prefix}_A_A_hero_time.npy')
    np.testing.assert_allclose(ts,np.load(media/f'{prefix}_A_A_course_time.npy'))
    n=last_video_frame(root/f'{prefix}_A_plan_A',ts)+1;ts=ts[:n]
    H=640;cw=640;ww=round(wide.shape[2]*H/wide.shape[1]);W=cw+ww+48;W+=W%2
    result=read(root/f'{prefix}_A_plan_A/result.json')
    def frame(i):
        im=Image.new('RGB',(W,H+105),'white');d=ImageDraw.Draw(im)
        d.text((20,12),'Material A · Plan using matched ID',font=font(30,True),fill=INK)
        im.paste(Image.fromarray(close[i]),(16,57))
        im.paste(Image.fromarray(wide[i]).resize((ww,H),Image.Resampling.LANCZOS),(cw+32,57))
        d.text((20,H+70),f'Real time · {ts[i]:.2f} s',font=font(22),fill=MUTED)
        if i==len(ts)-1:d.text((cw+45,H+70),f"Stopping error {result['error_mm']:.1f} mm",font=font(22),fill=INK)
        return np.asarray(im)
    # One-pixel padding keeps both dimensions even for H.264 4:2:0.
    fps=1/(ts[1]-ts[0])
    with imageio.get_writer(media/'paper_putting_demo.mp4',fps=fps,codec='libx264',quality=8,
            macro_block_size=2,pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
        for i in range(len(ts)):
            a=frame(i);writer.append_data(np.pad(a,((0,a.shape[0]%2),(0,0),(0,0)),constant_values=255))
        for _ in range(round(fps)):writer.append_data(np.pad(a,((0,a.shape[0]%2),(0,0),(0,0)),constant_values=255))
    i=int(np.argmin(abs(ts-np.mean(result['contact_time_span_s']))))
    Image.fromarray(frame(i)).save(media/'paper_putting_demo.png')

def swing_sequence(root,prefix='execution'):
    """Four recorded states at one common view and scale."""
    media=root/'media';name=f'{prefix}_A_A_detail'
    frames=np.load(media/f'{name}_frames.npy',mmap_mode='r');ts=np.load(media/f'{name}_time.npy')
    cfg=read(root/f'{prefix}_A_plan_A/config.json');p=cfg['protocol']
    result=read(root/f'{prefix}_A_plan_A/result.json')
    contact=np.mean(result['contact_time_span_s'])
    start=p['settle']+p['backswing_duration']+p['backswing_pause']
    states=[(0.,'Address'),(p['settle']+p['backswing_duration'],'Backswing'),
            (contact,'Contact'),(start+cfg['controls'][0],'Follow-through')]
    im=Image.new('RGB',(1600,360),'white');d=ImageDraw.Draw(im)
    for j,(t,label) in enumerate(states):
        i=int(np.argmin(abs(ts-t)))
        im.paste(Image.fromarray(frames[i]).resize((400,320),Image.Resampling.LANCZOS),(j*400,40))
        d.text((j*400+16,8),label,font=font(24,True),fill=INK)
    im.save(media/'swing_sequence.png')

def overview(root,prefix='execution'):
    media=root/'media';name=f'{prefix}_A_A_overview';frames=np.load(media/f'{name}_frames.npy',mmap_mode='r');t=np.load(media/f'{name}_time.npy')
    frames=frames[:last_video_frame(root/f'{prefix}_A_plan_A',t)+1]
    with imageio.get_writer(media/'franka_putting.mp4',fps=25,codec='libx264',quality=8,macro_block_size=2,pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
        for i,frame in enumerate(frames):
            im=Image.fromarray(frame[70:730]);d=ImageDraw.Draw(im);d.text((25,22),'Franka · Flexible putter · Material A',font=font(28,True),fill=INK)
            d.text((25,im.height-35),f'Real time · {t[i]:.2f} s',font=font(20),fill=MUTED);writer.append_data(np.asarray(im))
        for _ in range(25):writer.append_data(np.asarray(im))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['render','compose','overview','sequence','demo','all']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--material',default='A');ap.add_argument('--plan',default='A');ap.add_argument('--prefix',default='execution')
    ap.add_argument('--slow',type=int,default=1);ap.add_argument('--view',default='course',choices=['course','detail','overview','hero']);ap.add_argument('--preview',action='store_true')
    a=ap.parse_args()
    if a.command=='all' and 'robot_base_shift' in read(a.out/'render_scene.json'):
        if not (a.out/'scene_robot/audit.json').exists():
            from experiments.elastic.putting_scene import compile_scene
            compile_scene(a.out)
        for m in 'AB':
            for k in 'AB':render_case(a.out,m,k)
        compose(a.out)
    elif a.command=='all':
        for view in ['course','detail']:
            for m in 'AB':
                for k in 'AB':render_case(a.out,m,k,view=view)
            compose(a.out,view=view)
        compose(a.out,view='detail',slow=2);swing_sequence(a.out)
        render_case(a.out,'A','A',view='overview');overview(a.out)
        render_case(a.out,'A','A',view='hero');paper_demo(a.out)
    elif a.command=='render':render_case(a.out,a.material,a.plan,a.prefix,a.view,a.preview)
    elif a.command=='compose':compose(a.out,a.prefix,a.view,a.slow)
    elif a.command=='demo':paper_demo(a.out,a.prefix)
    elif a.command=='sequence':swing_sequence(a.out,a.prefix)
    else:overview(a.out,a.prefix)

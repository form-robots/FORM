"""Measured-size initial geometry and camera checks for real pressing.

The supplied ball diameter is a nominal unloaded geometry datum, not an
independent measurement of loaded plate gap. Camera translation is fitted only
from initial visible contours; orientation/intrinsics come from the recording.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.optimize import least_squares
from experiments.robotics.press_hardware_observe import read, save, segment

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'press_real_data'
OBS = ROOT / 'out/press_observation_assessment_20260913'
BALL_DIAMETER = .045
PAD_DIAMETER = .100


def frame(ep):
    obs = np.load(OBS / ep / 'hand_observations.npz')
    j = np.argmin(abs(obs['time']))
    idx = int(obs['frame_idx'][j])
    cap = cv2.VideoCapture(str(RAW / ep / 'hand_rgb.mp4'))
    cap.set(1, idx)
    ok, rgb = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(ep)
    depth = cv2.imread(str(RAW / ep / f'hand_depth/{idx:06}.png'), -1)
    return rgb, depth, float(obs['plate_white_edge_y_px'][j]), idx


def initial_mask(rgb, depth, edge, material, center):
    mask, _ = segment(rgb, depth, 'hand', material)
    if material == 'plasticine':
        # Compact seed around the measured initial centroid excludes background.
        yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
        cy = (edge + 267) / 2
        ell = ((xx-center)/48)**2 + ((yy-cy)/44)**2
        labels = np.full(mask.shape, cv2.GC_BGD, np.uint8)
        labels[(ell < 1.5) & (yy > edge+9) & (yy < 271)] = cv2.GC_PR_FGD
        labels[(ell < .25) & (yy > edge+12)] = cv2.GC_FGD
        cv2.grabCut(rgb, labels, None, np.zeros((1,65)), np.zeros((1,65)), 4, cv2.GC_INIT_WITH_MASK)
        mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    return mask


def fit_camera(ep):
    rgb, depth, edge, idx = frame(ep)
    a = read(OBS / ep / 'assessment.json')
    obs = np.load(OBS / ep / 'hand_observations.npz')
    center = np.nanmedian(obs['mask_centroid_px'][(obs['time'] >= -.5) & (obs['time'] <= .2),0])
    mask = initial_mask(rgb, depth, edge, a['material'], center)
    points = []
    for y in range(int(edge)+15, 264):
        xx = np.flatnonzero(mask[y])
        xx = xx[(xx > center-70) & (xx < center+70)]
        if len(xx) > 20:
            points.extend([[xx.min(),y],[xx.max(),y]])
    points = np.array(points, float)
    circ = least_squares(lambda v: np.linalg.norm(points-v[:2],axis=1)-v[2],
                         [center, (edge+267)/2,45], loss='soft_l1', f_scale=1.)
    meta = read(RAW / ep / 'meta.json')
    intr = meta['cameras']['hand']['intrinsics']
    fx,fy,cx,cy = [intr[k] for k in ['fx','fy','ppx','ppy']]
    rays = np.c_[(points[:,0]-cx)/fx, (points[:,1]-cy)/fy, np.ones(len(points))]
    rays /= np.linalg.norm(rays,axis=1)[:,None]
    z = BALL_DIAMETER/2 * (fx+fy)/2 / circ.x[2]
    C0 = np.array([(circ.x[0]-cx)/fx*z, (circ.x[1]-cy)/fy*z,z])
    fit = least_squares(lambda C: np.linalg.norm(C[None]-(rays@C)[:,None]*rays,axis=1) - BALL_DIAMETER/2,
                        C0, loss='soft_l1',f_scale=.0005)
    C = fit.x
    R = np.array(meta['extrinsics']['cameras']['hand']['T_base_cam']['matrix'])[:3,:3]
    # The common local frame retains the calibrated world vertical; translation
    # is independently anchored to the initial visible specimen at supplied size.
    world_center = np.array([.09,.09,.05+BALL_DIAMETER/2])
    camera_position = world_center - R@C
    return dict(episode=ep,material=a['material'],frame_idx=idx,
                ball_diameter_m=BALL_DIAMETER,plate_diameter_m=PAD_DIAMETER,
                camera_center_of_ball_m=C.tolist(),camera_position_m=camera_position.tolist(),
                camera_forward=R[:,2].tolist(),camera_up=(-R[:,1]).tolist(),
                intrinsics=intr,circle_center_px=circ.x[:2].tolist(),circle_radius_px=float(circ.x[2]),
                contour_radial_rms_px=float(np.sqrt(np.mean(circ.fun**2))),
                sphere_ray_rms_mm=float(np.sqrt(np.mean(fit.fun**2))*1000),
                orientation_source='Recorded hand-camera extrinsic rotation; translation fit only to initial contour at supplied ball size',
                contact_scope='45 mm sphere tangent to platform and pad is a candidate initialization; loaded gap/contact flattening not independently established'),rgb


def configure_camera(plotter, camera, width=848, height=480):
    import vtk
    intr=camera['intrinsics'];pos=np.array(camera['camera_position_m'])
    plotter.camera.position=pos;plotter.camera.focal_point=pos+np.array(camera['camera_forward']);plotter.camera.up=camera['camera_up']
    near,far=.01,2.;P=np.zeros((4,4));P[0,0]=2*intr['fx']/width;P[1,1]=2*intr['fy']/height;P[0,2]=1-2*intr['ppx']/width;P[1,2]=2*intr['ppy']/height-1
    P[2,2]=-(far+near)/(far-near);P[2,3]=-2*far*near/(far-near);P[3,2]=-1
    mat=vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):mat.SetElement(i,j,P[i,j])
    plotter.camera.SetExplicitProjectionTransformMatrix(mat);plotter.camera.SetUseExplicitProjectionTransformMatrix(True)


def render(out, episodes):
    import pyvista as pv
    panels=[]
    colors={'Play-Doh':'#bc2429','butter slime':'#e5c833','plasticine':'#454442'}
    for ep in episodes:
        d,rgb=fit_camera(ep);dest=out/ep;dest.mkdir(parents=True,exist_ok=True);save(dest/'camera.json',d)
        w,h=848,480;p=pv.Plotter(off_screen=True,window_size=(w,h));p.set_background('#d8dee0')
        p.add_mesh(pv.Plane(center=(.09,.09,.05),direction=(0,0,1),i_size=.3,j_size=.3),color='#8a9899')
        p.add_mesh(pv.Sphere(radius=BALL_DIAMETER/2,center=(.09,.09,.05+BALL_DIAMETER/2),theta_resolution=100,phi_resolution=100),color=colors[d['material']],smooth_shading=True)
        # Thickness remains a displayed assumption; lower surface is at the
        # candidate physical contact plane regardless of displayed thickness.
        half=.003
        p.add_mesh(pv.Cylinder(center=(.09,.09,.05+BALL_DIAMETER+half),direction=(0,0,1),radius=PAD_DIAMETER/2,height=2*half,resolution=120),color='#eeeeee',smooth_shading=False)
        configure_camera(p,d,w,h)
        p.render();im=cv2.cvtColor(p.screenshot(return_img=True),cv2.COLOR_RGB2BGR);p.close()
        left=cv2.resize(rgb[125:290,335:585],(600,396));right=cv2.resize(im[125:290,335:585],(600,396))
        for img,label in [(left,ep+' recording'),(right,'45 mm sphere / 100 mm pad: setup check')]:cv2.putText(img,label,(8,22),0,.52,(0,0,0),2)
        panel=np.concatenate([left,right],axis=1);panels.append(panel);cv2.imwrite(str(dest/'initial_comparison.jpg'),panel)
    cv2.imwrite(str(out/'initial_geometry_comparison.jpg'),np.concatenate(panels,axis=0))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,required=True);a.add_argument('--episodes',nargs='+',default=['ep0002','ep0006','ep0010']);args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True);render(args.out,args.episodes)

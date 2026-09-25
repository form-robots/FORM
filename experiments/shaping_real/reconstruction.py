"""RGB-D scan completion, photographic foreground filtering, and rigid XY IoU.

Missing surfaces are interpolated. IoU measures reconstructed shape agreement,
not independently measured robot-frame placement accuracy.
"""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import pyvista as pv
import shapely
from pymeshfix import _meshfix
from scipy.ndimage import binary_fill_holes
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, mapping

from experiments.shaping_real.outline_metric import footprint
from reproduce.run import ROOT, read

TARGET = None

def polys(g):
    return [p for p in shapely.get_parts(g) if p.geom_type=='Polygon' and not p.is_empty and p.area>1e-12]

def metric(g):
    i=g.intersection(TARGET).area;u=g.union(TARGET).area
    return dict(iou=i/u,error_percent=100*(1-i/u),intersection_mm2=i,union_mm2=u,
                material_area_mm2=g.area,target_area_mm2=TARGET.area,
                excess_mm2=g.difference(TARGET).area,missing_mm2=TARGET.difference(g).area,
                bounds_mm=list(g.bounds),components=len(polys(g)),holes=sum(len(p.interiors) for p in polys(g)))

def segmented(m,threshold):
    rgb=m['RGB'].astype(float)
    m=m.copy();m['saturation']=(rgb.max(1)-rgb.min(1))/np.maximum(rgb.max(1),1)
    # Exact linear cut through triangle vertices, rather than whole-face rejection.
    return m.clip_scalar(scalars='saturation',value=threshold,invert=False).extract_surface(algorithm='dataset_surface').triangulate()

def fill_internal(g):
    # RGB views show solid objects at the central depth holes. Preserve every
    # exterior concavity and component; do not use a convex hull or close gaps.
    return shapely.union_all([Polygon(p.exterior) for p in polys(g)])

def mask_for(root,pose):
    im=cv2.cvtColor(cv2.imread(str(root/'color'/pose['color'])),cv2.COLOR_BGR2RGB)
    rgb=im.astype(float);sat=(rgb.max(2)-rgb.min(2))/np.maximum(rgb.max(2),1)
    mask=(sat>=.35)&(rgb[:,:,0]>=rgb[:,:,1])&(rgb[:,:,0]>=rgb[:,:,2])
    # Fixed world-space volume projects to a conservative image ROI.
    intr=json.loads((root/'meta.json').read_text())['camera']['color_intrinsics']
    box=np.array([[x,y,z] for x in [-.055,.055] for y in [-.055,.055] for z in [0,.05]])
    T=np.array(pose['T_cam_page']);cam=box@T[:3,:3].T+T[:3,3]
    uv=cam[:,:2]/cam[:,2,None]*[intr['fx'],intr['fy']]+[intr['ppx'],intr['ppy']]
    lo=np.maximum(np.floor(uv.min(0)).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)).astype(int),[im.shape[1]-1,im.shape[0]-1])
    roi=np.zeros(mask.shape,bool);roi[lo[1]:hi[1]+1,lo[0]:hi[0]+1]=True
    n,l,s,c=cv2.connectedComponentsWithStats((roi&mask).astype(np.uint8))
    mask=binary_fill_holes(l==(1+np.argmax(s[1:,cv2.CC_STAT_AREA])))
    return im,mask

def complete(source):
    scan=segmented(source,.35).connectivity(extraction_mode='largest').extract_surface(algorithm='dataset_surface').triangulate()
    t=_meshfix.PyTMesh()
    t.load_array(np.asarray(scan.points,np.float64),scan.faces.reshape(-1,4)[:,1:].astype(np.int32))
    count=int(t.fill_small_boundaries(nbe=0,refine=True))
    v,f=t.return_arrays()
    mesh=pv.PolyData(v,np.column_stack([np.full(len(f),3),f]).ravel()).fill_holes(.1).triangulate()
    # Track geometry changes and classify new triangles, including ones whose
    # vertices all lie on original hole boundaries.
    original=np.asarray(scan.points,dtype=float)
    distance,index=cKDTree(original).query(mesh.points)
    f=mesh.faces.reshape(-1,4)[:,1:]
    old_faces={tuple(sorted(x)) for x in scan.faces.reshape(-1,4)[:,1:].tolist()}
    inferred=np.array([bool((distance[tri]>1e-9).any()) or tuple(sorted(index[tri])) not in old_faces for tri in f])
    mesh.cell_data['inferred_patch']=inferred.astype(np.uint8)
    reverse_distance=cKDTree(mesh.points).query(original)[0]
    # Ensure every retained observed vertex was preserved, to numerical precision.
    assert np.max(reverse_distance)<1e-8
    mesh=mesh.compute_normals(auto_orient_normals=True,consistent_normals=True,split_vertices=False)
    audit=dict(holes_filled=count,observed_vertices=len(original),completed_vertices=mesh.n_points,
               maximum_retained_vertex_displacement_mm=float(np.max(reverse_distance)*1000),
               added_triangles=int(inferred.sum()),total_triangles=len(f),open_edges=mesh.n_open_edges)
    return mesh,scan,audit

def photo_color(mesh,scan,root):
    poses=[p for p in json.loads((root/'merge/poses.json').read_text())['frames'] if p['used']]
    # Twelve uniformly sampled saved views, selected without shape or IoU scores.
    poses=[poses[k] for k in np.linspace(0,len(poses)-1,12).astype(int)]
    intr=json.loads((root/'meta.json').read_text())['camera']['color_intrinsics']
    points=np.asarray(mesh.points);normals=np.asarray(mesh.point_data['Normals'])
    faces=mesh.faces.reshape(-1,4)[:,1:]
    colors=np.zeros((mesh.n_points,3));weights=np.zeros(mesh.n_points);support=np.zeros(mesh.n_points,int)
    for p in poses:
        im,mask=mask_for(root,p);mask=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
        T=np.array(p['T_cam_page']);cam=points@T[:3,:3].T+T[:3,3]
        uv=cam[:,:2]/cam[:,2,None]*[intr['fx'],intr['fy']]+[intr['ppx'],intr['ppy']]
        pixel=np.round(uv).astype(np.int32);u=pixel[:,0];v=pixel[:,1]
        inside=(u>=0)&(u<im.shape[1])&(v>=0)&(v<im.shape[0])&(cam[:,2]>0)
        u=np.clip(u,0,im.shape[1]-1);v=np.clip(v,0,im.shape[0]-1)
        # Approximate z-buffer using near-to-far triangle ordering. Refined scan
        # triangles are small. A 1.5 mm depth tolerance avoids edge raster gaps.
        zbuf=np.full(im.shape[:2],np.inf,np.float32)
        face_z=cam[faces,2].mean(1)
        for f in np.argsort(face_z)[::-1]:
            cv2.fillConvexPoly(zbuf,pixel[faces[f]],float(face_z[f]))
        camera=np.array(p['T_page_cam'])[:3,3]
        direction=camera-points;length=np.linalg.norm(direction,axis=1)
        cosine=(normals*(direction/length[:,None])).sum(1)
        valid=inside&mask[v,u]&(cosine>.05)&(np.abs(cam[:,2]-zbuf[v,u])<.0015)
        weight=np.where(valid,np.maximum(cosine,0)**6/length**2,0)
        sampled=np.concatenate([cv2.remap(im,uv[k:k+16000,0].astype(np.float32)[:,None],uv[k:k+16000,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0,:] for k in range(0,len(uv),16000)])
        colors+=sampled*weight[:,None];weights+=weight;support+=valid
    good=weights>0
    colors[good]/=weights[good,None]
    # Complete texture where no eligible view exists using the closest vertex
    # with an actual photographic color. This is appearance interpolation; it
    # avoids copying paper/background colors from the supplied incomplete scan.
    nearest=cKDTree(points[good]).query(points[~good])[1]
    colors[~good]=colors[good][nearest]
    mesh['RGB_photo']=np.clip(colors,0,255).astype(np.uint8)
    mesh['photo_view_count']=support
    return dict(frames=[p['frame_idx'] for p in poses],photo_colored_vertex_fraction=float(good.mean()),
                fallback='Nearest photograph-supported vertex color for vertices without an eligible RGB view; appearance interpolation, not a direct texture observation',
                visibility='Front-facing normal and approximate triangle z-buffer, 1.5 mm tolerance')

def aligned(mesh,row):
    q=mesh.copy();q.points*=1000
    p=row['rotation_deg_then_translation_mm'];r=np.deg2rad(p['rotation_deg'])
    R=np.array([[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]])
    q.points[:,:2]=q.points[:,:2]@R.T+p['translation_mm']
    return q

def gray_score(rgb):
    rgb=np.asarray(rgb,dtype=float);mx=rgb.max(-1);mn=rgb.min(-1)
    return np.minimum((mx-mn)/np.maximum(mx,1)-.20,(150-mx)/255)

def segment(mesh,threshold=None):
    m=mesh.copy();m['gray_foreground']=gray_score(m['RGB'])
    return m.clip_scalar(scalars='gray_foreground',value=0,invert=False).extract_surface(algorithm='dataset_surface').triangulate()

def gray_mask(root,pose):
    im=cv2.cvtColor(cv2.imread(str(root/'color'/pose['color'])),cv2.COLOR_BGR2RGB)
    intr=json.loads((root/'meta.json').read_text())['camera']['color_intrinsics']
    box=np.array([[x,y,z] for x in [-.055,.055] for y in [-.055,.055] for z in [0,.05]])
    T=np.asarray(pose['T_cam_page']);cam=box@T[:3,:3].T+T[:3,3]
    uv=cam[:,:2]/cam[:,2,None]*[intr['fx'],intr['fy']]+[intr['ppx'],intr['ppy']]
    lo=np.maximum(np.floor(uv.min(0)).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)).astype(int),[im.shape[1]-1,im.shape[0]-1])
    roi=np.zeros(im.shape[:2],bool);roi[lo[1]:hi[1]+1,lo[0]:hi[0]+1]=True
    n,l,s,c=cv2.connectedComponentsWithStats(((gray_score(im)>0)&roi).astype(np.uint8))
    mask=binary_fill_holes(l==(1+np.argmax(s[1:,cv2.CC_STAT_AREA])))
    return im,mask

def transform(g,p):
    return translate(rotate(g,float(p[2]),origin=(0,0)),float(p[0]),float(p[1]))

def register(g):
    # The X target is 180-degree symmetric. Sweep one full unique half-turn.
    candidates=[]
    for theta in np.arange(-90,90,5.):
        h=rotate(g,theta,origin=(0,0));c=h.centroid
        p=[-c.x,-c.y,theta]
        candidates.append((1-metric(transform(g,p))['iou'],p))
    # Refine six distinct coarse angles, including each trial's best coarse pose.
    attempts=[]
    for _,p in sorted(candidates)[:6]:
        result=minimize(lambda q:1-metric(transform(g,q))['iou'],p,method='Powell',
                        bounds=[(-30,30),(-30,30),(-100,100)],
                        options=dict(xtol=1e-5,ftol=1e-9,maxiter=120))
        attempts.append(dict(parameters=result.x.tolist(),error=float(result.fun),nfev=int(result.nfev),success=bool(result.success)))
    best=min(attempts,key=lambda q:q['error'])
    assert best['error']<=min(x[0] for x in candidates)+1e-6
    return transform(g,best['parameters']),best['parameters'],attempts

def reconstruct(output, *, refit_alignment=False):
    global TARGET, segmented, mask_for
    TARGET = translate(footprint(pv.read(ROOT / 'out/press_iou_independent_20260915/study/target.vtp')),
                       xoff=-80, yoff=-80)
    rows = read(ROOT / 'out/shaping_hardware_iou_recheck_20260915/phototextured/material_row/metrics.json')['trials']
    output.mkdir(parents=True, exist_ok=False)
    result = []
    for row in rows:
        source = ROOT / 'shaping_real_data/extracted' / row['trial'] / row['scan']
        raw = pv.read(source / 'merge/mesh.ply')
        original_segment, original_mask = segmented, mask_for
        gray = row['label'] == 'Gray'
        if gray:
            segmented, mask_for = segment, gray_mask
        try:
            mesh, retained, audit = complete(raw)
            audit['photo_color'] = photo_color(mesh, retained, source)
            registration = row['registration']
            if refit_alignment:
                _, parameters, _ = register(fill_internal(footprint(segmented(raw, .35))))
                registration = dict(rotation_deg=parameters[2], translation_mm=parameters[:2])
            if gray:
                mesh['foreground'] = gray_score(mesh['RGB_photo'])
                threshold = 0.
            else:
                rgb = mesh['RGB_photo'].astype(float)
                mesh['foreground'] = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1)
                threshold = .35
            clean = mesh.clip_scalar(scalars='foreground', value=threshold, invert=False).extract_surface(
                algorithm='dataset_surface').triangulate()
            clean = clean.connectivity(extraction_mode='largest').extract_surface(algorithm='dataset_surface')
            clean.save(output / (row['label'].lower() + '.vtp'))
            projected = aligned(clean, {'rotation_deg_then_translation_mm': registration})
            projected.points /= 1000
            shape = footprint(projected)
            result.append(dict(material=row['label'], metric=metric(shape), completion=audit,
                               registration=registration, alignment_refitted=refit_alignment))
            (output / (row['label'].lower() + '.geojson')).write_text(json.dumps(dict(
                type='FeatureCollection', features=[dict(type='Feature', properties=dict(role=role),
                geometry=mapping(geometry)) for role, geometry in [('target', TARGET), ('reconstruction', shape)]])))
        finally:
            segmented, mask_for = original_segment, original_mask
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'out/reproduced/scan-reconstruction')
    parser.add_argument('--refit-alignment', action='store_true')
    args = parser.parse_args()
    reconstruct(args.out, refit_alignment=args.refit_alignment)

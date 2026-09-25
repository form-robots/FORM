"""XY silhouettes: symmetric arc-length-weighted boundary distance in mm.

Silhouettes are unions of projected triangles, not convex hulls or point clouds.
Height is absent from the score. There is no translation or scale registration.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon

STEP_MM = .25

def footprint(mesh):
    mesh = mesh.triangulate()
    triangles = mesh.points[mesh.faces.reshape(-1,4)[:,1:], :2].astype(float)*1000
    edges1,edges2 = triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]
    area2 = np.abs(edges1[:,0]*edges2[:,1]-edges1[:,1]*edges2[:,0])
    polygons = shapely.polygons(triangles[area2>1e-10])
    result = shapely.union_all(polygons)
    assert result.is_valid and not result.is_empty and result.area>0
    return result

def rings(shape):
    for poly in shapely.get_parts(shape):
        yield poly.exterior
        yield from poly.interiors

def samples(shape, step=STEP_MM):
    coords=[];weights=[]
    for ring in rings(shape):
        points=np.asarray(ring.coords)
        for a,b in zip(points[:-1],points[1:]):
            length=float(np.linalg.norm(b-a))
            if length <= 1e-12:continue
            n=max(1,int(np.ceil(length/step)))
            coords.append(a+(b-a)*(np.arange(n)+.5)[:,None]/n)
            weights.extend([length/n]*n)
    return shapely.points(np.concatenate(coords)), np.array(weights)

def directed(source, destination, step=STEP_MM):
    points,weights=samples(source,step)
    return float(np.average(shapely.distance(points,destination.boundary),weights=weights))

def compare(actual, target, step=STEP_MM):
    ab=directed(actual,target,step);ba=directed(target,actual,step)
    return {'outline_mm':.5*(ab+ba), 'actual_to_target_mm':ab,'target_to_actual_mm':ba,
            'footprint_iou':float(actual.intersection(target).area/actual.union(target).area),
            'actual_area_mm2':float(actual.area),'target_area_mm2':float(target.area),
            'actual_width_depth_mm':(np.array(actual.bounds)[2:]-np.array(actual.bounds)[:2]).tolist()}

def measure(mesh,target):
    return compare(footprint(mesh),footprint(target))

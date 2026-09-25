"""Fixed boundary reconstruction and area-weighted surface distance for shaping.

Distances are in world coordinates, without registration or rescaling. Triangle
centroids provide deterministic surface quadrature. This is a reconstructed
boundary metric, distinct from the unsquared full-particle nearest-neighbor
metric retained in the experiment records.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from experiments.robotics.hex_shaping_figure import prism
from experiments.robotics.plastic_shaping_figure import surface

METRIC = dict(
    name="symmetric area-weighted mean surface distance",
    units="mm",
    reconstruction_voxel_m=.0025,
    reconstruction_gaussian_sigma_voxels=1.3,
    reconstruction_density_level=.5,
    target="Exact analytic prism, five linear triangle subdivisions",
    quadrature="Triangle centroids weighted by triangle area",
    distance="Absolute exact closest-point distance to the opposing triangle mesh",
    registration="None",
)


def triangle_quadrature(mesh):
    mesh = mesh.triangulate()
    triangles = mesh.points[mesh.faces.reshape(-1, 4)[:, 1:]]
    areas = .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.all(np.isfinite(areas)) or areas.sum() <= 0:
        raise ValueError("Surface must have finite, positive area")
    return triangles.mean(axis=1), areas


def directed_mean_m(source, destination):
    points, areas = triangle_quadrature(source)
    distances = pv.PolyData(points).compute_implicit_distance(destination)["implicit_distance"]
    return float(np.average(np.abs(distances), weights=areas))


def mesh_distance_mm(actual, target):
    return 500. * (directed_mean_m(actual, target) + directed_mean_m(target, actual))


def target_surface(target):
    return prism(target).triangulate().subdivide(5, subfilter="linear")


def surface_error_mm(data, target_mesh, key="x_after_1s"):
    actual = surface(data[key], data["vol0"], h=METRIC["reconstruction_voxel_m"])
    return mesh_distance_mm(actual, target_mesh)

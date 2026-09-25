"""Analytic geometry checks for the new boundary objective."""
import numpy as np
import pyvista as pv
from experiments.robotics.hex_shaping_surface import (
    directed_mean_m,
    mesh_distance_mm,
    target_surface,
    triangle_quadrature,
)


def test_parallel_planes_and_units():
    a = pv.Plane(center=(0., 0., 0.), direction=(0., 0., 1.), i_size=1., j_size=1.)
    b = a.translate((0., 0., .012), inplace=False)
    np.testing.assert_allclose(mesh_distance_mm(a, b), 12., atol=1e-5)
    np.testing.assert_allclose(directed_mean_m(a, b), .012, atol=1e-8)


def test_area_weighting_is_not_vertex_weighting():
    # Two disconnected triangles at distances 1 and 3, with areas 1 and 3.
    points = np.array([[0., 0., 1.], [2., 0., 1.], [0., 1., 1.],
                       [0., 0., 3.], [6., 0., 3.], [0., 1., 3.]])
    a = pv.PolyData(points, [3, 0, 1, 2, 3, 3, 4, 5])
    b = pv.Plane(center=(3., 0., 0.), direction=(0., 0., 1.), i_size=20., j_size=20.)
    np.testing.assert_allclose(directed_mean_m(a, b), 2.5)
    np.testing.assert_allclose(directed_mean_m(a.subdivide(2), b), 2.5)


def test_analytic_prism_geometry_and_self_distance():
    target = dict(diameter=.09, floor=.025, height=.08166669505936196)
    mesh = target_surface(target)
    _, areas = triangle_quadrature(mesh)
    side = target["diameter"] / np.sqrt(3)
    expected = 6 * side * target["height"] + np.sqrt(3) * target["diameter"]**2
    np.testing.assert_allclose(areas.sum(), expected, rtol=1e-6)
    assert mesh.n_open_edges == 0
    assert mesh_distance_mm(mesh, mesh) < 1e-5
    expected_volume = np.sqrt(3) / 2 * target["diameter"]**2 * target["height"]
    np.testing.assert_allclose(abs(mesh.volume), expected_volume, rtol=1e-6)

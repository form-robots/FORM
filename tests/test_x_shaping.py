"""Physical geometry checks for cylindrical pinching and the concave X target."""
import numpy as np
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.x_shaping import CONFIG, centers, cylinder_distance, specimen, target_mesh


def test_capped_cylinder_distances():
    x = np.array([[0, 0, 0], [.017, 0, 0], [0, 0, .0255], [.017, 0, .0265]])
    np.testing.assert_allclose(
        cylinder_distance(x, .014, .0225), [-.014, .003, .003, .005], atol=1e-12
    )


def test_physical_gap_and_clearance():
    for a in np.linspace(0, np.pi, 5):
        c = centers(a, .036, CONFIG["finger_bottom"])
        np.testing.assert_allclose(np.linalg.norm(c[1]-c[0])-2*CONFIG["radius"], .036)
        np.testing.assert_allclose(c[:, 2]-CONFIG["finger_height"]/2,
                                   CONFIG["floor"]+CONFIG["finger_bottom"])


def test_x_is_closed_and_volume_matched():
    mesh = target_mesh()
    assert mesh.n_open_edges == 0
    np.testing.assert_allclose(mesh.volume, np.prod(CONFIG["size"]), rtol=2e-6)
    assert mesh_distance_mm(mesh, mesh) < 1e-5
    for grid in [64, 80]:
        x, volume = specimen(grid)
        np.testing.assert_allclose(volume.astype(float).sum(), mesh.volume, rtol=2e-6)
        assert np.min(x[:, 2]) > CONFIG["floor"]

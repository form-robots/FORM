"""set_x/set_v must copy IN PLACE into the existing warp arrays. Replacing the array
object leaves captured CUDA graphs holding a stale pointer (CUDA 700 on the GPU, the
Vista pour crash), and the old alias-a-temporary pattern dangled once the cloned torch
tensor was collected. Pointer stability is the contract the graph path relies on."""
from __future__ import annotations

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import newtonian


def test_set_x_in_place_and_physics_continues():
    grid = GridConfig(n_grid=32, grid_lim=0.4)
    rng = np.random.default_rng(0)
    pts = (rng.random((300, 3), dtype=np.float32) * 0.1 + 0.15).astype(np.float32)
    s = Solver(grid=grid, device="cpu").load_particles(pts, np.full(300, 1e-6, np.float32))
    s.set_material(newtonian(eta=5.0, density=1000.0, bulk_modulus=5.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "slip")
    s.step(2.0e-4, substeps=3)

    ptr_x = s._sim.mpm_state.particle_x.ptr
    ptr_v = s._sim.mpm_state.particle_v.ptr
    x = s.x()
    shifted = x + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    s.set_x(shifted)
    s.set_v(np.zeros_like(x))

    assert s._sim.mpm_state.particle_x.ptr == ptr_x, "set_x replaced the array"
    assert s._sim.mpm_state.particle_v.ptr == ptr_v, "set_v replaced the array"
    np.testing.assert_allclose(s.x(), shifted, atol=1e-7)
    s.step(2.0e-4, substeps=3)          # physics continues on the same buffers
    assert np.isfinite(s.x()).all()

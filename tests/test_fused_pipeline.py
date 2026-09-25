"""The claymore-fused tick (g2p+stress+p2g in one particle pass, split grid zeroing)
must be BITWISE equal to the normal three-pass pipeline on CPU: the fused kernel calls
the same wp.funcs in the same per-particle order, gathers from grid_v_out while
scattering into the disjoint grid_v_in/grid_m, and the split zero preserves what
sub-threshold nodes read. Any bit of drift here means the pipelines diverged.
Covers the fluid path (newtonian, SVD skipped), the elastic path (jelly, FCR), and the
plastic path (metal, von Mises return map + StVK), with a moving-cup collider and
wrench readout in the mix."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import GlassProfile, GridConfig, Solver, cup_fill, newtonian
from warpmpm.materials import elastic, vonmises

PROF = GlassProfile(outer_radius=0.05, inner_radius=0.04, height=0.12,
                    base_thickness=0.02, fillet_radius=0.006)


def _blob(n_grid=48, lim=0.4, seed=0, n=2000):
    rng = np.random.default_rng(seed)
    pts = (rng.random((n, 3), dtype=np.float32) * 0.10 + 0.15).astype(np.float32)
    vol = np.full(n, 1.0e-7, np.float32)
    return GridConfig(n_grid=n_grid, grid_lim=lim), pts, vol


def _run(material, fused: bool, ticks=6, substeps=8, cup=False):
    grid, pts, vol = _blob()
    s = Solver(grid=grid, device="cpu", fused=fused).load_particles(pts, vol)
    s.set_material(material)
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    h = None
    if cup:
        h = s.add_cup(PROF, np.array([0.2, 0.2, 3 * grid.dx + 0.001]))
        s.set_cup(h, center=np.array([0.2, 0.2, 3 * grid.dx + 0.001]),
                  quat=(1.0, 0.0, 0.0, 0.0), velocity=(0.02, 0.0, 0.0),
                  omega=(0.0, 0.0, 0.0))
    wrench = None
    for _ in range(ticks):
        if h is not None:
            s.reset_cup_wrench(h)
        s.step(2.0e-4, substeps)
        if h is not None:
            wrench = s.cup_wrench(h, 2.0e-4 * substeps)
    st = s._sim.mpm_state
    return (s.x(), s.v(), st.particle_F.numpy().copy(),
            st.particle_stress.numpy().copy(), wrench)


MATERIALS = [
    ("newtonian", newtonian(eta=2.0, density=1000.0, bulk_modulus=2.0e5)),
    ("jelly", elastic(E=5.0e4, nu=0.3, density=800.0)),
    ("metal", vonmises(E=1.0e5, nu=0.3, yield_stress=2.0e3, density=1200.0)),
]


@pytest.mark.parametrize("label,material", MATERIALS, ids=[m[0] for m in MATERIALS])
def test_fused_bitwise_equals_normal(label, material):
    xa, va, Fa, sa, _ = _run(material, fused=False)
    xb, vb, Fb, sb, _ = _run(material, fused=True)
    assert np.array_equal(xa, xb), f"{label}: positions diverged"
    assert np.array_equal(va, vb), f"{label}: velocities diverged"
    assert np.array_equal(Fa, Fb), f"{label}: F diverged"
    assert np.array_equal(sa, sb), f"{label}: stress diverged"


def test_fused_bitwise_with_moving_cup_and_wrench():
    xa, va, _, _, wa = _run(MATERIALS[0][1], fused=False, cup=True)
    xb, vb, _, _, wb = _run(MATERIALS[0][1], fused=True, cup=True)
    assert np.array_equal(xa, xb)
    assert np.array_equal(va, vb)
    np.testing.assert_array_equal(wa["force"], wb["force"])
    np.testing.assert_array_equal(wa["torque"], wb["torque"])


def test_fused_conserves_mass_momentum_sanity():
    x, v, F, stress, _ = _run(MATERIALS[0][1], fused=True)
    assert np.isfinite(x).all() and np.isfinite(v).all()
    assert np.isfinite(F).all() and np.isfinite(stress).all()


def test_fused_graph_replay_matches_live_on_cuda():
    """CUDA atomic accumulation order can vary even for identical kernels.
    Require agreement to float precision; CPU split/fused equality above remains
    bitwise. The old bitwise CUDA assertion also fails on the untouched engine.
    """
    import warp as wp

    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA graph capture needs a GPU")

    def run(graphs: bool):
        grid, pts, vol = _blob()
        s = Solver(grid=grid, device="cuda:0", fused=True).load_particles(pts, vol)
        s._sim.use_cuda_graph = graphs
        s.set_material(newtonian(eta=5.0, density=1000.0))
        s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "separate", friction=0.3)
        s.add_domain_walls()
        h = s.add_cup(PROF, np.array([0.2, 0.2, 3 * grid.dx + 0.001]))
        s.set_cup(h, center=np.array([0.2, 0.2, 3 * grid.dx + 0.001]),
                  quat=(1.0, 0.0, 0.0, 0.0), velocity=(0.02, 0.0, 0.0),
                  omega=(0.0, 0.0, 0.0))
        for _ in range(6):
            s.step(2.0e-4, 8)
        return s.x(), s.v(), s.F()

    x_g, v_g, F_g = run(graphs=True)
    x_l, v_l, F_l = run(graphs=False)
    np.testing.assert_allclose(x_g, x_l, rtol=0, atol=2e-8)
    np.testing.assert_allclose(v_g, v_l, rtol=0, atol=1e-7)
    np.testing.assert_allclose(F_g, F_l, rtol=3e-7, atol=2e-8)

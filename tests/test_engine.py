"""Foundational engine tests: packaging, no-NaN stepping, momentum, volume, pressure.

The conservation tests are deliberately physics-grounded rather than vacuous: free fall
isolates numerical volume drift (no load -> det(F) must stay ~1) from the physical settling
compression a floor would introduce, and exercises momentum (mean v_z = -g*t). A settled
column checks hydrostatic pressure growth with depth (the 3D-trace pressure convention).
"""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import GridConfig, Solver, block, dough


def _dough_solver(n_grid: int = 40):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    pos, vol, floor = block(grid, size=(0.10, 0.05, 0.05), ppc=2)
    s = Solver(grid=grid).load_particles(pos, vol)
    s.set_material(dough())
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    return s


def _free_blob(n_grid: int = 40):
    """A blob lifted to mid-domain with NO floor collider -> clean free fall, far from
    every grid boundary, so any det(F) change or transverse drift is purely numerical."""
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    pos, vol, _ = block(grid, size=(0.08, 0.05, 0.05), ppc=2)
    pos[:, 2] += grid.grid_lim * 0.5 - pos[:, 2].mean()
    s = Solver(grid=grid).load_particles(pos, vol)
    s.set_material(dough())
    return s


def test_packaging_import_and_step():
    # imports work with no caller sys.path manipulation, and a tiny sim steps cleanly
    s = _dough_solver(32)
    assert s.n_particles > 100
    s.step(2.0e-5, 5)
    assert np.isfinite(s.x()).all()


def test_no_nan_under_gravity():
    s = _dough_solver(40)
    s.step(2.0e-5, 50)
    assert np.isfinite(s.x()).all()
    assert np.isfinite(s.stress()).all()


def test_free_fall_momentum():
    # no floor: a free blob accelerates at g, so mean v_z = -g*t (momentum / integration);
    # an APIC transfer that lost or injected momentum would fail this.
    s = _free_blob()
    dt, n = 2.0e-5, 100
    s.step(dt, n)
    vz = float(s.v()[:, 2].mean())
    expect = -9.81 * dt * n
    assert abs(vz - expect) < 0.05 * abs(expect), f"v_z={vz:.5f}, expected {expect:.5f}"
    # no transverse forcing -> horizontal momentum stays ~0
    assert abs(float(s.v()[:, 0].mean())) < 0.02 * abs(expect)


def test_volume_conserved_in_free_fall():
    # free fall applies no stress, so det(F) must stay ~1. This isolates numerical drift
    # from the physical settling compression that a floor-resting blob would conflate.
    s = _free_blob()
    s.step(2.0e-5, 5)
    F0 = float(np.abs(np.linalg.det(s.F())).mean())
    s.step(2.0e-5, 200)
    F1 = float(np.abs(np.linalg.det(s.F())).mean())
    assert abs(F1 - 1.0) < 5e-3, f"det(F) drifted to {F1:.5f} in free fall (expected ~1)"
    assert abs(F1 - F0) / F0 < 5e-3


def test_inverted_state_policies_warn_reject_or_mark():
    import torch

    s = _free_blob(24)
    F = s.F()
    np.testing.assert_allclose(np.linalg.det(F), 1.0)
    F[0, 0, 0] = -1.0
    s._sim.import_particle_F_from_torch(torch.from_numpy(F), device=s.device)

    with pytest.warns(RuntimeWarning, match=r"using \|det\(F\)\|"):
        assert np.isfinite(s.vol()).all()
    assert np.isfinite(s.cauchy()).all()

    s.inversion_policy = "raise"
    with pytest.raises(RuntimeError, match=r"det\(F\)"):
        s.vol()
    with pytest.raises(RuntimeError, match=r"det\(F\)"):
        s.cauchy()

    s.inversion_policy = "nan"
    assert np.isnan(s.vol()[0])
    assert np.isnan(s.cauchy()[0]).all()
    assert np.isfinite(s.vol()[1:]).all()


def test_settles_downward():
    # under gravity on a sticky floor the blob's centroid should not rise
    s = _dough_solver(40)
    z0 = s.x()[:, 2].mean()
    s.step(2.0e-5, 200)
    z1 = s.x()[:, 2].mean()
    assert z1 <= z0 + 1e-4


@pytest.mark.slow
def test_static_column_pressure_grows_with_depth():
    # a settled column carries more (compressive) pressure at its base than its top, of
    # hydrostatic order rho*g*h. Pressure is the 3D trace -(s_xx+s_yy+s_zz)/3 (the binding
    # convention; the 2D trace is forbidden).
    grid = GridConfig(n_grid=48, grid_lim=0.4)
    pos, vol, floor = block(grid, size=(0.08, 0.08, 0.12), ppc=2)
    s = Solver(grid=grid).load_particles(pos, vol)
    s.set_material(dough())
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    s.step(2.0e-5, 2000)  # settle toward static equilibrium
    x = s.x()
    c = s.cauchy()
    p = -(c[:, 0, 0] + c[:, 1, 1] + c[:, 2, 2]) / 3.0
    z = x[:, 2]
    lo = z < np.quantile(z, 0.33)
    hi = z > np.quantile(z, 0.67)
    assert p[lo].mean() > p[hi].mean(), "base should carry more pressure than the top"
    assert p[lo].mean() > 0.0, "base pressure should be compressive (positive)"
    p_hydro = 1000.0 * 9.81 * (z.max() - z.min())
    assert 0.1 * p_hydro < p[lo].mean() < 10.0 * p_hydro, (
        f"base pressure {p[lo].mean():.1f} not within an order of magnitude of "
        f"hydrostatic {p_hydro:.1f}"
    )

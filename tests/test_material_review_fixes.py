"""Regression gates for the 2026-07 material-review findings: retired ids stay
unreachable, von-Mises hardening actually evolves the yield stress, tabulated
factories validate their tables, and ranged tabulated assignment is refused."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.kernels.mpm_solver_warp import MATERIAL_NAME_TO_ID
from warpmpm.materials import tabulated_viscous, vonmises

G = GridConfig(n_grid=32, grid_lim=0.4)


def _block(s: Solver):
    h = G.dx / 2
    ax = [np.arange(a + h / 2, b, h) for a, b in ((0.16, 0.24),) * 3]
    pos = np.stack(np.meshgrid(*[ax[0], ax[1], np.arange(0.16 + h / 2, 0.24, h)],
                               indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    s.load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    return pos


def test_retired_material_ids_unreachable():
    """foam (3) and snow (4) shipped broken (unfinished return map on a wrong
    stress; no stress branch at all) and are retired: the name map must refuse
    them so nothing silently simulates stress-free material."""
    assert "foam" not in MATERIAL_NAME_TO_ID
    assert "snow" not in MATERIAL_NAME_TO_ID
    assert 3 not in MATERIAL_NAME_TO_ID.values()
    assert 4 not in MATERIAL_NAME_TO_ID.values()


def _squeeze(hardening_xi: float):
    """Crush a von-Mises block against the floor with a moving plate; return the
    per-particle yield-stress array after sustained plastic flow."""
    s = Solver(grid=G, device="cpu")
    _block(s)
    s.set_material(vonmises(E=5.0e5, nu=0.3, yield_stress=2.0e3,
                            hardening_xi=hardening_xi), g=[0.0, 0.0, 0.0])
    s.add_plane((0, 0, 0.16), (0, 0, 1), "sticky")
    plate = s.add_box(center=(0.2, 0.2, 0.30), half_size=(0.1, 0.1, 0.06),
                      velocity=(0.0, 0.0, -0.2))
    for _ in range(30):                       # plate descends 12 mm into the block
        s.step(2e-4, 10)
    _ = plate
    return s._sim.mpm_model.yield_stress.numpy()


def test_vonmises_hardening_evolves_yield():
    """hardening_xi must harden: the public factory documents isotropic
    hardening, but the kernel gates the update on a flag the factory previously
    never set, so xi was silently inert."""
    y_hard = _squeeze(hardening_xi=5.0e4)
    y_soft = _squeeze(hardening_xi=0.0)
    assert float(y_soft.max()) == pytest.approx(2.0e3), "xi=0 must stay perfectly plastic"
    assert float(y_hard.max()) > 2.1e3, (
        f"yield never evolved under plastic flow with xi set "
        f"(max {y_hard.max():.1f} Pa)")


def test_tabulated_factory_validates():
    with pytest.raises(ValueError):
        tabulated_viscous([10.0]).resolve()                    # < 2 samples
    with pytest.raises(ValueError):
        tabulated_viscous([10.0, float("nan")]).resolve()      # non-finite
    with pytest.raises(ValueError):
        tabulated_viscous([10.0, -1.0]).resolve()              # negative viscosity
    with pytest.raises(ValueError):
        tabulated_viscous([10.0, 20.0], smin=1.0, smax=1.0).resolve()  # smin == smax
    tabulated_viscous([10.0, 20.0, 30.0]).resolve()            # valid passes


def test_ranged_tabulated_material_refused():
    """Only one global eta table exists on the model and only set_material
    installs it; the ranged path used to accept the material and silently keep
    the default zero table."""
    s = Solver(grid=G, device="cpu")
    _block(s)
    with pytest.raises(NotImplementedError):
        s.set_material_range(0, 10, tabulated_viscous([10.0, 20.0, 30.0]))

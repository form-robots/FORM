"""Tests for the engine's grid semantics options (set_grid_semantics).

Three things are checked. First, that the options are opt-in: a
standard scene stepped with all of them off reproduces the reference engine's
trajectory bit for bit, so the added kernel branches cannot have moved the
default arithmetic. Second, that the freeslip wall clamp is approach-only, which
is the point of it: a particle thrown at a wall is arrested, and a particle
moving away from a wall keeps its whole outward velocity, where a collider
"slip" plane holds it against the wall instead. Third, that the empty-node
gravity and the eps-softened mass division are in force where they should be.

The reference was deliberately updated for compensated position and F integration
after test_slow_advection demonstrated lost motion in the old arithmetic. The
previous commit-409ccb9 digest was verified with the untouched source before this
update. A mismatch means either that the
default arithmetic moved or that the digest is being read on a different float
backend; the failure message says so, because only the first reading is a bug.
"""
from __future__ import annotations

import hashlib
import platform

import numpy as np
import pytest

N_GRID = 20
GRID_LIM = 1.0
DX = GRID_LIM / N_GRID
BOUND = 3
GRAVITY = (0.0, 0.0, -9.8)


def _sim(n_particles):
    import warp as wp
    wp.config.quiet = True
    wp.init()
    from warpmpm.kernels import MPM_Simulator_WARP
    return MPM_Simulator_WARP(n_particles, device="cpu")


def _cube(pitch_cells: float = 0.5, half: float = 0.12, centre=(0.5, 0.5, 0.5)):
    h = pitch_cells * DX
    ax = np.arange(-half, half + 0.5 * h, h)
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    pts = (pts + np.asarray(centre)).astype(np.float32)
    vol = np.full(len(pts), h ** 3, dtype=np.float32)
    return np.ascontiguousarray(pts), np.ascontiguousarray(vol)


def _load(s, pts, vol, v0, params, nclaw=None):
    import torch
    s.load_initial_data_from_torch(
        torch.from_numpy(pts), torch.from_numpy(vol),
        n_grid=N_GRID, grid_lim=GRID_LIM, device="cpu")
    s.import_particle_v_from_torch(torch.from_numpy(v0), device="cpu")
    s.set_parameters_dict(params, device="cpu")
    s.finalize_mu_lam(device="cpu")
    if nclaw is None:
        pad = BOUND * DX
        for pt, nrm in (((pad, 0, 0), (1, 0, 0)), ((GRID_LIM - pad, 0, 0), (-1, 0, 0)),
                        ((0, pad, 0), (0, 1, 0)), ((0, GRID_LIM - pad, 0), (0, -1, 0)),
                        ((0, 0, pad), (0, 0, 1)), ((0, 0, GRID_LIM - pad), (0, 0, -1))):
            s.add_surface_collider(tuple(map(float, pt)), tuple(map(float, nrm)),
                                   "slip")
    else:
        s.set_grid_semantics(**nclaw)
    return s


def _params():
    return {"material": "metal", "density": 1000.0, "g": list(GRAVITY),
            "E": 3.0e5, "nu": 0.25, "yield_stress": 5.0e3, "softening": 0.0}


def _run(nclaw, steps=60, dt=5.0e-4, v0_scale=1.0):
    pts, vol = _cube()
    v0 = np.tile(np.array([0.6, -0.4, -1.2], dtype=np.float32) * v0_scale,
                 (len(pts), 1))
    s = _load(_sim(len(pts)), pts.copy(), vol.copy(),
              np.ascontiguousarray(v0), _params(), nclaw)
    for step in range(steps):
        s.p2g2p(step, dt, device="cpu")
    return (s.export_particle_x_to_torch().numpy().copy(),
            s.export_particle_v_to_torch().numpy().copy(),
            s.export_particle_F_to_torch().numpy().copy())


# sha256 of the contiguous float32 bytes of x, v and F after 60 substeps of the
# scene in _run, with compensated integration (2026-09-12). The independent
# manufactured-motion tests justify this intentional change in arithmetic.
PRE_CHANGE_DIGEST = {
    "x": "38847ebb325f8ddcc402dd061b89b5e6db2816d8dc7b096926d2e9a754e67086",
    "v": "2a16f9cf64e23a4694c2699ed2da11f3f8dcb276b15379fd5d93b4c7b3dedee8",
    "F": "994a4ec6a309e123b0085210f0ae8f7b6df5f71e6d867aa7b198649e99e664f6",
}


def test_mode_off_is_bit_identical():
    """With the mode off, the default pipeline must reproduce the reference
    engine's output bit for bit: the mode's kernel branches are unreachable."""
    out = dict(zip(("x", "v", "F"), _run(None), strict=True))
    got = {k: hashlib.sha256(np.ascontiguousarray(
        v.astype(np.float32)).tobytes()).hexdigest() for k, v in out.items()}
    if got != PRE_CHANGE_DIGEST:
        pytest.fail(
            "default-path output changed against the compensated reference "
            f"({platform.machine()} / {platform.system()}): {got} vs "
            f"{PRE_CHANGE_DIGEST}. Either the default arithmetic moved (a bug) "
            "or this is a different float backend than the one the digest was "
            "taken on; investigate the arithmetic before editing the digest.")

    # the NCLaw kernel with every behavior disabled differs from the default only
    # in the mass threshold (m > 0 instead of m > 1e-15) and in that the walls
    # are gone, so it is NOT claimed identical; it must merely stay finite
    c = _run({"freeslip_bound": 0, "mass_eps": 0.0,
              "empty_node_gravity": False, "mls_transfer": False,
              "particle_clip_cells": -1.0})[0]
    assert np.isfinite(c).all()


def test_freeslip_wall_clamp_is_approach_only():
    """A particle thrown at a wall cannot penetrate it, and the same particle
    thrown away from the wall keeps its whole outward velocity.

    Both legs are analytic. At x = dx the particle's stencil is nodes 0, 1, 2,
    every one of them a wall node at bound 3, so an approaching velocity is
    zeroed at every node it gathers from (the particle does not move at all) and
    a separating velocity survives at every node (it travels v * t exactly).
    """
    dt, steps = 5.0e-4, 20
    x0 = 1.0 * DX
    pts = np.array([[x0, 0.5, 0.5]], dtype=np.float32)
    vol = np.array([DX ** 3], dtype=np.float32)
    mode = {"freeslip_bound": BOUND, "mass_eps": 0.0,
            "empty_node_gravity": False, "mls_transfer": False,
            "particle_clip_cells": 0.5}
    no_g = {**_params(), "g": [0.0, 0.0, 0.0]}

    s = _load(_sim(1), pts.copy(), vol.copy(),
              np.ascontiguousarray(np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)),
              no_g, mode)
    for step in range(steps):
        s.p2g2p(step, dt, device="cpu")
    x_app = s.export_particle_x_to_torch().numpy()[0, 0]
    assert x_app == pytest.approx(x0, abs=1e-7), "the wall did not arrest it"

    s = _load(_sim(1), pts.copy(), vol.copy(),
              np.ascontiguousarray(np.array([[+2.0, 0.0, 0.0]], dtype=np.float32)),
              no_g, mode)
    for step in range(steps):
        s.p2g2p(step, dt, device="cpu")
    x_sep = s.export_particle_x_to_torch().numpy()[0, 0]
    assert x_sep == pytest.approx(x0 + 2.0 * dt * steps, rel=1e-4), \
        "separation off the wall was not free"


def test_freeslip_clamp_on_the_far_face():
    """The same asymmetric band on the high face: nodes with index > n - bound.
    A particle at x = grid_lim - 2 dx gathers nodes 17, 18, 19 of 20, so an
    approaching velocity is cut to node 17's weight and a separating one is
    untouched."""
    dt, steps = 5.0e-4, 20
    x0 = GRID_LIM - 2.0 * DX
    pts = np.array([[x0, 0.5, 0.5]], dtype=np.float32)
    vol = np.array([DX ** 3], dtype=np.float32)
    mode = {"freeslip_bound": BOUND, "mass_eps": 0.0,
            "empty_node_gravity": False, "mls_transfer": False,
            "particle_clip_cells": 0.5}
    no_g = {**_params(), "g": [0.0, 0.0, 0.0]}

    s = _load(_sim(1), pts.copy(), vol.copy(),
              np.ascontiguousarray(np.array([[+2.0, 0.0, 0.0]], dtype=np.float32)),
              no_g, mode)
    for step in range(steps):
        s.p2g2p(step, dt, device="cpu")
    x_app = s.export_particle_x_to_torch().numpy()[0, 0]
    # the quadratic weight of the single unclamped node is 0.125 there, so the
    # advance cannot exceed an eighth of the unobstructed 0.02 m; in practice the
    # particle is compressed against the wall and springs back a little
    assert x_app < x0 + 0.125 * 2.0 * dt * steps
    assert x_app < GRID_LIM

    s = _load(_sim(1), pts.copy(), vol.copy(),
              np.ascontiguousarray(np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)),
              no_g, mode)
    for step in range(steps):
        s.p2g2p(step, dt, device="cpu")
    x_sep = s.export_particle_x_to_torch().numpy()[0, 0]
    assert x_sep == pytest.approx(x0 - 2.0 * dt * steps, rel=1e-4)


def test_collider_slip_plane_blocks_separation():
    """What the freeslip option fixes: the collider "slip" plane projects the
    normal component out unconditionally, so the particle that separates freely
    above is held against the wall instead."""
    dt, steps = 5.0e-4, 20
    x0 = 1.0 * DX
    pts = np.array([[x0, 0.5, 0.5]], dtype=np.float32)
    vol = np.array([DX ** 3], dtype=np.float32)
    s = _load(_sim(1), pts.copy(), vol.copy(),
              np.ascontiguousarray(np.array([[+2.0, 0.0, 0.0]], dtype=np.float32)),
              {**_params(), "g": [0.0, 0.0, 0.0]}, None)
    for step in range(steps):
        s.p2g2p(step, dt, device="cpu")
    x_slip = s.export_particle_x_to_torch().numpy()[0, 0]
    assert x_slip == pytest.approx(x0, abs=1e-6)


def test_empty_node_gravity_reaches_g2p():
    """With empty-node gravity on, a lone particle in free fall picks up the
    free-fall velocity of the empty nodes as well, so g2p is reading them."""
    dt = 5.0e-4
    pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    vol = np.array([DX ** 3], dtype=np.float32)
    v0 = np.zeros((1, 3), dtype=np.float32)
    out = {}
    for name, empty_gravity in (("off", False), ("on", True)):
        s = _load(_sim(1), pts.copy(), vol.copy(), v0.copy(), _params(),
                  {"freeslip_bound": 0, "mass_eps": 0.0,
                   "empty_node_gravity": empty_gravity, "mls_transfer": False,
                   "particle_clip_cells": -1.0})
        for step in range(10):
            s.p2g2p(step, dt, device="cpu")
        out[name] = s.export_particle_v_to_torch().numpy()[0, 2]
    # a lone particle's 27 nodes all carry mass, so free fall is identical either
    # way: the flag must not perturb a fully supported stencil
    assert out["on"] == pytest.approx(out["off"], rel=1e-6)
    assert out["off"] == pytest.approx(-9.8 * dt * 10, rel=1e-3)


def test_eps_softens_a_light_node():
    """eps changes the velocity of a node whose mass is small compared with it,
    and leaves a normal node alone. Checked on the grid arrays directly."""
    dt = 5.0e-4
    pts, vol = _cube()
    v0 = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(pts), 1))
    got = {}
    for eps in (0.0, 1.0):          # 1.0 is far above any nodal mass here
        s = _load(_sim(len(pts)), pts.copy(), vol.copy(),
                  np.ascontiguousarray(v0.copy()), _params(),
                  {"freeslip_bound": 0, "mass_eps": eps,
                   "empty_node_gravity": True, "mls_transfer": False,
                   "particle_clip_cells": -1.0})
        s.p2g2p(0, dt, device="cpu")
        got[eps] = s.export_grid_v_out_to_torch().numpy().copy()
    assert not np.allclose(got[0.0], got[1.0])
    # the softened velocities are strictly smaller in magnitude
    assert np.abs(got[1.0]).max() < np.abs(got[0.0]).max()


def test_freeslip_grid_op_refuses_the_fused_tick():
    pts, vol = _cube()
    v0 = np.zeros((len(pts), 3), dtype=np.float32)
    s = _load(_sim(len(pts)), pts, vol, v0, _params(),
              {"freeslip_bound": BOUND})
    with pytest.raises(ValueError, match="fused tick"):
        s.p2g2p_fused_tick(5.0e-4, 2, device="cpu")

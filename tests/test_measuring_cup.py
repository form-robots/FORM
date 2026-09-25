"""Measured 500 mL measuring-cup tests (the pour twin's real vessel): the parametric
geometry must reproduce the measurement sheet and hold physical volumes; the analytic
collision SDF must contain a settled fill with zero penetration and read its weight on
the Newton-exact wrench; a hard tilt toward the spout must pour the water OUT of the
+x side while conserving the particle count. Coarse grids for speed; the true-scale
run lives in examples/pour_franka.py."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import GridConfig, Solver, newtonian
from warpmpm.colliders.glass import angular_velocity_between, quat_from_axis_angle
from warpmpm.geometry.measuring_cup import (
    MeasuringCupSpec,
    build_cup_sdf,
    cavity_mask,
    cavity_sdf_local,
    cup_fill,
    make_cup_mesh,
    project_out_of_solid,
    solid_mask,
    solid_sdf_local,
)

SPEC = MeasuringCupSpec()
Q_ID = (1.0, 0.0, 0.0, 0.0)


def _q_xyzw(q):
    return (q[1], q[2], q[3], q[0])


# ---- geometry (no solver) ----------------------------------------------------------
def test_spec_reproduces_measurement_sheet():
    # the derived quantities the sheet states explicitly, to 10 um
    assert abs(SPEC.rim_width - 0.00501) < 1e-5
    assert abs(SPEC.a_top_inner - 0.04479) < 1e-5
    assert abs(SPEC.spout_len - 0.02217) < 1e-5
    assert abs(SPEC.tip_x - 0.07197) < 1e-5
    assert abs(SPEC.handle_x_min - (-0.091606)) < 1e-5
    assert abs(SPEC.handle_z_max - 0.098608) < 1e-5
    assert abs(SPEC.spout_z0 - 0.07506) < 1e-5


def test_cavity_volume_is_a_500ml_cup():
    # brim capacity must exceed the 500 mL graduation but stay a sane margin over it
    brim = SPEC.brim_volume
    assert 0.50e-3 < brim < 0.62e-3, f"brim volume {brim*1e6:.0f} mL"
    # monotone in depth, zero at zero
    d = np.linspace(0.0, SPEC.rim_z - SPEC.floor_z, 20)
    v = np.array([SPEC.cavity_volume(float(x)) for x in d])
    assert v[0] == 0.0 and (np.diff(v) > 0).all()


def test_fill_stays_inside_cavity_and_outside_solid():
    h = 0.002
    pos, vol = cup_fill(SPEC, h, fill_fraction=0.80)
    assert len(pos) > 1000
    assert (cavity_sdf_local(pos, SPEC) < 0).all(), "fill left the cavity"
    assert (solid_sdf_local(pos, SPEC) > 0).all(), "fill inside the plastic"
    # count-implied volume within the cavity volume at the fill level
    lvl = float(pos[:, 2].max()) - SPEC.floor_z + 0.5 * h
    assert vol.sum() < SPEC.cavity_volume(lvl)


def test_collision_sdf_field_and_margin():
    extra = 0.008
    sdf = build_cup_sdf(SPEC, res=96, margin=0.010, extra_wall=extra, extra_base=extra)
    vals = sdf.values
    # the stored boundary margin protects the collider's containment guard
    bmin = min(vals[0].min(), vals[-1].min(), vals[:, 0, :].min(),
               vals[:, -1, :].min(), vals[:, :, 0].min(), vals[:, :, -1].min())
    assert bmin > 0.005, f"SDF box margin too thin ({bmin*1e3:.1f} mm)"
    # interior must dip negative about half the thickened wall
    assert vals.min() < -0.3 * (SPEC.wall + extra)
    # cavity centre voxel is OUTSIDE the solid (open cup, not a plug)
    c = ((np.array([0.0, 0.0, 0.05]) - sdf.origin) / sdf.cell).round().astype(int)
    assert vals[c[0], c[1], c[2]] > 0.01


def test_render_mesh_watertight():
    _verts, faces = make_cup_mesh(SPEC)
    edges = {}
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            edges[e] = edges.get(e, 0) + 1
    assert all(n == 1 for n in edges.values()), "duplicate directed edge (bad winding)"
    twin = sum(1 for (a, b) in edges if (b, a) not in edges)
    assert twin == 0, f"{twin} boundary edges (mesh not closed)"


def test_project_out_of_solid_rescues():
    x = np.array([[0.0442, 0.0, 0.05], [0.0, 0.0, 0.002]])   # in wall, in base
    v = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    xn, _vn, n = project_out_of_solid(x, v, np.zeros(3), np.array(Q_ID), SPEC,
                                     clearance=5e-4)
    assert n == 2
    assert (solid_sdf_local(xn, SPEC) >= 4e-4).all()


# ---- solver-coupled ------------------------------------------------------------------
def _water_in_cup(n_grid=40, grid_lim=0.35, fill=0.5, device="auto"):
    grid = GridConfig(n_grid=n_grid, grid_lim=grid_lim)
    h = grid.dx / 2
    extra_wall = max(0.0, 3.0 * grid.dx - SPEC.wall)
    extra_base = max(0.0, 3.0 * grid.dx - SPEC.base)
    pos_local, vol = cup_fill(SPEC, h, fill_fraction=fill)
    cup_pos = np.array([0.5 * grid_lim, 0.5 * grid_lim, 0.16])
    s = Solver(grid=grid, device=device).load_particles(
        (pos_local + cup_pos).astype(np.float32), vol
    )
    s.set_material(newtonian(eta=1.0e-3, density=1000.0, bulk_modulus=9.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    sdf = build_cup_sdf(SPEC, res=96, margin=0.010,
                        extra_wall=extra_wall, extra_base=extra_base)
    cup = s.add_sdf_collider(sdf, center=cup_pos, quat=_q_xyzw(Q_ID),
                             band=0.5 * grid.dx, surface="separable", friction=0.05)
    return s, cup, cup_pos, vol, h, (extra_wall, extra_base)


def test_static_cup_holds_water_and_reads_weight():
    # a settled cup of water: nothing penetrates the wall, everything stays in the
    # cavity, and the accumulated grid impulse reads the water's WEIGHT (Newton check)
    s, cup, cup_pos, vol, h, extras = _water_in_cup()
    m_water = float(1000.0 * vol.sum())
    dt = 5.0e-5
    fz = []
    for _ in range(6):
        s.reset_sdf_force(cup)
        s.step(dt, 500)  # 25 ms per batch
        fz.append(s.sdf_wrench(cup, dt * 500)["force"][2])
        assert int(solid_mask(s.x(), cup_pos, Q_ID, SPEC, extra_wall=extras[0],
                              extra_base=extras[1]).sum()) == 0, "wall penetration"
    x = s.x()
    assert cavity_mask(x, cup_pos, Q_ID, SPEC, pad=0.75 * h).all(), "water left the cup"
    f_settled = float(np.mean(fz[-3:]))
    # the voxel field is trilinear, so allow more than the analytic cup's 10%
    assert abs(f_settled + 9.81 * m_water) < 0.2 * 9.81 * m_water, (
        f"cup wrench {f_settled:.2f} N != -weight {-9.81 * m_water:.2f} N"
    )


@pytest.mark.slow
def test_hard_tilt_pours_out_the_spout_and_conserves_mass():
    # tilt 120 deg about +y (the spout side +x goes DOWN): most of the water must
    # LEAVE, the count is exactly conserved, nothing ends embedded in the wall, and
    # the poured puddle lands on the spout side of the cup axis
    s, cup, cup_pos, _, h, extras = _water_in_cup(n_grid=48)
    n0 = s.n_particles
    dt, fps = 5.0e-5, 50
    substeps = round(1.0 / fps / dt)
    s.step(dt, 6 * substeps)

    def q_at(t):  # 120 deg over 1.2 s, then hold
        a = np.deg2rad(120.0) * min(t / 1.2, 1.0)
        return quat_from_axis_angle([0, 1, 0], a)

    max_embedded = 0
    n_ticks = int(2.0 * fps)
    for tick in range(n_ticks):
        t = tick / fps
        q0, q1 = q_at(t), q_at(t + 1.0 / fps)
        omega = angular_velocity_between(q0, q1, 1.0 / fps)
        s.set_sdf_pose(cup, center=cup_pos, quat=_q_xyzw(q0), velocity=(0, 0, 0),
                       omega=omega)
        s.step(dt, substeps)
        max_embedded = max(
            max_embedded,
            int(solid_mask(s.x(), cup_pos, q1, SPEC, extra_wall=extras[0],
                           extra_base=extras[1]).sum()),
        )
    x = s.x()
    assert len(x) == n0, "particle count must be exactly conserved"
    assert np.isfinite(x).all()
    q_end = q_at(99.0)
    in_cup = int(cavity_mask(x, cup_pos, q_end, SPEC, pad=0.75 * h).sum())
    assert in_cup < 0.4 * n0, f"cup should have poured out (still holds {in_cup}/{n0})"
    assert max_embedded <= 0.005 * n0, f"{max_embedded} particles embedded in the wall"
    out = ~cavity_mask(x, cup_pos, q_end, SPEC, pad=0.75 * h)
    assert x[out, 0].mean() > cup_pos[0], "pour did not exit on the spout (+x) side"
    assert x[:, 2].min() > 0.0 and x.max() < 0.35, "material escaped the domain"

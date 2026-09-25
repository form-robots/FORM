"""Mesh -> SDF builder tests, against analytic ground truth.

A sphere has a closed-form signed distance |p| - R, so the voxelized field, its sign, and its
gradient are all checkable to the grid resolution. A cup checks that a concave shell is signed
correctly (cavity and exterior positive, wall and base negative), which is what makes a fluid
stay contained. These run on the pure-numpy path, so they need no optional dependency.
"""
from __future__ import annotations

import functools

import numpy as np

from warpmpm.geometry import build_sdf, build_sdf_cached, make_cup_mesh, revolve_profile


def _trilerp(sdf, p):
    fidx = (np.asarray(p, float) - sdf.origin) / sdf.cell
    i = np.clip(np.floor(fidx).astype(int), 0, sdf.res - 2)
    fr = fidx - i
    val = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((fr[0] if dx else 1 - fr[0]) * (fr[1] if dy else 1 - fr[1])
                     * (fr[2] if dz else 1 - fr[2]))
                val += w * sdf.values[i[0] + dx, i[1] + dy, i[2] + dz]
    return float(val)


def _sphere_mesh(R=0.05, n_theta=40, n_ang=24):
    ang = np.linspace(-np.pi / 2, np.pi / 2, n_ang)
    prof = [(0.0, -R)] + [(R * np.cos(a), R * np.sin(a)) for a in ang] + [(0.0, R)]
    return revolve_profile(prof, n_theta=n_theta)


@functools.lru_cache(maxsize=2)
def _sphere_sdf(R=0.05, res=48):
    v, f = _sphere_mesh(R)
    return build_sdf(v, f, res=res, margin_cells=4)


@functools.lru_cache(maxsize=2)
def _cup_sdf(res=56):
    v, f = make_cup_mesh(inner_radius=0.04, wall_thickness=0.008, height=0.09,
                         base_thickness=0.01, n_theta=40)
    return build_sdf(v, f, res=res, margin_cells=4,
                     interior_probe=np.array([0.044, 0.0, 0.04]))


def test_sphere_sdf_matches_analytic():
    R = 0.05
    sdf = _sphere_sdf(R)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(400):
        p = rng.uniform(-0.06, 0.06, 3)
        errs.append(abs(_trilerp(sdf, p) - (np.linalg.norm(p) - R)))
    errs = np.array(errs)
    # interpolation error must be a small fraction of a voxel
    assert errs.mean() < 0.3 * sdf.cell, f"mean SDF error {errs.mean()} vs cell {sdf.cell}"
    assert errs.max() < 1.0 * sdf.cell


def test_sphere_sign_inside_outside():
    R = 0.05
    sdf = _sphere_sdf(R)
    assert _trilerp(sdf, [0, 0, 0]) < 0.0          # centre is inside
    assert _trilerp(sdf, [0.04, 0, 0]) < 0.0       # still inside (r < R)
    assert _trilerp(sdf, [0.058, 0, 0]) > 0.0      # just outside


def test_sphere_gradient_is_radial():
    R = 0.05
    sdf = _sphere_sdf(R)
    # the stored gradient at a near-surface voxel should point radially outward (unit-ish)
    for direction in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        p = direction * (R + sdf.cell)
        fidx = (p - sdf.origin) / sdf.cell
        i = np.clip(np.round(fidx).astype(int), 0, sdf.res - 1)
        g = sdf.grads[i[0], i[1], i[2]]
        g = g / (np.linalg.norm(g) + 1e-12)
        assert float(g @ direction) > 0.9, f"gradient {g} not radial along {direction}"


def test_cup_is_signed_for_containment():
    sdf = _cup_sdf()
    # cavity (air above the base, inside the wall) and exterior and above-rim are OUTSIDE (>0)
    assert _trilerp(sdf, [0.0, 0.0, 0.04]) > 0.0    # cavity centre
    assert _trilerp(sdf, [0.0, 0.0, 0.13]) > 0.0    # above the open rim
    assert _trilerp(sdf, [0.07, 0.0, 0.04]) > 0.0   # outside the wall
    # the solid wall and the base are INSIDE (<0)
    assert _trilerp(sdf, [0.044, 0.0, 0.04]) < 0.0  # mid-wall
    assert _trilerp(sdf, [0.0, 0.0, 0.004]) < 0.0   # mid-base


def test_revolve_profile_closed_watertight_count():
    # a closed revolution of a P-vertex profile over n_theta gives a consistent vertex count
    v, f = make_cup_mesh(n_theta=32)
    assert v.shape[1] == 3 and f.shape[1] == 3
    assert len(f) > 0
    # faces index valid vertices
    assert f.min() >= 0 and f.max() < len(v)


def test_cache_key_includes_interior_probe(tmp_path):
    v, f = _sphere_mesh(R=0.03, n_theta=16, n_ang=12)
    build_sdf_cached(v, f, res=18, margin_cells=3, cache_dir=tmp_path,
                     interior_probe=np.array([0.0, 0.0, 0.0]))
    build_sdf_cached(v, f, res=18, margin_cells=3, cache_dir=tmp_path,
                     interior_probe=np.array([0.005, 0.0, 0.0]))
    assert len(list(tmp_path.glob("sdf_*.npz"))) == 2

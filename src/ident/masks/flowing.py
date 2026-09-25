"""Flowing mask and rheology-validity mask (MATH_REFERENCE.md Section 4).

Flowing: |gamma_dot|_eps > gamma_min AND I > I_min.

Validity, per particle, in units of grain diameter d:
  distance to front > c_f d, distance to free surface and base > c_s d,
  flowing-layer thickness h_flow / d > c_h along the local inward surface
  normal. For the quasi-2D column the surface normal is approximated as
  vertical, so surface distance and flowing-layer thickness are measured
  along z within x bins. This approximation is documented here and is the
  only deviation from the verbatim rule; it is exact wherever the free
  surface is locally horizontal.

Gate transient rule: drop t < max(2 sqrt(d / g_mag), measured gate-clearance
time, closure-diagnostic spike interval).
"""

from __future__ import annotations

import numpy as np

from common.conventions import (
    C_F_DEFAULT,
    C_H_DEFAULT,
    C_S_DEFAULT,
    G_MAG,
    GAMMA_MIN_DEFAULT,
    I_MIN_DEFAULT,
)


def flowing_mask(
    gamma_dot_eps: np.ndarray,
    I: np.ndarray,
    gamma_min: float = GAMMA_MIN_DEFAULT,
    I_min: float = I_MIN_DEFAULT,
) -> np.ndarray:
    return (np.asarray(gamma_dot_eps) > gamma_min) & (np.asarray(I) > I_min)


def validity_mask_quasi2d(
    x: np.ndarray,
    z: np.ndarray,
    flowing: np.ndarray,
    d: float,
    c_f: float = C_F_DEFAULT,
    c_s: float = C_S_DEFAULT,
    c_h: float = C_H_DEFAULT,
    bin_width: float | None = None,
    resolution_dx: float | None = None,
    min_cells: float = 2.0,
) -> np.ndarray:
    """Per-particle validity for one frame of a quasi-2D collapse.

    Clearance lengths are c_* d as specified, but floored at min_cells *
    resolution_dx when the grid spacing is given: below a couple of grid
    cells the MPM stress and velocity gradient are grid-smoothed and the
    inertial number is unreliable however small the physical grain is. In
    well-resolved experiments (d comparable to or larger than dx) the floor
    is inactive and the pure grain-diameter rule is recovered.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    flowing = np.asarray(flowing, dtype=bool)
    if bin_width is None:
        bin_width = max(4.0 * d, (resolution_dx or 0.0))
    floor = min_cells * resolution_dx if resolution_dx is not None else 0.0
    clear_front = max(c_f * d, floor)
    clear_surface = max(c_s * d, floor)
    clear_height = max(c_h * d, floor)

    x0 = x.min()
    bin_idx = np.floor((x - x0) / bin_width).astype(int)
    n_bins = int(bin_idx.max()) + 1

    h_surf = np.full(n_bins, -np.inf)
    np.maximum.at(h_surf, bin_idx, z)
    z_base = np.full(n_bins, np.inf)
    np.minimum.at(z_base, bin_idx, z)

    # flowing-layer thickness per bin, vertical extent of flowing particles
    h_flow = np.zeros(n_bins)
    if np.any(flowing):
        top = np.full(n_bins, -np.inf)
        bot = np.full(n_bins, np.inf)
        np.maximum.at(top, bin_idx[flowing], z[flowing])
        np.minimum.at(bot, bin_idx[flowing], z[flowing])
        has = np.isfinite(top) & np.isfinite(bot)
        h_flow[has] = (top - bot)[has]

    # front position: outermost x among flowing particles
    x_front = x[flowing].max() if np.any(flowing) else np.inf

    dist_surface = h_surf[bin_idx] - z
    dist_base = z - z_base[bin_idx]
    dist_front = x_front - x

    valid = (
        (dist_front > clear_front)
        & (dist_surface > clear_surface)
        & (dist_base > clear_surface)
        & (h_flow[bin_idx] > clear_height)
    )
    return valid


def gate_transient_cutoff(
    d: float,
    gate_clearance_time: float = 0.0,
    closure_spike_time: float = 0.0,
) -> float:
    """Earliest admissible time after release."""
    return max(2.0 * np.sqrt(d / G_MAG), gate_clearance_time, closure_spike_time)

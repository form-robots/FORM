"""Scene builders: particle blocks + material presets (reused by tests/benchmarks/examples)."""
from __future__ import annotations

import numpy as np

from warpmpm.core.solver import GridConfig
from warpmpm.materials import Material, newtonian


def dough(eta: float = 40.0, tau_y: float = 200.0) -> Material:
    """Return the Bingham material used in the squeeze-flow study.

    This composes the Newtonian model with yield stress; it does not add a material type.
    """
    return newtonian(eta=eta, density=1000.0).with_yield(tau_y)


def block(grid: GridConfig, size=(0.12, 0.05, 0.06), center=None, ppc: int = 2, seed: int = 0):
    """A jittered particle box resting on the floor. Returns (pos[N,3], vol[N])."""
    dx = grid.dx
    floor = 3 * dx
    h = dx / ppc
    cx = grid.grid_lim * 0.5 if center is None else center[0]
    cy = grid.grid_lim * 0.5 if center is None else center[1]
    sx, sy, sz = size
    xs = np.arange(cx - 0.5 * sx + 0.5 * h, cx + 0.5 * sx, h)
    ys = np.arange(cy - 0.5 * sy + 0.5 * h, cy + 0.5 * sy, h)
    zs = np.arange(floor + 0.5 * h, floor + sz, h)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    g = g + rng.uniform(-0.25 * h, 0.25 * h, size=g.shape)
    pos = g.astype(np.float32)
    vol = np.full(len(pos), h**3, dtype=np.float32)
    return pos, vol, floor

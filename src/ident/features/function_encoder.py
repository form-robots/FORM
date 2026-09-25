"""Mode F dictionary, table-backed (Phase 1 stub of the deployed artifact).

The deployed function-encoder artifact is NOT a network: it is a frozen
256-point tabulation of phi_k on s = log10 I in [-4, 0] with cubic
interpolation (docs/FUNCTION_ENCODER.md Section 1). This class implements
exactly that wrapper so assembly and solve code can exercise Mode F today.
Training code (torch) is Phase 3 and lives only under
ident/features/function_encoder_training/.

Internally everything is s = log10 I; the public interface takes physical I
per the Section 7 contract. Outside the tabulated support the basis is
clamped to its edge values with zero derivative, and metadata records the
support so downstream code can refuse extrapolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from common.conventions import LN10
from ident.features.base import Dictionary


class FunctionEncoderDict(Dictionary):
    def __init__(self, s_grid: np.ndarray, table: np.ndarray) -> None:
        """s_grid: (n,) ascending log10 I values; table: (n, K) basis samples."""
        s_grid = np.asarray(s_grid, dtype=float)
        table = np.asarray(table, dtype=float)
        if s_grid.ndim != 1 or table.ndim != 2 or table.shape[0] != s_grid.shape[0]:
            raise ValueError("need s_grid (n,) and table (n, K)")
        if np.any(np.diff(s_grid) <= 0.0):
            raise ValueError("s_grid must be strictly ascending")
        self.s_grid = s_grid
        self.table = table
        self._spline = CubicSpline(s_grid, table, axis=0, bc_type="natural")
        self._dspline = self._spline.derivative()

    @property
    def K(self) -> int:
        return self.table.shape[1]

    def _s_clamped(self, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        I = np.atleast_1d(np.asarray(I, dtype=float))
        with np.errstate(divide="ignore"):
            s = np.log10(np.maximum(I, np.finfo(float).tiny))
        inside = (s >= self.s_grid[0]) & (s <= self.s_grid[-1])
        return np.clip(s, self.s_grid[0], self.s_grid[-1]), inside

    def phi(self, I: np.ndarray) -> np.ndarray:
        s, _ = self._s_clamped(I)
        return self._spline(s)

    def dphi_dI(self, I: np.ndarray) -> np.ndarray:
        """Analytic chain rule: dphi_dI = dphi_ds / (I ln 10); zero outside support."""
        I = np.atleast_1d(np.asarray(I, dtype=float))
        s, inside = self._s_clamped(I)
        dphi_ds = self._dspline(s)
        out = np.zeros_like(dphi_ds)
        Ii = I[inside]
        out[inside] = dphi_ds[inside] / (Ii[:, None] * LN10)
        return out

    def dphi_dlogI(self, I: np.ndarray) -> np.ndarray:
        """Native log-coordinate derivative; zero outside support."""
        I = np.atleast_1d(np.asarray(I, dtype=float))
        s, inside = self._s_clamped(I)
        dphi_ds = self._dspline(s)
        dphi_ds[~inside] = 0.0
        return dphi_ds

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "F",
            "support": (float(10.0 ** self.s_grid[0]), float(10.0 ** self.s_grid[-1])),
            "nonnegativity": [False] * self.K,
            "table_points": int(self.s_grid.shape[0]),
        }

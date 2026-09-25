"""Dictionary interface, the binding contract of MATH_REFERENCE.md Section 7.

Assembly and solve code never sees log coordinates. Implementations take
physical inertial number I everywhere. The three derivative channels
(dphi_dI, dphi_dlogI, and the pressure channel dphi_dp built downstream from
dphi_dI) are the designated bug class for this module; tests/test_features.py
cross-checks them against finite differences for every implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from common.conventions import LN10


class Dictionary(ABC):
    """mu(I) = sum_k theta_k phi_k(I), linear in theta (a project invariant)."""

    @property
    @abstractmethod
    def K(self) -> int:
        """Number of basis functions."""

    @abstractmethod
    def phi(self, I: np.ndarray) -> np.ndarray:
        """Basis values, shape (len(I), K)."""

    @abstractmethod
    def dphi_dI(self, I: np.ndarray) -> np.ndarray:
        """Physical-I derivative, shape (len(I), K).

        Consumed by the pressure sensitivity operator via
        d phi / d p = (d phi / d I) * (-I / (2 p)).
        """

    def dphi_dlogI(self, I: np.ndarray) -> np.ndarray:
        """Derivative with respect to log10 I, shape (len(I), K).

        Exact chain rule: dphi_dlogI = dphi_dI * I * ln(10).
        Used only for monotonicity constraint rows.
        """
        I = np.atleast_1d(np.asarray(I, dtype=float))
        return self.dphi_dI(I) * I[:, None] * LN10

    def gram(self, weight: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """G_kl = INT w(s) phi_k phi_l ds on s = log10 I, by trapezoid.

        weight is (I_grid, w_values) with I_grid > 0 ascending; the
        integration coordinate is s = log10(I_grid).
        """
        I_grid, w_vals = weight
        I_grid = np.asarray(I_grid, dtype=float)
        w_vals = np.asarray(w_vals, dtype=float)
        if I_grid.ndim != 1 or I_grid.shape != w_vals.shape:
            raise ValueError("weight must be (I_grid, w_values) of equal 1d shapes")
        if np.any(I_grid <= 0.0) or np.any(np.diff(I_grid) <= 0.0):
            raise ValueError("I_grid must be positive and strictly ascending")
        s = np.log10(I_grid)
        P = self.phi(I_grid)  # (n, K)
        # trapezoid weights on s
        ds = np.diff(s)
        tw = np.zeros_like(s)
        tw[:-1] += 0.5 * ds
        tw[1:] += 0.5 * ds
        return np.einsum("n,nk,nl->kl", tw * w_vals, P, P)

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """At least: mode, support, nonnegativity pattern."""

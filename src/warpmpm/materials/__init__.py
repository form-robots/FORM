"""Composable, physics-named constitutive models.

A `Material` is built by composing components and then `.resolve()`d to the
engine's (material_name, params). Compose, do not subclass:

    newtonian(eta=40)                          # pure viscous fluid
    newtonian(eta=40).with_yield(200)          # Bingham
    newtonian(eta=40).with_yield(200).with_powerlaw(K=5, n=0.8)   # Herschel-Bulkley
    granular(mu_s=0.38)                         # mu(I) sand (delta_mu=0 -> constant mu)
    granular(mu_s=0.38, delta_mu=0.26, I0=0.3)  # Pouliquen mu(I)
    granular(mu_s=0.38).with_dilatancy(phi_init=0.6, chi=0.2)     # compressible mu(I)-Phi(I)
    elastic(E=1e5, nu=0.3)                       # Neo-Hookean-ish solid

Materials are named for their PHYSICS (newtonian, granular, elastic), never for an
application. Each `.with_*` returns a new frozen Material, so they compose freely. Not
every combination is realized by the current engine; `.resolve()` validates and maps to the
nearest engine material (newtonian=10, mu_i_sand=9, mu_i_phi=11, jelly=0).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Material:
    base: str  # "newtonian" | "granular" | "elastic" | "tabulated"
    E: float = 1.0e5
    nu: float = 0.3
    density: float = 1000.0
    bulk_modulus: float = 9.0e5
    # generalized-Newtonian components (base="newtonian")
    viscosity: float = 0.0       # eta  [Pa.s]
    yield_stress: float = 0.0    # tau_y [Pa] (Bingham)
    powerlaw_K: float = 0.0      # consistency (hardening)
    powerlaw_n: float = 1.0      # exponent (softening); n<1 shear-thinning
    # von-Mises (J2) elasto-plastic solid (base="vonmises", fork "metal"=1): HOLDS a shape
    hardening_xi: float = 0.0    # isotropic hardening modulus (xi); 0 = perfectly plastic
    # granular mu(I) components (base="granular")
    mu_s: float = 0.0
    delta_mu: float = 0.0
    I0: float = 0.3
    grain_diameter: float = 1.0e-3
    grain_density: float = 2650.0
    # dilatancy component (compressible mu(I)-Phi(I))
    dilatant: bool = False
    phi_init: float = 0.6
    phi_chi: float = 0.2
    # tabulated apparent viscosity (base="tabulated"): eta_app(gd) samples on a uniform
    # log10(gd) grid in [eta_smin, eta_smax]; used to re-simulate an FE-recovered curve.
    eta_table: Any = None        # 1D array-like of eta_app samples (Pa.s)
    eta_smin: float = -1.0
    eta_smax: float = 2.0

    # ---- composable modifiers (return a new Material) -------------------------------
    def with_viscosity(self, eta: float) -> Material:
        return dataclasses.replace(self, viscosity=eta)

    def with_yield(self, tau_y: float) -> Material:
        return dataclasses.replace(self, yield_stress=tau_y)

    def with_powerlaw(self, K: float, n: float) -> Material:
        return dataclasses.replace(self, powerlaw_K=K, powerlaw_n=n)

    def with_friction(self, mu_s: float, delta_mu: float = 0.0, I0: float = 0.3) -> Material:
        return dataclasses.replace(self, mu_s=mu_s, delta_mu=delta_mu, I0=I0)

    def with_dilatancy(self, phi_init: float = 0.6, chi: float = 0.2) -> Material:
        return dataclasses.replace(self, dilatant=True, phi_init=phi_init, phi_chi=chi)

    def with_density(self, rho: float) -> Material:
        return dataclasses.replace(self, density=rho)

    # ---- map to the warp-mpm engine ---------------------------------------------------
    def resolve(self) -> tuple[str, dict[str, float]]:
        if self.base == "newtonian":
            return "newtonian", dict(
                E=self.E, nu=self.nu, density=self.density, bulk_modulus=self.bulk_modulus,
                plastic_viscosity=self.viscosity, yield_stress=self.yield_stress,
                hardening=self.powerlaw_K, softening=self.powerlaw_n,
            )
        if self.base == "granular":
            params = dict(
                E=self.E, nu=self.nu, density=self.density,
                mu_s=self.mu_s, delta_mu=self.delta_mu, I0=self.I0,
                grain_diameter=self.grain_diameter, grain_density=self.grain_density,
            )
            if self.dilatant:
                params.update(phi_init=self.phi_init, phi_chi=self.phi_chi)
                return "mu_i_phi", params
            return "mu_i_sand", params
        if self.base == "elastic":
            return "jelly", dict(E=self.E, nu=self.nu, density=self.density)
        if self.base == "vonmises":
            # engine "metal" (id 1): Hencky elastic predictor + von-Mises (J2) radial
            # return. Identified by (G via E,nu) and yield_stress, with optional xi.
            # The kernel gates the yield update on the hardening flag, so xi alone
            # would be inert; the flag follows xi here.
            return "metal", dict(E=self.E, nu=self.nu, density=self.density,
                                 yield_stress=self.yield_stress, xi=self.hardening_xi,
                                 hardening=1.0 if self.hardening_xi != 0.0 else 0.0)
        if self.base == "tabulated":
            if self.eta_table is None:
                raise ValueError("tabulated material needs eta_table")
            table = [float(v) for v in self.eta_table]
            if len(table) < 2:
                raise ValueError("eta_table needs at least 2 samples")
            if not all(math.isfinite(v) for v in table):
                raise ValueError("eta_table contains non-finite values")
            if any(v < 0.0 for v in table):
                raise ValueError("eta_table viscosities must be nonnegative")
            if not (math.isfinite(self.eta_smin) and math.isfinite(self.eta_smax)
                    and self.eta_smax > self.eta_smin):
                raise ValueError("tabulated material needs smax > smin (finite)")
            return "tabulated_viscous", dict(
                E=self.E, nu=self.nu, density=self.density, bulk_modulus=self.bulk_modulus,
                eta_table=table, eta_table_smin=self.eta_smin,
                eta_table_smax=self.eta_smax,
            )
        raise ValueError(f"unknown material base {self.base!r}")


def newtonian(eta: float = 0.0, density: float = 1000.0, bulk_modulus: float = 9.0e5,
              E: float = 1.0e5, nu: float = 0.3) -> Material:
    """Weakly-compressible generalized-Newtonian fluid (EOS + 2 eta dev D). Add .with_yield
    for Bingham, .with_powerlaw for Herschel-Bulkley."""
    return Material(base="newtonian", viscosity=eta, density=density,
                    bulk_modulus=bulk_modulus, E=E, nu=nu)


def granular(mu_s: float, delta_mu: float = 0.0, I0: float = 0.3, density: float = 1590.0,
             grain_diameter: float = 1.0e-3, grain_density: float = 2650.0,
             E: float = 1.0e6, nu: float = 0.3) -> Material:
    """Local mu(I) granular rheology. delta_mu=0 -> constant friction. Add .with_dilatancy
    for the compressible mu(I)-Phi(I) variant."""
    return Material(base="granular", mu_s=mu_s, delta_mu=delta_mu, I0=I0, density=density,
                    grain_diameter=grain_diameter, grain_density=grain_density, E=E, nu=nu)


def elastic(E: float = 1.0e5, nu: float = 0.3, density: float = 1000.0) -> Material:
    """Hyperelastic solid."""
    return Material(base="elastic", E=E, nu=nu, density=density)


def vonmises(E: float = 5.0e5, nu: float = 0.3, yield_stress: float = 2.0e3,
             hardening_xi: float = 0.0, density: float = 1000.0) -> Material:
    """von-Mises (J2) elasto-plastic solid: the elastic shear modulus G=E/2(1+nu) sets the
    pre-yield stiffness, yield_stress sets the plastic flow threshold, xi the isotropic
    hardening. It holds a shape after the tool releases (true plasticity), the dough/plasticine
    analog used by RoboCraft/RoboCook. Identified by the convex weak-form solve as (G, yield)."""
    return Material(base="vonmises", E=E, nu=nu, yield_stress=yield_stress,
                    hardening_xi=hardening_xi, density=density)


def tabulated_viscous(eta_table, smin: float = -1.0, smax: float = 2.0,
                      density: float = 1000.0, bulk_modulus: float = 9.0e5) -> Material:
    """Weakly-compressible fluid whose apparent viscosity eta_app(gd) is read from a table
    on a uniform log10(gd) grid in [smin, smax] (clamped linear interp). Lets an FE-recovered
    eta_app curve be re-simulated directly, with no parametric fit."""
    return Material(base="tabulated", eta_table=eta_table, eta_smin=smin, eta_smax=smax,
                    density=density, bulk_modulus=bulk_modulus)


__all__ = ["Material", "elastic", "granular", "newtonian", "tabulated_viscous", "vonmises"]

"""Buffered, off-hot-path dump writer producing schema-valid npz files.

Lives under sim/ (imports warp/torch indirectly via the solver) and writes
the format validated by ident/io/schema.py. Frames are snapshotted into
host-side numpy buffers during the run and the single npz is written at
finalize, so nothing touches disk on the simulation hot path.

Stress conversion: the solver stores Kirchhoff stress tau = J sigma in
particle_stress; the dump carries the full 3D CAUCHY stress sigma = tau / J,
whose trace gives the pressure the constitutive update consumed (verified in
sim/verify_mu_i.py). Current particle volume is J * V_p^0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ident.io.schema import SCHEMA_VERSION
from common.conventions import (
    LOG10_I_TABLE_MIN,
    LOG10_I_TABLE_MAX,
    MU_TABLE_POINTS,
    EPS_GAMMA_DEFAULT,
)

# Config A axis tags inside the 3D arrays: x=0 horizontal, z=2 vertical up,
# y=1 out of plane.
IN_PLANE_AXES = (0, 2)
OUT_OF_PLANE_AXIS = 1
COORDINATE_CONVENTION = "configA_x0_z2_yout1"


class DumpWriter:
    def __init__(
        self,
        solver,
        *,
        grain_diameter: float,
        rho_s: float,
        rho_bulk: float,
        packing_fraction: float,
        gravity_inplane: tuple[float, float],
        law: str,
        law_params: dict,
        theta_true: np.ndarray | None,
        l_convention: str = "pending",
        extra: dict | None = None,
        store_F: bool = False,
        compress: bool = False,
        xonly: bool = False,
    ):
        self.compress = bool(compress)
        # scoring (nclaw_position_mse) and rendering (blender_scene.py) only
        # ever read "x" back from a rollout's own dump; v/L/stress/F/volume/
        # active/mass and the mu-table/flowing-I diagnostics exist for
        # identification input (truth/tier dumps), never for a comparison
        # rollout's prediction. xonly skips exporting and storing all of that
        # for a large disk and write-time saving on predictions.
        self.xonly = bool(xonly)
        self.solver = solver
        # the elastic and elastoplastic identifications need the deformation
        # gradient and the REFERENCE particle volume, neither of which the
        # granular schema carries; they ride along as extra arrays ("F",
        # "volume0") that validate_dump_schema ignores
        self.store_F = bool(store_F)
        self._F: list[np.ndarray] = []
        self._vol0: np.ndarray | None = None
        self.globals = dict(
            grain_diameter=float(grain_diameter),
            rho_s=float(rho_s),
            rho_bulk=float(rho_bulk),
            packing_fraction=float(packing_fraction),
            gravity_inplane=np.asarray(gravity_inplane, dtype=float),
            law=law,
            law_params=law_params,
            theta_true=None if theta_true is None else np.asarray(theta_true, float),
            l_convention=l_convention,
            extra=extra or {},
        )
        self._times: list[float] = []
        self._x: list[np.ndarray] = []
        self._v: list[np.ndarray] = []
        self._L: list[np.ndarray] = []
        self._stress: list[np.ndarray] = []  # Cauchy, (P, 9)
        self._vol: list[np.ndarray] = []     # current volume (P,)
        self._active: list[np.ndarray] = []
        self._mass: np.ndarray | None = None

    def snapshot(self, t: float) -> bool:
        """Capture one frame. Returns False if NaN is detected (caller aborts)."""
        x = self.solver.export_particle_x_to_torch().cpu().numpy()
        if not np.isfinite(x).all():
            return False
        if self.xonly:
            self._times.append(float(t))
            self._x.append(x.astype(np.float32))
            return True
        v = self.solver.export_particle_v_to_torch().cpu().numpy()
        L = self.solver.export_particle_L_to_torch().cpu().numpy()      # (P, 9)
        tau = self.solver.export_particle_stress_to_torch().cpu().numpy()  # Kirchhoff (P, 9)
        F = self.solver.export_particle_F_to_torch().cpu().numpy().reshape(-1, 3, 3)
        sel = self.solver.export_particle_selection_to_torch().cpu().numpy()

        J = np.linalg.det(F)
        J = np.where(np.abs(J) < 1e-12, 1.0, J)
        sigma = tau / J[:, None]              # Cauchy = Kirchhoff / J
        vol = J * self.solver.export_particle_vol_to_torch().cpu().numpy()

        if self._mass is None:
            self._mass = (
                self.solver.mpm_state.particle_mass.numpy().astype(np.float32).copy()
            )
        if self.store_F:
            self._F.append(F.astype(np.float32).reshape(-1, 9).copy())
            if self._vol0 is None:
                self._vol0 = (
                    self.solver.export_particle_vol_to_torch()
                    .cpu().numpy().astype(np.float32).copy()
                )

        self._times.append(float(t))
        self._x.append(x.astype(np.float32))
        self._v.append(v.astype(np.float32))
        self._L.append(L.astype(np.float32))
        self._stress.append(sigma.astype(np.float32))
        self._vol.append(vol.astype(np.float32))
        self._active.append((sel == 0))
        return True

    def _mu_table(self) -> tuple[np.ndarray, np.ndarray]:
        log10I = np.linspace(LOG10_I_TABLE_MIN, LOG10_I_TABLE_MAX, MU_TABLE_POINTS)
        I = 10.0**log10I
        lp = self.globals["law_params"]
        if self.globals["law"] == "constant":
            mu = np.full_like(I, lp["mu_s"])
        elif self.globals["law"] == "pouliquen":
            mu = lp["mu_s"] + lp["delta_mu"] * I / (I + lp["I0"])
        else:
            # non-granular law: mu(I) is not defined, and the schema requires the
            # table to be present and the same length as its abscissae
            mu = np.zeros_like(I)
        return log10I, mu

    def _flowing_I_hist(self, x, v, L, stress, vol, active):
        """Realized flowing-I histogram from the in-plane fields, for the report."""
        from common.conventions import (
            equivalent_shear_rate,
            inertial_number,
            pressure_from_cauchy_3d_trace,
            sym,
        )

        ax, az = IN_PLANE_AXES
        Is = []
        d, rho_s = self.globals["grain_diameter"], self.globals["rho_s"]
        for fi in range(x.shape[0]):
            m = active[fi]
            if not np.any(m):
                continue
            Lf = L[fi, m].reshape(-1, 3, 3)
            Lip = Lf[:, [ax, az]][:, :, [ax, az]]  # in-plane 2x2 block
            D = sym(Lip)
            gd = equivalent_shear_rate(D, EPS_GAMMA_DEFAULT)
            p = pressure_from_cauchy_3d_trace(stress[fi, m].reshape(-1, 3, 3))
            I = inertial_number(gd, p, d, rho_s)
            good = np.isfinite(I) & (gd > 0.5) & (I > 1e-4)
            Is.append(I[good])
        if not Is or sum(len(a) for a in Is) == 0:
            edges = np.logspace(LOG10_I_TABLE_MIN, LOG10_I_TABLE_MAX, 41)
            return edges, np.zeros(40)
        allI = np.concatenate(Is)
        edges = np.logspace(LOG10_I_TABLE_MIN, LOG10_I_TABLE_MAX, 41)
        counts, _ = np.histogram(allI, bins=edges)
        return edges, counts.astype(float)

    def finalize(self, path: str | Path, frame_dt: float) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        times = np.asarray(self._times, dtype=float)
        x = np.stack(self._x)
        if not self.xonly:
            v = np.stack(self._v)
            L = np.stack(self._L)
            stress = np.stack(self._stress)
            vol = np.stack(self._vol)
            active = np.stack(self._active)
            log10I, mu = self._mu_table()
            edges, counts = self._flowing_I_hist(x, v, L, stress, vol, active)

        units = {
            "x": "m", "v": "m/s", "L": "1/s", "stress": "Pa",
            "volume": "m^3", "mass": "kg", "times": "s",
            "grain_diameter": "m", "rho_s": "kg/m^3", "rho_bulk": "kg/m^3",
        }
        meta_json = json.dumps(
            {
                "law_params": self.globals["law_params"],
                "units": units,
                **self.globals["extra"],
            },
            default=float,
        )

        arrays = dict(
            schema_version=np.array(SCHEMA_VERSION),
            coordinate_convention=np.array(COORDINATE_CONVENTION),
            in_plane_axes=np.asarray(IN_PLANE_AXES, dtype=int),
            out_of_plane_axis=np.array(OUT_OF_PLANE_AXIS, dtype=int),
            L_convention=np.array(self.globals["l_convention"]),
            frame_dt=np.array(float(frame_dt)),
            grain_diameter=np.array(self.globals["grain_diameter"]),
            rho_s=np.array(self.globals["rho_s"]),
            rho_bulk=np.array(self.globals["rho_bulk"]),
            packing_fraction=np.array(self.globals["packing_fraction"]),
            gravity_inplane=self.globals["gravity_inplane"],
            pressure_source=np.array("true_mpm_trace"),
            law=np.array(self.globals["law"]),
            meta_json=np.array(meta_json),
            times=times,
            x=x,
        )
        if not self.xonly:
            arrays.update(
                mu_table_log10I=log10I,
                mu_table_mu=mu,
                flowing_I_hist_edges=edges,
                flowing_I_hist_counts=counts,
                v=v,
                L=L,
                stress=stress,
                volume=vol,
                mass=self._mass,
                active=active,
            )
        if self.globals["theta_true"] is not None:
            arrays["theta_true"] = self.globals["theta_true"]
        if self.store_F and self._F and not self.xonly:
            arrays["F"] = np.stack(self._F)
            arrays["volume0"] = self._vol0

        if self.compress:
            np.savez_compressed(path, **arrays)
        else:
            # zlib dominated the write (267 of 278 s on a bunny dump, Vista);
            # uncompressed is ~10 percent bigger and 24x faster to write
            np.savez(path, **arrays)
        return path

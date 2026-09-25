"""Dump schema: the contract between sim and ident.

One file per run (npz here; the validator also accepts HDF5 via h5py if a
.h5/.hdf5 path is given). schema_version is REQUIRED. Gate scripts never read
raw npz/HDF5 keys; everything goes through validate_dump_schema and load_dump.

This module is under ident/ and therefore must not import warp, torch, jax,
taichi, or sim. It is pure numpy plus an optional h5py read path.

Layout (Config A, quasi-2D plane strain; full 3D fields are dumped and the
in-plane axes are tagged):

  globals (scalars / small arrays + a JSON metadata blob under "meta_json"):
    schema_version           str
    coordinate_convention    str   e.g. "configA_x0_z2_yout1"
    in_plane_axes            int[2] indices of (x, z) in the 3D arrays
    out_of_plane_axis        int
    L_convention             str   acceleration relation, set by the probe
    frame_dt                 float seconds between dumped frames (1e-3)
    grain_diameter           float d, metres
    rho_s                    float grain density
    rho_bulk                 float bulk density
    packing_fraction         float
    gravity_inplane          float[2]  (g_x, g_z)
    pressure_source          str   "true_mpm_trace"
    law                      str   "constant" | "pouliquen"
    law_params               JSON in meta_json: mu_s, delta_mu, I0, ...
    theta_true               float[K] (law-dependent; may be absent for real data)
    mu_table_log10I          float[256] ground-truth mu(I) sample abscissae
    mu_table_mu              float[256] ground-truth mu(I) values
    flowing_I_hist_edges     float[nb+1]
    flowing_I_hist_counts    float[nb]
    units                    JSON in meta_json (SI tags per field)

  per-frame, per-particle arrays (F frames, P particles):
    times    float[F]
    x        float[F, P, 3]   positions
    v        float[F, P, 3]   velocities
    L        float[F, P, 9]   velocity gradient, row-major 3x3, L_ij=dv_i/dx_j
    stress   float[F, P, 9]   full 3D Cauchy stress, row-major (incl sigma_yy)
    volume   float[F, P]      current particle volume (J * V_p^0)
    mass     float[P]         particle mass (constant)
    active   bool[F, P]       selection == 0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "trackeuclid-dump-1.0"

# Constitutive law tag carried by a dump. The granular pair came first and is
# what every mu(I) gate reads; the rest were added for the NCLaw-matched
# comparison, which dumps the same fields for three non-granular materials. Gate
# code branches on "constant" and "pouliquen" and ignores the others, so adding
# a name changes no existing behaviour.
KNOWN_LAWS = (
    "constant",         # constant-friction mu(I)
    "pouliquen",        # three-parameter mu(I)
    "corotated",        # fixed corotated hyperelasticity (jelly)
    "vonmises",         # corotated elasticity plus a von Mises return map
    "drucker_prager",   # corotated elasticity plus a Drucker-Prager return map
    "eos_fluid",        # weakly compressible pressure-only fluid
    "tabulated_mu_i",   # mu(I) read from a table: a recovered curve, re-simulated
    "tabulated_viscous",  # eta_app(gamma_dot) read from a table, likewise
)

REQUIRED_ARRAYS = {
    "times": 1,
    "x": 3,
    "v": 3,
    "L": 3,
    "stress": 3,
    "volume": 2,
    "mass": 1,
    "active": 2,
}

REQUIRED_GLOBALS = [
    "schema_version",
    "coordinate_convention",
    "in_plane_axes",
    "out_of_plane_axis",
    "L_convention",
    "frame_dt",
    "grain_diameter",
    "rho_s",
    "rho_bulk",
    "packing_fraction",
    "gravity_inplane",
    "pressure_source",
    "law",
    "mu_table_log10I",
    "mu_table_mu",
]


class DumpSchemaError(ValueError):
    """Raised when a dump fails validation; message names the exact failure."""


@dataclass
class DumpMetadata:
    schema_version: str
    coordinate_convention: str
    in_plane_axes: tuple[int, int]
    out_of_plane_axis: int
    L_convention: str
    frame_dt: float
    grain_diameter: float
    rho_s: float
    rho_bulk: float
    packing_fraction: float
    gravity_inplane: np.ndarray
    pressure_source: str
    law: str
    law_params: dict[str, Any]
    n_frames: int
    n_particles: int
    is_3d: bool
    has_pressure: bool
    theta_true: np.ndarray | None
    units: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dump:
    meta: DumpMetadata
    times: np.ndarray      # (F,)
    x: np.ndarray          # (F, P, 3)
    v: np.ndarray          # (F, P, 3)
    L: np.ndarray          # (F, P, 3, 3)
    stress: np.ndarray     # (F, P, 3, 3)
    volume: np.ndarray     # (F, P)
    mass: np.ndarray       # (P,)
    active: np.ndarray     # (F, P) bool
    mu_table_log10I: np.ndarray
    mu_table_mu: np.ndarray


def _load_raw(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in (".npz",):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    if suffix in (".h5", ".hdf5"):
        import h5py  # local import; not a hard dependency of ident/

        out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = f[k][()]
        return out
    raise DumpSchemaError(f"unsupported dump extension: {path.suffix}")


def _as_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return str(v.reshape(-1)[0]) if v.shape else str(v.item())
    return str(v)


def _as_float(v: Any) -> float:
    return float(np.asarray(v).reshape(-1)[0])


def validate_dump_schema(path: str | Path) -> DumpMetadata:
    """Validate a dump file and return its metadata, or raise DumpSchemaError."""
    path = Path(path)
    if not path.exists():
        raise DumpSchemaError(f"dump not found: {path}")
    raw = _load_raw(path)

    # schema version first
    if "schema_version" not in raw:
        raise DumpSchemaError("missing required global 'schema_version'")
    version = _as_str(raw["schema_version"])
    if version != SCHEMA_VERSION:
        raise DumpSchemaError(
            f"schema_version {version!r} != supported {SCHEMA_VERSION!r}"
        )

    meta_blob: dict[str, Any] = {}
    if "meta_json" in raw:
        meta_blob = json.loads(_as_str(raw["meta_json"]))

    # required globals present
    for g in REQUIRED_GLOBALS:
        if g not in raw:
            raise DumpSchemaError(f"missing required global {g!r}")

    # required arrays present with correct ndim
    for name, ndim in REQUIRED_ARRAYS.items():
        if name not in raw:
            raise DumpSchemaError(f"missing required array {name!r}")
        if raw[name].ndim != ndim:
            raise DumpSchemaError(
                f"array {name!r} has ndim {raw[name].ndim}, expected {ndim}"
            )

    times = raw["times"]
    x = raw["x"]
    F, P = x.shape[0], x.shape[1]

    # frame and particle count consistency
    if times.shape[0] != F:
        raise DumpSchemaError(f"times length {times.shape[0]} != n_frames {F}")
    if np.any(np.diff(times) <= 0):
        raise DumpSchemaError("times must be strictly increasing")
    for name, expect in {
        "v": (F, P, 3),
        "L": (F, P, 9),
        "stress": (F, P, 9),
        "volume": (F, P),
        "active": (F, P),
    }.items():
        if raw[name].shape != expect:
            raise DumpSchemaError(
                f"array {name!r} shape {raw[name].shape} != expected {expect}"
            )
    if x.shape[2] != 3:
        raise DumpSchemaError(f"x last dim {x.shape[2]} != 3 (full 3D required)")
    if raw["mass"].shape != (P,):
        raise DumpSchemaError(f"mass shape {raw['mass'].shape} != ({P},)")

    # stress must be full 3D (9 components) so sigma_yy is present
    if raw["stress"].shape[2] != 9:
        raise DumpSchemaError(
            "stress must carry 9 components (full 3D Cauchy incl sigma_yy); "
            "the 2D trace is forbidden by convention"
        )

    # pressure availability: derived from the 3D stress trace
    pressure_source = _as_str(raw["pressure_source"])
    has_pressure = pressure_source in ("true_mpm_trace",)

    in_plane = tuple(int(i) for i in np.asarray(raw["in_plane_axes"]).reshape(-1)[:2])
    out_axis = int(np.asarray(raw["out_of_plane_axis"]).reshape(-1)[0])
    if len(in_plane) != 2 or out_axis in in_plane or out_axis not in (0, 1, 2):
        raise DumpSchemaError(
            f"bad axis tags: in_plane={in_plane} out_of_plane={out_axis}"
        )

    g_inplane = np.asarray(raw["gravity_inplane"], dtype=float).reshape(-1)
    if g_inplane.shape[0] != 2:
        raise DumpSchemaError("gravity_inplane must have 2 components (g_x, g_z)")

    mu_log = np.asarray(raw["mu_table_log10I"], dtype=float).reshape(-1)
    mu_val = np.asarray(raw["mu_table_mu"], dtype=float).reshape(-1)
    if mu_log.shape != mu_val.shape:
        raise DumpSchemaError("mu_table_log10I and mu_table_mu shapes differ")

    law = _as_str(raw["law"])
    if law not in KNOWN_LAWS:
        raise DumpSchemaError(f"unknown law {law!r}")
    law_params = meta_blob.get("law_params", {})

    theta_true = None
    if "theta_true" in raw:
        theta_true = np.asarray(raw["theta_true"], dtype=float).reshape(-1)

    units = meta_blob.get("units", {})
    extra = {
        k: v for k, v in meta_blob.items() if k not in ("law_params", "units")
    }

    return DumpMetadata(
        schema_version=version,
        coordinate_convention=_as_str(raw["coordinate_convention"]),
        in_plane_axes=in_plane,
        out_of_plane_axis=out_axis,
        L_convention=_as_str(raw["L_convention"]),
        frame_dt=_as_float(raw["frame_dt"]),
        grain_diameter=_as_float(raw["grain_diameter"]),
        rho_s=_as_float(raw["rho_s"]),
        rho_bulk=_as_float(raw["rho_bulk"]),
        packing_fraction=_as_float(raw["packing_fraction"]),
        gravity_inplane=g_inplane,
        pressure_source=pressure_source,
        law=law,
        law_params=law_params,
        n_frames=F,
        n_particles=P,
        is_3d=True,
        has_pressure=has_pressure,
        theta_true=theta_true,
        units=units,
        extra=extra,
    )


def load_dump(path: str | Path) -> Dump:
    """Validate and load a dump into a Dump dataclass with 3D arrays reshaped."""
    meta = validate_dump_schema(path)
    raw = _load_raw(Path(path))
    F, P = meta.n_frames, meta.n_particles
    return Dump(
        meta=meta,
        times=np.asarray(raw["times"], dtype=float),
        x=np.asarray(raw["x"], dtype=float),
        v=np.asarray(raw["v"], dtype=float),
        L=np.asarray(raw["L"], dtype=float).reshape(F, P, 3, 3),
        stress=np.asarray(raw["stress"], dtype=float).reshape(F, P, 3, 3),
        volume=np.asarray(raw["volume"], dtype=float),
        mass=np.asarray(raw["mass"], dtype=float),
        active=np.asarray(raw["active"]).astype(bool),
        mu_table_log10I=np.asarray(raw["mu_table_log10I"], dtype=float).reshape(-1),
        mu_table_mu=np.asarray(raw["mu_table_mu"], dtype=float).reshape(-1),
    )

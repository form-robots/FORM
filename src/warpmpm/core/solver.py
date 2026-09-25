"""Typed wrapper around the warp-mpm fork.

Centralizes device handling, Warp init, and the common load/material/collider/step/export
calls, so scenes, tests, and the coupling backend never touch the raw fork or sys.path.
``device="auto"`` selects ``cuda:0`` when CUDA is available and ``cpu`` otherwise.
Pass an explicit device string to override it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from warpmpm.kernels import MPM_Simulator_WARP

_INITED = False
_DEVICE_ANNOUNCED = False   # print the auto-resolved device once per process


def _ensure_warp() -> None:
    global _INITED
    if not _INITED:
        wp.config.quiet = True
        wp.init()
        _INITED = True


# upper-triangular covariance packing used by the kernels: xx, xy, xz, yy, yz, zz
_COV6_IDX = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _cov_to_six(cov: np.ndarray) -> np.ndarray:
    """Normalize a covariance array to (N, 6) float32 in upper-triangular order
    (xx, xy, xz, yy, yz, zz). Accepts (N, 6) already packed or (N, 3, 3) symmetric."""
    cov = np.asarray(cov)
    if cov.ndim == 2 and cov.shape[1] == 6:
        return np.ascontiguousarray(cov, dtype=np.float32)
    if cov.ndim == 3 and cov.shape[1:] == (3, 3):
        out = np.stack([cov[:, i, j] for (i, j) in _COV6_IDX], axis=1)
        return np.ascontiguousarray(out, dtype=np.float32)
    raise ValueError(f"cov must be (N, 6) or (N, 3, 3), got shape {cov.shape}")


@dataclass
class GridConfig:
    n_grid: int = 64
    grid_lim: float = 0.4  # cubic domain edge, metres

    @property
    def dx(self) -> float:
        return self.grid_lim / self.n_grid


@dataclass
class Solver:
    """Wrapper for one MPM_Simulator_WARP instance. Device selection occurs at load time."""

    grid: GridConfig = field(default_factory=GridConfig)
    device: str = "auto"
    # Control-tick interval for the particle-box update and grid-edge guard. Both read x and v
    # from the device. The readback is inexpensive on CPU but synchronizes a CUDA pipeline.
    # A larger interval reduces those synchronizations but delays the edge check equally.
    guard_interval: int = 1
    # Run grid sweeps over active 4^3 blocks, rebuilt each tick, rather than a dense grid or
    # bounding box. This is intended for separated bodies, spread fluid, and large empty
    # domains. Storage remains dense, and sparse mode takes precedence over CUDA graphs.
    sparse: bool = False
    # Claymore-style fused particle pass; see docs/performance.md. Interior substeps use one
    # g2p+stress+p2g kernel instead of three particle passes. Rigid bodies, particle modifiers,
    # and sparse mode fall back to the split path for that tick. Set fused=False to use the
    # split path globally and enable CUDA graph capture.
    fused: bool = True
    # claymore-style block sort (5a): every `sort_interval` ticks, reorder particles by
    # their 4^3 grid block so P2G atomics from neighboring threads hit neighboring
    # nodes and G2P gathers coalesce (the locality AoSoA buys, in SoA layout). 0 = off.
    # Sorting changes particle index identity. Keep this at 0 when dumps pair frames by
    # particle index, including trajectory-based identification.
    sort_interval: int = 0
    # Per-phase profiling records zero/stress/p2g/grid-update/BC/g2p timings. It synchronizes
    # around each phase and forces live launches, making the run slower. Read the accumulated
    # timings with profile_report().
    profile: bool = False
    # Exported volumes and Cauchy stresses are not physical for det(F) <= 0. "warn"
    # preserves the legacy |det(F)| behavior while making the event visible;
    # "raise" rejects it and "nan" marks invalid particles in fixed-shape exports.
    inversion_policy: str = "warn"
    # Periodic boundary along x: transfers wrap the x node index and advection wraps
    # particle x into [0, Lx). For steady chute flows (tilted gravity + periodic
    # streamwise direction). Incompatible with CDF colliders and rigid bodies.
    periodic_x: bool = False
    _sim: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)
    _vol0: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.inversion_policy not in {"warn", "raise", "nan"}:
            raise ValueError("inversion_policy must be 'warn', 'raise', or 'nan'")

    def load_particles(self, pos: np.ndarray, vol: np.ndarray, cov: np.ndarray | None = None,
                       cov_mode: str = "step") -> Solver:
        """Load particle positions and volumes, optionally with per-particle covariances
        for Gaussian-splat coupling.

        cov: (N, 6) upper-triangular (xx, xy, xz, yy, yz, zz) or (N, 3, 3), in sim-space
        units (the caller applies any world->sim scaling first). Normalized to (N, 6)
        float32. cov=None reproduces the covariance-free behavior.

        cov_mode selects how the covariance evolves:
          "step":   the covariance advects each substep as
                    Sigma_{n+1} = Sigma_n + dt (L Sigma_n + Sigma_n L^T). The fused and
                    split step pipelines advect it identically (see step()).
          "from_F": the covariance is reconstructed at export time as Sigma' = F Sigma0
                    F^T from the stored rest-frame covariance.
        """
        import torch

        _ensure_warp()
        if self.device == "auto":
            self.device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
            global _DEVICE_ANNOUNCED
            if not _DEVICE_ANNOUNCED:
                _DEVICE_ANNOUNCED = True
                print(f"warpmpm: device auto -> {self.device}")
        self._vol0 = vol.astype(np.float32).copy()
        self._sim = MPM_Simulator_WARP(len(pos), device=self.device)
        tensor_cov = None
        if cov is not None:
            if cov_mode not in ("step", "from_F"):
                raise ValueError(f"cov_mode must be 'step' or 'from_F', got {cov_mode!r}")
            cov6 = _cov_to_six(cov)
            if cov6.shape[0] != len(pos):
                raise ValueError(f"cov has {cov6.shape[0]} rows, expected {len(pos)}")
            tensor_cov = torch.from_numpy(cov6)
        self._sim.load_initial_data_from_torch(
            torch.from_numpy(pos.astype(np.float32)),
            torch.from_numpy(vol.astype(np.float32)),
            tensor_cov=tensor_cov,
            n_grid=self.grid.n_grid,
            grid_lim=self.grid.grid_lim,
            device=self.device,
        )
        if self.periodic_x:
            self._sim.mpm_model.periodic_x = 1
        if cov is not None and cov_mode == "step":
            # Set update_cov_with_F=True after loading, because initialize() resets it.
            # particle_cov is cloned (not aliased) from init_cov so that advection modifies
            # a distinct copy, leaving init_cov untouched.
            self._sim.mpm_model.update_cov_with_F = True
            self._sim.mpm_state.particle_cov = wp.clone(self._sim.mpm_state.particle_init_cov)
        return self

    def set_material(self, material, **overrides: float) -> Solver:
        """Accepts a composable warpmpm.materials.Material (preferred) or a material
        name string. Resolves to the engine's (name, params) and applies it."""
        if hasattr(material, "resolve"):
            name, params = material.resolve()
        else:
            name, params = str(material), {}
        params = {**params, **overrides}
        self._sim.set_parameters_dict(
            {"material": name, "g": [0.0, 0.0, -9.81], **params}, device=self.device
        )
        self._sim.finalize_mu_lam(device=self.device)
        return self

    def set_material_range(self, start: int, end: int, material, **overrides) -> Solver:
        """Assign a material to particles [start, end), overriding the global material.
        Accepts a Material or a material name plus per-range parameters (density
        updates per-particle mass in the range). For material "rigid", pass obj_id to
        group particles into one body, then call finalize_rigid_bodies once all rigid
        ranges are assigned."""
        if hasattr(material, "resolve"):
            name, params = material.resolve()
        else:
            name, params = str(material), {}
        # one global eta table exists on the model and only set_material installs
        # it; the per-range path would silently keep the default zero table
        if name in ("tabulated_viscous", "tabulated_mu_i"):
            raise NotImplementedError(
                "tabulated materials are global-only (one eta table on the model); "
                "use set_material")
        params = {"material": name, **params, **overrides}
        self._sim.set_parameters_for_particles(start, end, params, device=self.device)
        return self

    def finalize_rigid_bodies(self) -> Solver:
        """Compute each rigid body's center of mass, mass, and inertia from the particles
        assigned material "rigid". Call after every set_material_range and before the
        first step. Rigid scenes run the split pipeline (the fused path excludes them)."""
        self._sim.finalize_rigid_bodies(device=self.device)
        return self

    def rigid_state(self, body: int = 0) -> dict:
        """State of one rigid body: com (3,), v (3,), omega (3,), R (3, 3). The
        displacement from the spawn pose is com minus the com recorded at
        finalize_rigid_bodies; rotation from spawn is R itself (bodies start at
        identity orientation)."""
        return {
            "com": self._sim.rigid_x_cm.numpy()[body].copy(),
            "v": self._sim.rigid_v_cm.numpy()[body].copy(),
            "omega": self._sim.rigid_omega.numpy()[body].copy(),
            "R": self._sim.rigid_orientation.numpy()[body].copy(),
        }

    def add_plane(self, point, normal, surface: str = "sticky", friction: float = 0.0,
                  restitution: float = 0.0) -> Solver:
        """Add a plane boundary condition for fluid and deformable particles.

        Grid boundary conditions do not affect rigid particles. A restitution value in
        (0, 1] also registers the plane as a rigid-body contact surface, with an impulse
        applied at the deepest penetrating particle.
        """
        self._sim.add_surface_collider(tuple(point), tuple(normal), surface,
                                       friction=friction, restitution=restitution)
        return self

    def add_box(self, center, half_size, velocity=(0.0, 0.0, 0.0),
                start_time: float = 0.0, end_time: float = 1.0e9) -> int:
        """Add a kinematic, axis-aligned box collider (volumetric grid-node velocity
        overwrite over the box, not surface-SDF contact; for oriented surface
        contact with friction modes use add_sdf_collider).

        The collider imposes its velocity on covered grid nodes. Returns a handle for
        updates through set_box; the coupling layer uses this collider for robot tools.
        """
        self._sim.set_velocity_on_cuboid(
            point=tuple(center), size=tuple(half_size), velocity=tuple(velocity),
            start_time=start_time, end_time=end_time,
        )
        return len(self._sim.collider_params) - 1

    def set_box(self, handle: int, center=None, velocity=None) -> Solver:
        """Update a kinematic box at the start of a control tick.

        The box integrates its velocity on every substep. Over one tick it sweeps
        from center -> center + dt_ctrl*velocity. Drive it with the start-of-tick
        center and the per-tick velocity to ensure it lands exactly on target by
        the end of the step. Passing the end-of-tick target as the center will
        double-apply the motion."""
        p = self._sim.collider_params[handle]
        self._sim._bc_box_cache = {}
        if center is not None:
            p.point = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        if velocity is not None:
            p.velocity = wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        return self

    # --- kinematic glass (revolved-SDF cup collider) ----------------------------------
    def add_cup(self, profile, center, quat=(1.0, 0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                omega=(0.0, 0.0, 0.0), friction: float = 0.05, sticky_cells: float = 1.5,
                contact_cells: float = 0.5, start_time: float = 0.0,
                end_time: float = 1.0e9) -> int:
        """Add a kinematic open-top glass from an analytic revolved SDF.

        ``profile`` is a ``colliders.glass.GlassProfile``, and ``quat`` uses wxyz order.
        Separable Coulomb contact begins ``contact_cells * dx`` outside the surface and
        extends to ``sticky_cells * dx`` inside it. Deeper nodes use a full velocity grab
        as an anti-tunneling backstop. Returns a handle for set_cup and accumulates the
        reaction impulse and torque read by cup_wrench.
        """
        from warpmpm.colliders.glass import quat_to_mat

        # the sticky core must survive inside the wall: cap the friction shell at just
        # under half the wall thickness so coarse grids keep an anti-tunneling backstop
        sticky_depth = min(sticky_cells * self.grid.dx, 0.45 * profile.wall_thickness)
        return self._sim.add_revolved_sdf_collider(
            point=tuple(center), rot=quat_to_mat(quat), velocity=tuple(velocity),
            omega=tuple(omega), outer_radius=profile.outer_radius,
            inner_radius=profile.inner_radius, half_height=profile.half_height,
            inner_floor_z=profile.inner_floor_z, fillet_radius=profile.fillet_radius,
            friction=friction, sticky_depth=sticky_depth,
            contact_band=contact_cells * self.grid.dx,
            start_time=start_time, end_time=end_time,
        )

    def set_cup(self, handle: int, center=None, quat=None, velocity=None, omega=None) -> Solver:
        """Update a cup using set_box's start-of-tick contract.

        ``quat`` and ``omega`` extend that contract to rotation.
        """
        from warpmpm.colliders.glass import quat_to_mat

        rot = None if quat is None else quat_to_mat(quat)
        self._sim.set_revolved_collider_pose(handle, point=center, rot=rot,
                                             velocity=velocity, omega=omega)
        return self

    def reset_cup_wrench(self, handle: int) -> Solver:
        """Zero a cup's reaction accumulators before the substeps to be measured."""
        p = self._sim.collider_params[handle]
        p.force.zero_()
        p.torque.zero_()
        return self

    def cup_wrench(self, handle: int, dt: float) -> dict:
        """Return the cup's accumulated grid-impulse wrench divided by elapsed time ``dt``.

        The result contains ``force[3]`` and ``torque[3]`` about the cup center. A static
        cup holding ``m`` kilograms of settled liquid reads approximately
        ``force = (0, 0, -m*g)``.
        """
        p = self._sim.collider_params[handle]
        return {
            "force": np.asarray(p.force.numpy()[0], dtype=float) / dt,
            "torque": np.asarray(p.torque.numpy()[0], dtype=float) / dt,
        }

    def add_domain_walls(self, start_time: float = 0.0, end_time: float = 1.0e9) -> Solver:
        """Zero outward velocity in a three-cell band at each domain face.

        This prevents splashes from leaving ``[0, grid_lim]^3`` and indexing the P2G grid
        out of bounds.
        """
        self._sim.add_bounding_box(start_time=start_time, end_time=end_time)
        return self

    def add_sdf_collider(self, sdf, center, quat=(0.0, 0.0, 0.0, 1.0),
                         velocity=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), band=None,
                         surface: str = "separable", friction: float = 0.4,
                         start_time: float = 0.0, end_time: float = 1.0e9) -> int:
        """Add a watertight mesh as a moving signed-distance-field collider.

        ``sdf`` is a ``warpmpm.geometry.SDFData`` built from a mesh. Use set_sdf_pose for
        translation and rotation, and sdf_wrench for its reaction wrench. Returns a handle.
        """
        return self._sim.add_sdf_collider(
            sdf.values, sdf.grads, sdf.origin, sdf.cell, center, quat=quat, velocity=velocity,
            omega=omega, band=band, surface=surface, friction=friction,
            start_time=start_time, end_time=end_time, device=self.device,
        )

    def set_sdf_pose(self, handle: int, center=None, quat=None, velocity=None, omega=None
                     ) -> Solver:
        """Update an SDF collider using set_box's start-of-tick contract.

        ``quat`` and ``omega`` extend that contract to rotation.
        """
        self._sim.set_sdf_pose(handle, center=center, quat=quat, velocity=velocity, omega=omega)
        return self

    def reset_sdf_force(self, handle: int) -> Solver:
        """Zero an SDF collider's reaction force and torque before step()."""
        self._sim.collider_params[handle].force.zero_()
        self._sim.collider_params[handle].torque.zero_()
        return self

    def sdf_wrench(self, handle: int, dt: float) -> dict:
        """Reaction wrench the material exerts on an SDF collider, from the grid impulse
        accumulated since the last reset: force = sum m*(v_free - v_new) / dt, torque =
        sum (x - center) x impulse / dt (about the collider centre, world frame). Returns
        {'force': (3,), 'torque': (3,)}. The general 6-DOF analogue of tool_force for the box."""
        f = np.asarray(self._sim.collider_params[handle].force.numpy()[0], dtype=float)
        t = np.asarray(self._sim.collider_params[handle].torque.numpy()[0], dtype=float)
        return {"force": f / dt, "torque": t / dt}

    def add_cdf_collider(self, cdf, center, quat=(0.0, 0.0, 0.0, 1.0),
                         velocity=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), band=None,
                         surface: str = "separable", friction: float = 0.4,
                         start_time: float = 0.0, end_time: float = 1.0e9) -> int:
        """Add an OPEN oriented mid-surface (warpmpm.geometry.CDFData) as a CPIC
        thin-boundary collider. Particle-node transfers are severed across the
        surface, making it watertight at any wall thickness.

        Drive the pose with set_cdf_pose and read the reaction wrench with cdf_wrench.
        Handles are a separate space from the grid-BC colliders."""
        if self.periodic_x:
            raise NotImplementedError("CDF colliders do not wrap the periodic x axis")
        return self._sim.add_cdf_collider(
            cdf.values, cdf.valid, cdf.origin, cdf.cell, cdf.band, center, quat=quat,
            velocity=velocity, omega=omega, band=band, surface=surface,
            friction=friction, start_time=start_time, end_time=end_time,
            device=self.device,
        )

    def set_cdf_pose(self, handle: int, center=None, quat=None, velocity=None,
                     omega=None) -> Solver:
        """Update a CDF collider's pose each control tick (set_sdf_pose's
        start-of-tick contract)."""
        self._sim.set_cdf_pose(handle, center=center, quat=quat, velocity=velocity,
                               omega=omega, device=self.device)
        return self

    def reset_cdf_wrench(self) -> Solver:
        """Zero the CDF reaction accumulators (all lanes; call before step())."""
        self._sim.reset_cdf_wrench()
        return self

    def cdf_wrench(self, handle: int, dt: float) -> dict:
        """Reaction wrench the material exerts on a CDF collider, accumulated from
        the BLOCKED p2g deposits (the momentum flux the sheet interrupts) since the
        last reset. Scales linearly with the supported load but reads a
        geometry-dependent fraction of the absolute value (~1/3 for a node-aligned
        sheet; see docs/performance.md): a calibrated scale, not an FT sensor. For
        calibrated absolute forces use an SDF collider. Returns
        {'force': (3,), 'torque': (3,)}."""
        f = np.asarray(self._sim.mpm_state.cdf_reaction_force.numpy()[handle],
                       dtype=float)
        t = np.asarray(self._sim.mpm_state.cdf_reaction_torque.numpy()[handle],
                       dtype=float)
        return {"force": f / dt, "torque": t / dt}

    def reset_tool_force(self, handle: int) -> Solver:
        """Zero a box collider's reaction accumulator before the measured substeps."""
        self._sim.collider_params[handle].force.zero_()
        return self

    def tool_force(self, handle: int, dt: float) -> np.ndarray:
        """Reaction force the material exerts on a box collider, from the grid impulse
        accumulated since the last reset: F = sum_substeps sum_nodes m*(v_free - v_imposed) /
        dt. dt is the elapsed time accumulated over (e.g. substeps*substep_dt). Returns
        force[3] (compression -> +z). Unlike the stress-integral estimator, this readout
        uses no contact band, T_layer, or stress gate."""
        impulse = self._sim.collider_params[handle].force.numpy()[0]
        return np.asarray(impulse, dtype=float) / dt

    def step(self, dt: float, substeps: int = 1) -> Solver:
        if self._tick % max(1, self.guard_interval) == 0:
            self._update_grid_box(dt, substeps)
        if self.sort_interval and self._tick % self.sort_interval == 0:
            self._sort_particles()
        if self.sparse:
            self._sim.rebuild_active_blocks(self.device)
        self._sim.profile = self.profile
        self._tick += 1
        # Both the split and fused pipelines advect covariance identically because the
        # fused kernel g2p_stress_p2g reuses the g2p_particle wp.func.
        fused_ok = (self.fused and not self.sparse
                    and not self._sim.pre_p2g_operations
                    and not self._sim.particle_velocity_modifiers
                    and self._sim.n_rigid_bodies == 0)
        if fused_ok:
            self._sim.p2g2p_fused_tick(dt, substeps, device=self.device)
            self._step += substeps
        else:
            for _ in range(substeps):
                self._sim.p2g2p(self._step, dt, device=self.device)
                self._step += 1
        return self

    def _sort_particles(self) -> bool:
        """Block-key sort (stable): key = lexicographic 4^3-block index of the stencil
        base. Skips the permutation when already ordered (particles move well under a
        cell per tick, so most sort ticks after the first are no-ops)."""
        x = self.x()
        base = (np.floor(x / self.grid.dx - 0.5).astype(np.int64)) >> 2
        nb = (self.grid.n_grid >> 2) + 2
        keys = (base[:, 0] * nb + base[:, 1]) * nb + base[:, 2]
        if np.all(np.diff(keys) >= 0):
            return False
        perm = np.argsort(keys, kind="stable")
        self._sim.permute_particles(perm, device=self.device)
        self._vol0 = self._vol0[perm]  # host-side pair of the device arrays
        return True

    def profile_report(self) -> str:
        """Aggregate the per-phase timings collected while profile=True into a table
        (total seconds, ms per substep, share of timed device work)."""
        prof = getattr(self._sim, "time_profile", {}) or {}
        rows = [(k, sum(v) / 1000.0, len(v)) for k, v in prof.items() if v]
        if not rows:
            return "profile_report: no samples (set solver.profile = True and step)"
        timed = sum(t for _, t, _ in rows)
        rows.sort(key=lambda r: -r[1])
        lines = [f"substep profile over {self._step} substeps "
                 f"(timed device work {timed:.1f}s; live launches + per-phase sync):"]
        for name, tot, n in rows:
            lines.append(f"  {name:<28s} {tot:7.1f}s  {tot / max(n, 1) * 1000:8.3f} ms/substep"
                         f"  {tot / timed * 100:5.1f}%")
        return "\n".join(lines)

    def _update_grid_box(self, dt: float, substeps: int) -> None:
        """Check the grid edge and update the particle launch box once per control tick.

        Particles near the edge can make the quadratic P2G stencil write out of bounds.
        The live launch box is padded for the current tick's motion and replaces a full
        dense-grid sweep for zeroing, normalization, and damping.
        """
        x = self.x()
        v = self.v()
        dx = self.grid.dx
        lim = self.grid.grid_lim
        # With periodic x, guard y/z only since transfers wrap the x index.
        g = x[:, 1:] if self.periodic_x else x
        if g.min() < 1.5 * dx or g.max() > lim - 2.5 * dx:
            raise RuntimeError(
                f"particles within 2 cells of the grid edge (x in "
                f"[{g.min():.4f}, {g.max():.4f}] m, domain [0, {lim}] m, dx={dx:.4f}): "
                f"the P2G stencil would write out of bounds. Enlarge grid_lim or add a "
                f"bounding box / wall collider.")
        pad = 3.0 * dx + 1.5 * float(np.abs(v).max()) * dt * substeps
        lo = x.min(0) - pad
        hi = x.max(0) + pad
        if self.periodic_x:
            lo[0], hi[0] = 0.0, lim
        self._sim.grid_launch_box = self._sim._grid_box(lo, hi, halo=0)
    # --- imports (numpy, off the hot path; e.g. the leak-projection rescue net) -------
    def set_x(self, pos: np.ndarray) -> Solver:
        import torch

        self._sim.import_particle_x_from_torch(
            torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32)), device=self.device
        )
        return self

    def set_v(self, vel: np.ndarray) -> Solver:
        import torch

        self._sim.import_particle_v_from_torch(
            torch.from_numpy(np.ascontiguousarray(vel, dtype=np.float32)), device=self.device
        )
        return self

    # --- exports (numpy, off the hot path) -------------------------------------------
    def x(self) -> np.ndarray:
        return self._sim.export_particle_x_to_torch().cpu().numpy()

    def v(self) -> np.ndarray:
        return self._sim.export_particle_v_to_torch().cpu().numpy()

    def F(self) -> np.ndarray:
        return self._sim.export_particle_F_to_torch().cpu().numpy().reshape(-1, 3, 3)

    def stress(self) -> np.ndarray:
        return self._sim.export_particle_stress_to_torch().cpu().numpy().reshape(-1, 3, 3)

    def L(self) -> np.ndarray:
        """Per-particle velocity gradient L_ij = dv_i/dx_j from the most recent G2P. Use the
        symmetric part D = sym(L) for the strain rate (and |gamma_dot| = sqrt(2 D:D + eps^2))."""
        return self._sim.export_particle_L_to_torch().cpu().numpy().reshape(-1, 3, 3)

    def inverted_count(self) -> int:
        """Particles with a non-finite or non-positive deformation Jacobian."""
        J = np.linalg.det(self.F())
        return int(np.count_nonzero(~np.isfinite(J) | (J <= 0.0)))

    def _checked_jacobian(self) -> np.ndarray:
        if self.inversion_policy not in {"warn", "raise", "nan"}:
            raise ValueError("inversion_policy must be 'warn', 'raise', or 'nan'")
        J = np.linalg.det(self.F())
        invalid = ~np.isfinite(J) | (J <= 0.0)
        n = int(np.count_nonzero(invalid))
        if not n:
            return J
        message = (
            f"{n} particles have non-finite or non-positive det(F); current volume "
            "and Cauchy stress are undefined for an inverted configuration"
        )
        if self.inversion_policy == "raise":
            raise RuntimeError(message)
        if self.inversion_policy == "warn":
            if not getattr(self, "_warned_inverted", False):
                self._warned_inverted = True
                warnings.warn(
                    f"{message}; using |det(F)| for compatibility. Set "
                    "inversion_policy='raise' to reject or 'nan' to mark these particles.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            compatible = np.abs(J)
            compatible[~np.isfinite(compatible)] = np.nan
            return compatible
        J = J.copy()
        J[invalid] = np.nan
        return J

    def vol(self) -> np.ndarray:
        """Current volume under the configured inversion policy."""
        return self._vol0 * self._checked_jacobian()

    def cauchy(self) -> np.ndarray:
        """Cauchy stress per particle = Kirchhoff / det(F), with explicit inversion policy."""
        J = self._checked_jacobian()
        if self.inversion_policy == "warn":
            J = np.clip(J, 1e-9, None)
        return self.stress() / J[:, None, None]

    def cov(self) -> np.ndarray:
        """Per-particle covariance, shape (N, 6) upper-triangular (xx, xy, xz, yy, yz, zz).
        In cov_mode "step" this is the advected covariance; in "from_F" it is rebuilt as
        F Sigma0 F^T at call time. Only meaningful when load_particles was given cov."""
        return self._sim.export_particle_cov_to_torch(device=self.device).cpu().numpy(
            ).reshape(-1, 6)

    def R(self) -> np.ndarray:
        """Per-particle polar rotation R of F (Sigma' = R Sigma0 R^T holds under rigid
        motion), shape (N, 3, 3). The splat SH view-direction trick applies R^T to the
        camera->splat direction. The kernel stores R^T internally, so this transposes it
        back to the polar rotation."""
        R = self._sim.export_particle_R_to_torch(device=self.device).cpu().numpy(
            ).reshape(-1, 3, 3)
        return np.transpose(R, (0, 2, 1))

    # --- torch-resident exports (tensors on self.device; read-only, per kernel behavior) --
    def x_torch(self):
        """Particle positions as a torch tensor on self.device, shape (N, 3)."""
        return self._sim.export_particle_x_to_torch()

    def v_torch(self):
        """Particle velocities as a torch tensor on self.device, shape (N, 3)."""
        return self._sim.export_particle_v_to_torch()

    def F_torch(self):
        """Deformation gradient as a torch tensor on self.device, shape (N, 3, 3)."""
        return self._sim.export_particle_F_to_torch().reshape(-1, 3, 3)

    def cov_torch(self):
        """Per-particle covariance as a torch tensor on self.device, shape (N, 6),
        upper-triangular (xx, xy, xz, yy, yz, zz). See cov() for the two cov_mode meanings."""
        return self._sim.export_particle_cov_to_torch(device=self.device).reshape(-1, 6)

    def R_torch(self):
        """Per-particle polar rotation R of F as a torch tensor on self.device, shape
        (N, 3, 3). Transposes the kernel's stored R^T; see R() for the convention."""
        R = self._sim.export_particle_R_to_torch(device=self.device).reshape(-1, 3, 3)
        return R.transpose(1, 2)

    @property
    def n_particles(self) -> int:
        return 0 if self._sim is None else self._sim.n_particles

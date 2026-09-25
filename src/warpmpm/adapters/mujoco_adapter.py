"""Kinematic MuJoCo Panda adapter with pose, velocity, and contact-wrench access."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np


def _default_mujoco_gl() -> None:
    """Use EGL for offscreen MuJoCo rendering on headless Linux."""
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class FrankaArm:
    """Franka Panda in MuJoCo. Scripted vertical descent."""

    # raised / lowered arm configs (7 arm joints); fingers held closed-ish
    Q_UP = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.6, 0.79])
    Q_DOWN = np.array([0.0, 0.35, 0.0, -2.4, 0.0, 2.75, 0.79])

    def __init__(self, height: int = 480, width: int = 640, ft_sensor: bool = False,
                 hide_gripper: bool = False, base_pos=None, max_geom: int = 10000,
                 sphere_detail: tuple[int, int] | None = None):
        _default_mujoco_gl()
        import mujoco
        bundled = Path(__file__).resolve().parents[3] / "assets/panda/panda.xml"
        if bundled.is_file():
            panda_xml = str(bundled)
        else:
            os.environ.setdefault("ROBOT_DESCRIPTION_COMMIT", "feadf76d42f8a2162426f7d226a3b539556b3bf5")
            from robot_descriptions import panda_mj_description
            panda_xml = panda_mj_description.MJCF_PATH

        self.mj = mujoco
        # optionally compile with a wrist force/torque sensor (a load cell at the hand)
        self._ft = None
        if ft_sensor or base_pos is not None:
            spec = mujoco.MjSpec.from_file(panda_xml)
            if base_pos is not None:
                # reposition the arm base (link0) so MuJoCo world == the scene's world
                root = next((b for b in spec.bodies if b.name == "link0"), None)
                if root is None:
                    root = spec.bodies[1]  # first body under the world
                root.pos = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
            hand = next((b for b in spec.bodies if b.name == "hand"), None)
            if ft_sensor and hand is not None:
                hand.add_site(name="wrist_ft", pos=[0.0, 0.0, 0.0])
                for nm, ty in (("wrist_force", mujoco.mjtSensor.mjSENS_FORCE),
                               ("wrist_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
                    sn = spec.add_sensor(); sn.name = nm; sn.type = ty
                    sn.objtype = mujoco.mjtObj.mjOBJ_SITE; sn.objname = "wrist_ft"
            self._customize_spec(spec, mujoco)
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(panda_xml)
        self.data = mujoco.MjData(self.model)
        # end-effector body = the hand (fall back to last body)
        try:
            self.ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        except Exception:
            self.ee = self.model.nbody - 1
        if self.ee < 0:
            self.ee = self.model.nbody - 1
        if ft_sensor and self.model.nsensor:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force")
            self._ft = int(self.model.sensor_adr[fid])
        if hide_gripper:
            # the squeeze tool is a flat PLATE (the box collider), not the Panda's jaws;
            # make the gripper fingers + hand flange invisible so the render matches the
            # physics (arm + mounted plate). Collision is unaffected (rgba is render-only).
            for gid in range(self.model.ngeom):
                bid = int(self.model.geom_bodyid[gid])
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if "finger" in bname or bname == "hand":
                    self.model.geom_rgba[gid, 3] = 0.0
        # the default offscreen framebuffer is 640x480; enlarge it so any requested size renders
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        if sphere_detail is not None:
            # tessellation of PROCEDURAL geoms (the particle spheres; robot meshes are
            # unaffected). The default 28x16 is ~500 triangles per sphere, far more than
            # particles a few pixels wide need; (8, 6) halves the GL frame time at 3e5 geoms
            # with no visible change at particle scale. Must be set before the Renderer
            # is built (the GL context bakes the tessellation).
            self.model.vis.quality.numslices = int(sphere_detail[0])
            self.model.vis.quality.numstacks = int(sphere_detail[1])
        self.renderer = mujoco.Renderer(self.model, height=height, width=width,
                                        max_geom=max_geom)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 1.8
        self.cam.azimuth = 135
        self.cam.elevation = -20
        self._prev_ee = None

    def _customize_spec(self, spec, mujoco) -> None:
        """Hook for subclasses to extend the MjSpec (extra assets/bodies) before
        compile. Only invoked on the MjSpec path (ft_sensor or base_pos set)."""

    def close(self) -> None:
        """Release MuJoCo renderer resources, suppressing backend shutdown noise."""
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            with contextlib.suppress(Exception):
                renderer.close()
            self.renderer = None

    def __del__(self):
        self.close()

    def set_descent(self, frac: float, dt: float, track_camera: bool = True) -> dict:
        """Set the arm to a scripted descent fraction in [0,1]; return EE world pose+vel.
        track_camera follows the EE with the camera (good for the 2-panel view); pass False
        when the caller drives a fixed camera (e.g. the composite single-view)."""
        q = (1.0 - frac) * self.Q_UP + frac * self.Q_DOWN
        self.data.qpos[:7] = q
        self.mj.mj_forward(self.model, self.data)
        ee_pos = self.data.xpos[self.ee].copy()
        ee_vel = (np.zeros(3) if (self._prev_ee is None or dt <= 0)
                  else (ee_pos - self._prev_ee) / dt)
        self._prev_ee = ee_pos
        if track_camera:
            self.cam.lookat[:] = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.2]
        return {"pos": ee_pos, "vel": ee_vel}

    def wrist_load_cell(self, frac: float, f_dough_world, settle: int = 300) -> np.ndarray:
        """Read the wrist force-torque sensor (the load cell) for the reaction the dough
        exerts on the arm. Position actuators hold the arm at descent fraction
        `frac`. The routine settles the arm twice: once with no load (the
        baseline is the gripper's own weight) and once with the dough reaction
        `f_dough_world` applied to the hand. It returns the difference, the
        dough's contribution at the wrist. By Newton's third law this equals the force we
        fed in (= the MPM grid-impulse reaction), now read at the wrist like a real robot.
        Requires ft_sensor=True. Uses forward dynamics, so it mutates the arm state."""
        if self._ft is None:
            raise RuntimeError("FrankaArm was built without ft_sensor=True")
        q = (1.0 - frac) * self.Q_UP + frac * self.Q_DOWN

        def _settle(F):
            self.data.qpos[:7] = q; self.data.qvel[:] = 0.0
            if self.model.nu >= 7:
                self.data.ctrl[:7] = q
            for _ in range(settle):
                self.data.xfrc_applied[self.ee, :3] = F
                self.mj.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.ee, :3] = 0.0
            return self.data.sensordata[self._ft:self._ft + 3].copy()

        base = _settle(np.zeros(3))
        loaded = _settle(np.asarray(f_dough_world, dtype=float))
        return loaded - base

    def render_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()

    def render_with_particles(self, pts_world, rgba, radius=0.004, table=None, boxes=None,
                              cylinders=None):
        """Composite render: the Franka + the MPM material as spheres in ONE camera view.
        pts_world: (M, 3) world-frame particle positions.
        rgba: (M, 4) per-particle colour.
        table: (cx, cy, z, half) draws a flat support box.
        boxes: list of (center3, half3, rgba4), drawn as solid boxes, e.g. a
            plate mounted on the gripper.
        cylinders: list of (center3, mat33 or None, radius, half_height,
            rgba4), drawn as cylinders, optionally transparent; the glasses of
            the pouring scene. Subsample pts to fit
        max_geom."""
        self.renderer.update_scene(self.data, self.cam)
        sc = self.renderer.scene
        eye = np.eye(3).flatten()
        if table is not None:
            cx, cy, z, half = table
            g = sc.geoms[sc.ngeom]
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_BOX,
                                 np.array([half, half, 0.01]), np.array([cx, cy, z - 0.01]),
                                 eye, np.array([0.55, 0.57, 0.6, 1.0], np.float32))
            sc.ngeom += 1
        for box in (boxes or []):
            if sc.ngeom >= sc.maxgeom:
                break
            center, half3, col = box[0], box[1], box[2]
            rot = np.asarray(box[3], np.float64).flatten() if len(box) > 3 else eye
            g = sc.geoms[sc.ngeom]
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_BOX,
                                 np.asarray(half3, np.float64), np.asarray(center, np.float64),
                                 rot, np.asarray(col, np.float32))
            sc.ngeom += 1
        for cyl_center, cyl_mat, cyl_r, cyl_half_h, cyl_col in (cylinders or []):
            if sc.ngeom >= sc.maxgeom:
                break
            g = sc.geoms[sc.ngeom]
            rot = eye if cyl_mat is None else np.asarray(cyl_mat, np.float64).flatten()
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_CYLINDER,
                                 np.array([cyl_r, cyl_r, cyl_half_h], np.float64),
                                 np.asarray(cyl_center, np.float64), rot,
                                 np.asarray(cyl_col, np.float32))
            sc.ngeom += 1
        room = sc.maxgeom - sc.ngeom
        n = len(pts_world)
        stride = max(1, int(np.ceil(n / max(room, 1))))
        idx = np.arange(0, n, stride)
        # hoist everything out of the hot loop: per-particle astype/attribute lookups
        # dominate at 1e5+ geoms (the loop is Python->C once per particle regardless)
        pts64 = np.ascontiguousarray(pts_world[idx], np.float64)
        rgba32 = np.ascontiguousarray(rgba[idx], np.float32)
        size = np.array([radius, 0.0, 0.0])
        init = self.mj.mjv_initGeom
        sphere = self.mj.mjtGeom.mjGEOM_SPHERE
        geoms = sc.geoms
        base = sc.ngeom
        m = min(len(idx), sc.maxgeom - base)
        for k in range(m):
            init(geoms[base + k], sphere, size, pts64[k], eye, rgba32[k])
        sc.ngeom = base + m
        return self.renderer.render()


class PandaPour(FrankaArm):
    """Scripted Franka POUR kinematics + the cup grasp transform. The robot ACTION is
    still the companion Genesis (SPH) study's Panda joint trajectory (FK bit-identical
    between this Menagerie panda and Genesis's panda.xml, verified at the upright /
    80% / full pour configs); the held CUP is the measured 500 mL measuring cup
    (geometry.measuring_cup, the real cup of the hardware pouring experiment), so the
    grasp constants below are the cup's own handle in its spec frame (origin on the
    ellipse axis, z = 0 at the external base, +x toward the spout). The held cup's
    pose is this fixed handle-grasp transform applied to the hand FK; drive the MPM
    collider with cup_pose_at(t).

    Default action: smoothstep joint interpolation upright -> POUR_POSE_FRACTION of the
    full-pour config over TILT_SECONDS, then back over RETURN_SECONDS. The default
    profile's peak pose-fraction rate (1.5x average) is far below the Panda's tightest
    joint-velocity ceiling for this motion (~2.2 fraction/s), so the scripted action is
    physically executable."""

    Q_UPRIGHT = np.array([-1.5916905403137207, -1.2717534303665161, -0.06664533913135529,
                          -2.951836109161377, 1.5030548572540283, 1.537463665008545,
                          2.2464425563812256, 0.026, 0.026])
    Q_FULL_POUR = np.array([-1.6516658067703247, -0.798383355140686, 0.6634261012077332,
                            -2.0772507190704346, 0.5259794592857361, 1.4164435863494873,
                            2.8647570610046387, 0.026, 0.026])
    POUR_POSE_FRACTION = 0.80
    TILT_SECONDS = 3.0
    RETURN_SECONDS = 1.6
    BASE_POS = (-0.15, 0.0, 0.0)
    TCP_LOCAL = np.array([0.0, 0.0, 0.092])        # tool centre point in the hand frame
    # grasp point: mid-height of the measured cup's handle column (column centreline
    # x = -86.3 mm, grip z = 60 mm in the cup spec frame)
    GRASP_LOCAL = np.array([-0.0863, 0.0, 0.060])
    # cup axes in hand axes: the Genesis grasp composed with Rz(-phi*), phi* = -6.65 deg
    # = the downhill azimuth of the tilt at peak (FK-measured; it wanders only -11 to
    # -6.6 deg over the active tilt), so the spout (+x cup) points where the pour goes
    CUP_TO_HAND = np.array([[0.0, -0.11579915, 0.99327265],
                            [0.0, 0.99327265, 0.11579915],
                            [-1.0, 0.0, 0.0]])

    def __init__(self, height: int = 480, width: int = 640, max_geom: int = 30000,
                 glass_mesh=None, glass_rgba=(0.76, 0.92, 1.0, 0.30),
                 sphere_detail: tuple[int, int] | None = None):
        # optional render glasses: the watertight open-top mesh (write_glass_obj) is
        # added as an asset with two mocap bodies ("glass_src", "glass_rcv") so MuJoCo
        # draws the REAL glass geometry (thick base, filleted cavity) with glass-like
        # transparency. Pose them per frame with set_glass_pose.
        self._glass_mesh = None if glass_mesh is None else str(glass_mesh)
        self._glass_rgba = tuple(float(c) for c in glass_rgba)
        super().__init__(height=height, width=width, base_pos=self.BASE_POS,
                         max_geom=max_geom, sphere_detail=sphere_detail)
        self.q_pour = self.Q_UPRIGHT + self.POUR_POSE_FRACTION * (
            self.Q_FULL_POUR - self.Q_UPRIGHT
        )
        self._glass_mocap = {}
        if self._glass_mesh is not None:
            for nm in ("glass_src", "glass_rcv"):
                bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, nm)
                self._glass_mocap[nm] = int(self.model.body_mocapid[bid])

    def _customize_spec(self, spec, mujoco) -> None:
        if self._glass_mesh is None:
            return
        mesh = spec.add_mesh()
        mesh.name = "pour_glass"
        mesh.file = self._glass_mesh
        for nm in ("glass_src", "glass_rcv"):
            b = spec.worldbody.add_body()
            b.name = nm
            b.mocap = True
            g = b.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = "pour_glass"
            g.rgba = self._glass_rgba
            g.contype = 0
            g.conaffinity = 0

    def set_glass_pose(self, name: str, pos, quat) -> None:
        """Pose a render glass ("glass_src" / "glass_rcv") in the world frame (wxyz
        quat). Takes effect at the next mj_forward (set_time does one)."""
        mid = self._glass_mocap[name]
        self.data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64)
        self.data.mocap_quat[mid] = np.asarray(quat, dtype=np.float64)

    @property
    def duration(self) -> float:
        return self.TILT_SECONDS + self.RETURN_SECONDS

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def motion_fraction(self, t: float) -> float:
        """Pose fraction upright->pour at time t (smoothstep tilt, then return)."""
        if t < 0.0:
            return 0.0
        if t < self.TILT_SECONDS:
            return self._smoothstep(t / self.TILT_SECONDS)
        t -= self.TILT_SECONDS
        if t < self.RETURN_SECONDS:
            return 1.0 - self._smoothstep(t / self.RETURN_SECONDS)
        return 0.0

    def q_at(self, t: float) -> np.ndarray:
        return self.Q_UPRIGHT + self.motion_fraction(t) * (self.q_pour - self.Q_UPRIGHT)

    def set_time(self, t: float) -> None:
        """Pose the arm at trajectory time t (kinematic; fingers held at the grasp)."""
        self.data.qpos[:9] = self.q_at(t)
        self.mj.mj_forward(self.model, self.data)

    def cup_pose_at(self, t: float):
        """World pose (pos, wxyz quat) of the held glass at time t: hand FK -> TCP ->
        fixed handle-grasp transform."""
        from warpmpm.colliders.glass import quat_from_mat, quat_to_mat

        self.set_time(t)
        hand_pos = self.data.xpos[self.ee].copy()
        hand_quat = self.data.xquat[self.ee].copy()
        r_hand = quat_to_mat(hand_quat)
        tcp = hand_pos + r_hand @ self.TCP_LOCAL
        r_cup = r_hand @ self.CUP_TO_HAND.T
        cup_pos = tcp - r_cup @ self.GRASP_LOCAL
        return cup_pos, quat_from_mat(r_cup)

    def tilt_degrees(self, quat) -> float:
        """Cup tilt from vertical (deg): angle of the cup z-axis to world z."""
        from warpmpm.colliders.glass import quat_to_mat

        return float(np.degrees(np.arccos(np.clip(quat_to_mat(quat)[2, 2], -1.0, 1.0))))

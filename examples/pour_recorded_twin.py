"""Digital twin of a recorded hardware pour episode (pouring_real_data/epNNNN).

Where pour_franka.py replays the Genesis study's scripted action, this twin replays the
real robot: the Franka's recorded joint trajectory drives the same MuJoCo Panda, and the
held measuring cup follows the fingertips through the measured grasp chain. The liquid is
the episode's dyed food-grade glycerol, filled to the measured initial volume. Everything
else (cup SDF collider with exact cavity + thickened walls, wrench readouts, leak audit,
level-volume ledger) is the pour_franka machinery unchanged.

Kinematic chain (verified against ep0001):
  recorded joints -> MuJoCo hand FK -> TCP at +0.1034 m along hand z. That offset is the
  Franka fingertip pinch point: FK(hand) + R_hand @ (0,0,0.1034) reproduces the recorded
  ee_pos/ee_quat to <1 mm / ~0 deg across the pour window, so arm render and cup physics
  share one pose source. The cup hangs from the pinch point at the same holding point as
  pour_franka (GRASP_LOCAL: handle column mid-height), and the constant hand->cup rotation
  is solved at the pour-start reference tick from the nominal insertion: cup upright,
  spout toward world -x (the tilt is +60 deg about world -y, so -x is downhill).

Scene constants measured from the episode's side camera (embedded AprilTag extrinsics,
1.8 mm / 0.36 deg residuals):
  tabletop      z = -0.063 m in the robot base frame (the base sits on a riser)
  receiver      the same 500 mL measuring-cup model, at the rim-ellipse fit of the
                pre-pour depth cloud (ep0001: x=0.2934, y=-0.0230), spout toward -x
experiments/pour/pour_calibrate_geometry.py reproduces both; --table-z / --receiver-xy override.

Glycerol: eta defaults to the 20 C literature value 1.41 Pa.s; temperature and absorbed
water move it by ~2x, so treat it as a prior, not truth (the ep0001 weak-form fit in
experiments/pour/pour_weakform_identify.py gives 3.03 -- pass it via --eta). rho = 1260.
bulk_modulus stays the artificial 9e5 of the honey twin (Ma < 0.1 at pour speeds; the
true K would cost ~50x).

Run:
  python examples/pour_recorded_twin.py                       # ep0001, 192^3, video
  python examples/pour_recorded_twin.py --fast --skip-video   # 96^3 smoke
  python examples/pour_recorded_twin.py --episode pouring_real_data/ep0002
  python examples/pour_recorded_twin.py --eta 3.03 --hold 1.5 --skip-video
      # planning: the recorded roll, then the cup HELD at its end pose for 1.5 s (the
      # robot's dwell_s), then the recorded return. Writes to <episode>_hold1.5_eta3.03/.

Outputs (out/pour_recorded_twin/<episode>/):
  twin.mp4           composite MuJoCo render from the real side camera's viewpoint
  side_by_side.mp4   real side-camera video (rotated upright) | twin render, time-locked
  metrics.csv        per-frame counts/fractions (n_rcv_pool = landed, below the 30 mL
                     graduation), tilt, wrenches, level volumes, leak audit
  metrics.png        transfer/level/wrench/audit plots
  final_n*.npz       end-state particles
  settled_n*.npz     cached settled fill (delete or --rebake to refresh)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warpmpm import GridConfig, Solver, newtonian
from warpmpm.adapters.mujoco_adapter import FrankaArm
from warpmpm.colliders.glass import (
    angular_velocity_between,
    quat_from_mat,
    quat_to_mat,
    world_to_local,
)
from warpmpm.geometry.measuring_cup import (
    MeasuringCupSpec,
    build_cup_sdf,
    cavity_sdf_local,
    project_out_of_solid,
    solid_sdf_local,
    write_cup_obj,
)

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "out" / "pour_recorded_twin"

SPEC = MeasuringCupSpec()
# landed liquid: receiver-cavity particles below the cup's 30 mL graduation, the floor
# region whose pixels pour_perception.py watches for the real onset (same definition)
POOL_ML = 30.0
_depths = np.linspace(0.0, 0.03, 301)
POOL_DEPTH = float(np.interp(POOL_ML * 1e-6, [SPEC.cavity_volume(d) for d in _depths], _depths))
# receiver: 180 deg about z -- spout toward -x, handle toward +x, matching the video
Q_RCV = np.array([0.0, 0.0, 0.0, 1.0])
# eta: Segur & Oberstar 1951, pure glycerol at 20 C (see module docstring); --eta overrides
GLYCEROL = dict(eta=1.41, density=1260.0, bulk_modulus=9.0e5)
VOLUME_ML = 300.0
CUP_FRICTION = 0.05
SDF_RES = 160
WALL_CELLS = 3.0
FPS = 60
PRE_ROLL = 1.0        # quiet seconds before the pour move (liquid settles on camera)
HOLD_SECONDS = 2.5    # keep simulating after the return (clamped before the lift step)
# settle to quiescence rather than for a fixed time: glycerol's viscous damping time
# rho*L^2/eta ~ 2.6 s is 7x honey's, and a honey-sized bake leaves the surface creeping
SETTLE_SPEED = 1.5e-3   # m/s mean particle speed; reached at ~2.5 s at 192^3
SETTLE_MAX_S = 5.0
GRID_LIM = 0.7
LIQUID_RGBA = np.array([0.93, 0.45, 0.10, 1.0])   # the dyed glycerol's amber
CUP_RGBA = (0.80, 0.88, 0.95, 0.28)
RENDER_MAX = 120_000

# ep0001 side-camera measurements (base frame); overridable per episode on the CLI.
# Rim-ellipse fits with the cup's known top semi-axes on the fused pre-pour depth cloud:
# receiver center 2.5 mm median residual (4.7k pts); held-cup center 1.4 mm (68 clean
# pts, gripper/spout/liquid excluded), giving the cup-in-hand shift below. The grasp
# height is separately confirmed by the pre-pour liquid-surface z (within 2 mm).
TABLE_Z = -0.063
RECEIVER_XY = (0.2934, -0.0230)
# world-frame offset of the real cup axis vs the nominal handle grasp at the reference
# pose (the human inserted the handle ~11 mm shallower than GRASP_LOCAL assumes)
CUP_SHIFT_XY = (0.0112, 0.0039)
# nominal insertion: cup upright, spout toward -x (downhill of the +60deg tilt about -y)
R_CUP_REF = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
# grasp roll about the tilt axis (deg): +ve leans the cup top toward the handle (+x),
# reducing the effective tilt and delaying pour onset. The one orientation DOF the rim
# fit cannot see; calibrated so the sim's receiver-arrival time matches the video
# (real: t_send + 1.70 s; sensitivity 0.042 s/deg; ep0001 sweep at eta=1.41). The
# arrival time couples to eta only weakly (second order, ~0.1 s over 1.4 -> 3.0 Pa.s).
GRASP_ROLL_DEG = 1.0


# --------------------------------------------------------------------------------------
# episode loading
# --------------------------------------------------------------------------------------
def recorded_pour_actions(ep_dir: Path) -> tuple[dict, dict, dict | None]:
    """Find pour, return and optional lift in planned or cached-trajectory logs."""
    actions = [json.loads(line) for line in
               (ep_dir / "actions.jsonl").read_text().splitlines() if line]
    moves = [a for a in actions if a.get("type") in
             {"go_to_pose", "execute_trajectory"}]
    upright = np.array([0.5, 0.5, -0.5, 0.5])
    for i, action in enumerate(moves[:-1]):
        angle = action.get("pour_angle_deg")
        quat = np.asarray(action.get("target_quat_xyzw", upright), dtype=float)
        tilted = (abs(float(angle)) > 1e-6 if angle is not None else
                  not np.isclose(abs(np.dot(quat / np.linalg.norm(quat), upright)),
                                 1.0, atol=1e-7, rtol=0.0))
        if tilted:
            return action, moves[i + 1], moves[i + 2] if i + 2 < len(moves) else None
    raise ValueError(f"No pour followed by a return in {ep_dir / 'actions.jsonl'}")


def load_episode(ep_dir: Path, pre_roll: float, hold: float) -> dict:
    """Parse actions/states into a replay window and a joint track.

    Window: [pour.t_send - pre_roll, return.t_ack + hold], clamped before the next
    go_to_pose (the unload lift) so the twin never replays the gripper opening."""
    pour, ret, lift = recorded_pour_actions(ep_dir)
    t0 = pour["t_send"] - pre_roll
    t1 = ret["t_ack"] + hold
    if lift is not None:
        t1 = min(t1, lift["t_send"] - 0.05)

    rows = [json.loads(ln) for ln in
            (ep_dir / "states.jsonl").read_text().splitlines() if ln]
    ts, qs, ws = [], [], []
    for r in rows:
        if t0 - 0.5 <= r["t"] <= t1 + 0.5:
            ts.append(r["t"])
            qs.append(r["state"]["joint_position"])
            ws.append(r["state"]["gripper_width"])
    ts = np.asarray(ts)
    gaps = np.diff(ts)
    max_gap = float(gaps.max()) if len(gaps) else float("inf")
    if len(ts) < 10 or max_gap > 0.3 or np.any(gaps <= 0.0):
        raise SystemExit(f"state stream too sparse in the replay window "
                         f"(n={len(ts)}, max gap {max_gap:.2f}s)")
    meta = json.loads((ep_dir / "meta.json").read_text())
    return dict(
        t0=t0, duration=t1 - t0, t_ref=pour["t_send"] - t0,
        t_pour=pour["t_send"] - t0, t_hold=pour["t_ack"] - t0,
        t_return_start=ret["t_send"] - t0,
        t_return_done=ret["t_ack"] - t0,
        ts=ts - t0, qs=np.asarray(qs), widths=np.asarray(ws), meta=meta,
    )


def side_camera_view(meta: dict) -> dict:
    """MuJoCo free-camera parameters approximating the episode's side camera (its
    mount roll is within ~3 deg of the zero-roll free camera, checked for ep0001)."""
    extr = meta["extrinsics"]["cameras"]["side"]
    T = np.asarray(extr["T_base_cam"]["matrix"])
    cam_pos, d = T[:3, 3], T[:3, 2]                     # OpenCV: z forward
    fx = meta["cameras"]["side"]["intrinsics"]["fx"]
    w = meta["cameras"]["side"]["width"]                # rotated upright: w is vertical
    dist = 0.55
    return dict(
        lookat=cam_pos + dist * d, distance=dist,
        azimuth=float(np.degrees(np.arctan2(d[1], d[0]))),
        elevation=float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))),
        fovy=float(np.degrees(2.0 * np.arctan(w / (2.0 * fx)))),
    )


# --------------------------------------------------------------------------------------
# recorded arm
# --------------------------------------------------------------------------------------
class RecordedPanda(FrankaArm):
    """MuJoCo Panda driven by the recorded joint track; base at the robot base frame
    origin so MuJoCo world == base frame. The held cup follows the fingertip pinch
    point (TCP) through the fixed handle grasp, like PandaPour but with the constant
    hand->cup rotation solved at the pour-start reference tick."""

    TCP_LOCAL = np.array([0.0, 0.0, 0.1034])       # hand -> pinch point (<1 mm vs ee_pos)
    GRASP_LOCAL = np.array([-0.0863, 0.0, 0.060])  # pour_franka's holding point

    def __init__(self, ep: dict, glass_mesh: Path, height: int = 848, width: int = 480,
                 max_geom: int = 360000, cup_shift_xy=CUP_SHIFT_XY,
                 grasp_roll_deg: float = GRASP_ROLL_DEG, hold: float = 0.0,
                 cup_reference_pos=None, cup_reference_quat=None):
        self._glass_mesh = str(glass_mesh)
        self._glass_rgba = CUP_RGBA
        self._ts, self._qs, self._ws = ep["ts"], ep["qs"], ep["widths"]
        # planning variant: freeze the joint track at the roll's end pose (the pour
        # move's ack) for `hold` seconds, then continue with the recorded return
        self._hold, self._t_hold = float(hold), float(ep["t_hold"])
        super().__init__(height=height, width=width, base_pos=(0.0, 0.0, 0.0),
                         max_geom=max_geom, sphere_detail=(8, 6))
        self._glass_mocap = {}
        for nm in ("glass_src", "glass_rcv"):
            bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, nm)
            self._glass_mocap[nm] = int(self.model.body_mocapid[bid])
        self.duration = float(ep["duration"]) + self._hold
        # constant hand->cup rotation from the insertion at the reference tick: the
        # nominal upright/spout(-x) pose, drooped by the calibrated grasp roll about
        # the world tilt axis (the pivot is the pinch point, which GRASP_LOCAL is)
        d = np.radians(grasp_roll_deg)
        r_ref = np.array([[np.cos(d), 0.0, np.sin(d)],
                          [0.0, 1.0, 0.0],
                          [-np.sin(d), 0.0, np.cos(d)]]) @ R_CUP_REF
        if cup_reference_quat is not None:
            r_ref = quat_to_mat(np.asarray(cup_reference_quat, dtype=float))
        self.set_time(ep["t_ref"])
        r_hand = quat_to_mat(self.data.xquat[self.ee])
        self._hand_cup = r_hand.T @ r_ref
        # fold the measured world-frame cup shift at the reference pose into the grasp
        # (cup_pos = TCP - R_cup @ grasp  =>  grasp -= R_ref^T @ shift)
        shift = np.array([cup_shift_xy[0], cup_shift_xy[1], 0.0])
        self._grasp = self.GRASP_LOCAL - r_ref.T @ shift
        if cup_reference_pos is not None:
            # A pose fitted to this session's cup outlines determines the complete
            # rigid grasp, including vertical insertion and both cup lean angles.
            tcp = self.data.xpos[self.ee] + r_hand @ self.TCP_LOCAL
            self._grasp = r_ref.T @ (tcp - np.asarray(cup_reference_pos, dtype=float))

    def _customize_spec(self, spec, mujoco) -> None:
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
        mid = self._glass_mocap[name]
        self.data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64)
        self.data.mocap_quat[mid] = np.asarray(quat, dtype=np.float64)

    def joints_at(self, t: float) -> tuple[np.ndarray, float]:
        if self._hold > 0.0 and t > self._t_hold:
            t = max(self._t_hold, t - self._hold)
        t = float(np.clip(t, self._ts[0], self._ts[-1]))
        q = np.array([np.interp(t, self._ts, self._qs[:, j]) for j in range(7)])
        return q, float(np.interp(t, self._ts, self._ws))

    def set_time(self, t: float) -> None:
        q, w = self.joints_at(t)
        self.data.qpos[:7] = q
        self.data.qpos[7:9] = w / 2.0
        self.mj.mj_forward(self.model, self.data)

    def cup_pose_at(self, t: float):
        """World (pos, wxyz quat) of the held cup: hand FK -> TCP -> handle grasp."""
        self.set_time(t)
        hand_pos = self.data.xpos[self.ee].copy()
        r_hand = quat_to_mat(self.data.xquat[self.ee])
        tcp = hand_pos + r_hand @ self.TCP_LOCAL
        r_cup = r_hand @ self._hand_cup
        return tcp - r_cup @ self._grasp, quat_from_mat(r_cup)

    @staticmethod
    def tilt_degrees(quat) -> float:
        return float(np.degrees(np.arccos(np.clip(quat_to_mat(quat)[2, 2], -1.0, 1.0))))


# --------------------------------------------------------------------------------------
# scene helpers (pour_franka semantics)
# --------------------------------------------------------------------------------------
def fill_to_volume(spec: MeasuringCupSpec, h: float, volume_m3: float, seed: int = 0):
    """Jittered lattice holding `volume_m3` of material (n = volume / h^3 particles,
    lowest-z first), cup_fill clearances. The settled level then lands where the
    cavity-volume curve puts that volume -- a free graduation cross-check."""
    n_target = round(volume_m3 / h**3)
    z0 = spec.floor_z + h
    z1 = spec.rim_z - 0.006
    r_max = max(spec.a_top_inner, spec.b_top_inner)
    xs = np.arange(-r_max + 0.5 * h, r_max, h)
    zs = np.arange(z0 + 0.5 * h, z1, h)
    g = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    g = g + np.random.default_rng(seed).uniform(-0.25 * h, 0.25 * h, size=g.shape)
    a_i, b_i = spec.inner_semi_axes(g[:, 2])
    rho = np.sqrt((g[:, 0] / (a_i - h)) ** 2 + (g[:, 1] / (b_i - h)) ** 2)
    g = g[(rho < 1.0) & (g[:, 2] >= z0)]
    g = g[np.argsort(g[:, 2], kind="stable")]
    if len(g) < n_target:
        raise ValueError(f"cup lattice holds only {len(g)} < {n_target} particles")
    pos = g[:n_target].astype(np.float32)
    vol = np.full(n_target, h**3, dtype=np.float32)
    return pos, vol


def collision_extras(dx: float) -> tuple[float, float]:
    return (max(0.0, WALL_CELLS * dx - SPEC.wall), max(0.0, WALL_CELLS * dx - SPEC.base))


def sound_speed(liq: dict) -> float:
    return float(np.sqrt(1.1 * liq["bulk_modulus"] / liq["density"]))


def substeps_per_tick(liq: dict, dx: float, dt_tick: float) -> int:
    acoustic = sound_speed(liq) / (0.28 * dx)
    viscous = 6.0 * liq["eta"] / (liq["density"] * dx * dx)
    return int(np.ceil(dt_tick * max(acoustic, viscous)))


def world_to_mpm_offset(arm: RecordedPanda, receiver_pos, dx: float) -> np.ndarray:
    """Offset such that the swept held cup, the receiver, and the table fit in the MPM
    domain with margin. Sampled over the full episode window (never a --frames cap) so
    the offset -- and with it the cached settled state -- is stable per episode."""
    n_all = round(arm.duration * FPS)
    centers = np.array([arm.cup_pose_at(k / FPS)[0] for k in range(0, n_all, 3)])
    reach = 0.16  # cup local AABB radius incl. handle/spout
    lo = np.minimum(centers.min(0) - reach,
                    np.asarray(receiver_pos) - np.array([0.13, 0.13, 0.0]))
    hi = np.maximum(centers.max(0) + reach,
                    np.asarray(receiver_pos) + np.array([0.13, 0.13, 0.14]))
    lo[2] = min(lo[2], TABLE_Z)
    pad = max(0.03, 4 * dx)
    span = (hi - lo) + 2 * pad
    if span.max() > GRID_LIM:
        raise SystemExit(f"replay does not fit the {GRID_LIM} m domain: span {span}")
    off = pad - lo + 0.5 * (GRID_LIM - span)
    return off.astype(np.float64)


def cup_audit(x_mpm, pos_world, quat, h: float, extras, w2m):
    local = world_to_local(x_mpm, np.asarray(pos_world) + w2m, quat)
    n_solid = int((solid_sdf_local(local, SPEC, *extras) < 0.0).sum())
    z = local[:, 2]
    cav = ((cavity_sdf_local(local, SPEC) < 0.75 * h)
           & (z >= SPEC.floor_z - 1e-3) & (z <= SPEC.rim_z))
    if int(cav.sum()) < 50:
        vol = 0.0
    else:
        depth = float(np.quantile(z[cav], 0.97) - SPEC.floor_z) + 0.5 * h
        vol = SPEC.cavity_volume(depth)
    n_pool = int((cav & (z < SPEC.floor_z + POOL_DEPTH)).sum())
    return n_solid, cav, vol, n_pool


def render_subsample(n: int, cap: int = RENDER_MAX):
    if not cap or n <= cap:
        return slice(None), 1.0
    idx = np.sort(np.random.default_rng(0).permutation(n)[:cap])
    return idx, float((n / cap) ** (1.0 / 3.0))


def render_frame(arm: RecordedPanda, x_world, spd, p_now, q_now, t_now, h,
                 radius_scale=1.0):
    arm.set_glass_pose("glass_src", p_now, q_now)
    arm.set_time(t_now)
    col = np.tile(LIQUID_RGBA, (len(x_world), 1)).astype(np.float32)
    col[:, :3] = np.clip(col[:, :3] + 0.30 * np.clip(spd / 1.5, 0, 1)[:, None], 0, 1)
    return arm.render_with_particles(
        x_world, col, radius=0.85 * h * radius_scale,
        boxes=[((0.35, -0.05, TABLE_Z - 0.01), (0.55, 0.55, 0.01),
                (0.13, 0.13, 0.15, 1.0))],  # the black lab table
    )


def write_mp4(frames_dir: Path, mp4: Path, fps: int) -> Path:
    import imageio.v2 as imageio

    with imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8,
                            macro_block_size=2,
                            output_params=["-movflags", "+faststart"]) as wtr:
        for p in sorted(frames_dir.glob("f_*.png")):
            wtr.append_data(imageio.imread(p))
    return mp4


def compose_side_by_side(ep_dir: Path, ep: dict, frames_dir: Path, out_mp4: Path,
                         n_frames: int) -> Path | None:
    """[real side camera (rotated upright) | twin render], time-locked at 60 fps."""
    import imageio.v2 as imageio

    rows = [json.loads(ln) for ln in
            (ep_dir / "frames_side.jsonl").read_text().splitlines() if ln]
    t_host = np.array([r["t_host"] for r in rows])
    needed = []
    for k in range(n_frames):
        i = int(np.argmin(np.abs(t_host - (ep["t0"] + k / FPS))))
        needed.append(rows[i]["frame_idx"] if abs(t_host[i] - (ep["t0"] + k / FPS)) < 0.06
                      else -1)
    reader = imageio.get_reader(ep_dir / "side_rgb.mp4")
    pos, cur = -1, None
    with imageio.get_writer(out_mp4, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=2,
                            output_params=["-movflags", "+faststart"]) as wtr:
        for k in range(n_frames):
            sim = imageio.imread(frames_dir / f"f_{k:04d}.png")
            idx = needed[k]
            while idx >= 0 and pos < idx:
                try:
                    cur = reader.get_next_data()
                except IndexError:
                    idx = -1
                    break
                pos += 1
            if cur is None:
                continue
            real = np.ascontiguousarray(np.rot90(cur, k=-1))
            if real.shape[0] != sim.shape[0]:
                sc = sim.shape[0] / real.shape[0]
                yi = (np.arange(sim.shape[0]) / sc).astype(int).clip(0, real.shape[0] - 1)
                xi = (np.arange(int(real.shape[1] * sc)) / sc).astype(int).clip(
                    0, real.shape[1] - 1)
                real = real[yi][:, xi]
            wtr.append_data(np.hstack([real, sim[:, :, :3]]))
    reader.close()
    return out_mp4


# --------------------------------------------------------------------------------------
# main run
# --------------------------------------------------------------------------------------
def run(episode: Path, device: str = "auto", n_grid: int = 192, video: bool = True,
        side_by_side: bool = True, rebake: bool = False, frames: int | None = None,
        eta: float = GLYCEROL["eta"], volume_ml: float = VOLUME_ML,
        receiver_xy=RECEIVER_XY, table_z: float = TABLE_Z,
        cup_shift_xy=CUP_SHIFT_XY, grasp_roll_deg: float = GRASP_ROLL_DEG,
        hold: float = 0.0, cup_reference_pos=None, cup_reference_quat=None) -> dict:
    liq = dict(GLYCEROL, eta=eta)
    # the settled cache lives with the episode; a planning variant (hold > 0) writes its
    # own directory, and the real video cannot be paired with it
    cache_dir = OUT_ROOT / episode.name
    out = cache_dir if hold == 0.0 else OUT_ROOT / f"{episode.name}_hold{hold:g}_eta{eta:.2f}"
    side_by_side = side_by_side and hold == 0.0
    out.mkdir(parents=True, exist_ok=True)
    ep = load_episode(episode, PRE_ROLL, HOLD_SECONDS)
    receiver_pos = np.array([receiver_xy[0], receiver_xy[1], table_z])

    arm = RecordedPanda(ep, write_cup_obj(SPEC, out / "cup_render.obj"),
                        cup_shift_xy=cup_shift_xy, grasp_roll_deg=grasp_roll_deg, hold=hold,
                        cup_reference_pos=cup_reference_pos,
                        cup_reference_quat=cup_reference_quat)
    cam = side_camera_view(ep["meta"])
    arm.model.vis.global_.fovy = cam["fovy"]
    arm.cam.lookat[:] = cam["lookat"]
    arm.cam.distance = cam["distance"]
    arm.cam.azimuth = cam["azimuth"]
    arm.cam.elevation = cam["elevation"]
    arm.set_glass_pose("glass_rcv", receiver_pos, Q_RCV)

    grid = GridConfig(n_grid=n_grid, grid_lim=GRID_LIM)
    h = grid.dx / 2
    dt_tick = 1.0 / FPS
    substeps = substeps_per_tick(liq, grid.dx, dt_tick)
    dt = dt_tick / substeps
    n_frames = round(arm.duration * FPS)
    if frames is not None:
        n_frames = min(n_frames, frames)
    w2m = world_to_mpm_offset(arm, receiver_pos, grid.dx)

    extras = collision_extras(grid.dx)
    sdf = build_cup_sdf(SPEC, res=SDF_RES, margin=0.010,
                        extra_wall=extras[0], extra_base=extras[1])
    cup_pos0, cup_quat0 = arm.cup_pose_at(0.0)
    pos_local, vol = fill_to_volume(SPEC, h, volume_ml * 1e-6)
    pos = (cup_pos0 + pos_local @ quat_to_mat(cup_quat0).T + w2m).astype(np.float32)

    def q_xyzw(q):
        return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))

    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(**liq))
    s.add_plane((0, 0, table_z + w2m[2]), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    band = 0.5 * grid.dx
    src = s.add_sdf_collider(sdf, center=cup_pos0 + w2m, quat=q_xyzw(cup_quat0),
                             band=band, surface="separable", friction=CUP_FRICTION)
    rcv = s.add_sdf_collider(sdf, center=receiver_pos + w2m, quat=q_xyzw(Q_RCV),
                             band=band, surface="separable", friction=CUP_FRICTION)

    n0 = s.n_particles
    m_liq = float(liq["density"] * vol.sum())
    print(f"{episode.name}: n_grid={n_grid} dx={grid.dx*1000:.2f}mm N={n0} "
          f"glycerol={m_liq:.3f}kg ({1e6*vol.sum():.0f}mL) eta={liq['eta']} "
          f"dt={dt:.2e} ({substeps} substeps/frame) {n_frames} frames "
          f"[pour @ {ep['t_pour']:.2f}s, hold {hold:g}s @ {ep['t_hold']:.2f}s, "
          f"return done @ {ep['t_return_done'] + hold:.2f}s]")
    print(f"world->mpm offset {np.round(w2m, 4)}; cup0 {np.round(cup_pos0, 4)} "
          f"tilt0 {arm.tilt_degrees(cup_quat0):.2f}deg; receiver {receiver_pos}")

    # ---- settle (cached; stored in world frame so a changed w2m cannot shift it) ----
    cache = cache_dir / f"settled_n{n_grid}_v{int(volume_ml)}.npz"
    loaded = False
    if cache.exists() and not rebake:
        d = np.load(cache)
        same_pose = ("cup_pos0" in d
                     and np.linalg.norm(d["cup_pos0"] - cup_pos0) < 1e-3
                     and np.linalg.norm(d["cup_quat0"] - cup_quat0) < 1e-3)
        if "x_world" in d and len(d["x_world"]) == n0 and same_pose:
            s.set_x((d["x_world"] + w2m).astype(np.float32))
            s.set_v(d["v"].astype(np.float32))
            print(f"loaded settled state from {cache.name}")
            loaded = True
    if not loaded:
        t0 = time.time()
        for k in range(round(SETTLE_MAX_S * FPS)):
            s.step(dt, substeps)
            if k % 12 == 11:
                spd = float(np.linalg.norm(s.v(), axis=1).mean())
                if spd < SETTLE_SPEED:
                    print(f"settle quiescent at {k/FPS:.1f}s (mean {spd*1000:.2f} mm/s)")
                    break
        x, v, npx = project_out_of_solid(s.x(), s.v(), cup_pos0 + w2m, cup_quat0, SPEC,
                                         clearance=0.35 * h,
                                         extra_wall=extras[0], extra_base=extras[1])
        s.set_x(x)
        s.set_v(v)
        np.savez_compressed(cache, x_world=s.x() - w2m, v=s.v(),
                            cup_pos0=cup_pos0, cup_quat0=cup_quat0)
        print(f"settled {time.time()-t0:.0f}s (projected {npx}) -> {cache.name}")

    # ---- replay ---------------------------------------------------------------------
    frames_dir = out / "_frames"
    if video:
        import imageio.v2 as imageio

        frames_dir.mkdir(exist_ok=True)
        for stale in frames_dir.glob("f_*.png"):
            stale.unlink()
    rows_out = []
    max_embedded = 0
    proj_total = 0
    sub, rscale = render_subsample(n0)
    t_start = time.time()
    for frame in range(n_frames):
        t = frame * dt_tick
        p0, q0 = arm.cup_pose_at(t)
        p1, q1 = arm.cup_pose_at(t + dt_tick)
        vel = (p1 - p0) / dt_tick
        omega = angular_velocity_between(q0, q1, dt_tick)
        s.set_sdf_pose(src, center=p0 + w2m, quat=q_xyzw(q0), velocity=vel, omega=omega)
        s.reset_sdf_force(src)
        s.reset_sdf_force(rcv)
        s.step(dt, substeps)
        w_src = s.sdf_wrench(src, dt_tick)
        w_rcv = s.sdf_wrench(rcv, dt_tick)
        t_now, p_now, q_now = t + dt_tick, p1, q1

        x = s.x()
        v = s.v()
        ns_src, cav_src, vol_src, _ = cup_audit(x, p_now, q_now, h, extras, w2m)
        ns_rcv, cav_rcv, vol_rcv, n_pool = cup_audit(x, receiver_pos, Q_RCV, h, extras, w2m)
        max_embedded = max(max_embedded, ns_src + ns_rcv)
        if ns_src or ns_rcv:
            x, v, n1 = project_out_of_solid(x, v, p_now + w2m, q_now, SPEC,
                                            clearance=0.35 * h,
                                            solid_velocity=(vel, omega),
                                            extra_wall=extras[0], extra_base=extras[1])
            x, v, n2 = project_out_of_solid(x, v, receiver_pos + w2m, Q_RCV, SPEC,
                                            clearance=0.35 * h,
                                            extra_wall=extras[0], extra_base=extras[1])
            proj_total += n1 + n2
            s.set_x(x)
            s.set_v(v)
            _, cav_src, vol_src, _ = cup_audit(x, p_now, q_now, h, extras, w2m)
            _, cav_rcv, vol_rcv, n_pool = cup_audit(x, receiver_pos, Q_RCV, h, extras, w2m)

        in_src, in_rcv = int(cav_src.sum()), int(cav_rcv.sum())
        tilt = arm.tilt_degrees(q_now)
        rows_out.append(dict(
            frame=frame, t=round(t_now, 5), tilt_deg=round(tilt, 2), n_src=in_src,
            n_rcv=in_rcv, n_rcv_pool=n_pool, n_air_spill=n0 - in_src - in_rcv,
            frac_rcv=round(in_rcv / n0, 5),
            frac_spill=round((n0 - in_src - in_rcv) / n0, 5),
            ml_rcv=round(1e6 * vol.sum() * in_rcv / n0, 2),
            src_fz=round(w_src["force"][2], 4), rcv_fz=round(w_rcv["force"][2], 4),
            level_vol_src_mL=round(vol_src * 1e6, 2),
            level_vol_rcv_mL=round(vol_rcv * 1e6, 2),
            embedded=ns_src + ns_rcv, projected=proj_total,
        ))

        if video:
            img = render_frame(arm, x[sub] - w2m, np.linalg.norm(v[sub], axis=1),
                               p_now, q_now, t_now, h, radius_scale=rscale)
            imageio.imwrite(frames_dir / f"f_{frame:04d}.png", img)

        if frame % 30 == 0 or frame == n_frames - 1:
            el = time.time() - t_start
            print(f"frame {frame+1:3d}/{n_frames} t={t_now:5.2f}s tilt={tilt:5.1f} "
                  f"src={in_src:6d} rcv={in_rcv:6d} spill={n0-in_src-in_rcv:5d} "
                  f"emb={ns_src+ns_rcv} rcv={1e6*vol.sum()*in_rcv/n0:5.1f}mL "
                  f"[{el/(frame+1)*1000:.0f}ms/frame]")

    # ---- outputs ----------------------------------------------------------------------
    csv_path = out / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    r = {k: np.array([row[k] for row in rows_out]) for k in rows_out[0]}

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax = axes[0]
    ax.plot(r["t"], r["ml_rcv"], color="tab:blue", label="transferred (count) [mL]")
    ax.plot(r["t"], r["level_vol_rcv_mL"], ":", color="tab:cyan",
            label="receiver level volume [mL]")
    ax.plot(r["t"], r["level_vol_src_mL"], ":", color="tab:orange",
            label="source level volume [mL]")
    ax.plot(r["t"], r["tilt_deg"], "--", color="grey", label="cup tilt [deg]")
    ax.axvline(ep["t_pour"], color="k", lw=0.5)
    if hold > 0.0:
        ax.axvspan(ep["t_hold"], ep["t_hold"] + hold, color="k", alpha=0.06)
    ax.axvline(ep["t_return_done"] + hold, color="k", lw=0.5)
    ax.set_ylabel("mL / deg")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(r["t"], -r["src_fz"], color="tab:green", label="held cup -Fz (wrist load)")
    ax.plot(r["t"], -r["rcv_fz"], color="tab:blue", label="receiver -Fz (scale)")
    ax.axhline(9.81 * m_liq, ls=":", color="k", label=f"total weight {9.81*m_liq:.2f}N")
    ax.set_ylabel("force (N)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(r["t"], r["embedded"], color="tab:red", label="embedded (audit)")
    ax.plot(r["t"], r["projected"], color="tab:orange", label="cumulative projected")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("particles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle(f"{episode.name} recorded-pour twin: glycerol eta={liq['eta']} Pa.s, "
                 f"{volume_ml:.0f} mL, N={n0}, dx={grid.dx*1000:.1f} mm"
                 + (f", held {hold:g} s at the roll's end" if hold > 0.0 else ""))
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=130)
    plt.close(fig)

    np.savez_compressed(out / f"final_n{n_grid}.npz", x=s.x(), v=s.v(), vol=vol,
                        rho0=liq["density"])
    arm.close()
    mp4 = sbs = None
    if video:
        mp4 = write_mp4(frames_dir, out / "twin.mp4", fps=FPS)
        print("wrote", mp4)
        if side_by_side:
            sbs = compose_side_by_side(episode, ep, frames_dir, out / "side_by_side.mp4",
                                       n_frames)
            print("wrote", sbs)
    print("wrote", csv_path, "and", out / "metrics.png")
    print(f"final: transferred {r['ml_rcv'][-1]:.1f} mL ({100*r['frac_rcv'][-1]:.1f}%) "
          f"spill/air {r['frac_spill'][-1]*100:.2f}% | max embedded {max_embedded} "
          f"| projected {proj_total}")
    return {"rows": rows_out, "mp4": mp4, "side_by_side": sbs, "out": out,
            "final_ml": float(r["ml_rcv"][-1])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path,
                    default=REPO / "pouring_real_data" / "ep0001")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-grid", type=int, default=192)
    ap.add_argument("--fast", action="store_true", help="coarse smoke run (96^3)")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--no-side-by-side", action="store_true")
    ap.add_argument("--rebake", action="store_true")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--eta", type=float, default=GLYCEROL["eta"],
                    help="glycerol viscosity Pa.s (temperature/purity move it ~2x)")
    ap.add_argument("--volume-ml", type=float, default=VOLUME_ML)
    ap.add_argument("--receiver-xy", type=float, nargs=2, default=list(RECEIVER_XY))
    ap.add_argument("--table-z", type=float, default=TABLE_Z)
    ap.add_argument("--cup-shift", type=float, nargs=2, default=list(CUP_SHIFT_XY),
                    help="world-frame xy offset of the real cup axis vs the nominal "
                         "grasp at the reference pose (rim-fit calibration)")
    ap.add_argument("--grasp-roll", type=float, default=GRASP_ROLL_DEG,
                    help="cup droop about the tilt axis, deg (+ve delays pour onset); "
                         "calibrated against the real receiver-arrival time")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="planning: hold the cup at the roll's end pose for this many "
                         "seconds before the recorded return (the robot's dwell_s)")
    ap.add_argument("--geometry-json", type=Path,
                    help="session geometry: table_z, receiver_xy, cup_reference_pos/quat (wxyz)")
    args = ap.parse_args()
    geo = json.loads(args.geometry_json.read_text()) if args.geometry_json else {}
    run(episode=args.episode.resolve(), device=args.device,
        n_grid=96 if args.fast else args.n_grid, video=not args.skip_video,
        side_by_side=not args.no_side_by_side, rebake=args.rebake, frames=args.frames,
        eta=args.eta, volume_ml=args.volume_ml,
        receiver_xy=tuple(geo.get("receiver_xy", args.receiver_xy)),
        table_z=geo.get("table_z", args.table_z),
        cup_shift_xy=tuple(args.cup_shift), grasp_roll_deg=args.grasp_roll, hold=args.hold,
        cup_reference_pos=geo.get("cup_reference_pos"),
        cup_reference_quat=geo.get("cup_reference_quat"))

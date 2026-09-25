"""Free-surface observables of a recorded pour episode, from the RGB videos.

The pour's transparent measuring cups make the dyed glycerol body visible through the
walls, so the episode's own cameras measure everything the weak-form identification
needs, with no learned tracker:

  source level/volume   the held cup's liquid, seen through the wall. Per frame, the
                        liquid body model {cavity(p) < 0 and world_z(p) < L} is
                        ray-marched against the analytic cup cavity (the pixel's
                        turn-on level z_on = the lowest horizontal surface making its
                        ray hit liquid); the level L* maximizes IoU between the
                        predicted mask {z_on < L} and the observed amber mask. Volume
                        follows from a cavity point lattice below the fitted plane
                        (the cup is tilted, so the upright graduation curve does not
                        apply). Diagnostic only: wall dye-film + lensing bias it by
                        +60-90 mL; the brink identification closes mass from
                        V0 - V_rcv instead.
  receiver level/volume the same match with a precomputed z_on map (the receiver is
                        static) and the exact upright cavity-volume curve. This V(t)
                        is the identification's only liquid observable, and the twin
                        comparison reads the same curve.
  onset                 the first frame with amber below the receiver's 30 mL
                        graduation (rays that light up only once liquid has landed on
                        the floor; no ray through that region sees the held cup, so
                        the count is zero until the pour arrives). The twin records the
                        same event as landed particles (n_rcv_pool), so the two onsets
                        compare like for like in pour_validate.py.

Direct field channels (free-jet silhouette widths, spout-film thickness) were built
and defeated by this episode's optics: the through-wall jet is refraction-distorted
by the receiver's rim band/graduations (+-40% width bulges -> negative eta), and the
spout film has the amber body directly behind it in image space (no silhouette).
Their extractors live in git history (40ce313); the slender-jet estimator itself
survives as pour_weakform_identify.py's manufactured-solution selftest.

Poses come from the twin's own chain (RecordedPanda: recorded joints -> hand FK ->
TCP -> calibrated handle grasp), and the camera model is the episode's embedded
AprilTag extrinsics, so the projected-cup overlays (--probe) validate calibration +
FK end to end in image space before any physics is read off.

Conventions: all image math in NATIVE camera frames (OpenCV axes; the side camera is
mounted rotated, so overlays are rotated upright only for display). World = robot
base frame. Amber = HSV threshold (hue wraps at red).

Run:
  python experiments/pour/pour_perception.py --probe          # overlay stills on 4 frames
  python experiments/pour/pour_perception.py                  # full extraction + overlay video
  python experiments/pour/pour_perception.py --no-video       # observations.npz only

Outputs (out/pour_wf/<episode>/):
  observations.npz   per-frame levels, volumes, poses, times, floor-crop amber counts;
                     rcv_onset_s = the real onset on the pour clock
  extract_overlay.mp4  extraction proof: masks + fitted boundaries
  levels.png         level/volume curves + fit quality
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))

from pour_recorded_twin import (
    CUP_SHIFT_XY,
    GRASP_ROLL_DEG,
    HOLD_SECONDS,
    POOL_DEPTH,
    POOL_ML,
    PRE_ROLL,
    Q_RCV,
    RECEIVER_XY,
    SPEC,
    TABLE_Z,
    RecordedPanda,
    load_episode,
    recorded_pour_actions,
)
from warpmpm.colliders.glass import quat_to_mat
from warpmpm.geometry.measuring_cup import cavity_sdf_local, write_cup_obj

OUT_ROOT = REPO / "out" / "pour_wf"

# ---- amber segmentation (HSV; hue wraps at red). Tuned on ep0001 probe frames. ------
HUE_LO, HUE_HI = 0.965, 0.115     # accept hue <= HI or >= LO (deep orange-red)
SAT_MIN, VAL_MIN = 0.42, 0.12     # generous amber mask (thin pools included)
# The cup walls carry a dyed film wherever liquid has been (and the filled cup lenses
# its apparent body upward), so the LEVEL fits match only the DEEP liquid: pixels
# whose ray crosses more than CHORD_* metres of liquid are saturated (measured 0.90+
# on ep0001) while film/lensing bands sit at 0.3-0.5. The nested-mask machinery is
# unchanged: the level at which a pixel's liquid chord crosses the threshold is the
# m-th smallest cavity-sample z along its ray.
SAT_DEEP_SRC, CHORD_SRC = 0.70, 0.015
SAT_DEEP_RCV, CHORD_RCV = 0.55, 0.008
IOU_MIN = 0.25                    # below this the level fit is starved -> NaN
POOL_PX_MIN = 50                  # amber px below the receiver's 30 mL graduation = onset
# ---- geometry / sampling -------------------------------------------------------------
RAY_STEPS = 64                    # samples per ray through a cup's bounding sphere
CAVITY_INSET = 0.0012             # m; require this depth inside the cavity (wall band)
LATTICE_H = 0.0012                # m; source-cup volume lattice pitch
LEVEL_STEP = 0.0005               # m; level-scan resolution (~4 mL on the receiver)
RIM_Z_WORLD = TABLE_Z + SPEC.rim_z
SIGMA_GLYCEROL = 0.063            # N/m at 20 C; stored in the npz for record


class Camera:
    """Pinhole + rigid transform from the episode's embedded extrinsics."""

    def __init__(self, meta: dict, name: str):
        cm = meta["cameras"][name]
        intr = cm["intrinsics"]
        self.fx, self.fy = float(intr["fx"]), float(intr["fy"])
        self.cx, self.cy = float(intr["ppx"]), float(intr["ppy"])
        self.w, self.h = int(cm["width"]), int(cm["height"])
        T = np.asarray(meta["extrinsics"]["cameras"][name]["T_base_cam"]["matrix"])
        self.R, self.t = T[:3, :3], T[:3, 3]          # camera -> base

    def project(self, p_base: np.ndarray):
        """(N,3) base-frame points -> (N,2) native pixels, (N,) camera depth."""
        pc = (np.atleast_2d(p_base) - self.t) @ self.R
        z = pc[:, 2]
        uv = np.stack([self.fx * pc[:, 0] / z + self.cx,
                       self.fy * pc[:, 1] / z + self.cy], axis=1)
        return uv, z

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """(N,2) native pixels -> (N,3) unit ray directions in the base frame."""
        d = np.stack([(uv[:, 0] - self.cx) / self.fx,
                      (uv[:, 1] - self.cy) / self.fy,
                      np.ones(len(uv))], axis=1)
        return (d / np.linalg.norm(d, axis=1, keepdims=True)) @ self.R.T


def amber_mask(rgb: np.ndarray):
    """(generous amber mask, saturation map). Level fits gate the mask by SAT_DEEP_*."""
    hsv = rgb_to_hsv(np.asarray(rgb, dtype=np.float32) / 255.0)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    amber = ((hue <= HUE_HI) | (hue >= HUE_LO)) & (sat >= SAT_MIN) & (val >= VAL_MIN)
    return amber, sat


# --------------------------------------------------------------------------------------
# liquid-body ray model
# --------------------------------------------------------------------------------------
def roi_from_pose(cam: Camera, pos, rot, pad_px: int = 12):
    """Image bbox of the cup's local AABB (handle included via total_length)."""
    lo = np.array([SPEC.handle_x_min - 0.004, -(SPEC.b_top_outer + 0.004), -0.004])
    hi = np.array([SPEC.tip_x + 0.004, SPEC.b_top_outer + 0.004, SPEC.height + 0.004])
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])
                        for c in (lo[2], hi[2])])
    uv, z = cam.project(corners @ rot.T + np.asarray(pos))
    if np.any(z <= 0.05):
        return None
    u0, v0 = np.floor(uv.min(0)).astype(int) - pad_px
    u1, v1 = np.ceil(uv.max(0)).astype(int) + pad_px
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, cam.w - 1), min(v1, cam.h - 1)
    if u1 <= u0 or v1 <= v0:
        return None
    return u0, u1, v0, v1


def z_on_map(cam: Camera, pos, quat_wxyz, roi, chord_min: float):
    """Per-pixel turn-on levels. Returns (z_first, z_deep, chord):
    z_first  lowest horizontal world plane L at which the ray touches ANY liquid
             {cavity < -inset, z < L} (the first cavity sample along the ray)
    z_deep   lowest L at which the ray crosses more than `chord_min` metres of
             liquid: the ceil(chord_min/step)-th smallest cavity-sample z
    chord    total ray-cavity chord (m); +inf levels where the ray misses."""
    u0, u1, v0, v1 = roi
    uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
    d = cam.rays(uv)
    o = cam.t
    rot = quat_to_mat(quat_wxyz)
    c = np.asarray(pos) + rot @ np.array([0.5 * (SPEC.tip_x - SPEC.a_top_outer), 0.0,
                                          0.5 * SPEC.height])
    rad = 0.085
    oc = o - c
    b = d @ oc
    disc = b * b - (oc @ oc - rad * rad)
    hit = disc > 0.0
    z_first = np.full(len(uv), np.inf)
    z_deep = np.full(len(uv), np.inf)
    chord = np.zeros(len(uv))
    if np.any(hit):
        sq = np.sqrt(disc[hit])
        t0, t1 = -b[hit] - sq, -b[hit] + sq
        s = np.linspace(0.0, 1.0, RAY_STEPS)
        pts = (o + (t0[:, None] + (t1 - t0)[:, None] * s[None, :])[..., None]
               * d[hit][:, None, :])                        # (M, K, 3) world samples
        local = (pts - np.asarray(pos)) @ rot               # world -> local
        inside = (cavity_sdf_local(local.reshape(-1, 3), SPEC) < -CAVITY_INSET)
        inside = inside.reshape(len(sq), RAY_STEPS)
        inside &= (local[..., 2] > SPEC.floor_z + 5e-4) & (local[..., 2] < SPEC.rim_z)
        step = (t1 - t0) / (RAY_STEPS - 1)
        chord[hit] = inside.sum(axis=1) * step
        wz = np.where(inside, pts[..., 2], np.inf)
        wz.sort(axis=1)
        z_first[hit] = wz[:, 0]
        m = np.ceil(chord_min / np.maximum(step, 1e-9)).astype(int).clip(1, RAY_STEPS - 1)
        z_deep[hit] = np.take_along_axis(wz, m[:, None], axis=1)[:, 0]
    shape = vv.shape
    return z_first.reshape(shape), z_deep.reshape(shape), chord.reshape(shape)


def fit_level(z_on: np.ndarray, chord: np.ndarray, observed: np.ndarray,
              domain_chord: float = 0.004, excluded: np.ndarray | None = None):
    """Level maximizing IoU between the liquid prediction {z_on < L} and the observed
    amber mask, over rays whose total cavity chord exceeds `domain_chord` (sliver
    rejection). Returns (level, iou, predicted mask at the level). NaN level when
    starved or the best IoU is below IOU_MIN. Excluded pixels (e.g. the falling
    stream) are removed from both the observation and prediction domains."""
    solid = (chord > domain_chord) & np.isfinite(z_on)
    if excluded is not None:
        solid &= ~excluded
    if solid.sum() < 40 or (observed & solid).sum() < 40:
        return np.nan, 0.0, np.zeros_like(observed)
    obs = observed & solid
    n_obs = obs.sum()
    z_obs = np.sort(z_on[obs])
    z_all = np.sort(z_on[solid])
    levels = np.arange(z_all[0], z_all[-1] + LEVEL_STEP, LEVEL_STEP)
    # nested masks: IoU(L) from two cumulative counts, no per-level mask builds
    inter = np.searchsorted(z_obs, levels, side="right")
    pred = np.searchsorted(z_all, levels, side="right")
    iou = inter / np.maximum(n_obs + pred - inter, 1)
    k = int(np.argmax(iou))
    if iou[k] < IOU_MIN:
        return np.nan, float(iou[k]), np.zeros_like(observed)
    level = float(levels[k])
    return level, float(iou[k]), solid & (z_on < level)


def source_volume_fn(pos, quat_wxyz, lattice_local: np.ndarray, cell_vol: float):
    """Volume of {cavity, world_z < L} for the tilted cup, via the local lattice."""
    rot = quat_to_mat(quat_wxyz)
    wz = lattice_local @ rot[2] + pos[2]
    wz = np.sort(wz)

    def vol(level: float) -> float:
        return float(np.searchsorted(wz, level) * cell_vol)

    return vol


def build_cavity_lattice(h: float = LATTICE_H):
    r = max(SPEC.a_top_inner, SPEC.b_top_inner)
    xs = np.arange(-r + 0.5 * h, SPEC.tip_x, h)
    ys = np.arange(-r + 0.5 * h, r, h)
    zs = np.arange(SPEC.floor_z + 0.5 * h, SPEC.rim_z, h)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = cavity_sdf_local(g, SPEC) < 0.0
    return g[keep].astype(np.float64), h**3


def rim_curve_local(spec=SPEC, half_width: float = 0.024, n: int = 33):
    """The pouring-lip transverse profile: rim-curve points near the spout apex,
    local frame, parametrized by y in [-half_width, half_width]. The brink
    identification integrates the head h(y)^3 over this curve."""
    ys = np.linspace(-half_width, half_width, n)
    # rim curve: x on the (spout-displaced) inner sheet at z = rim
    a_i, b_i = spec.inner_semi_axes(spec.rim_z)
    x0 = a_i * np.sqrt(np.clip(1.0 - (ys / b_i) ** 2, 0.0, None))
    xs = x0 + spec.spout_dx(ys, np.full_like(ys, spec.rim_z))
    return np.stack([xs, ys, np.full_like(ys, spec.rim_z)], axis=1)


# --------------------------------------------------------------------------------------
# episode plumbing
# --------------------------------------------------------------------------------------
def load_all(ep_dir: Path):
    ep = load_episode(ep_dir, PRE_ROLL, HOLD_SECONDS)
    meta = ep["meta"]
    pour, ret, lift = recorded_pour_actions(ep_dir)
    t_send, t_ret = pour["t_send"], ret["t_ack"]
    t_lift = lift["t_send"] if lift is not None else t_ret + 60.0
    rows = [json.loads(ln) for ln in
            (ep_dir / "frames_side.jsonl").read_text().splitlines() if ln]
    return ep, meta, rows, t_send, t_ret, t_lift


def frame_iter(ep_dir: Path, rows: list[dict], t_lo: float, t_hi: float, stride: int):
    """Yield (row, rgb) for side frames with t_host in [t_lo, t_hi], every `stride`."""
    import imageio.v2 as imageio

    picked = [r for r in rows if t_lo <= r["t_host"] <= t_hi][::stride]
    want = {r["frame_idx"]: r for r in picked}
    if not want:
        return
    reader = imageio.get_reader(ep_dir / "side_rgb.mp4")
    last = max(want)
    for idx, frame in enumerate(reader):
        if idx in want:
            yield want[idx], frame
        if idx >= last:
            break
    reader.close()


# --------------------------------------------------------------------------------------
# main extraction
# --------------------------------------------------------------------------------------
def run(ep_dir: Path, stride: int = 1, video: bool = True, probe: bool = False,
        t_pre: float = 6.0, t_post: float = 8.0, table_z: float = TABLE_Z,
        receiver_xy=RECEIVER_XY, cup_shift_xy=CUP_SHIFT_XY,
        grasp_roll_deg: float = GRASP_ROLL_DEG, probe_times=None,
        cup_reference_pos=None, cup_reference_quat=None):
    out = OUT_ROOT / ep_dir.name
    out.mkdir(parents=True, exist_ok=True)
    ep, meta, rows, t_send, t_ret, t_lift = load_all(ep_dir)
    cam = Camera(meta, "side")
    arm = RecordedPanda(ep, write_cup_obj(SPEC, out / "cup_render.obj"),
                        height=64, width=64, max_geom=4000,
                        cup_shift_xy=cup_shift_xy, grasp_roll_deg=grasp_roll_deg,
                        cup_reference_pos=cup_reference_pos,
                        cup_reference_quat=cup_reference_quat)
    t0 = t_send - PRE_ROLL                       # twin's episode clock zero
    rcv_pos = np.array([receiver_xy[0], receiver_xy[1], table_z])
    lattice, cell_vol = build_cavity_lattice()

    # receiver turn-on map: static, computed once. The level uses the first-touch
    # model + generous amber mask (reads the settled end state at IoU ~0.96 on ep0001).
    rcv_roi = roi_from_pose(cam, rcv_pos, quat_to_mat(Q_RCV))
    rcv_zon, _rcv_deep, rcv_chord = z_on_map(cam, rcv_pos, Q_RCV, rcv_roi, CHORD_RCV)
    # onset watch: the floor crop, pixels lit once the level reaches the 30 mL graduation
    rcv_pool = ((rcv_chord > 0.004) & np.isfinite(rcv_zon)
                & (rcv_zon < table_z + SPEC.floor_z + POOL_DEPTH))

    t_lo, t_hi = t_send - t_pre, min(t_ret + t_post, t_lift - 0.2)
    tip_local = np.array([SPEC.tip_x, 0.0, SPEC.rim_z])

    if probe:
        probe_times = ([t_send + t for t in probe_times] if probe_times is not None else
                       [t_send - 0.5, t_send + 2.0, t_send + 3.2, t_ret + 4.0])
    frames_out = out / "_overlay"
    if video:
        import imageio.v2 as imageio
        frames_out.mkdir(exist_ok=True)
        for stale in frames_out.glob("f_*.png"):
            stale.unlink()

    rec: dict[str, list] = {k: [] for k in (
        "t", "frame_idx", "cup_pos", "cup_quat", "tilt_deg", "lip",
        "src_level", "src_vol", "src_vol_loose", "src_iou",
        "rcv_level", "rcv_vol", "rcv_iou", "rcv_pool_px")}
    zon_cache_pose = None
    t_start = time.time()
    n_done = 0

    it = (frame_iter(ep_dir, rows, t - 0.02, t + 0.02, 1) for t in probe_times) \
        if probe else [frame_iter(ep_dir, rows, t_lo, t_hi, stride)]
    for gen in it:
        for row, rgb in gen:
            t_abs = row["t_host"]
            t_rel = t_abs - t0
            amber, sat = amber_mask(rgb)
            p_cup, q_cup = arm.cup_pose_at(t_rel)
            rot_cup = quat_to_mat(q_cup)
            lip = rot_cup @ tip_local + p_cup
            tilt = arm.tilt_degrees(q_cup)

            # ---- source cup (diagnostic channel: film/lensing bias both fits) ---------
            src_roi = roi_from_pose(cam, p_cup, rot_cup)
            key = (np.round(p_cup, 4).tobytes(), np.round(q_cup, 4).tobytes())
            if zon_cache_pose is None or zon_cache_pose[0] != key:
                maps = z_on_map(cam, p_cup, q_cup, src_roi, CHORD_SRC)
                zon_cache_pose = (key, src_roi, maps)
            else:
                _, src_roi, maps = zon_cache_pose
            src_first, src_deep, src_chord = maps
            u0, u1, v0, v1 = src_roi
            vol_fn = source_volume_fn(p_cup, q_cup, lattice, cell_vol)
            src_obs = (amber & (sat >= SAT_DEEP_SRC))[v0:v1 + 1, u0:u1 + 1]
            src_level, src_iou, src_pred = fit_level(src_deep, src_chord, src_obs)
            src_vol = vol_fn(src_level) if np.isfinite(src_level) else np.nan
            lvl_loose, _, _ = fit_level(src_first, src_chord,
                                        amber[v0:v1 + 1, u0:u1 + 1])
            src_vol_loose = vol_fn(lvl_loose) if np.isfinite(lvl_loose) else np.nan

            # ---- receiver -------------------------------------------------------------
            ru0, ru1, rv0, rv1 = rcv_roi
            rcv_obs = amber[rv0:rv1 + 1, ru0:ru1 + 1]
            # exclude the falling stream: rays passing within 20 mm of the vertical
            # line through the lip's xy
            uu, vv = np.meshgrid(np.arange(ru0, ru1 + 1), np.arange(rv0, rv1 + 1))
            d = cam.rays(np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64))
            oxy, dxy = cam.t[:2] - lip[:2], d[:, :2]
            tt = -(dxy @ oxy) / np.maximum((dxy * dxy).sum(1), 1e-12)
            miss = np.linalg.norm(oxy[None, :] + tt[:, None] * dxy, axis=1)
            stream_band = (miss < 0.020).reshape(vv.shape)
            # Exclude stream rays from BOTH sides of IoU. Removing only their amber
            # pixels penalizes correctly predicted liquid there and biases the level
            # downward, especially for camera views with a wide projected stream band.
            rcv_level, rcv_iou, rcv_pred = fit_level(
                rcv_zon, rcv_chord, rcv_obs, excluded=stream_band)
            rcv_vol = (SPEC.cavity_volume(rcv_level - (table_z + SPEC.floor_z))
                       if np.isfinite(rcv_level) else np.nan)

            rec["t"].append(t_abs - t_send)      # pour clock: 0 = pour move sent
            rec["frame_idx"].append(row["frame_idx"])
            rec["cup_pos"].append(p_cup)
            rec["cup_quat"].append(q_cup)
            rec["tilt_deg"].append(tilt)
            rec["lip"].append(lip)
            rec["src_level"].append(src_level)
            rec["src_vol"].append(src_vol)
            rec["src_vol_loose"].append(src_vol_loose)
            rec["src_iou"].append(src_iou)
            rec["rcv_level"].append(rcv_level)
            rec["rcv_vol"].append(rcv_vol)
            rec["rcv_iou"].append(rcv_iou)
            rec["rcv_pool_px"].append(int((rcv_obs & rcv_pool).sum()))

            if probe or video:
                img = draw_overlay(rgb, amber, src_roi, src_pred, rcv_roi, rcv_pred,
                                   stream_band, cam, lip)
                name = (f"probe_{len(rec['t']):02d}_t{t_abs - t_send:+.2f}.png"
                        if probe else f"f_{n_done:05d}.png")
                import imageio.v2 as imageio
                imageio.imwrite((out if probe else frames_out) / name, img)
            n_done += 1
            if n_done % 120 == 0:
                el = time.time() - t_start
                print(f"  frame {n_done} t={t_abs - t_send:+6.2f}s "
                      f"src={src_vol * 1e6 if np.isfinite(src_vol) else -1:6.1f}mL "
                      f"rcv={rcv_vol * 1e6 if np.isfinite(rcv_vol) else -1:6.1f}mL "
                      f"[{el / n_done * 1000:.0f}ms/f]")

    # ---- pack + save -------------------------------------------------------------------
    n = len(rec["t"])
    payload = {k: np.asarray(v) for k, v in rec.items()}
    hit = np.where(payload["rcv_pool_px"] > POOL_PX_MIN)[0]
    onset = float(payload["t"][hit[0]]) if len(hit) else np.nan
    payload.update(t_send=t_send, t_ret_done=t_ret - t_send, table_z=table_z,
                   receiver_xy=np.asarray(receiver_xy), sigma=SIGMA_GLYCEROL,
                   cup_shift_xy=np.asarray(cup_shift_xy),
                   grasp_roll_deg=grasp_roll_deg, stride=stride,
                   rcv_onset_s=onset, pool_ml=POOL_ML, pool_px_min=POOL_PX_MIN)
    if cup_reference_pos is not None:
        payload["cup_reference_pos"] = np.asarray(cup_reference_pos)
    if cup_reference_quat is not None:
        payload["cup_reference_quat"] = np.asarray(cup_reference_quat)
    print(f"onset: first frame with >{POOL_PX_MIN} amber px below the {POOL_ML:.0f} mL "
          f"graduation at t = {onset:+.2f} s (pour clock)")
    npz = out / ("observations_probe.npz" if probe else "observations.npz")
    np.savez_compressed(npz, **payload)
    print(f"wrote {npz} ({n} frames)")

    if not probe:
        plot_levels(payload, out / "levels.png")
        if video:
            write_overlay_video(frames_out, out / "extract_overlay.mp4",
                                fps=max(1, 60 // stride))
    arm.close()
    return payload


# --------------------------------------------------------------------------------------
# overlay / plots
# --------------------------------------------------------------------------------------
def mask_boundary(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask)


def draw_overlay(rgb, amber, src_roi, src_pred, rcv_roi, rcv_pred, stream_band,
                 cam: Camera, lip):
    """Extraction proof: amber tint, fitted level boundaries, stream band, lip marker."""
    img = rgb.copy()
    img[amber] = (0.65 * img[amber] + 0.35 * np.array([255, 255, 0])).astype(np.uint8)
    u0, u1, v0, v1 = src_roi
    sb = mask_boundary(src_pred)
    img[v0:v1 + 1, u0:u1 + 1][sb] = (255, 0, 200)
    ru0, ru1, rv0, rv1 = rcv_roi
    rb = mask_boundary(rcv_pred)
    img[rv0:rv1 + 1, ru0:ru1 + 1][rb] = (0, 220, 255)
    img[rv0:rv1 + 1, ru0:ru1 + 1][stream_band & ~rcv_pred] //= 2
    uv, z = cam.project(lip[None, :])
    if z[0] > 0:
        u, v = round(uv[0, 0]), round(uv[0, 1])
        if 2 <= u < cam.w - 2 and 2 <= v < cam.h - 2:
            img[v - 2:v + 3, u - 2:u + 3] = (0, 255, 0)
    return np.ascontiguousarray(np.rot90(img, k=1 if cam.R[2, 0] > 0 else -1))


def plot_levels(p: dict, path: Path):
    t = p["t"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax = axes[0]
    ax.plot(t, p["src_vol"] * 1e6, label="source volume, deep fit [mL]",
            color="tab:orange")
    ax.plot(t, p["src_vol_loose"] * 1e6, label="source volume, loose fit [mL]",
            color="tab:red", alpha=0.5)
    ax.plot(t, p["rcv_vol"] * 1e6, label="receiver volume [mL]", color="tab:blue")
    tot = p["src_vol"] + p["rcv_vol"]
    ax.plot(t, tot * 1e6, ":", color="k", label="sum (deep + receiver)")
    ax.set_ylabel("mL")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t, p["src_iou"], label="source fit IoU", color="tab:orange")
    ax.plot(t, p["rcv_iou"], label="receiver fit IoU", color="tab:blue")
    ax.plot(t, p["tilt_deg"] / 100, "--", color="grey", label="tilt/100 [deg]")
    ax.set_ylabel("quality")
    ax.set_xlabel("t - t_send [s]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def write_overlay_video(frames_dir: Path, mp4: Path, fps: int):
    import imageio.v2 as imageio
    with imageio.get_writer(mp4, fps=fps, codec="libx264", quality=7,
                            macro_block_size=2,
                            output_params=["-movflags", "+faststart"]) as wtr:
        for f in sorted(frames_dir.glob("f_*.png")):
            wtr.append_data(imageio.imread(f))
    print("wrote", mp4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, default=REPO / "pouring_real_data" / "ep0001")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="overlay stills on 4 key frames, no full run")
    ap.add_argument("--probe-times", type=float, nargs="+", help="probe times after pour send")
    ap.add_argument("--table-z", type=float, default=TABLE_Z)
    ap.add_argument("--receiver-xy", type=float, nargs=2, default=RECEIVER_XY)
    ap.add_argument("--cup-shift", type=float, nargs=2, default=CUP_SHIFT_XY)
    ap.add_argument("--grasp-roll", type=float, default=GRASP_ROLL_DEG)
    ap.add_argument("--geometry-json", type=Path,
                    help="session geometry: table_z, receiver_xy, cup_reference_pos/quat (wxyz)")
    args = ap.parse_args()
    geo = json.loads(args.geometry_json.read_text()) if args.geometry_json else {}
    run(args.episode.resolve(), stride=args.stride, video=not args.no_video,
        probe=args.probe, table_z=geo.get("table_z", args.table_z),
        receiver_xy=geo.get("receiver_xy", args.receiver_xy),
        cup_shift_xy=args.cup_shift, grasp_roll_deg=args.grasp_roll,
        probe_times=args.probe_times, cup_reference_pos=geo.get("cup_reference_pos"),
        cup_reference_quat=geo.get("cup_reference_quat"))

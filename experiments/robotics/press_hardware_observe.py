"""Audit real pressing measurements and reconstruct visible geometry, without fitting a law.

RGB masks, depth samples and tentative image-feature correspondences are kept
separate from the optional axisymmetric silhouette interpretation. No interior
material motion, constant volume, no-slip condition or stress-free state is imposed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ['Play-Doh'] * 4 + ['butter slime'] * 4 + ['plasticine'] * 4
FROZEN = {('ep0005', 'side'), ('ep0010', 'side')}
CAMERA_ROI = {'side': (410, 299), 'hand': (462, 275)}


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def finite(value):
    return float(value) if np.isfinite(value) else None


def deproject(uv, depth, camera):
    k = camera['intrinsics']
    return np.column_stack(((uv[:, 0] - k['ppx']) / k['fx'] * depth,
                            (uv[:, 1] - k['ppy']) / k['fy'] * depth, depth))


def to_base(x, transform):
    return x @ transform[:3, :3].T + transform[:3, 3]


def force_audit(ep, dest):
    meta = read(ep / 'meta.json')
    records = jsonl(ep / 'states.jsonl')
    states = [r['state'] for r in records]
    absolute = np.array([r['t'] for r in records])
    active = np.array([s['force_control']['active'] for s in states])
    phase = np.array([s['force_control']['phase'] for s in states])
    t0 = absolute[np.flatnonzero(active & (phase == 'baseline'))[0]]
    t = absolute - t0
    action = next(a for a in jsonl(ep / 'actions.jsonl') if a['type'] == 'press')
    raw = np.array([s['ext_force_base'] for s in states])
    xyz = np.array([s['ee_pos'] for s in states])
    quat = np.array([s['ee_quat_xyzw'] for s in states])
    reported = np.array([s['force_control']['measured_n'] for s in states])
    baseline = float(action['baseline_raw_n'])
    load = meta['press']['force_sign'] * (raw[:, 2] - baseline)
    check = active & np.isin(phase, ['ramp', 'hold', 'ramp_down'])
    base = active & (phase == 'baseline')
    z0 = np.median(xyz[base, 2])
    np.savez_compressed(dest / 'robot_force.npz', time=t, time_host=absolute,
                        ee_pos=xyz, ee_quat_xyzw=quat, ext_force_base=raw,
                        incremental_normal_force=load, controller_force=reported,
                        controller_active=active, phase=phase)
    np.savetxt(dest / 'robot_force.csv', np.column_stack((t, load, reported, xyz, active)),
               delimiter=',', header='time_s,incremental_normal_force_N,controller_force_N,ee_x_m,ee_y_m,ee_z_m,controller_active', comments='')
    result = dict(material=MATERIALS[int(ep.name[2:])], t0_host=t0,
                  t0_definition='First recorded active baseline state; onset quantized by robot logging interval',
                  force_command_N=action['force_cmd_n'], baseline_raw_z_N=baseline,
                  baseline_std_N=float(np.std(raw[base, 2])),
                  controller_match_max_N=float(np.max(abs(load[check] - reported[check]))),
                  controller_match_rms_N=float(np.sqrt(np.mean((load[check] - reported[check]) ** 2))),
                  state_dt_median_s=float(np.median(np.diff(t))),
                  state_dt_max_s=float(np.max(np.diff(t))),
                  phase_intervals={name: [float(t[active & (phase == name)].min()),
                                         float(t[active & (phase == name)].max())]
                                   for name in ['baseline', 'ramp', 'hold', 'ramp_down']},
                  robot_sink_3_to_10_s_mm=float((np.interp(3, t, xyz[:, 2]) - np.interp(10, t, xyz[:, 2])) * 1000),
                  mean_force_3_to_4_s_N=float(np.mean(load[(t >= 3) & (t <= 4)])),
                  mean_force_9_to_10_s_N=float(np.mean(load[(t >= 9) & (t <= 10)])),
                  logged_sink_mm=action['sink_m'] * 1000,
                  robot_z_baseline_m=float(z0),
                  force_scope='Incremental vertical robot force relative to recorded contact baseline; no independent sensor calibration or tool-inertia correction',
                  inactive_controller_force_is_stale=True)
    assert np.all(np.diff(t) > 0)
    return meta, result, (t, xyz, load, active, phase)


def segment(image, depth, cam, material):
    """Fixed image ROIs checked visually; contact-edge pixels remain uncertain."""
    cx, bottom = CAMERA_ROI[cam]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 80) & (hsv[..., 2] > 100)
    flank = white[:, cx - 80:cx - 55].sum(1) + white[:, cx + 55:cx + 80].sum(1)
    rows = np.flatnonzero((flank > 30) & (np.arange(len(flank)) > 70) &
                         (np.arange(len(flank)) < bottom - 10))
    plate = int(rows[-1]) if len(rows) else 140
    if material == 'Play-Doh':
        valid = ((hsv[..., 0] < 15) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 45)
    elif material == 'butter slime':
        valid = (hsv[..., 0] > 15) & (hsv[..., 0] < 42) & (hsv[..., 1] > 100) & (hsv[..., 2] > 45)
    else:
        valid = (hsv[..., 2] < 105) & (hsv[..., 1] < 165)
        # Frozen depth cannot act as a dynamic segmentation cue.
        if depth is not None:
            valid &= (depth > 180) & (depth < 400)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[max(plate + 7, 100):bottom, cx - 90:cx + 90] = valid[max(plate + 7, 100):bottom, cx - 90:cx + 90]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    ids = [i for i in range(1, n) if stats[i, 4] > 150 and abs(centers[i, 0] - cx) < 45]
    mask[:] = 0
    if ids:
        j = max(ids, key=lambda i: stats[i, 4])
        contours, _ = cv2.findContours(np.uint8(labels == j), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 1, -1)
    return mask, plate


def sphere_reference(points):
    """Reference-frame geometric fit, not a stress-free or material assumption."""
    mean = points.mean(0)
    q = points - mean
    a = np.column_stack((2 * q, np.ones(len(q))))
    coef = np.linalg.lstsq(a, np.sum(q * q, axis=1), rcond=None)[0]
    radius = np.sqrt(max(1e-6, coef[3] + np.dot(coef[:3], coef[:3])))
    initial = np.r_[mean + coef[:3], np.clip(radius, .01, .06)]
    result = least_squares(lambda v: np.linalg.norm(points - v[:3], axis=1) - v[3],
                           initial, bounds=(np.r_[mean - .1, .008], np.r_[mean + .1, .07]),
                           loss='soft_l1', f_scale=.0015)
    return result.x[:3], result.x[3], np.sqrt(np.mean(result.fun ** 2))


def silhouette_profile(mask, camera, transform, center):
    """Intersect silhouette rays with the vertical plane through a fitted axis.

    This is an approximate axisymmetric shape interpretation. It does not
    conserve volume, establish correspondences, or fill observed clouds.
    """
    rows = np.flatnonzero(mask.sum(1) >= 8)
    if len(rows) < 8:
        return np.full(64, np.nan), np.full(64, np.nan), np.nan, np.nan
    left = np.array([np.flatnonzero(mask[y])[0] for y in rows])
    right = np.array([np.flatnonzero(mask[y])[-1] for y in rows])
    origin = transform[:3, 3]
    normal = center - origin
    normal[2] = 0
    normal /= np.linalg.norm(normal)
    uv = np.vstack((np.column_stack((left, rows)), np.column_stack((right, rows))))
    rays = deproject(uv, np.ones(len(uv)), camera) @ transform[:3, :3].T
    distances = np.dot(center - origin, normal) / (rays @ normal)
    points = origin + rays * distances[:, None]
    l, r = np.split(points, 2)
    z = (l[:, 2] + r[:, 2]) / 2
    radius = np.linalg.norm(l - r, axis=1) / 2
    order = np.argsort(z)
    z, radius = z[order], radius[order]
    grid = np.linspace(z[0], z[-1], 64)
    rr = np.interp(grid, z, radius)
    volume = np.trapezoid(np.pi * rr ** 2, grid)
    return grid, rr, volume, float((z[-1] - z[0]) * 1000)


def depth_at(points, depth, scale):
    out = np.full(len(points), np.nan)
    for j, (x, y) in enumerate(np.rint(points).astype(int)):
        if not (2 <= x < depth.shape[1] - 2 and 2 <= y < depth.shape[0] - 2):
            continue
        patch = depth[y - 1:y + 2, x - 1:x + 2].ravel().astype(float) * scale
        patch = patch[(patch > .18) & (patch < .4)]
        if len(patch) >= 5 and np.ptp(patch) <= .01:
            out[j] = np.median(patch)
    return out


def feature_step(previous, current, points, alive, mask, reference, seeds):
    ids = np.flatnonzero(alive)
    updated = points.copy()
    valid = np.zeros(len(points), bool)
    if not len(ids):
        return updated, valid
    src = points[ids].astype('float32').reshape(-1, 1, 2)
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, src, None,
                                        winSize=(21, 21), maxLevel=3)
    back, okb, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None,
                                         winSize=(21, 21), maxLevel=3)
    good = ok[:, 0].astype(bool) & okb[:, 0].astype(bool)
    good &= np.linalg.norm(back[:, 0] - src[:, 0], axis=1) < .7
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
    for j, idx in enumerate(ids):
        u, v = nxt[j, 0]
        if not np.isfinite([u, v]).all() or not (5 < u < current.shape[1] - 5 and 5 < v < current.shape[0] - 5):
            good[j] = False
            continue
        good[j] &= bool(eroded[int(round(v)), int(round(u))])
        if good[j]:
            a = cv2.getRectSubPix(reference, (9, 9), tuple(seeds[idx])).astype(float).ravel()
            b = cv2.getRectSubPix(current, (9, 9), (float(u), float(v))).astype(float).ravel()
            a -= a.mean(); b -= b.mean()
            ncc = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-10)
            good[j] &= ncc >= .7
    updated[ids] = nxt[:, 0]
    valid[ids[good]] = True
    return updated, valid


def camera_audit(ep, cam, meta, force, dest, force_data):
    frames = jsonl(ep / f'frames_{cam}.jsonl')
    hosts = np.array([r['t_host'] for r in frames])
    sensors = np.array([r['t_sensor_ms'] / 1000 for r in frames])
    relative = hosts - force['t0_host']
    chosen = np.flatnonzero((relative >= -.9) & (relative <= 15.1))[::6]
    assert len(chosen) > 100
    frozen = (ep.name, cam) in FROZEN
    camera = meta['cameras'][cam]
    transform = np.array(meta['extrinsics']['cameras'][cam]['T_base_cam']['matrix'])
    cap = cv2.VideoCapture(str(ep / f'{cam}_rgb.mp4'))
    cloud_sequence, image_sequence, masks, plates, depths, times = [], [], [], [], [], []
    coverage, areas, zlimits, centers2d, widths = [], [], [], [], []
    for idx in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, image = cap.read()
        assert ok
        depth = cv2.imread(str(ep / f'{cam}_depth/{idx:06}.png'), cv2.IMREAD_UNCHANGED)
        mask, plate = segment(image, None if frozen else depth, cam, force['material'])
        inner = cv2.erode(mask, np.ones((7, 7), np.uint8)).astype(bool)
        good = inner & (depth > 180) & (depth < 400) & (not frozen)
        y, x = np.nonzero(good)
        q = deproject(np.column_stack((x, y)), depth[y, x] * camera['depth_scale'], camera)
        q = to_base(q, transform)
        # Deterministic bounded-size samples; no unobserved points are created.
        selected = np.linspace(0, max(0, len(q) - 1), min(400, len(q))).astype(int)
        cloud = np.full((400, 3), np.nan, np.float32)
        cloud[:len(selected)] = q[selected]
        cloud_sequence.append(cloud)
        coverage.append(float(good.sum() / max(1, inner.sum())))
        ys, xs = np.nonzero(mask)
        areas.append(len(xs))
        centers2d.append([float(np.mean(xs)), float(np.mean(ys))] if len(xs) else [np.nan, np.nan])
        zlimits.append([float(ys.min()), float(ys.max())] if len(xs) else [np.nan, np.nan])
        widths.append(float(np.ptp(xs)) if len(xs) else np.nan)
        image_sequence.append(image)
        depths.append(depth)
        masks.append(mask)
        plates.append(plate)
        times.append(relative[idx])
    cap.release()
    times = np.array(times)
    clouds = np.array(cloud_sequence)
    initial = clouds[(times < .1)]
    points = initial[np.isfinite(initial).all(-1)]
    if len(points) > 100:
        center, radius, rms = sphere_reference(points)
        reference_fit = dict(center_base_m=center.tolist(), radius_mm=radius * 1000,
                             residual_rms_mm=rms * 1000,
                             scope='Approximate sphere fit to initial visible depth; camera frame independently anchored, no registration correction')
    else:
        center = radius = rms = None
        reference_fit = None
    profiles, profile_z, volumes, heights = [], [], [], []
    for mask in masks:
        if center is None:
            z, r, v, h = np.full(64, np.nan), np.full(64, np.nan), np.nan, np.nan
        else:
            z, r, v, h = silhouette_profile(mask, camera, transform, center.copy())
        profiles.append(r); profile_z.append(z); volumes.append(v); heights.append(h)

    reference = cv2.cvtColor(image_sequence[0], cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(reference, 200, .01, 5,
                                    mask=cv2.erode(masks[0], np.ones((11, 11), np.uint8)) * 255,
                                    blockSize=7)
    seeds = np.empty((0, 2), np.float32) if corners is None else corners[:, 0]
    points2d = seeds.copy()
    alive = np.ones(len(seeds), bool)
    tracks = np.full((len(times), len(seeds), 3), np.nan, np.float32)
    pixels = np.full((len(times), len(seeds), 2), np.nan, np.float32)
    accepted = np.zeros((len(times), len(seeds)), bool)
    previous = reference
    for i, image in enumerate(image_sequence):
        current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if i:
            points2d, alive = feature_step(previous, current, points2d, alive, masks[i], reference, seeds)
        pixels[i, alive] = points2d[alive]
        ids = np.flatnonzero(alive)
        if not frozen and len(ids):
            depth = depth_at(points2d[ids], depths[i], camera['depth_scale'])
            valid = np.isfinite(depth)
            tracks[i, ids[valid]] = to_base(deproject(points2d[ids[valid]], depth[valid], camera), transform)
            accepted[i, ids[valid]] = True
        previous = current
    profiles, profile_z, heights, volumes = map(np.asarray, (profiles, profile_z, heights, volumes))
    np.savez_compressed(dest / f'{cam}_observations.npz', time=times,
                        frame_idx=chosen, time_host=hosts[chosen], surface_points_base=clouds,
                        silhouette_radius_m=profiles, silhouette_z_base_m=profile_z,
                        silhouette_volume_m3=volumes, silhouette_height_mm=heights,
                        mask_area_px=areas, mask_centroid_px=centers2d, mask_y_limits_px=zlimits,
                        mask_width_px=widths, plate_white_edge_y_px=plates,
                        depth_valid_fraction=coverage, candidate_track_positions_base=tracks,
                        candidate_track_pixels=pixels, candidate_track_valid=accepted)
    target_times = [-.5, 1, 5, 10, 12, 14, 15]
    tiles = []
    cx, _ = CAMERA_ROI[cam]
    for t in target_times:
        i = int(np.argmin(abs(times - t)))
        image = image_sequence[i].copy()
        contours, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
        for uv in pixels[i, np.isfinite(pixels[i]).all(-1)]:
            cv2.circle(image, tuple(np.rint(uv).astype(int)), 2, (255, 120, 0), -1)
        crop = image[95:325, cx - 115:cx + 115]
        tile = np.full((254, 230, 3), 255, np.uint8); tile[24:] = crop
        cv2.putText(tile, f'{ep.name} {cam} {times[i]:.2f}s', (4, 16), 0, .43, (0, 0, 0), 1)
        tiles.append(tile)
    cv2.imwrite(str(dest / f'{cam}_segmentation.jpg'), np.hstack(tiles))
    if ep.name in ['ep0001', 'ep0004', 'ep0009']:
        writer = cv2.VideoWriter(str(dest / f'{cam}_observation_preview.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), 10, (400, 400))
        for i, image in enumerate(image_sequence):
            image = image.copy()
            contours, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            for uv in pixels[i, np.isfinite(pixels[i]).all(-1)]:
                cv2.circle(image, tuple(np.rint(uv).astype(int)), 2, (255, 120, 0), -1)
            tile = cv2.resize(image[85:325, cx - 120:cx + 120], (400, 400))
            cv2.putText(tile, f'{ep.name} {cam} t={times[i]:.2f}s', (8, 25), 0, .55, (0, 255, 255), 1)
            writer.write(tile)
        writer.release()
    host_latency = hosts - sensors
    selected_t = np.array(times)
    robot_t, xyz, load, active, phase = force_data
    result = dict(depth_frozen=frozen, frames_used=len(times), time_range_s=times[[0, -1]].tolist(),
                  host_minus_sensor_median_ms=float(np.median(host_latency) * 1000),
                  host_minus_sensor_p01_p99_ms=(np.percentile(host_latency, [1, 99]) * 1000).tolist(),
                  sensor_dt_max_ms=float(np.diff(sensors).max() * 1000),
                  depth_coverage_median=float(np.median(coverage)),
                  depth_coverage_min=float(np.min(coverage)),
                  initial_sphere=reference_fit, feature_candidates=len(seeds),
                  tracked_2d_at_s={str(t): int(np.isfinite(pixels[np.argmin(abs(times - t))]).all(-1).sum()) for t in [0, 1, 3, 5, 10, 14, 15]},
                  tracked_3d_at_s={str(t): int(accepted[np.argmin(abs(times - t))].sum()) for t in [0, 1, 3, 5, 10, 14, 15]},
                  shape_samples={str(t): dict(height_mm=finite(heights[np.argmin(abs(times - t))]),
                                             volume_ml=finite(volumes[np.argmin(abs(times - t))] * 1e6),
                                             top_image_y=finite(np.array(zlimits)[np.argmin(abs(times - t)), 0]),
                                             bottom_image_y=finite(np.array(zlimits)[np.argmin(abs(times - t)), 1]))
                                 for t in [0, 1, 3, 5, 10, 12, 14, 15]},
                  surface_cloud_scope='Visible eroded-mask depth only, in supplied calibration coordinates; no fusion or hole completion; mask contact edges may include background',
                  shape_scope='Axisymmetric silhouette volume is a diagnostic assumption, independently evaluated per camera; no conservation imposed. Contact-edge segmentation and calibration uncertainty prevent treating it as measured volume.',
                  depth_coverage_scope='Fraction within selected eroded mask; plasticine mask itself uses depth, so this is not independent whole-surface coverage.',
                  track_scope='Tentative RGB feature correspondences, forward/backward LK <0.7 px, reference 9x9 patch NCC >=0.7 and current object mask; not independently validated material-point identities')
    return result


def plot_episode(dest, force, force_data):
    t, xyz, load, active, phase = force_data
    window = (t >= -1) & (t <= 15.2)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].plot(t[window], load[window], label='Baseline-corrected robot force')
    axes[0, 0].axhline(force['force_command_N'], color='gray', ls=':', label='Command')
    axes[0, 0].set(ylabel='Force (N)', title='Force channel; unloading includes tool-motion bias')
    axes[0, 1].plot(t[window], (force['robot_z_baseline_m'] - xyz[window, 2]) * 1000)
    axes[0, 1].set(ylabel='Downward EEF displacement (mm)', title='Measured robot motion')
    for cam in ['side', 'hand']:
        d = np.load(dest / f'{cam}_observations.npz')
        tt = d['time']
        axes[0, 2].plot(tt, d['silhouette_height_mm'], label=cam)
        axes[1, 0].plot(tt, d['silhouette_volume_m3'] * 1e6, label=cam)
        axes[1, 1].plot(tt, d['depth_valid_fraction'], label=cam)
        axes[1, 2].plot(tt, d['candidate_track_valid'].sum(1), label=cam)
    axes[0, 2].set(title='Silhouette height diagnostic; edges uncertain', ylabel='Height (mm)')
    axes[1, 0].set(title='Volume diagnostic; NOT a measured volume', ylabel='Volume (mL)')
    axes[1, 1].set(title='Valid depth within selected specimen mask', ylabel='Fraction')
    axes[1, 2].set(title='Tentative feature tracks with valid depth', ylabel='Count')
    for ax in axes.ravel():
        ax.set_xlim(-1, 15.2); ax.set_xlabel('Time from first active baseline sample (s)')
        ax.axvspan(.3, 10.8, alpha=.06, color='orange'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); axes[0, 2].legend(); axes[1, 2].legend()
    fig.suptitle(f"{dest.name}: {force['material']}, {force['force_command_N']:g} N command")
    fig.savefig(dest / 'measurement_summary.png', dpi=140)
    plt.close(fig)


def process(root, out, episode):
    ep = root / episode
    dest = out / episode
    dest.mkdir(exist_ok=False)
    meta, result, force_data = force_audit(ep, dest)
    result['cameras'] = {}
    for cam in ['side', 'hand']:
        result['cameras'][cam] = camera_audit(ep, cam, meta, result, dest, force_data)
    a, b = [result['cameras'][cam]['initial_sphere'] for cam in ['side', 'hand']]
    if a is not None and b is not None:
        result['independent_initial_center_disagreement_mm'] = float(np.linalg.norm(np.array(a['center_base_m']) - b['center_base_m']) * 1000)
    save(dest / 'assessment.json', result)
    plot_episode(dest, result, force_data)
    print(episode, 'complete', 'creep mm', result['robot_sink_3_to_10_s_mm'],
          'camera center disagreement mm', result.get('independent_initial_center_disagreement_mm'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'press_real_data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', default=[f'ep{i:04}' for i in range(12)])
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    for episode in args.episodes:
        process(args.root, args.out, episode)


if __name__ == '__main__':
    main()

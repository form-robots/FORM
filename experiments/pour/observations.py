"""Extract the paper's 60-degree optical observations from the recorded video."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from matplotlib.patches import Rectangle
from examples import pour_recorded_twin as twin
from experiments.pour import pour_perception as perception
from reproduce.run import ROOT, read
MESH = ROOT / 'out/pour_wf/09-04-60-2s/cup_render.obj'
OUT = None

def extract(run, scene, hand_cup, grasp):
    episode_path = ROOT / "pouring_real_data" / run["episode"]
    folder = OUT / run["episode"]
    folder.mkdir()
    camera = perception.Camera(json.loads((episode_path / "meta.json").read_text()), "side")
    geom = run["receiver_geometry"]
    pos = np.r_[geom["receiver_xy"], geom["table_z"]]
    roi = list(map(int, perception.roi_from_pose(camera, pos, twin.R_CUP_REF)))
    roi[2:] = run.get("roi_native_v", [238, 260])
    u0, u1, v0, v1 = roi
    perception.RAY_STEPS = 4096
    perception.LEVEL_STEP = .0000078125
    z_on, _, chord = perception.z_on_map(camera, pos, twin.Q_RCV, roi, .001)
    rows = [json.loads(s) for s in (episode_path / "frames_side.jsonl").read_text().splitlines()]
    lo, hi = run["tilt_ack_s"] - 2., run["return_send_s"] + .5
    send = run["t_send"]
    times, levels, ious, frame_indices, previews = [], [], [], [], []
    preview_times = np.linspace(*run["fit_window_s"], 3)
    preview_rows = {min(rows, key=lambda r: abs(r["t_host"] - send - t))["frame_idx"]
                    for t in preview_times}
    for row, rgb in perception.frame_iter(episode_path, rows, send + lo, send + hi, 1):
        hsv = rgb_to_hsv(rgb[v0:v1+1, u0:u1+1] / 255.)
        mask = (((hsv[:, :, 0] < .115) | (hsv[:, :, 0] > .965))
                & (hsv[:, :, 1] > .55) & (hsv[:, :, 2] > .12))
        level, iou, predicted = perception.fit_level(z_on, chord, mask)
        times.append(row["t_host"] - send)
        levels.append(level); ious.append(iou); frame_indices.append(row["frame_idx"])
        if row["frame_idx"] in preview_rows:
            previews.append((rgb.copy(), predicted.copy(), times[-1], level, iou))
    times, levels, ious = np.array(times), np.array(levels), np.array(ious)
    volumes = np.array([twin.SPEC.cavity_volume(z - pos[2] - twin.SPEC.floor_z) * 1e6
                        if np.isfinite(z) else np.nan for z in levels])
    np.savez_compressed(folder / "optical_series.npz", t=times, level_m=levels,
                        receiver_ml=volumes, iou=ious, frame_idx=frame_indices, roi_native=roi)
    sampled = [r for r in rows if r["t_host"] >= send - 6.][::2]
    sampled = [r for r in sampled if send + lo <= r["t_host"] <= send + hi]
    obs_t = np.array([r["t_host"] - send for r in sampled])
    good = np.isfinite(volumes)
    if good.sum() < 12:
        raise ValueError(f"{run['episode']}: insufficient liquid visibility in fixed strip")
    obs_volume = np.interp(obs_t, times[good], volumes[good], left=np.nan, right=np.nan)
    depths = np.linspace(0, twin.SPEC.rim_z - twin.SPEC.floor_z, 2001)
    physical_volume = np.array([twin.SPEC.cavity_volume(d) * 1e6 for d in depths])
    episode = twin.load_episode(episode_path, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(episode, MESH, height=64, width=64, max_geom=4000)
    arm._hand_cup, arm._grasp = hand_cup.copy(), grasp.copy()
    poses = [arm.cup_pose_at(float(t + episode["t_pour"])) for t in obs_t]
    obs = dict(t=obs_t, frame_idx=np.array([r["frame_idx"] for r in sampled]),
               rcv_vol=obs_volume * 1e-6,
               rcv_level=pos[2] + twin.SPEC.floor_z + np.interp(obs_volume, physical_volume, depths),
               cup_pos=np.array([p for p, q in poses]), cup_quat=np.array([q for p, q in poses]),
               tilt_deg=np.array([arm.tilt_degrees(q) for p, q in poses]),
               lip=np.array([p + twin.quat_to_mat(q) @ np.array([twin.SPEC.tip_x, 0, twin.SPEC.rim_z])
                             for p, q in poses]),
               t_send=np.array(send), table_z=np.array(pos[2]), receiver_xy=pos[:2],
               cup_reference_pos=np.array(scene["cup_reference_pos"]),
               cup_reference_quat=np.array(scene["cup_reference_quat"]))
    arm.close()
    np.savez_compressed(folder / "observations.npz", **obs)
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    for ax, (rgb, pred, t, level, iou) in zip(axes, previews):
        whole = np.full(rgb.shape[:2], np.nan)
        whole[v0:v1+1, u0:u1+1] = pred.astype(float)
        ax.imshow(np.rot90(rgb))
        if pred.any():
            ax.contour(np.rot90(whole), levels=[.5], colors=["lime"], linewidths=1)
        ax.add_patch(Rectangle((v0, camera.w - 1 - u1), v1-v0, u1-u0,
                               fill=False, edgecolor="cyan", linewidth=1))
        ax.set(xlim=(110, 300), ylim=(590, 350), title=f"t={t:.2f} s; IoU={iou:.2f}")
        ax.axis("off")
    fig.suptitle(f"{run['angle_deg']}°: fixed optical strip (cyan), fitted liquid boundary (green)")
    fig.tight_layout(); fig.savefig(folder / "optical_check.png", dpi=150); plt.close(fig)
    in_fit = (times >= run["fit_window_s"][0]) & (times <= run["fit_window_s"][1])
    optical = dict(roi_native=roi, valid_fraction_fit=float(np.mean(good[in_fit])),
                   median_iou_fit=float(np.median(ious[in_fit])),
                   longest_valid_gap_s=float(np.max(np.diff(times[good]))),
                   number_of_optical_frames=len(times))
    return obs, optical

def main(output):
    global OUT
    OUT = output
    OUT.mkdir(parents=True, exist_ok=False)
    episode = ROOT / 'pouring_real_data/09-04-60-2s'
    scene = read(ROOT / 'out/pour_weakform_recovery/identified/geometry.json')
    geometry = read(ROOT / 'out/pour_wf/09-04-60-2s/endpoint_geometry.json')
    pour, ret, _ = twin.recorded_pour_actions(episode)
    ack, back = pour['t_ack'] - pour['t_send'], ret['t_send'] - pour['t_send']
    run = dict(episode=episode.name, angle_deg=60, receiver_geometry=geometry,
               t_send=pour['t_send'], tilt_ack_s=ack, return_send_s=back,
               fit_window_s=[ack + .15, back - .15])
    ep = twin.load_episode(episode, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ep, MESH, height=64, width=64, max_geom=4000,
                            cup_reference_pos=scene['cup_reference_pos'],
                            cup_reference_quat=scene['cup_reference_quat'])
    hand_cup, grasp = arm._hand_cup.copy(), arm._grasp.copy()
    arm.close()
    return extract(run, scene, hand_cup, grasp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)

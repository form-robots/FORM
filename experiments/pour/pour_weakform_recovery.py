"""Recover the September 60-degree spout-edge weak-form identification.

The receiver's unobstructed right side supplies V(t). The fit is exactly the
documented two-pass, integrated brink balance. No measured endpoint or other
pour enters the fit. The optical region is an explicit per-camera calibration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from examples.pour_recorded_twin import (
    SPEC,
    RecordedPanda,
    load_episode,
    quat_to_mat,
    recorded_pour_actions,
)

from experiments.pour.pour_weakform_identify import (
    brink_forcing,
    fit_eta_integrated,
    plot_brink_integrated,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "out/pour_weakform_recovery/identified"
EPISODE = ROOT / "pouring_real_data/09-04-60-2s"
WF = ROOT / "out/pour_wf/09-04-60-2s"


def fit(obs, window):
    first = brink_forcing(obs, 300.0, window)
    eta0, *_ = fit_eta_integrated(first)
    forcing = brink_forcing(obs, 300.0, window, eta_flight=eta0)
    eta, se, keep, dv, cumulative = fit_eta_integrated(forcing)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Weak-form fit must give a finite positive viscosity")
    rms = float(np.sqrt(np.mean((dv[keep] - cumulative[keep] / eta) ** 2)))
    return eta, se, keep, dv, cumulative, forcing, rms


def identify():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    optical_path = OUTPUT.parent / "side_strip_probe.npz"
    optical = dict(np.load(optical_path))
    original_path = WF / "observations_two_pose_diagnostic.npz"
    original = dict(np.load(original_path))
    pour, ret, _ = recorded_pour_actions(EPISODE)
    ack = pour["t_ack"] - pour["t_send"]
    return_send = ret["t_send"] - pour["t_send"]
    window = (ack + 0.15, return_send - 0.15)
    receiver_geom = json.loads((WF / "endpoint_geometry.json").read_text())
    source_geom = json.loads((WF / "grasp_two_poses.json").read_text())
    scene = {k: source_geom[k] for k in ["cup_reference_pos", "cup_reference_quat"]}
    scene.update({k: receiver_geom[k] for k in ["receiver_xy", "table_z"]})
    # Rebuild poses from the recorded joints: the historical diagnostic NPZ kept
    # stale reference-pose metadata even when its per-frame arrays were updated.
    ep = load_episode(EPISODE, 1.0, 2.5)
    arm = RecordedPanda(ep, WF / "cup_render.obj", height=64, width=64, max_geom=4000,
                        cup_reference_pos=scene["cup_reference_pos"],
                        cup_reference_quat=scene["cup_reference_quat"])
    poses = [arm.cup_pose_at(float(t) + ep["t_pour"]) for t in original["t"]]
    original["cup_pos"] = np.array([p for p, q in poses])
    original["cup_quat"] = np.array([q for p, q in poses])
    original["tilt_deg"] = np.array([arm.tilt_degrees(q) for p, q in poses])
    original["lip"] = np.array([p + quat_to_mat(q) @ np.array([SPEC.tip_x, 0, SPEC.rim_z])
                                for p, q in poses])
    arm.close()
    for key in ["cup_reference_pos", "cup_reference_quat"]:
        original[key] = np.array(scene[key])
    depths = np.linspace(0, SPEC.rim_z - SPEC.floor_z, 2001)
    physical_volumes = np.array([SPEC.cavity_volume(d) * 1e6 for d in depths])

    def observations(saturation):
        obs = {k: v.copy() for k, v in original.items()}
        volume = optical[f"right_{saturation}"]
        valid = np.isfinite(volume)
        obs["rcv_vol"] = np.interp(obs["t"], optical["t"][valid], volume[valid],
                                    left=np.nan, right=np.nan) * 1e-6
        obs["rcv_level"] = scene["table_z"] + SPEC.floor_z + np.interp(
            obs["rcv_vol"] * 1e6, physical_volumes, depths)
        return obs

    obs = observations(0.55)
    eta, se, keep, dv, cumulative, forcing, rms = fit(obs, window)
    variations = []
    for lo, hi in [(ack, ack + 1.0), window, (ack + 0.45, return_send - 0.15),
                   (ack - 0.5, ack + 0.5)]:
        value, _, mask, delta, _, _, residual = fit(obs, (lo, hi))
        variations.append(dict(t_fit=[lo, hi], eta=value, rms_mL=residual,
                               n_kept=int(mask.sum()), dV_total_mL=float(delta[-1])))
    optical_eta, *_ = fit(observations(0.42), window)
    prefix = []
    for stop in np.arange(window[0] + 0.5, window[1] + 1e-8, 0.2):
        ep, sp, kp, vp, _ = fit_eta_integrated(forcing, t_hi=float(stop))
        prefix.append(dict(t_hi=float(stop), eta=ep, se=sp, n=int(kp.sum()),
                           dV_mL=float(vp[-1])))
    inputs = [original_path, optical_path, WF / "grasp_two_poses.json",
              WF / "endpoint_geometry.json", EPISODE / "states.jsonl",
              EPISODE / "actions.jsonl", EPISODE / "meta.json",
              EPISODE / "side_rgb.mp4", Path(__file__),
              Path(__file__).with_name("pour_weakform_identify.py"),
              Path(__file__).with_name("pour_perception.py"),
              ROOT / "src/warpmpm/geometry/measuring_cup.py",
              OUTPUT.parent / "side_strip_probe.py"]
    result = dict(channel="brink-integrated", eta=eta, eta_se=se, rms_mL=rms,
                  n_frames=len(dv), n_kept=int(keep.sum()), v0_ml=300.0,
                  t_fit=list(window), h_tip_max_mm=float(forcing["h_tip"].max()*1000),
                  dV_total_mL=float(dv[-1]), prefix_sweep=prefix, v0_scan=[],
                  method="PDF equations 8-14: measured head, two-pass time-weak brink fit",
                  identification_episode=EPISODE.name, calibration_pour_count=1,
                  measured_endpoints_used=[], validation_measurements_used=[],
                  viscosity_fitted_by_simulator=False, volume_rescaling_applied=False,
                  source_pose="two-pose external-contour fit on the same recording",
                  optical_region_upright_x=[238, 260], saturation_min=0.55,
                  optical_model="first-intersection cavity model on a fixed right-side ROI",
                  fit_window_rule="pour acknowledgement + 0.15 s to return send - 0.15 s",
                  window_sensitivity=variations,
                  alternate_saturation_042_eta=optical_eta,
                  status="weak-form candidate; physical accuracy and MPM transfer unvalidated",
                  caveats=["Optical region developed on this identification recording.",
                           "Statistical error excludes optical and geometry systematics.",
                           "Prior exploratory comparisons are disclosed in repository history."],
                  input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in inputs})
    np.savez_compressed(OUTPUT / "observations.npz", **obs)
    np.savez_compressed(OUTPUT / "brink_fit.npz", **forcing, keep=keep,
                        delta_volume_ml=dv, cumulative_forcing_ml_pa_s=cumulative)
    (OUTPUT / "geometry.json").write_text(json.dumps(scene, indent=2) + "\n")
    (OUTPUT / "identify.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_brink_integrated(forcing, eta, keep, dv, cumulative, result, OUTPUT / "identify.png")
    print(json.dumps({k: result[k] for k in ["eta", "eta_se", "rms_mL", "t_fit",
                     "n_kept", "window_sensitivity", "alternate_saturation_042_eta"]}), flush=True)
    return result


if __name__ == "__main__":
    identify()

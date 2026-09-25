"""Extract receiver volume from fixed, unobstructed side strips in the 60° video.

The cavity and camera are calibrated from external outlines. There is no liquid
endpoint input, area-ratio normalization, monotonic projection, or volume gain.
The left/both regions and alternate thresholds are retained as diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.colors
import numpy as np
from examples.pour_recorded_twin import Q_RCV, R_CUP_REF, SPEC

from experiments.pour.pour_perception import Camera, fit_level, roi_from_pose, z_on_map

ROOT = Path(__file__).resolve().parents[2]


def extract(output):
    episode = ROOT / "pouring_real_data/09-04-60-2s"
    calibration = ROOT / "out/pour_wf/09-04-60-2s"
    observations = dict(np.load(calibration / "observations_two_pose_diagnostic.npz"))
    geometry = json.loads((calibration / "endpoint_geometry.json").read_text())
    camera = Camera(json.loads((episode / "meta.json").read_text()), "side")
    position = np.r_[geometry["receiver_xy"], geometry["table_z"]]
    roi = roi_from_pose(camera, position, R_CUP_REF)
    u0, u1, v0, v1 = roi
    first_intersection, _, chord = z_on_map(camera, position, Q_RCV, roi, .001)
    _, vertical = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
    # The native video is sideways: native vertical is upright horizontal.
    left = (vertical >= 152) & (vertical <= 174)
    right = (vertical >= 238) & (vertical <= 260)
    regions = {"left": left, "right": right, "both": left | right}
    timestamps = [json.loads(line) for line in
                  (episode / "frames_side.jsonl").read_text().splitlines()]
    send = float(observations["t_send"])
    frames = [row for row in timestamps if 4.5 <= row["t_host"] - send <= 14.3][::3]
    times = []
    thresholds = [.3, .42, .55]
    result = {f"{name}_{threshold}": [] for name in regions for threshold in thresholds}
    reader = imageio.get_reader(episode / "side_rgb.mp4")
    try:
        for row in frames:
            rgb = np.asarray(reader.get_data(row["frame_idx"]))
            hsv = matplotlib.colors.rgb_to_hsv(rgb[v0:v1 + 1, u0:u1 + 1] / 255.)
            for name, region in regions.items():
                for threshold in thresholds:
                    mask = (((hsv[:, :, 0] < .115) | (hsv[:, :, 0] > .965))
                            & (hsv[:, :, 1] > threshold) & (hsv[:, :, 2] > .12))
                    level, _, _ = fit_level(first_intersection, chord, mask, excluded=~region)
                    volume = (SPEC.cavity_volume(level - position[2] - SPEC.floor_z) * 1e6
                              if np.isfinite(level) else np.nan)
                    result[f"{name}_{threshold}"].append(volume)
            times.append(row["t_host"] - send)
    finally:
        reader.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, t=np.asarray(times),
                        **{key: np.asarray(value) for key, value in result.items()})
    print(f"Wrote {len(times)} observations to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    extract(parser.parse_args().output)

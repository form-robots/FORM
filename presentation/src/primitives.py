"""Primitives for the FORM video presentation."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
from observation import (
    Canvas as Canvas,
    S as S,
    BG as BG,
    INK as INK,
    TEAL as TEAL,
    BLUE as BLUE,
    ORANGE as ORANGE,
    MUTED as MUTED,
    LINE as LINE,
    data as data,
    Projection,
    lerp_frame as lerp_frame,
    centered_positions,
    DISPLACEMENT,
)

TEXT_RECORDS = []


class ReviewCanvas(Canvas):
    def text(self, x, y, s, n=18, c=INK, bold=False, anchor=None):
        from observation import font

        bounds = self.d.textbbox((x * S, y * S), s, font=font(n, bold), anchor=anchor)
        TEXT_RECORDS.append({"text": s, "size_px": n, "bbox": [v / S for v in bounds]})
        super().text(x, y, s, n, c, bold, anchor)


def motion(c, source_time=1.3):
    D = data()
    source, _, _ = lerp_frame(D["hp"]["pos"], D["hp"]["times"], source_time)
    points = centered_positions(source, D["reference"])
    proj = Projection(D["display_pos"][D["hsel"][::3], ::3].reshape(-1, 3), (327, 196, 203, 132))
    xy = proj(points)
    dep = points @ proj.view
    brightness = 0.8 + 0.2 * (dep - dep.min()) / np.ptp(dep)
    displacement = np.linalg.norm(source - D["reference"], axis=1)
    colors = DISPLACEMENT(np.clip(displacement / D["disp_max_m"], 0, 1))[:, :3] * 255
    for j in np.argsort(dep):
        c.dot(xy[j], 0.85, tuple((colors[j] * brightness[j]).astype(int)))


def force(c, source_time=2.0):
    f = data()["force"]
    use = (f["time"] >= 0) & (f["time"] <= source_time)
    t, v = (f["time"][use], f["incremental_normal_force"][use])
    c.line([(76, 211), (76, 312), (252, 312)], "#a9bac3", 1.2)
    if len(t) > 1:
        c.line(np.c_[76 + t / 2 * 176, 312 - v / 12 * 101], ORANGE, 2.7)
    c.text(66, 201, "N", 16, MUTED, anchor="rt")
    c.text(68, 228, "10", 16, MUTED, anchor="rm")
    c.text(76, 319, "0", 16, MUTED, anchor="mt")
    c.text(252, 319, "2 s", 16, MUTED, anchor="mt")

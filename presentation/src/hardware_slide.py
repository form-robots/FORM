"""Hardware slide for the FORM video presentation."""

from functools import lru_cache
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageOps

MATERIALS = [
    ("play_doh", "Play-Doh", "#c93632", 75.7),
    ("butter_slime", "Butter slime", "#b88b00", 72.4),
    ("plasticine", "Plasticine", "#4c5149", 77.8),
]
DURATION = 30


def opacity(t, start):
    u = np.clip((t - start) / 0.6, 0, 1)
    return 0.06 + 0.94 * float(u * u * (3 - 2 * u))


@lru_cache(None)
def laws(root):
    return {
        m: json.loads((root / f"press_hardware/{m}/fit.json").read_text())["selected"]
        for m, _, _, _ in MATERIALS
    }


def compression_time(t):
    """Three-second forward-only loop: 0.2 s initial hold, 2.5 s at 0.8x,
    0.3 s compressed hold. Restart the recording, never reverse plastic flow.
    Only display timing changes; identification still uses its full source.
    """
    return float(np.clip((max(0, t) % 3 - 0.2) * 0.8, 0, 2.0))


def law_card(b, im):
    b.box(im, (812, 166, 1236, 380), b.WHITE, 12, b.LINE)
    b.text(im, (1024, 178), "Identified material laws", 24, b.TEAL, True, anchor="mt")
    b.text(im, (835, 215), "Dev. stress (kPa)", 19, b.INK)
    b.text(im, (1092, 215), "Stiffness", 18, b.INK, anchor="mt")
    b.text(im, (1180, 215), "Yield stress", 18, b.INK, anchor="mt")
    b.text(im, (1092, 240), "E (kPa)", 19, b.INK, anchor="mt")
    b.text(im, (1180, 240), "Y (kPa)", 19, b.INK, anchor="mt")
    x0, x1, y0, y1 = (859, 1026, 330, 247)

    def xy(e, s):
        return (x0 + (x1 - x0) * e / 0.4, y0 - (y0 - y1) * s / 30)

    d = ImageDraw.Draw(im)
    for stress in (0, 15, 30):
        y = xy(0, stress)[1]
        d.line((x0, y, x1, y), fill=b.LINE, width=1)
        b.text(im, (851, y), str(stress), 18, b.MUTED, anchor="rm")
    d.line((x0, y1, x0, y0, x1, y0), fill=b.MUTED, width=2)
    for strain in (0, 0.2, 0.4):
        b.text(im, (xy(strain, 0)[0], 335), f"{strain:g}", 18, b.MUTED, anchor="mt")
    b.text(im, (942, 357), "Dev. log strain", 18, b.INK, anchor="mt")
    curves = []
    for i, (m, _, color, _) in enumerate(MATERIALS):
        law = laws(b.ROOT)[m]
        E = law["E_Pa"] / 1000
        Y = law["Y_Pa"] / 1000
        slope = E / (1 + law["nu_assumed"])
        pts = [xy(0, 0), xy(Y / slope, Y), xy(0.4, Y)]
        curves.append((m, color, pts))
        y = 270 + 30 * i
        d.ellipse((1046, y + 3, 1053, y + 10), fill=color)
        b.text(im, (1092, y), f"{E:.2f}", 20, color, anchor="mt")
        b.text(im, (1180, y), f"{Y:.2f}", 20, color, anchor="mt")
    for m, color, pts in sorted(curves, key=lambda curve: curve[0] == "play_doh"):
        if m != "play_doh":
            d.line(pts, fill=color, width=3)
            continue
        distance = 0.0
        for a, z in zip(pts, pts[1:]):
            length = math.dist(a, z)
            offset = 0.0
            while offset < length:
                phase = distance % 14
                step = min(length - offset, 8 - phase if phase < 8 else 14 - phase)
                if phase < 8:
                    p = tuple((a[k] + (z[k] - a[k]) * offset / length for k in (0, 1)))
                    q = tuple((a[k] + (z[k] - a[k]) * (offset + step) / length for k in (0, 1)))
                    d.line((p, q), fill=color, width=3)
                offset += step
                distance += step


def draw(b, t, elapsed):
    im = Image.new("RGB", (b.W, b.H), b.BG)
    d = ImageDraw.Draw(im)
    prefix = "Results · Hardware / "
    b.text(im, (44, 22), prefix, 34, b.TEAL, True)
    b.text(
        im,
        (44 + d.textlength(prefix, font=b.font(34, True)), 22),
        "Pressing and shaping",
        34,
        b.INK,
        True,
    )
    b.text(
        im,
        (44, 79),
        "Identify material laws from pressing. Plan and execute shaping of fresh specimens into an X.",
        23,
        b.MUTED,
        width=1192,
    )
    d.line((44, 119, 1236, 119), fill=b.LINE, width=2)
    d.rectangle((0, 716, b.W, 719), fill=b.LINE)
    d.rectangle((0, 716, int(b.W * elapsed / b.TOTAL), 719), fill=b.TEAL)
    for j, (_, name, color, _) in enumerate(MATERIALS):
        b.text(im, (156 + 236 * j, 136), name, 22, color, True, anchor="mt")
    b.text(im, (44, 160), "Identification presses" if t < 6 else "Recorded pressing", 19, b.INK)
    b.SOURCES["hardware_identification"] = b.ASSETS / "hardware_identification.mp4"
    idpic = b.frame("hardware_identification", compression_time(t) * 149 / 250)
    evaluation_t = compression_time(t - 6)
    realpic = b.frame("real", evaluation_t)
    transition = np.clip((t - 6) / 0.4, 0, 1)
    for j in range(3):
        a = ImageOps.pad(idpic.crop((460 * j, 0, 460 * j + 448, 280)), (224, 130), color=b.BG)
        q = ImageOps.pad(realpic.crop((610 * j, 50, 610 * j + 600, 410)), (224, 130), color=b.BG)
        im.paste(Image.blend(a, q, float(transition)), (44 + 236 * j, 182))
    layer = im.copy()
    law_card(b, layer)
    b.arrow(layer, (737, 245), (812, 245), b.TEAL, 3)
    im = Image.blend(im, layer, opacity(t, 6))
    layer = im.copy()
    b.text(layer, (44, 319), "MPM prediction from identified material laws", 19, b.INK)
    simpic = b.frame("sim", evaluation_t)
    for j in range(3):
        b.fit(layer, simpic.crop((610 * j, 50, 610 * j + 600, 410)), (44 + 236 * j, 343, 224, 130))
    b.arrow(layer, (812, 365), (737, 365), b.TEAL, 3)
    im = Image.blend(im, layer, opacity(t, 6))
    layer = im.copy()
    b.arrow(layer, (1024, 383), (1024, 407), b.TEAL, 3)
    b.text(layer, (1024, 414), "Execute planned shaping", 23, b.INK, True, anchor="mt")
    b.SOURCES["hardware_execution"] = b.ASSETS / "hardware_execution.mp4"
    pic = b.frame("hardware_execution", max(0, t - 12))
    score_bottom = ImageDraw.Draw(layer).textbbox(
        (156, 635), "75.7%", font=b.font(22, True), anchor="mt"
    )[3]
    b.fit(layer, pic, (812, score_bottom - 210, 424, 210))
    im = Image.blend(im, layer, opacity(t, 12))
    layer = im.copy()
    b.text(layer, (44, 479), "Final shapes · target overlap (IoU)", 19, b.INK)
    for j, (key, _, color, iou) in enumerate(MATERIALS):
        b.fit(layer, b.still(key), (44 + 236 * j, 505, 224, 125))
        b.text(layer, (156 + 236 * j, 635), f"{iou:.1f}%", 22, b.INK, True, anchor="mt")
    b.arrow(layer, (806, 567), (746, 567), b.TEAL, 3)
    im = Image.blend(im, layer, opacity(t, 26))
    return im

"""Equations for the FORM video presentation."""

from paths import ASSET_ROOT

P = ASSET_ROOT / "identification"
from functools import lru_cache
import numpy as np
from PIL import Image, ImageColor
from primitives import (
    Canvas as Canvas,
    ReviewCanvas as ReviewCanvas,
    S as S,
    BG as BG,
    INK as INK,
    BLUE as BLUE,
    TEAL as TEAL,
    ORANGE as ORANGE,
    MUTED as MUTED,
    LINE as LINE,
    force as force,
    motion as motion,
    TEXT_RECORDS as TEXT_RECORDS,
)

LATEX = P / "revision_latex"
VOLUME = "#92733f"
BASIS = "#796399"
PARTICLES = "#5a788b"
CONNECTORS = []
MATH_BOUNDS = []


@lru_cache(None)
def latex_raster(index, width=None, height=None):
    path = LATEX / f"equation-{index:02d}.png"
    if not path.exists():
        path = LATEX / f"equation-{index}.png"
    im = Image.open(path).convert("RGBA")
    im = im.crop(im.getbbox())
    scale = width * S / im.width if width else height * S / im.height
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    return im


def tex(c, index, center, width=None, height=None):
    im = latex_raster(index, width, height)
    x, y = (round(center[0] * S - im.width / 2), round(center[1] * S - im.height / 2))
    c.im.paste(im, (x, y), im)
    MATH_BOUNDS.append(
        {"equation": index, "bbox": [x / S, y / S, (x + im.width) / S, (y + im.height) / S]}
    )
    anchors = {}
    a = np.asarray(im)
    for name, col in [
        ("load", ORANGE),
        ("volume", VOLUME),
        ("basis", BASIS),
        ("stress", BLUE),
        ("weight", TEAL),
        ("sum", PARTICLES),
        ("time", "#485862"),
    ]:
        mask = (abs(a[:, :, :3].astype(int) - ImageColor.getrgb(col)).max(2) < 5) & (
            a[:, :, 3] > 180
        )
        yy, xx = np.where(mask)
        if len(xx):
            anchors[name] = {
                "top": ((x + (xx.min() + xx.max()) / 2) / S, (y + yy.min()) / S),
                "bottom": ((x + (xx.min() + xx.max()) / 2) / S, (y + yy.max()) / S),
            }
    return anchors


def arrow(c, start, end, color, width=1.9, head=7):
    CONNECTORS.append({"from": start, "to": end, "color": color})
    c.arrow(start, end, color, width, head)

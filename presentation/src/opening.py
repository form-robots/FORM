"""Opening for the FORM video presentation."""

from paths import ASSET_ROOT

HERE = ASSET_ROOT / "opening"
from functools import lru_cache
from PIL import Image
from captions import caption_at, paint_caption

DURATION = 13
FADE_SECONDS = 0.4
REVEALS = [
    {"step": "01", "start_s": 3.0, "box": (40, 390, 554, 466)},
    {"step": "02", "start_s": 6.0, "box": (40, 490, 554, 591)},
    {"step": "03", "start_s": 10.0, "box": (40, 618, 554, 689)},
]


@lru_cache(maxsize=1)
def approved_frame():
    return Image.open(HERE / "opening_proposal_720p.png").convert("RGB")


def draw_opening(t):
    approved = approved_frame()
    out = approved.copy()
    for reveal in REVEALS:
        alpha = min(1.0, max(0.0, (t - reveal["start_s"]) / FADE_SECONDS))
        if alpha < 1:
            crop = approved.crop(reveal["box"])
            blank = Image.new("RGB", crop.size, "#f4f6f7")
            out.paste(Image.blend(blank, crop, alpha), reveal["box"][:2])
    body = out.crop((40, 119, 1240, 706))
    scale = (653 - 119) / body.height
    body = body.resize(
        (round(body.width * scale), round(body.height * scale)), Image.Resampling.LANCZOS
    )
    framed = Image.new("RGB", out.size, "#f4f6f7")
    framed.paste(out.crop((0, 0, 1280, 119)), (0, 0))
    framed.paste(body, ((1280 - body.width) // 2, 119))
    return paint_caption(framed, caption_at("opening", t))

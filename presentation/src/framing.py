"""Framing for the FORM video presentation."""

from paths import FONTS
from PIL import Image, ImageDraw, ImageFont
from captions import paint_caption

BG = "#f4f6f7"
INK = "#21333e"
TEAL = "#167b76"
LINE = "#d9e1e5"
SHIFT = 60


def frame_method(source, number, title, caption):
    scale = source.width / 1280
    out = Image.new("RGB", source.size, BG)
    top = round(135 * scale)
    body = source.crop((0, top, source.width, round(715 * scale)))
    out.paste(body, (0, round((135 - SHIFT) * scale)))
    draw = ImageDraw.Draw(out)

    def text(x, y, value, size, color, bold=False, anchor=None):
        font = ImageFont.truetype(
            str(FONTS) + "/Lato-" + ("Bold" if bold else "Regular") + ".ttf", round(size * scale)
        )
        xy = (round(x * scale), round(y * scale))
        box = draw.textbbox(xy, value, font=font, anchor=anchor)
        assert box[0] >= 0 and box[2] <= out.width, (value, box)
        draw.text(xy, value, font=font, fill=color, anchor=anchor)
        return draw.textlength(value, font=font) / scale

    prefix = f"FORM / {number:02d}: "
    width = text(44, 22, prefix, 34, TEAL, True)
    text(44 + width, 22, title, 34, INK, True)
    draw.line(
        [(round(44 * scale), round(65 * scale)), (round(1236 * scale), round(65 * scale))],
        fill=LINE,
        width=max(1, round(scale)),
    )
    return paint_caption(out, caption)

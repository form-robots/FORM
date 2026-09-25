"""Conclusion for the FORM video presentation."""

from paths import FONTS
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SCALE = (1280, 720, 25, 2)
DURATION = 12
REVEAL_TIMES = (0, 3, 6, 9)
FADE_SECONDS = 0.45
BG, INK, TEAL = ("#f4f6f7", "#21333e", "#167b76")
MUTED, LINE, HIGHLIGHT = ("#566975", "#d9e1e5", "#e7f1ef")
CAPTION_CUES = [
    (0, 3, "FORM identifies a material law from one robot interaction."),
    (3, 6, "We identify material laws without backpropagation."),
    (6, 9, "We use the law in MPM to predict motion and plan actions."),
    (9, 12, "We reuse the identified material law for new actions and geometries."),
]
CAPTION_RATE_LIMITS = {9: 23}
POINTS = [
    ("Recover material laws", "from one interaction.", "From observed motion and contact loads."),
    (
        "No repeated simulation or",
        "differentiation during identification.",
        "Fit coefficients using weak-form linear least squares.",
    ),
    (
        "Use the identified law",
        "directly in MPM.",
        "Shared discretization for identification and prediction.",
    ),
    (
        "Plan new actions and geometries",
        "with the same law.",
        "Demonstrated with elastic, elastoplastic, and fluid materials.",
    ),
]
ROWS = [
    (
        "NCLaw",
        "Neural constitutive model",
        ["5.2 × 10⁻⁴", "1.6 × 10⁻⁴", "1.2 × 10⁻⁴", "5.7 × 10⁻⁴"],
        ["1513.2", "1305.6", "1300.2", "1306.8"],
    ),
    (
        "Diff. system ID (oracle)",
        "Known constitutive model",
        ["1.2 × 10⁻⁸", "6.3 × 10⁻¹³", "1.6 × 10⁻¹⁰", "4.0 × 10⁻⁹"],
        ["715.8", "1132.8", "1028.4", "556.2"],
    ),
    (
        "FORM (Fn Enc.)",
        "Pretrained stress responses",
        ["3.5 × 10⁻⁴", "3.3 × 10⁻⁸", "2.9 × 10⁻⁶", "2.3 × 10⁻⁶"],
        ["5.2", "0.1", "5.3", "4.8"],
    ),
    (
        "FORM (Full)",
        "Full-information inputs",
        ["5.1 × 10⁻⁶", "1.4 × 10⁻⁸", "5.2 × 10⁻⁷", "2.3 × 10⁻⁶"],
        ["4.4", "2.4", "4.7", "2.2"],
    ),
]


@lru_cache(None)
def font(size, bold=False):
    weight = "Bold" if bold else "Regular"
    return ImageFont.truetype(f"{FONTS}/Lato-{weight}.ttf", round(size * SCALE))


class Canvas:
    def __init__(self, image=None):
        self.im = image if image is not None else Image.new("RGB", (W * SCALE, H * SCALE), BG)
        self.draw = ImageDraw.Draw(self.im)
        self.text_records = []

    def text(self, x, y, value, size=24, color=INK, bold=False, anchor="lt", max_width=None):
        face = font(size, bold)
        width = self.draw.textlength(value, font=face) / SCALE
        if max_width is not None:
            assert width <= max_width, (value, width, max_width)
        bbox = self.draw.textbbox((x * SCALE, y * SCALE), value, font=face, anchor=anchor)
        bbox = [v / SCALE for v in bbox]
        assert 0 <= bbox[0] < bbox[2] <= W and 0 <= bbox[1] < bbox[3] <= H, (value, bbox)
        self.draw.text((x * SCALE, y * SCALE), value, font=face, fill=color, anchor=anchor)
        self.text_records.append(dict(text=value, size=size, bbox=bbox))
        return width

    def line(self, bounds, color=LINE, width=1):
        self.draw.line(tuple((v * SCALE for v in bounds)), fill=color, width=round(width * SCALE))

    def rect(self, bounds, fill):
        self.draw.rectangle(tuple((v * SCALE for v in bounds)), fill=fill)


def table(c):
    c.text(44, 323, "Identification accuracy and solve time", 25, INK, True)
    c.text(1236, 329, "Table II · lower is better", 19, MUTED, anchor="rt")
    c.line((44, 362, 1236, 362), width=2)
    first_column = 324
    group_width = (1236 - first_column) / 4
    c.text(56, 399, "Method", 21, INK, True, anchor="lm")
    for j, material in enumerate(["Jell-O", "Sand", "Plasticine", "Water"]):
        left = first_column + j * group_width
        c.text(left + group_width / 2, 382, material, 22, INK, True, anchor="mm")
        c.text(left + group_width * 0.25, 412, "Mean loss ↓", 18, MUTED, anchor="mm")
        c.text(left + group_width * 0.75, 412, "Time (s) ↓", 18, MUTED, anchor="mm")
    c.line((44, 435, 1236, 435))
    for i, (name, explanation, losses, times) in enumerate(ROWS):
        top = 436 + i * 49
        if i >= 2:
            c.rect((44, top, 1236, top + 48), HIGHLIGHT)
            c.rect((44, top, 48, top + 48), TEAL)
        display_name = "FORM (learned bases)" if name == "FORM (Fn Enc.)" else name
        c.text(56, top + 5, display_name, 22, TEAL if i >= 2 else INK, True, max_width=260)
        c.text(56, top + 29, explanation, 18, MUTED, max_width=260)
        for j, (loss, time) in enumerate(zip(losses, times)):
            left = first_column + j * group_width
            c.text(left + group_width * 0.25, top + 25, loss, 20, INK, anchor="mm", max_width=108)
            c.text(left + group_width * 0.75, top + 25, time, 22, INK, anchor="mm", max_width=108)
        c.line((44, top + 48, 1236, top + 48))
    for j in range(4):
        x = first_column + j * group_width
        c.line((x, 370, x, 631))


def caption_at(t):
    for start, end, value in CAPTION_CUES:
        if start <= t < end:
            return value
    return CAPTION_CUES[-1][2]


def compose(t, full=False):
    c = Canvas()
    prefix_width = c.text(44, 22, "FORM / ", 34, TEAL, True)
    c.text(46 + prefix_width, 22, "From one interaction to new robot actions", 34, INK, True)
    c.line((44, 65, 1236, 65))
    for i, (line1, line2, support) in enumerate(POINTS):
        x, y = (44 if i % 2 == 0 else 654, 86 if i < 2 else 203)
        layer = Canvas(c.im.copy())
        layer.text(x, y + 3, f"{i + 1:02}", 22, TEAL, True)
        layer.text(x + 42, y, line1, 26, INK, True, max_width=540)
        layer.text(x + 42, y + 31, line2, 26, INK, True, max_width=540)
        layer.text(x + 42, y + 70, support, 20, MUTED, max_width=540)
        u = 1.0 if full else min(1.0, max(0.0, (t - REVEAL_TIMES[i]) / FADE_SECONDS))
        opacity = 0.07 + 0.93 * u * u * (3 - 2 * u)
        c.im = Image.blend(c.im, layer.im, opacity)
        c.draw = ImageDraw.Draw(c.im)
        c.text_records.extend(layer.text_records)
    table(c)
    value = caption_at(t)
    width = c.draw.textlength(value, font=font(24)) / SCALE
    assert width <= 1192
    c.draw.rounded_rectangle(
        tuple((v * SCALE for v in (640 - width / 2 - 14, 665, 640 + width / 2 + 14, 710))),
        radius=5 * SCALE,
        fill="#26353e",
    )
    c.text(640, 687.5, value, 24, "#ffffff", anchor="mm")
    return c


def draw_frame(t):
    return compose(t).im.resize((W, H), Image.Resampling.LANCZOS)

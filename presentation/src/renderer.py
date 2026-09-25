"""Renderer for the FORM video presentation."""

from paths import ASSET_ROOT, DATA, FONTS

ROOT = DATA
ASSETS = ASSET_ROOT / "illustrations"
import json, math, sys
from functools import lru_cache
from opening import draw_opening, DURATION as OPENING_DURATION
from observation import draw_method01, DURATION as OBSERVE_DURATION
from identification import draw_frame as draw_method02, DURATION as IDENTIFY_PLAN_DURATION
from conclusion import draw_frame as draw_takeaway, DURATION as TAKEAWAY_DURATION
from captions import ALL_SECTIONS, caption_at, paint_caption
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = (1280, 720, 25)
BG, WHITE, INK = ("#f4f6f7", "#ffffff", "#21333e")
MUTED, LINE, BLUE, ORANGE, TEAL = ("#566975", "#d9e1e5", "#2375aa", "#c96932", "#167b76")
FONT = str(FONTS) + "/"
SOURCES = {
    k: ROOT / "footage" / v
    for k, v in {
        "insert": "01_insertion_matched_swapped.mp4",
        "golf": "02_golf_matched_swapped.mp4",
        "pour": "03_pouring_glycerol_real_vs_simulation.mp4",
        "real": "04_pressing_real_red_yellow_gray.mp4",
        "sim": "05_pressing_simulation_red_yellow_gray.mp4",
    }.items()
}
TIMELINE = [
    ("opening", OPENING_DURATION, "Identify material laws. Plan robot actions."),
    ("observe", OBSERVE_DURATION, "Observe one interaction"),
    ("balance", IDENTIFY_PLAN_DURATION, "Identify the material law and plan robot actions"),
    ("insertion", 12, "Elastic rod insertion"),
    ("golf", 10, "Putting with a flexible club"),
    ("simshape", 15, "Plan plastic shaping"),
    ("hardware", 30, "Hardware pressing and shaping"),
    ("pouring", 26, "Identify and plan target-volume pouring"),
    ("takeaway", TAKEAWAY_DURATION, "From one interaction to new robot actions"),
]
TOTAL = sum((d for _, d, _ in TIMELINE))
assert [(name, duration) for name, duration, _ in TIMELINE] == ALL_SECTIONS
TEXT_LOG = []


@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype(FONT + ("Lato-Bold.ttf" if bold else "Lato-Regular.ttf"), size)


def text(im, xy, s, size=26, color=INK, bold=False, width=None, anchor=None):
    d = ImageDraw.Draw(im)
    f = font(size, bold)
    if width is not None:
        assert d.textlength(s, font=f) <= width, (s, size, width, d.textlength(s, font=f))
    bounds = d.textbbox(xy, s, font=f, anchor=anchor)
    assert bounds[0] >= 0 and bounds[1] >= 0 and (bounds[2] <= W) and (bounds[3] <= H), (s, bounds)
    d.text(xy, s, font=f, fill=color, anchor=anchor)
    TEXT_LOG.append({"text": s, "size": size, "bbox": bounds})


def lines(im, xy, ss, size=26, color=INK, bold=False, step=None, width=None):
    for i, s in enumerate(ss):
        text(im, (xy[0], xy[1] + i * (step or size + 9)), s, size, color, bold, width)


def box(im, bounds, fill=WHITE, radius=16, outline=None):
    ImageDraw.Draw(im).rounded_rectangle(bounds, radius, fill=fill, outline=outline, width=2)


def arrow(im, a, b, color=TEAL, width=4):
    d = ImageDraw.Draw(im)
    d.line([a, b], fill=color, width=width)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    l = 13
    d.polygon(
        [
            b,
            (b[0] - l * math.cos(angle - 0.5), b[1] - l * math.sin(angle - 0.5)),
            (b[0] - l * math.cos(angle + 0.5), b[1] - l * math.sin(angle + 0.5)),
        ],
        fill=color,
    )


@lru_cache(None)
def still(name):
    source = Image.open(ASSETS / (name + ".png")).convert("RGBA")
    out = Image.new("RGBA", source.size, WHITE)
    out.alpha_composite(source)
    return out.convert("RGB")


def fit(im, source, bounds):
    x, y, w, h = bounds
    scale = min(w / source.width, h / source.height)
    new = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS
    )
    im.paste(new, (round(x + (w - new.width) / 2), round(y + (h - new.height) / 2)))


class Clip:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        assert self.cap.isOpened(), path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = -1
        self.last = None

    def get(self, t):
        idx = min(self.n - 1, max(0, int(t * self.fps + 1e-06)))
        if idx == self.index:
            return self.last
        if idx != self.index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        assert ok, (idx, self.n)
        self.index = idx
        self.last = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return self.last


CLIPS = {}


@lru_cache(None)
def insertion_probe_data():
    with np.load(ASSETS / "insertion_probe.npz") as data:
        return {key: data[key] for key in data.files}


def insertion_probe(t):
    """A then B: each actual loading episode, followed by a neutral B hold."""
    data = insertion_probe_data()
    material = "A" if t < 2 else "B"
    local = t if t < 2 else t - 2
    index = round(min(1, max(0, local / 1.6)) * 40)
    rgb = data[material][index].astype(float)
    mask = data[material + "_mask"][index].astype(float) / 255
    tint = max(0, min(1, (4 - t) / 0.2))
    color = np.array([35, 117, 170] if material == "A" else [201, 105, 50])
    shade = rgb.mean(axis=2, keepdims=True) / 180
    colored = np.clip(shade * color, 0, 255)
    alpha = (0.72 * tint * mask)[..., None]
    result = np.clip(rgb * (1 - alpha) + colored * alpha, 0, 255).astype("uint8")
    return (Image.fromarray(result), material)


@lru_cache(None)
def insertion_math_token(source, size, color):
    """Common-size math raster with its baseline retained across color spans."""
    from matplotlib.mathtext import MathTextParser
    from matplotlib.font_manager import FontProperties
    from PIL import ImageColor

    if color in (INK, "#000000"):
        color = "#000000"
    raster = MathTextParser("agg").parse(
        "$" + source + "$", dpi=144, prop=FontProperties(size=size)
    )
    mask = np.asarray(raster.image).copy()
    rgba = np.empty((*mask.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = ImageColor.getrgb(color)
    rgba[:, :, 3] = mask
    tile = Image.fromarray(rgba).resize(
        (round(mask.shape[1] / 2), round(mask.shape[0] / 2)), Image.Resampling.LANCZOS
    )
    return (tile, raster.depth / 2)


def insertion_math(im, parts, center, baseline, size=22, max_width=224):
    tokens = [insertion_math_token(source, size, color) for source, color in parts]
    width = sum((tile.width for tile, _ in tokens))
    assert width <= max_width, (parts, width, max_width)
    x = round(center - width / 2)
    for tile, depth in tokens:
        im.paste(tile, (x, round(baseline - tile.height + depth)), tile)
        x += tile.width


def frame(key, t):
    if key not in CLIPS:
        CLIPS[key] = Clip(SOURCES[key])
    return CLIPS[key].get(t)


@lru_cache(None)
def shaping_identification():
    path = ASSETS / "shaping_identification.npz"
    with np.load(path) as data:
        frames = {k: data[k] for k in data.files}
    laws = {
        m: json.loads((ROOT / f"press_simulation/separated_{m}/identification.json").read_text())
        for m in "AB"
    }
    results = json.loads((ROOT / "shaping/execution_summary.json").read_text())["results"]
    errors = {(r["material"], r["planned_for"]): r["surface_mm"] for r in results}
    return (frames, laws, errors)


def shaping_law_graph(im, laws, t):
    """Exact scalar reduction of the saved Hencky/J2 law under monotone
    isochoric coaxial loading: ||dev tau|| = min(2 mu ||dev log F||, Y).
    The engine defines Y as the deviatoric Kirchhoff-stress norm, not the
    sqrt(3/2)-scaled equivalent stress. This is a constitutive response,
    not a force/strain measurement from the nonuniform pressing experiment.
    """
    box(im, (44, 411, 540, 648), fill=WHITE, radius=12, outline=LINE)
    text(im, (292, 422), "Identified material laws", 24, TEAL, True, anchor="mt")
    text(im, (66, 458), "Deviatoric stress (kPa)", 19, INK)
    text(im, (324, 458), "E: stiffness · Y: yield", 19, INK)
    d = ImageDraw.Draw(im)
    x0, x1, y0, y1 = (91, 302, 593, 491)

    def xy(strain, stress):
        return (x0 + (x1 - x0) * strain / 0.22, y0 - (y0 - y1) * stress / 12)

    for stress in (0, 6, 12):
        y = xy(0, stress)[1]
        d.line((x0, y, x1, y), fill=LINE, width=1)
        text(im, (82, y), str(stress), 18, MUTED, anchor="rm")
    d.line((x0, y1, x0, y0, x1, y0), fill=MUTED, width=2)
    for strain in (0, 0.1, 0.2):
        x = xy(strain, 0)[0]
        text(im, (x, 599), f"{strain:g}", 18, MUTED, anchor="mt")
    text(im, (195, 623), "Deviatoric log strain", 18, INK, anchor="mt")
    for m, y, start, color in [("A", 490, 1.86, BLUE), ("B", 568, 3.86, ORANGE)]:
        layer = im.copy()
        ld = ImageDraw.Draw(layer)
        E = laws[m]["E_pa"] / 1000
        Y = laws[m]["yield_pa"] / 1000
        slope = E / (1 + 0.3)
        ld.line([xy(0, 0), xy(Y / slope, Y), xy(0.22, Y)], fill=color, width=4)
        text(layer, (329, y), m, 21, color, True)
        text(layer, (361, y), f"E = {E:.2f} kPa", 20, color)
        text(layer, (361, y + 25), f"Y = {Y:.2f} kPa", 20, color)
        im = Image.blend(im, layer, max(0, min(1, (t - start) / 0.14)))
    return im


def draw_section_content(name, t):
    start = sum((d for n, d, _ in TIMELINE[: [n for n, _, _ in TIMELINE].index(name)]))
    elapsed = start + t
    if name == "opening":
        return draw_opening(t)
    if name == "observe":
        return draw_method01(t)
    if name == "balance":
        return draw_method02(t)
    if name == "insertion":
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        prefix = "Results · Simulation / "
        text(im, (44, 22), prefix, 34, TEAL, True)
        prefix_width = d.textlength(prefix, font=font(34, True))
        text(im, (44 + prefix_width, 22), "Elastic rod insertion", 34, INK, True)
        text(
            im,
            (44, 79),
            "Identify the material law from bending. Use it to insert the rod through the hole by controlling gripper height and tilt.",
            23,
            MUTED,
            width=1192,
        )
        d.line((44, 119, 1236, 119), fill=LINE, width=2)
        d.rectangle((0, 716, W, 719), fill=LINE)
        d.rectangle((0, 716, int(W * elapsed / TOTAL), 719), fill=TEAL)
        probe, active_material = insertion_probe(t)
        text(im, (44, 137), "Identify material law", 25, INK, True)
        text(im, (44, 172), "Same bending probe for A and B", 22, MUTED)
        fit(im, probe, (44, 250, 174, 265))
        if t < 4:
            text(
                im,
                (131, 515),
                f"Material {active_material}",
                22,
                BLUE if active_material == "A" else ORANGE,
                True,
                anchor="mt",
            )
        text(im, (131, 550), "Stereo motion", 25, INK, True, anchor="mt")
        text(im, (131, 581), "+ force", 25, INK, True, anchor="mt")
        arrow(im, (203, 428), (252, 428), TEAL, 3)
        box(im, (260, 238, 495, 604), fill=WHITE, radius=12, outline=LINE)
        text(im, (377, 257), "Identified", 23, TEAL, True, anchor="mt")
        text(im, (377, 287), "material laws", 23, TEAL, True, anchor="mt")
        for material, value, y, color, start in [
            ("A", "80.6", 352, BLUE, 1.6),
            ("B", "248.1", 403, ORANGE, 3.6),
        ]:
            values = im.copy()
            insertion_math(
                values,
                [
                    (f"\\sigma_{material}=", INK),
                    (value + "\\,\\mathrm{kPa}", color),
                    ("\\,T_\\nu(F)", INK),
                ],
                377,
                y,
            )
            im = Image.blend(im, values, max(0, min(1, (t - start) / 0.16)))
        ImageDraw.Draw(im).line((278, 428, 477, 428), fill=LINE, width=1)
        text(im, (377, 446), "Colored values:", 22, "#000000", anchor="mt")
        prefix = "identified stiffness "
        prefix_width = ImageDraw.Draw(im).textlength(prefix, font=font(22))
        symbol, depth = insertion_math_token("E", 22, INK)
        left = 377 - (prefix_width + symbol.width) / 2
        text(im, (left, 495), prefix, 22, "#000000", anchor="ls")
        im.paste(symbol, (round(left + prefix_width), round(495 - symbol.height + depth)), symbol)
        insertion_math(im, [("T_\\nu(F)", INK)], 377, 541, size=22, max_width=100)
        text(im, (377, 558), "Deformation response", 22, "#000000", anchor="mt")
        layer = im.copy()
        arrow(layer, (495, 428), (544, 428), TEAL, 3)
        pic = frame("insert", max(0, t - 4))
        for col, label in enumerate(["Matched ID", "Swapped ID"]):
            text(layer, (713 + 354 * col, 137), label, 25, INK, True, anchor="mt")
            text(
                layer,
                (713 + 354 * col, 169),
                ["Plan with own material model", "Plan with other material’s model"][col],
                21,
                MUTED,
                anchor="mt",
            )
        for row, material in enumerate("AB"):
            text(
                layer,
                (523, 312 + 229 * row),
                material,
                25,
                BLUE if row == 0 else ORANGE,
                True,
                anchor="mm",
            )
            for col in range(2):
                rect = (56 + 784 * col, 56 + 540 * row, 816 + 784 * col, 576 + 540 * row)
                fit(layer, pic.crop(rect), (544 + 354 * col, 202 + 229 * row, 338, 222))
        opacity = max(0, min(1, (t - 4) / 0.5))
        opacity = 0.07 + 0.93 * opacity * opacity * (3 - 2 * opacity)
        im = Image.blend(im, layer, opacity)
        return im
    if name == "golf":
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        prefix = "Results · Simulation / "
        text(im, (44, 22), prefix, 34, TEAL, True)
        prefix_width = d.textlength(prefix, font=font(34, True))
        text(im, (44 + prefix_width, 22), "Putting with a flexible club", 34, INK, True)
        text(
            im,
            (44, 79),
            "Reuse the previously identified material laws. Plan forward-stroke duration and aim angle to stop the ball in the target.",
            23,
            MUTED,
            width=1192,
        )
        d.line((44, 119, 1236, 119), fill=LINE, width=2)
        d.rectangle((0, 716, W, 719), fill=LINE)
        d.rectangle((0, 716, int(W * elapsed / TOTAL), 719), fill=TEAL)
        if "golf" not in CLIPS:
            CLIPS["golf"] = Clip(SOURCES["golf"])
        clip = CLIPS["golf"]
        source_t = min(t % 5 / 4.5, 1) * (clip.n - 1) / clip.fps
        pic = clip.get(source_t)
        for col, label in enumerate(["Matched ID", "Swapped ID"]):
            text(im, (368 + 584 * col, 137), label, 25, INK, True, anchor="mt")
            text(
                im,
                (368 + 584 * col, 169),
                ["Plan with own material model", "Plan with other material’s model"][col],
                21,
                MUTED,
                anchor="mt",
            )
        for row, material in enumerate("AB"):
            text(
                im,
                (53, 311 + 230 * row),
                material,
                25,
                BLUE if row == 0 else ORANGE,
                True,
                anchor="mm",
            )
            for col in range(2):
                rect = (34 + 790 * col, 113 + 466 * row, 818 + 790 * col, 425 + 466 * row)
                fit(im, pic.crop(rect), (84 + 584 * col, 198 + 230 * row, 568, 226))
        return im
    if name == "simshape":
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        prefix = "Results · Simulation / "
        text(im, (44, 22), prefix, 34, TEAL, True)
        text(
            im,
            (44 + d.textlength(prefix, font=font(34, True)), 22),
            "Plastic shaping",
            34,
            INK,
            True,
        )
        text(
            im,
            (44, 79),
            "Identify stiffness and yield stress from pressing. Plan six pinches to shape a larger block into an X.",
            23,
            MUTED,
            width=1192,
        )
        d.line((44, 119, 1236, 119), fill=LINE, width=2)
        d.rectangle((0, 716, W, 719), fill=LINE)
        d.rectangle((0, 716, int(W * elapsed / TOTAL), 719), fill=TEAL)
        frames, laws, errors = shaping_identification()
        material = "A" if t < 2 else "B"
        source_t = min(2, t if t < 2 else t - 2)
        index = min(50, round(source_t / 0.04))
        text(im, (44, 137), "Identify material laws", 25, INK, True)
        if t < 4:
            text(
                im, (404, 137), material, 25, BLUE if material == "A" else ORANGE, True, anchor="rt"
            )
        rgb = frames[material][index].astype(float)
        color = np.array([35, 117, 170] if material == "A" else [201, 105, 50])
        tint = np.clip(rgb.mean(2)[..., None] / 185 * color, 0, 255)
        amount = 1 - max(0, min(1, (t - 4) / 0.25))
        alpha = frames[material + "_mask"][index][..., None] / 255 * amount
        rgb = rgb * (1 - alpha) + tint * alpha
        fit(im, Image.fromarray(np.rint(rgb).astype("uint8")), (44, 174, 496, 208))
        arrow(im, (292, 384), (292, 405), TEAL, 3)
        im = shaping_law_graph(im, laws, t)
        layer = im.copy()
        arrow(layer, (545, 411), (640, 411), TEAL, 3)
        for col, label in enumerate(["Matched ID", "Swapped ID"]):
            text(layer, (799 + 280 * col, 137), label, 25, INK, True, anchor="mt")
        for row, m in enumerate("AB"):
            for col, model in enumerate([m, "B" if m == "A" else "A"]):
                key = f"shaping_hand_{m}_{model}"
                if key not in SOURCES:
                    SOURCES[key] = ASSETS / "shaping_hand" / f"{m}_plan_{model}.mp4"
                pic = frame(key, max(0, min(15.96, 2 * (t - 4))))
                fit(
                    layer, pic.crop((64, 0, 576, 440)), (678 + 280 * col, 178 + 246 * row, 242, 208)
                )
                if t >= 12:
                    prefix = "Surface error: "
                    value = f"{errors[m, model]:.3f} mm"
                    ld = ImageDraw.Draw(layer)
                    prefix_width = ld.textlength(prefix, font=font(21))
                    label_width = prefix_width + ld.textlength(value, font=font(21, True))
                    assert label_width <= 242
                    left = 799 + 280 * col - label_width / 2
                    y = 407 + 246 * row
                    text(layer, (left, y), prefix, 21, "#000000", anchor="ls")
                    text(layer, (left + prefix_width, y), value, 21, INK, True, anchor="ls")
            text(
                layer,
                (656, 280 + 246 * row),
                m,
                25,
                BLUE if m == "A" else ORANGE,
                True,
                anchor="mm",
            )
        opacity = max(0, min(1, (t - 4) / 0.5))
        opacity = 0.07 + 0.93 * opacity * opacity * (3 - 2 * opacity)
        im = Image.blend(im, layer, opacity)
        return im
    if name == "hardware":
        import hardware_slide

        return hardware_slide.draw(sys.modules[__name__], t, elapsed)
    if name == "pouring":
        import pouring_slide

        return pouring_slide.draw(sys.modules[__name__], t, elapsed)
    if name == "takeaway":
        return draw_takeaway(t)
    raise ValueError(name)


def draw_section(name, t):
    im = draw_section_content(name, t)
    if name not in ("opening", "observe", "balance", "takeaway"):
        paint_caption(im, caption_at(name, t))
    return im

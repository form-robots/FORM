"""Pouring slide for the FORM video presentation."""

from functools import lru_cache
import csv, json, hashlib
import numpy as np
from PIL import Image, ImageDraw

DURATION = 26
IDENTIFY = 2.0
PLAN_START = 6.0
LOWER_PLAYBACK_RATE = 1.2
CYCLE = 3.0
TRANSFER = 2.0
ARRIVAL = 2.5
WATER_START = 25.0
GLYCEROL = "#c96932"
WATER = "#2375aa"
CUP_CROP = (500, 1840, 5290, 2840)
CUP_BOUNDARIES = (500, 1440, 2140, 2820, 3590, 4350, 5290)


@lru_cache(None)
def cup_photos(root):
    folder = root / "pouring/photos"
    photo = folder / "all_6_levels.jpg"
    mapping = folder / "group_photo_mapping.csv"
    provenance = json.loads((folder / "group_photo_provenance.json").read_text())
    assert hashlib.sha256(photo.read_bytes()).hexdigest() == provenance["group_photo_sha256"]
    assert hashlib.sha256(mapping.read_bytes()).hexdigest() == provenance["mapping_csv_sha256"]
    rows = list(csv.DictReader(mapping.open()))
    assert [int(r["target_ml"]) for r in rows] == [60, 80, 100, 120, 140, 160]
    assert all((int(r["seed"]) == 2 for r in rows))
    photo = Image.open(photo).convert("RGB").crop(CUP_CROP)
    return (
        photo.resize((424, round(424 * photo.height / photo.width)), Image.Resampling.LANCZOS),
        rows,
    )


def draw_cups(b, im, t):
    photo, rows = cup_photos(b.ROOT)
    x, y = (396, 533)
    scale = photo.width / (CUP_CROP[2] - CUP_CROP[0])
    for i, row in enumerate(rows):
        left = round((CUP_BOUNDARIES[i] - CUP_CROP[0]) * scale)
        right = round((CUP_BOUNDARIES[i + 1] - CUP_CROP[0]) * scale)
        tile = photo.crop((left, 0, right, photo.height))
        background = Image.new("RGB", tile.size, b.BG)
        alpha = 0.06 + 0.94 * smooth((t - (PLAN_START + i * CYCLE + ARRIVAL)) / 0.35)
        im.paste(Image.blend(background, tile, alpha), (x + left, y))


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def reveal(t, start):
    return 0.06 + 0.94 * smooth((t - start) / 0.55)


@lru_cache(None)
def data(root):
    base = root / "pouring/measurements"
    rows = list(csv.DictReader((base / "summary.csv").open()))
    commands = list(csv.DictReader((base / "verified_commands.csv").open()))[:6]
    water = list(csv.DictReader((root / "pouring/water/water_receiver_readings.csv").open()))
    records = []
    for r, c, w in zip(rows, commands, water, strict=True):
        assert float(r["target_ml"]) == float(c["target_ml"]) == float(w["target_ml"])
        result = root / c["result_path"]
        metrics = result.parent / "planned_episode/metrics.csv"
        history = np.genfromtxt(metrics, names=True, delimiter=",")
        frozen = json.loads(result.read_text())
        assert abs(float(c["mpm_receiver_ml"]) - frozen["receiver_ml"]) < 1e-08
        assert abs(history["ml_rcv"][-1] - frozen["receiver_ml"]) < 0.01
        records.append(
            dict(
                target=float(r["target_ml"]),
                angle=float(c["command_angle_deg"]),
                mean=float(r["mean_ml"]),
                sd=float(r["sample_sd_ml"]),
                water=float(w["receiver_ml"]),
                history=history,
                result=result,
                metrics=metrics,
                eta=frozen["eta_pa_s"],
                contact=frozen["source_coulomb_friction"],
            )
        )
    return records


def state(t):
    elapsed = max(0, t - PLAN_START)
    index = min(5, int(elapsed / CYCLE))
    phase = elapsed - index * CYCLE
    return (index, phase)


def completed(t):
    return sum((t >= PLAN_START + i * CYCLE + ARRIVAL for i in range(6)))


@lru_cache(None)
def law_equation():
    import io
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties

    stream = io.BytesIO()
    math_to_image(
        "$\\tau = 2\\eta\\,\\mathrm{dev}\\,D$",
        stream,
        dpi=180,
        format="png",
        color="#21333e",
        prop=FontProperties(size=26),
    )
    stream.seek(0)
    return Image.open(stream).convert("RGBA")


def planning_frame(b, index, t):
    path = (
        b.ASSETS / "pouring_targets" / f"target_{int(data(b.ROOT)[index]['target'])}" / "target.mp4"
    )
    key = f"pouring_target_{index}"
    if key not in b.CLIPS:
        b.CLIPS[key] = b.Clip(path)
    return b.CLIPS[key].get(min(1.96, max(0, t)))


def video_panel(b, im, source, sim=False):
    offset = 744 if sim else 0
    crop = source.crop((36 + offset, 308, 756 + offset, 898))
    x = 906 if sim else 44
    b.fit(im, crop, (x, 166, 330, 188))


def identification(b, im):
    b.box(im, (446, 159, 834, 355), b.WHITE, 12, b.LINE)
    b.text(im, (640, 171), "Identified material law", 24, b.TEAL, True, anchor="mt")
    equation = law_equation()
    scale = 31 / equation.height
    equation = equation.resize((round(equation.width * scale), 31), Image.Resampling.LANCZOS)
    im.paste(equation, (640 - equation.width // 2, 210), equation)
    b.text(im, (640, 255), "η = 3.44 Pa·s · effective viscosity", 24, b.BLUE, anchor="mt")
    b.text(im, (640, 294), "τ: viscous stress", 24, b.INK, anchor="mt")
    b.text(im, (640, 324), "dev D: rate of shape change", 24, b.INK, anchor="mt")


def results_plot(b, im, t, records):
    d = ImageDraw.Draw(im)
    x0, x1, y0, y1 = (937, 1226, 610, 445)

    def xy(x, y):
        return (x0 + (x - 50) / 120 * (x1 - x0), y0 - (y - 50) / 140 * (y0 - y1))

    b.text(im, (1071, 413), "Measured volume (mL)", 24, b.INK, True, anchor="mt")
    for y in [60, 100, 140, 180]:
        py = xy(60, y)[1]
        d.line((x0, py, x1, py), fill=b.LINE, width=1)
        b.text(im, (927, py), str(y), 18, b.MUTED, anchor="rm")
    d.line((x0, y1, x0, y0, x1, y0), fill=b.MUTED, width=2)
    for r in records:
        b.text(im, (xy(r["target"], 60)[0], 617), str(int(r["target"])), 18, b.MUTED, anchor="mt")
    b.text(im, (1082, 641), "Target volume (mL)", 19, b.INK, anchor="mt")
    a = np.array(xy(50, 50))
    z = np.array(xy(170, 170))
    length = np.linalg.norm(z - a)
    for s in np.arange(0, length, 13):
        p = a + (z - a) * s / length
        q = a + (z - a) * min(length, s + 7) / length
        d.line([tuple(p), tuple(q)], fill="#9aa8b0", width=2)
    d.ellipse((950, 454, 959, 463), fill=GLYCEROL)
    b.text(im, (966, 448), "Glycerol · 5 trials", 19, GLYCEROL)
    for i, r in enumerate(records):
        appear = PLAN_START + i * CYCLE + ARRIVAL
        if t < appear:
            continue
        x, y = xy(r["target"], r["mean"])
        _, top = xy(r["target"], r["mean"] + r["sd"])
        _, bottom = xy(r["target"], r["mean"] - r["sd"])
        d.line((x, top, x, bottom), fill=GLYCEROL, width=2)
        d.line((x - 5, top, x + 5, top), fill=GLYCEROL, width=2)
        d.line((x - 5, bottom, x + 5, bottom), fill=GLYCEROL, width=2)
        rad = 5 + 6 * (1 - smooth((t - appear) / 0.35))
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=GLYCEROL, outline=b.WHITE, width=1)
    if t >= WATER_START:
        layer = im.copy()
        w = ImageDraw.Draw(layer)
        w.polygon([(955, 475), (949, 486), (961, 486)], fill=WATER)
        b.text(layer, (966, 471), "Water · 1 trial", 19, WATER)
        for r in records:
            x, y = xy(r["target"], r["water"])
            w.polygon([(x, y - 6), (x - 6, y + 5), (x + 6, y + 5)], fill=WATER)
        im.paste(Image.blend(im, layer, smooth((t - WATER_START) / 0.65)))


def draw(b, t, elapsed):
    records = data(b.ROOT)
    im = Image.new("RGB", (b.W, b.H), b.BG)
    d = ImageDraw.Draw(im)
    prefix = "Results · Hardware / "
    b.text(im, (44, 22), prefix, 34, b.TEAL, True)
    b.text(
        im,
        (44 + d.textlength(prefix, font=b.font(34, True)), 22),
        "Pouring a target volume",
        34,
        b.INK,
        True,
    )
    b.text(
        im,
        (44, 79),
        "Identify from one glycerol pour. Reuse the model to plan the tilt for each target volume.",
        23,
        b.MUTED,
        width=1192,
    )
    d.line((44, 119, 1236, 119), fill=b.LINE, width=2)
    b.box(im, (34, 128, 1246, 369), b.BG, 12, b.LINE)
    replay_end = 386 / 30 - 1 / 30
    source = b.frame("pour", min(t / PLAN_START, 1.0) * replay_end)
    b.text(im, (209, 135), "Recorded pour · 60°", 23, b.INK, True, anchor="mt")
    video_panel(b, im, source)
    layer = im.copy()
    identification(b, layer)
    b.text(layer, (1071, 135), "MPM replay · 60°", 23, b.INK, True, anchor="mt")
    video_panel(b, layer, source, True)
    b.arrow(layer, (324, 260), (446, 260), b.TEAL, 3)
    b.arrow(layer, (834, 260), (956, 260), b.TEAL, 3)
    im = Image.blend(im, layer, reveal(t, IDENTIFY))
    if t >= PLAN_START:
        t = PLAN_START + (t - PLAN_START) * LOWER_PLAYBACK_RATE
    layer = im.copy()
    for bounds in [(34, 405, 320, 661), (384, 405, 832, 661), (896, 405, 1246, 661)]:
        b.box(layer, bounds, b.BG, 12, b.LINE)
    ld = ImageDraw.Draw(layer)
    ld.line([(640, 355), (640, 384), (177, 384)], fill=b.TEAL, width=3)
    b.arrow(layer, (177, 384), (177, 405), b.TEAL, 3)
    b.text(layer, (177, 413), "Plan with MPM", 24, b.INK, True, anchor="mt")
    b.text(layer, (608, 413), "Hardware execution", 24, b.INK, True, anchor="mt")
    index, phase = state(t)
    r = records[index]
    replay = planning_frame(b, index, min(1.96, phase))
    b.fit(layer, replay, (48, 454, 258, 176))
    b.text(layer, (177, 641), f"Target: {r['target']:.0f} mL", 22, b.INK, anchor="mt")
    tilt_box = (502, 448, 714, 520)
    arrow_y = (tilt_box[1] + tilt_box[3]) / 2
    b.box(layer, tilt_box, b.WHITE, 12, b.LINE)
    b.text(layer, (608, 454), "Selected tilt", 20, b.MUTED, anchor="mt")
    done = completed(t)
    transferring = TRANSFER <= phase < ARRIVAL and t >= PLAN_START
    value_layer = layer.copy()
    if done:
        b.text(
            value_layer,
            (608, 480),
            f"{records[done - 1]['angle']:.2f}°",
            34,
            b.TEAL,
            True,
            anchor="mt",
        )
    else:
        b.text(value_layer, (608, 480), "θ", 34, b.TEAL, anchor="mt")
    previous_opacity = 1 - smooth((phase - TRANSFER) / 0.15) if transferring else 1
    layer.paste(Image.blend(layer, value_layer, previous_opacity))
    draw_cups(b, layer, t)
    b.arrow(layer, (320, arrow_y), (tilt_box[0], arrow_y), b.TEAL, 3)
    b.arrow(layer, (tilt_box[2], arrow_y), (896, arrow_y), b.TEAL, 3)
    if transferring:
        u = smooth((phase - TRANSFER) / (ARRIVAL - TRANSFER))
        x = 350 + (608 - 350) * u
        y = 450 + 30 * smooth(u / 0.55)
        size = round(24 + 10 * u)
        label = f"{r['angle']:.2f}°"
        width = ld.textlength(label, font=b.font(size, True))
        ld.rounded_rectangle(
            (x - width / 2 - 4, y - 2, x + width / 2 + 4, y + size + 2), radius=5, fill=b.WHITE
        )
        b.text(layer, (x, y), label, size, b.TEAL, True, anchor="mt")
    results_plot(b, layer, t, records)
    im = Image.blend(im, layer, reveal(t, PLAN_START))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 716, b.W, 719), fill=b.LINE)
    d.rectangle((0, 716, int(b.W * elapsed / b.TOTAL), 719), fill=b.TEAL)
    return im

"""Identification layout for the FORM video presentation."""

from equations import VOLUME as VOLUME, PARTICLES as PARTICLES
from PIL import Image
from equations import (
    Canvas,
    ReviewCanvas,
    S,
    BG,
    INK,
    BLUE,
    TEAL,
    ORANGE,
    MUTED,
    BASIS,
    force,
    motion,
    tex,
    arrow,
    TEXT_RECORDS,
    CONNECTORS,
    MATH_BOUNDS,
)


def fitted(c, source, bounds):
    x, y, w, h = bounds
    scale = min(w * S / source.width, h * S / source.height)
    pic = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS
    )
    c.im.paste(
        pic, (round((x + w / 2) * S - pic.width / 2), round((y + h / 2) * S - pic.height / 2))
    )


def route(c, points, color, width=1.4, arrowhead=False):
    """Explicit orthogonal routes and short term leaders, all audited for text."""
    if arrowhead:
        c.line(points[:-1], color, width)
    else:
        c.line(points, color, width)
    for a, b in zip(points, points[1:]):
        CONNECTORS.append(
            {"from": a, "to": b, "color": color, "kind": "process" if arrowhead else "leader"}
        )
    if arrowhead:
        c.arrow(points[-2], points[-1], color, width, 6)


def construction(c, source_time=None, operators=None, compact=False):
    if compact:
        text_start = len(TEXT_RECORDS)
        math_start = len(MATH_BOUNDS)
        line_start = len(CONNECTORS)
        scratch = Image.new("RGB", c.im.size, BG)
        anchors = construction(ReviewCanvas(scratch), source_time, operators=lambda *_: None)
        region = scratch.crop((67 * S, 239 * S, 579 * S, 586 * S))
        c.im.paste(region, (67 * S, 215 * S))
        TEXT_RECORDS[text_start:] = [r for r in TEXT_RECORDS[text_start:] if r["bbox"][1] >= 239]
        for record in TEXT_RECORDS[text_start:] + MATH_BOUNDS[math_start:]:
            record["bbox"][1] -= 24
            record["bbox"][3] -= 24
        for record in CONNECTORS[line_start:]:
            for key in ("from", "to"):
                x, y = record[key]
                record[key] = (x, y - 24)
        anchors = {
            name: {side: (xy[0], xy[1] - 24) for side, xy in ends.items()}
            for name, ends in anchors.items()
        }
        c.text(64, 187, "by weighting Newton’s law and integrating over space and time.", 17, INK)
        c.text(566, 455, "Weak balance", 15, TEAL, True, "rt")
        if operators is not None:
            operators(c, anchors, source_time)
        return anchors
    c.text(64, 189, "Motion and contact forces constrain internal stress.", 18, INK)
    c.text(64, 216, "Chosen test functions eliminate the pressure term.", 17.5, MUTED)
    c.rect((67, 239, 225, 464), BG, 7, edge="#dcb69d")
    c.text(146, 248, "Reconstructed", 17, BLUE, True, "mt")
    c.text(146, 268, "motion", 17, BLUE, True, "mt")
    c.text(146, 365, "Contact force", 17, ORANGE, True, "mt")
    for draw, crop, box in [
        (motion, (320, 192, 538, 335), (79, 289, 134, 74)),
        (force, (47, 191, 268, 337), (79, 387, 134, 58)),
    ]:
        src = Image.new("RGB", (1280 * S, 720 * S), BG)
        if source_time is None:
            draw(Canvas(src))
        else:
            draw(Canvas(src), source_time=source_time)
        fitted(c, src.crop(tuple((v * S for v in crop))), box)
    c.text(146, 447, "Motion + contact loads", 13.7, ORANGE, True, "mt")
    c.rect((250, 239, 366, 373), "#eaf0f3", 6)
    c.text(308, 248, "Material state", 16, BLUE, True, "mt")
    tex(c, 7, (308, 280), width=102)
    arrow(c, (225, 326), (250, 326), BLUE, 1.8, 5)
    for yy, symbol, first, second in [
        (299, "F:", "deformation", "gradient"),
        (338, "D:", "strain-rate", "tensor"),
    ]:
        c.text(260, yy, symbol, 13.5, MUTED, True)
        c.text(280, yy, first, 13.5, MUTED)
        c.text(280, yy + 16, second, 13.5, MUTED)
    c.rect((378, 239, 576, 373), "#efedf3", 6)
    c.text(477, 248, "Chosen stress basis", 16, BASIS, True, "mt")
    c.text(421, 272, "Analytic", 14.5, INK, anchor="mt")
    c.text(421, 289, "material laws", 14.5, INK, anchor="mt")
    c.text(390, 307, "Stress", 12.5, MUTED)
    c.line([(390, 323), (390, 353), (459, 353)], "#a1adbb", 1)
    c.line([(395, 350), (421, 328), (456, 328)], BASIS, 2.2)
    c.text(459, 357, "Strain", 12.5, MUTED, anchor="rt")
    c.text(471, 325, "or", 13, MUTED, anchor="mt")
    c.text(530, 272, "Function", 14.5, INK, anchor="mt")
    c.text(530, 289, "encoder", 14.5, INK, anchor="mt")
    layers = [
        [(504, 318), (504, 333), (504, 348)],
        [(528, 314), (528, 325), (528, 337), (528, 352)],
        [(552, 324), (552, 342)],
    ]
    for left, right in zip(layers, layers[1:]):
        for a in left:
            for b in right:
                c.line([a, b], "#b5a2c8", 0.8)
    for layer in layers:
        for point in layer:
            c.dot(point, 3.3, BASIS, "#fcfbfe")
    c.text(530, 357, "Pretrained offline", 11.5, MUTED, anchor="mt")
    c.rect((250, 377, 576, 464), "#eaf0f3", 7)
    c.text(413, 383, "Stress model", 18, BLUE, True, "mt")
    stress_anchors = tex(c, 9, (413, 421), width=250)
    c.text(413, 443, "θ: unknown material coefficients", 15.5, INK, anchor="mt")
    for key, start_y, color in [("stress", 373, BLUE), ("basis", 373, BASIS)]:
        x, y = stress_anchors[key]["top"]
        arrow(c, (x, start_y), (x, y - 7), color, 1.8, 6)
    c.rect((67, 473, 578, 585), "#eaf0f3", 9, edge="#91bab3")
    anchors = tex(c, 1, (330, 525), width=420)
    for key, col, start_y in [("load", ORANGE, 464), ("stress", BLUE, 464)]:
        x, y = anchors[key]["top"]
        arrow(c, (x, start_y), (x, y - 8), col, 1.8, 6)
    if operators is not None:
        operators(c, anchors, source_time)
        return anchors
    return anchors


def recovered_law(c):
    c.rect((641, 185, 868, 338), "#eaf0f3", 7)
    c.text(754, 195, "Identified law", 22, BLUE, True, "mt")
    c.line([(666, 229), (666, 313), (847, 313)], "#98aab5", 1.2)
    c.text(669, 223, "Stress", 14.5, MUTED)
    c.text(849, 319, "Strain", 15.5, MUTED, anchor="rt")
    c.line([(673, 306), (742, 246), (840, 246)], BLUE, 2.8)
    c.line([(742, 246), (742, 313)], "#bfd1dc", 1)
    c.text(711, 278, "E", 19, BLUE, True)
    c.text(832, 221, "Y", 19, BLUE, True, "mt")
    c.text(660, 319, "Schematic", 13.5, MUTED)


def identification_solve(c):
    """A single shared panel, read bottom-up from assembly to the solve."""
    c.rect((641, 369, 868, 703), BG, 7, edge="#ccd9e0")
    c.text(754, 389, "Linear", 22, INK, True, "mt")
    c.text(754, 416, "least-squares solve", 22, INK, True, "mt")
    tex(c, 4, (754, 471), width=220)
    arrow(c, (754, 535), (754, 510), BLUE, 2.2, 7)
    c.text(754, 551, "Assemble equations", 21, INK, True, "mt")
    tex(c, 6, (754, 601), width=130)
    c.text(654, 627, "A:", 15.5, BLUE, True)
    c.text(677, 627, "integrated stress bases", 14.5, MUTED)
    c.text(677, 645, "on observed motion", 14.5, MUTED)
    c.text(654, 664, "b:", 15.5, ORANGE, True)
    c.text(677, 664, "motion + contact loads", 14.5, MUTED)
    c.text(654, 683, "θ:", 15.5, INK, True)
    c.text(677, 683, "material coefficients", 14.5, MUTED)
    arrow(c, (754, 369), (754, 338), BLUE, 2.2, 7)

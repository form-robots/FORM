"""Opening layout for the FORM video presentation."""

from paths import ASSET_ROOT, FONTS

P = ASSET_ROOT / "opening"


def draw_static_opening():
    from PIL import Image, ImageDraw, ImageFont

    S = 2
    im = Image.new("RGB", (1280 * S, 720 * S), "#f4f6f7")
    d = ImageDraw.Draw(im)
    ink, teal, muted = ("#21333e", "#167b76", "#566975")
    text_boxes = []

    def text(x, y, s, size=24, color=ink, bold=False, anchor=None):
        f = ImageFont.truetype(
            str(FONTS) + "/Lato-" + ("Bold" if bold else "Regular") + ".ttf", size * S
        )
        bb = d.textbbox((x * S, y * S), s, font=f, anchor=anchor)
        assert 0 <= bb[0] < bb[2] <= im.width and 0 <= bb[1] < bb[3] <= im.height, (s, bb)
        d.text((x * S, y * S), s, font=f, fill=color, anchor=anchor)
        text_boxes.append((s, bb))

    def line(x0, y0, x1, y1, c, w=1):
        d.line((x0 * S, y0 * S, x1 * S, y1 * S), fill=c, width=w * S)

    text(44, 65, "FORM", 54, teal, True, "ls")
    text(
        227,
        65,
        "Robot Manipulation through Direct Material Law Identification",
        36,
        ink,
        True,
        "ls",
    )
    text(44, 97, "From Observed Response to Material laws", 20, teal, False, "ls")
    line(44, 112, 1236, 112, "#d9e1e5")
    text(44, 151, "Identify material laws.", 43, teal, True)
    text(44, 210, "Plan robot actions.", 43, teal, True)
    text(44, 289, "Same action. Different materials.", 27, ink)
    text(44, 326, "Different outcomes.", 27, ink)
    steps = [
        (392, "01", "Observe one interaction", "Material motion and contact forces"),
        (492, "02", "Recover an explicit material law", "Linear least squares using the weak form"),
        (
            620,
            "03",
            "Use that law to plan robot actions",
            "New tasks and geometries, without refitting",
        ),
    ]
    for y, num, title, subtitle in steps:
        text(44, y + 1, num, 22, teal, True)
        text(88, y, title, 27, ink, True)
        text(88, y + 37, subtitle, 24, muted)
    text(88, 566, "No differentiation through simulation", 23, muted)
    records = []
    x0, tw, dx = (676, 180, 188)
    rows = [
        (["Pinch to", "25 mm"], 153, 135, [2, 1, 0], ["Plasticine", "Play-Doh", "Butter slime"]),
        (["Press at", "10 N"], 305, 135, [5, 3, 4], None),
        (["Pour at", "10°/s"], 490, 210, [7, 8, 6], ["Glycerin", "90% glycerin", "Water"]),
    ]
    for heading, y, th, ids, labels in rows:
        for line_index, label in enumerate(heading):
            text(x0 - 18, y + th / 2 - 22 + line_index * 26, label, 23, ink, line_index == 0, "rm")
        for col, num in enumerate(ids):
            x = x0 + dx * col
            src = Image.open(P / "source_images" / f"figure1-{num:03d}.jpg").convert("RGB")
            rotation = 0
            if num >= 6:
                src = src.transpose(Image.Transpose.ROTATE_90)
                rotation = 90
            if num < 6:
                ch = src.width * th / tw
                rect = (0, round((src.height - ch) / 2), src.width, round((src.height + ch) / 2))
                pic = src.crop(rect).resize((tw * S, th * S), Image.Resampling.LANCZOS)
                im.paste(pic, (x * S, y * S))
                display = [x, y, tw, th]
            else:
                rect = (0, 0, src.width, src.height)
                height = round(tw * src.height / src.width)
                pic = src.resize((tw * S, height * S), Image.Resampling.LANCZOS)
                im.paste(pic, (x * S, y * S))
                display = [x, y, tw, height]
            material = (labels or ["Plasticine", "Play-Doh", "Butter slime"])[col]
            if labels:
                text(x + tw / 2, y - 29, material, 23, teal, True, "mt")
            records.append(
                {
                    "source_pdf_image": num,
                    "material": material,
                    "rotation_ccw": rotation,
                    "crop_in_oriented_pixels": rect,
                    "display_box": display,
                }
            )
    for i, (s, a) in enumerate(text_boxes):
        for t, b in text_boxes[i + 1 :]:
            assert not (max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])), (
                s,
                t,
            )
    return im.resize((1280, 720), Image.Resampling.LANCZOS)

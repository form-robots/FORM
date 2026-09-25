"""Captions for the FORM video presentation."""

from paths import FONTS
from functools import lru_cache
from PIL import ImageDraw, ImageFont
from conclusion import CAPTION_CUES as TAKEAWAY_CUES, DURATION as TAKEAWAY_DURATION

OPENING_CAPTION_RATE_LIMITS = {10: 22}
INSERTION_CAPTION_RATE_LIMITS = {8: 21}
SECTIONS = [("opening", 13), ("observe", 32), ("balance", 28)]
ALL_SECTIONS = SECTIONS + [
    ("insertion", 12),
    ("golf", 10),
    ("simshape", 15),
    ("hardware", 30),
    ("pouring", 26),
    ("takeaway", TAKEAWAY_DURATION),
]
CUES = {
    "opening": [
        (0, 3, "Different materials respond differently to the same action."),
        (3, 6, "We observe how the material moves during one interaction."),
        (6, 10, "Our method, FORM, identifies a material law."),
        (10, 13, "We use the identified law to plan new actions for new geometries."),
    ],
    "observe": [
        (0, 4, "First, we record the material as it deforms."),
        (4, 8, "With RGB-D, we reconstruct the shape assuming symmetry around the pressing axis."),
        (8, 12, "We keep the volume fixed, so the material spreads sideways as it flattens."),
        (12, 16, "We infer the flow inside from the surface motion, assuming no slip at contacts."),
        (16, 20, "Then we follow particles through this flow to recover their 3D paths."),
        (20, 24, "With stereo, we track the same texture in two views to recover surface motion."),
        (24, 28, "From the surface motion, we estimate how the interior moves."),
        (28, 32, "The motion and measured contact forces then go into material identification."),
    ],
    "balance": [
        (0, 4, "Which material law explains the motion we just observed?"),
        (4, 8, "We weight Newton’s law and integrate over space and time."),
        (8, 12, "Our test fields cancel the pressure term, leaving only deviatoric stress."),
        (12, 16, "We express stress using known responses and unknown coefficients."),
        (16, 20, "Linear least squares gives us the coefficients of the material law."),
        (20, 24, "Keeping the law fixed, we plan new robot actions in simulation."),
        (24, 28, "Finally, we execute that plan on the real material."),
    ],
    "insertion": [
        (0, 4, "From bending and force, we identify each material’s stiffness."),
        (4, 8, "Our task is rod insertion: a new action and geometry."),
        (
            8,
            12,
            "Matched ID uses the rod’s identified law; swapped ID uses the other material’s law.",
        ),
    ],
    "golf": [
        (0, 3, "We reuse the material laws identified from bending."),
        (3, 7, "We plan forward-stroke duration and aim angle to stop the ball on target."),
        (7, 10, "Matched plans succeed; swapped plans miss."),
    ],
    "simshape": [
        (0, 4, "From pressing, we identify stiffness and yield stress for each material."),
        (4, 12, "Keeping those laws fixed, we plan six pinches to shape a larger block into an X."),
        (12, 15, "Both matched plans give lower surface error."),
    ],
    "hardware": [
        (0, 6, "For each material, one press identifies stiffness and yield stress."),
        (6, 12, "We use those laws to predict a separate, lower-force press."),
        (12, 16, "Next, we plan four pinches to shape fresh specimens into an X."),
        (16, 21, "Here, the robot executes the plan on a fresh Play-Doh specimen."),
        (21, 26, "The identified laws stay fixed throughout planning and execution."),
        (26, 30, "After rigid alignment, footprint overlap is 72 to 78 percent."),
    ],
    "pouring": [
        (0, 6, "We identify an effective viscosity from a single glycerol pour."),
        (6, 14, "We keep it fixed and plan the cup’s tilt for each of six target volumes."),
        (14, 21 + 5 / 6, "Across all six targets, the largest mean error is 3.8 milliliters."),
        (21 + 5 / 6, 26, "Water overpours when we use the same commands planned for glycerol."),
    ],
    "takeaway": TAKEAWAY_CUES,
}


def caption_at(section, t):
    for start, end, caption in CUES[section]:
        if start <= t < end:
            return caption
    return CUES[section][-1][2] if t >= CUES[section][-1][1] else ""


@lru_cache(None)
def subtitle_font(size):
    return ImageFont.truetype(str(FONTS) + "/Lato-Regular.ttf", size)


def paint_caption(image, caption):
    """Readable one-line subtitles in reserved space below the diagrams."""
    if not caption:
        return image
    scale = image.width / 1280
    draw = ImageDraw.Draw(image)
    font = subtitle_font(round(24 * scale))
    width = draw.textlength(caption, font=font)
    assert width <= 1192 * scale, caption
    center = image.width / 2
    draw.rounded_rectangle(
        (
            center - width / 2 - 14 * scale,
            665 * scale,
            center + width / 2 + 14 * scale,
            710 * scale,
        ),
        radius=5 * scale,
        fill="#26353e",
    )
    draw.text((center, 687.5 * scale), caption, font=font, fill="#ffffff", anchor="mm")
    return image


def timestamp(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def export_captions(folder, sections=SECTIONS, stem="first_three_slides"):
    offset = 0
    subtitles = []
    transcript = []
    index = 1
    for section, duration in sections:
        transcript.append(section.upper())
        for start, end, caption in CUES[section]:
            subtitles.append(
                f"{index}\n{timestamp(offset + start)} --> {timestamp(offset + end)}\n{caption}\n"
            )
            transcript.append(f"{offset + start:05.1f}–{offset + end:05.1f}  {caption}")
            index += 1
        offset += duration
        transcript.append("")
    (folder / f"{stem}.srt").write_text("\n".join(subtitles), encoding="utf-8")
    (folder / f"{stem}_transcript.txt").write_text("\n".join(transcript) + "\n", encoding="utf-8")

"""Encoding and assembly, independent of the original research repository."""

import subprocess
import cv2
from paths import OUTPUT
from captions import export_captions, ALL_SECTIONS
import renderer


def render(name):
    folder = OUTPUT / "sections"
    folder.mkdir(parents=True, exist_ok=True)
    duration = next(d for n, d, _ in renderer.TIMELINE if n == name)
    target = folder / f"{name}.mp4"
    temporary = folder / f"{name}.rendering.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1280x720",
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "17",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(round(duration * renderer.FPS)):
            process.stdin.write(renderer.draw_section(name, index / renderer.FPS).tobytes())
            if index % 100 == 0:
                print(f"{name}: {index / 25:g}/{duration:g}s", flush=True)
        process.stdin.close()
        if process.wait():
            raise RuntimeError(f"ffmpeg failed for {name}")
        temporary.replace(target)
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise


def assemble():
    folder = OUTPUT / "sections"
    for name, duration, _ in renderer.TIMELINE:
        cap = cv2.VideoCapture(str(folder / f"{name}.mp4"))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == round(duration * 25), (name, "frame count")
        cap.release()
    listing = OUTPUT / "sections.txt"
    listing.write_text("".join(f"file 'sections/{name}.mp4'\n" for name, _, _ in renderer.TIMELINE))
    joined = OUTPUT / "assembly.temporary.mp4"
    target = OUTPUT / "ICRA2027_video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-map_metadata",
            "-1",
            str(joined),
        ],
        check=True,
    )
    total = renderer.TOTAL
    filters = f"[0:v]drawbox=x=0:y=716:w=iw:h=4:color=0xd9e1e5:t=fill[base];[base][1:v]overlay=x=-w+W*t/{total}:y=716:shortest=1[out]"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(joined),
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x167b76:s=1280x4:r=25:d={total}",
            "-filter_complex_threads",
            "2",
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "fast",
            "-threads",
            "6",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )
    joined.unlink()
    export_captions(OUTPUT, ALL_SECTIONS, "ICRA2027_video")
    print(target, flush=True)


def preview(names):
    folder = OUTPUT / "previews"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        duration = next(d for n, d, _ in renderer.TIMELINE if n == name)
        renderer.draw_section(name, duration - 0.4).save(folder / f"{name}.png")
        print(folder / f"{name}.png", flush=True)

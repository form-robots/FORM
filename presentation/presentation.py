#!/usr/bin/env python3
"""Render, assemble, or export the FORM presentation."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["slides", "render", "assemble", "all", "preview", "check"]
    )
    parser.add_argument("sections", nargs="*", help="Optional slide names for render/preview")
    args = parser.parse_args()
    if args.command == "check":
        from validate_package import check

        check()
        return
    if args.command == "slides":
        from export_slides import export

        export()
        return
    from video_io import render, assemble, preview
    import renderer

    names = args.sections or [n for n, _, _ in renderer.TIMELINE]
    unknown = set(names) - {n for n, _, _ in renderer.TIMELINE}
    if unknown:
        parser.error("Unknown slide names: " + ", ".join(sorted(unknown)))
    if args.command in ("render", "all"):
        for name in names:
            render(name)
    if args.command in ("assemble", "all"):
        assemble()
    if args.command == "preview":
        preview(names)
    if args.command == "all":
        from export_slides import export

        export()


if __name__ == "__main__":
    main()

"""Check the bundled inputs, section timing, and self-contained slide media."""

import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
import cv2
from paths import PACKAGE, OUTPUT
from captions import ALL_SECTIONS


def check():
    records = json.loads((PACKAGE / "asset_manifest.json").read_text())
    for record in records:
        file = PACKAGE / record["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"], file
    print(f"Asset hashes: {len(records)} passed", flush=True)
    for name, duration in ALL_SECTIONS:
        cap = cv2.VideoCapture(str(OUTPUT / "sections" / f"{name}.mp4"))
        assert cap.isOpened(), name
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == duration * 25, name
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280, name
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720, name
        assert cap.get(cv2.CAP_PROP_FPS) == 25, name
        cap.release()
    print("Section timing and resolution: passed", flush=True)
    deck = OUTPUT / "FORM_slides.pptx"
    if deck.exists():
        with zipfile.ZipFile(deck) as z:
            slide_names = [
                n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            ]
            assert len(slide_names) == len(ALL_SECTIONS)
            for filename in z.namelist():
                if filename.endswith(".rels"):
                    for rel in ET.fromstring(z.read(filename)):
                        assert rel.get("TargetMode") != "External", (filename, rel.attrib)
            for filename in slide_names:
                root = ET.fromstring(z.read(filename))
                ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
                conditions = root.findall(".//p:video/p:cMediaNode/p:cTn/p:stCondLst/p:cond", ns)
                assert conditions and all(c.get("delay") == "0" for c in conditions), filename
        print(
            "PowerPoint: nine slides, embedded media, simultaneous start conditions passed",
            flush=True,
        )

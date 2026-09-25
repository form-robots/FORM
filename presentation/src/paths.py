"""All inputs are bundled; outputs are separate from source assets."""

from pathlib import Path
import os

PACKAGE = Path(__file__).resolve().parent.parent
ASSET_ROOT = PACKAGE / "assets"
DATA = ASSET_ROOT / "data"
FONTS = ASSET_ROOT / "fonts"
OUTPUT = Path(os.environ.get("FORM_OUTPUT", PACKAGE / "outputs"))

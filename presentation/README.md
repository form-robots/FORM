# FORM presentation

Python composition code for the 2:58 video. Install the `presentation` data
bundle first. Requires Python 3.12 and `ffmpeg`.

```bash
uv pip install -r presentation/requirements.txt
cd presentation
python presentation.py preview
python presentation.py all
```

Outputs include the captioned MP4, still-slide PDF, and PowerPoint with editable
main titles and separately movable embedded video panels. Content inside each
panel remains rendered pixels. Edit `src/captions.py` for narration or the relevant
slide module for visuals. `render observe` rebuilds only the first methods slide.
The bundled inputs reproduce the animation without rerunning physics simulations.

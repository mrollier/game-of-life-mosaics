"""
Gradio web app for gol_mosaics.

Upload an image, pick a few settings, and get back a Conway's Game of Life
still-life mosaic with an Elementary Cellular Automaton (ECA) background.

Deployable as a free Hugging Face Gradio Space (`python app.py` runs it locally).

Design notes:
- Background removal is intentionally NOT performed here. rembg + onnxruntime +
  the 176 MB u2net model would make the free Space heavy and slow. Instead we ask
  users to upload a background-free image and always call the pipeline with
  remove_background=False. (Automatic removal may return in a future version.)
- The pattern libraries (levels 1-5) are loaded once at startup and shared across
  requests, since they are read-only after loading.
"""

import os
import re
import random
import tempfile
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image
import gradio as gr

from gol_mosaics import MosaicGenerator, PatternLibrary, ColorScheme
from gol_mosaics.eca import ECABackground

# --- Safety / resource limits -------------------------------------------------

# Guard against decompression-bomb images: refuse to decode anything absurdly
# large. The mosaic pixelates to a coarse grid, so huge inputs are never needed.
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP

# Inputs larger than this (longest side) are downscaled before processing. The
# pipeline resizes to the grid anyway, so this only saves memory/time.
MAX_INPUT_DIM = 1600

# Compute caps exposed in the UI.
LEVELS = [3, 4, 5]
DEFAULT_LEVEL = 4
MIN_GRID, MAX_GRID, DEFAULT_GRID = 40, 120, 60

# --- One-time startup work ----------------------------------------------------

# Preload every pattern library once. MosaicGenerator otherwise loads these
# lazily per instance, which would re-read the 19 MB level-5 file on each request.
PATTERN_LIBRARIES = {level: PatternLibrary.load(level) for level in (1, 2, 3, 4, 5)}

# Colour scheme UI labels. UGent and monochrome are deterministic; Warhol picks
# random pop colours every call.
UGENT = "UGent (yellow/blue)"
MONOCHROME = "Monochrome (black/white)"
WARHOL = "Warhol (random pop colours)"
MANUAL = "Manual (pick your own)"
COLOR_SCHEME_LABELS = [UGENT, MONOCHROME, WARHOL, MANUAL]

# Default manual colours = the UGent palette (a sensible starting point).
DEFAULT_MANUAL = ColorScheme.ugent()


def _to_hex(color: str) -> str:
    """Normalise a colour-picker value to '#RRGGBB'.

    gr.ColorPicker usually returns '#rrggbb', but can hand back 'rgb(...)' /
    'rgba(...)' or shorthand '#abc'; the renderer needs full 6-digit hex.
    """
    if not color:
        return "#000000"
    color = color.strip()
    if color.startswith("#"):
        h = color[1:]
        if len(h) == 3:  # #abc -> #aabbcc
            h = "".join(c * 2 for c in h)
        return "#" + h[:6].lower()
    if color.startswith("rgb"):
        r, g, b = (int(round(float(n))) for n in re.findall(r"[\d.]+", color)[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    return color


@lru_cache(maxsize=256)
def _warhol_for_seed(seed: int) -> ColorScheme:
    """A Warhol palette that stays fixed for a given seed.

    ColorScheme.warhol() draws from its own RNG (independent of numpy's global
    seed), so live tweaking would otherwise reshuffle the colours on every
    change. Caching by the effective seed keeps the palette stable while the
    user adjusts other settings, and a new seed (the "New variation" button)
    yields fresh colours. Bounded by lru_cache so it can't grow without limit.
    """
    return ColorScheme.warhol()


def _scheme_for(label: str, seed: int, manual_colors=None) -> ColorScheme:
    """Build the ColorScheme for a UI label, keeping Warhol stable per seed.

    manual_colors is a (gol_background, gol_pixel, eca_background, eca_pixel)
    tuple of colour-picker values, used only for the 'Manual' scheme.
    """
    if label == MANUAL:
        gol_bg, gol_px, eca_bg, eca_px = manual_colors
        return ColorScheme.custom(
            gol_background=_to_hex(gol_bg),
            gol_pixel=_to_hex(gol_px),
            eca_background=_to_hex(eca_bg),
            eca_pixel=_to_hex(eca_px),
        )
    if label == WARHOL:
        return _warhol_for_seed(seed)
    if label == MONOCHROME:
        return ColorScheme.monochrome()
    return ColorScheme.ugent()

# ECA rule dropdown: a curated set of interesting rules, plus a random option.
# Values are ints, or the sentinel "random" which lets the generator choose.
ECA_CHOICES = (
    [("Random (recommended)", "random")]
    + [(f"Rule {r} — complex", r) for r in ECABackground.COMPLEX_RULES]
    + [(f"Rule {r} — chaotic", r) for r in ECABackground.CHAOTIC_RULES]
)


def _bound_input(img: Image.Image) -> Image.Image:
    """Downscale very large uploads, preserving aspect ratio and mode."""
    width, height = img.size
    longest = max(width, height)
    if longest > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / longest
        img = img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    return img


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple:
    """Convert '#RRGGBB' to an (R, G, B, alpha) tuple."""
    h = _to_hex(hex_color).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _fit_to_aspect(img: Image.Image, target_ratio: float, fill_rgba: tuple
                   ) -> Image.Image:
    """Place the mosaic on a solid `fill_rgba` backdrop sized to target_ratio.

    The shorter axis is padded (never cropped) so width/height matches the
    original aspect ratio, and `fill_rgba` fills that padding. The mosaic itself
    is an opaque square (its rotation rim is the GoL background colour), so it is
    pasted centred on top of the backdrop.
    """
    width, height = img.size
    if width == 0 or height == 0:
        return img

    if width / height < target_ratio:        # too narrow -> widen (add columns)
        new_width, new_height = round(height * target_ratio), height
    else:                                     # too wide -> heighten (add rows)
        new_width, new_height = width, round(width / target_ratio)
    new_width, new_height = max(new_width, width), max(new_height, height)

    canvas = Image.new("RGBA", (new_width, new_height), fill_rgba)
    offset = ((new_width - width) // 2, (new_height - height) // 2)
    # The mosaic is opaque, so this simply centres it; the backdrop shows only
    # in the aspect-ratio padding around it.
    canvas.paste(img, offset, mask=img)
    return canvas


def render_mosaic(image, level, color_scheme, grid_size,
                  empty_tiles_cutoff, alpha_cutoff, eca_rule, seed, auto_seed,
                  gol_background, gol_pixel, eca_background, eca_pixel
                  ) -> Optional[Image.Image]:
    """Core generation: returns the RGBA mosaic (or None if there's no image).

    With no image yet (e.g. a live trigger fired before upload) it returns None
    silently rather than erroring, so adjusting controls is smooth. Genuine
    failures raise gr.Error.

    The effective seed keeps live tweaking stable: an explicit seed (if given)
    wins, otherwise the session's auto_seed is used, so changing one control
    only changes that aspect. The "New variation" button rerolls auto_seed.
    """
    if image is None:
        return None

    # Normalise and bound the settings.
    level = int(level)
    grid_size = int(grid_size)
    if grid_size % 2:  # the diamond layout requires an even grid
        grid_size += 1
    grid_size = max(MIN_GRID, min(MAX_GRID, grid_size))

    rule = None if eca_rule == "random" else int(eca_rule)
    effective_seed = int(seed) if seed not in (None, "") else int(auto_seed)
    manual_colors = (gol_background, gol_pixel, eca_background, eca_pixel)

    try:
        image = _bound_input(image)
        target_ratio = image.width / image.height  # original upload's aspect
        scheme = _scheme_for(color_scheme, effective_seed, manual_colors)
        # Seed before constructing the generator: with a "random" ECA rule the
        # rule is drawn in __init__ (before generate_from_pil reseeds), so this
        # is what keeps the background stable across live tweaks.
        np.random.seed(effective_seed)
        generator = MosaicGenerator(
            level=level,
            grid_size=grid_size,
            color_scheme=scheme,
            eca_rule=rule,
        )
        # Inject the preloaded library instead of reloading it from disk.
        generator._pattern_library = PATTERN_LIBRARIES[level]

        mosaic = generator.generate_from_pil(
            image,
            empty_tiles_cutoff=float(empty_tiles_cutoff),
            alpha_cutoff=float(alpha_cutoff),
            remove_background=False,  # the app never runs rembg
            seed=effective_seed,
        )
        # Fit to the original aspect ratio on a solid ECA-background backdrop
        # (fills the aspect padding around the opaque mosaic).
        return _fit_to_aspect(
            mosaic, target_ratio, _hex_to_rgba(scheme.eca_background)
        )
    except Exception as exc:  # surface a friendly message, keep the app alive
        raise gr.Error(f"Could not generate the mosaic: {exc}")


def generate(*args) -> Optional[str]:
    """UI handler: render the mosaic and save it so the download is named
    'gol-mosaic.png'. Returns a filepath (or None when there's no image).

    A unique temp directory per call avoids collisions between concurrent users
    while keeping the basename fixed for the download.
    """
    mosaic = render_mosaic(*args)
    if mosaic is None:
        return None
    out_path = os.path.join(tempfile.mkdtemp(prefix="gol_"), "gol-mosaic.png")
    mosaic.save(out_path)
    return out_path


ABOUT = """
### About these mosaics

Each tile is a **Game of Life still life** — a pattern that, under Conway's Game
of Life rules, never changes from one generation to the next. Darker areas of
your image are filled with denser still lifes, lighter areas with sparser ones,
so the portrait emerges from the arrangement of stable patterns.

The coloured backdrop behind the subject is an **Elementary Cellular Automaton
(ECA)** — a one-dimensional automaton (Wolfram's rules) evolved row by row.
"Complex" rules (e.g. 110) make structured patterns; "chaotic" rules (e.g. 30)
make noisier ones.

**Tip:** upload an image whose background has already been removed
(e.g. with [remove.bg](https://www.remove.bg)). Automatic background removal may
be added in a future version.
"""


def build_demo() -> gr.Blocks:
    """Build the Gradio interface.

    Layout: a compact controls column on the left and the large mosaic on the
    right, so the controls and result are visible together without scrolling.
    The mosaic regenerates live whenever a control changes.
    """
    with gr.Blocks(title="Game of Life Mosaics") as demo:
        gr.Markdown(
            "# Game of Life Mosaics\n"
            "Turn a portrait into a mosaic of Conway's Game of Life still lifes. "
            "**For best results upload a background-free image** "
            "(e.g. via [remove.bg](https://www.remove.bg)): the subject on a "
            "transparent background."
        )

        # Session seed used when the seed field is blank, so live tweaks stay
        # stable. Randomised per session and rerolled by "New variation".
        auto_seed = gr.State(random.randrange(2**31))

        with gr.Row(equal_height=False):
            # --- Controls (compact, left) ------------------------------------
            with gr.Column(scale=2, min_width=280):
                image_in = gr.Image(
                    label="Upload image (PNG/JPG)",
                    type="pil",
                    image_mode="RGBA",  # preserve transparency of bg-free PNGs
                    sources=["upload"],
                    height=170,  # small thumbnail — the mosaic is the star
                )
                level_in = gr.Dropdown(
                    label="Detail level",
                    choices=LEVELS,
                    value=DEFAULT_LEVEL,
                    info="Higher = finer tiles. Level 5 is noticeably slower.",
                )
                color_in = gr.Dropdown(
                    label="Colour scheme",
                    choices=COLOR_SCHEME_LABELS,
                    value=UGENT,
                )
                # Manual colour pickers, shown only when "Manual" is selected.
                with gr.Group(visible=False) as manual_group:
                    with gr.Row():
                        gol_bg_in = gr.ColorPicker(
                            label="GoL background", value=DEFAULT_MANUAL.gol_background)
                        gol_px_in = gr.ColorPicker(
                            label="GoL pixel", value=DEFAULT_MANUAL.gol_pixel)
                    with gr.Row():
                        eca_bg_in = gr.ColorPicker(
                            label="ECA background", value=DEFAULT_MANUAL.eca_background)
                        eca_px_in = gr.ColorPicker(
                            label="ECA pixel", value=DEFAULT_MANUAL.eca_pixel)

                grid_in = gr.Slider(
                    label="Grid size (tiles across)",
                    minimum=MIN_GRID, maximum=MAX_GRID, value=DEFAULT_GRID, step=2,
                    info="Must be even; larger = more, smaller tiles.",
                )

                # Above the accordion so it doesn't shift when Advanced opens.
                reroll_btn = gr.Button("🎲 New variation", variant="primary")

                with gr.Accordion("Advanced settings", open=False):
                    empty_in = gr.Slider(
                        label="Empty-tiles cutoff",
                        minimum=0.0, maximum=1.0, value=0.65, step=0.01,
                        info="Greyscale brighter than this becomes empty space.",
                    )
                    alpha_in = gr.Slider(
                        label="Alpha cutoff",
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        info="Transparency below this is filled with the ECA background.",
                    )
                    eca_in = gr.Dropdown(
                        label="ECA background rule",
                        choices=ECA_CHOICES,
                        value="random",
                    )
                    seed_in = gr.Number(
                        label="Seed (optional)",
                        value=None, precision=0,
                        info="Set for a fully reproducible result. Blank = a stable "
                             "per-session look; use 'New variation' to reroll.",
                    )

            # --- Result (prominent, right) -----------------------------------
            with gr.Column(scale=3, min_width=320):
                image_out = gr.Image(
                    label="Mosaic (use the toolbar to download as gol-mosaic.png)",
                    type="filepath",  # return a path so the download keeps its name
                    format="png",
                    # No fixed height: display the image at its true aspect ratio
                    # so the preview shows exactly the saved file (no letterbox
                    # bars on the sides of a portrait mosaic).
                )
                with gr.Accordion("About", open=False):
                    gr.Markdown(ABOUT)

        # Inputs passed to every generation call (order matches render_mosaic).
        gen_inputs = [image_in, level_in, color_in, grid_in,
                      empty_in, alpha_in, eca_in, seed_in, auto_seed,
                      gol_bg_in, gol_px_in, eca_bg_in, eca_px_in]

        # Show the manual colour pickers only for the "Manual" scheme.
        color_in.change(
            fn=lambda label: gr.update(visible=(label == MANUAL)),
            inputs=color_in, outputs=manual_group,
        )

        # Live regeneration: sliders fire on release (not every pixel of drag),
        # dropdowns / number / image / colour pickers fire on change.
        live_triggers = [
            image_in.change, level_in.change, color_in.change,
            grid_in.release, empty_in.release, alpha_in.release,
            eca_in.change, seed_in.change,
            gol_bg_in.change, gol_px_in.change, eca_bg_in.change, eca_px_in.change,
        ]
        gr.on(triggers=live_triggers, fn=generate,
              inputs=gen_inputs, outputs=image_out)

        # "New variation": pick a fresh session seed, then regenerate.
        reroll_btn.click(
            fn=lambda: random.randrange(2**31), outputs=auto_seed
        ).then(
            fn=generate, inputs=gen_inputs, outputs=image_out
        )

    return demo


demo = build_demo()
# Cap concurrency so simultaneous requests don't oversubscribe a small (2-vCPU)
# box; queue extra requests rather than running them all at once.
demo.queue(default_concurrency_limit=2, max_size=20)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

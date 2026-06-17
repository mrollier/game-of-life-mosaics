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

# Colour schemes, by UI label. Each value is a factory so Warhol re-randomises
# per request (it deliberately picks fresh colours every call).
COLOR_SCHEMES = {
    "UGent (yellow/blue)": ColorScheme.ugent,
    "Monochrome (black/white)": ColorScheme.monochrome,
    "Warhol (random pop colours)": ColorScheme.warhol,
}

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


def generate(image, level, color_scheme, grid_size,
             empty_tiles_cutoff, alpha_cutoff, eca_rule, seed):
    """Generate a mosaic from the uploaded image and UI settings.

    Returns the RGBA result for display/download. Raises gr.Error with a
    friendly message on bad input rather than crashing the request.
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Normalise and bound the settings.
    level = int(level)
    grid_size = int(grid_size)
    if grid_size % 2:  # the diamond layout requires an even grid
        grid_size += 1
    grid_size = max(MIN_GRID, min(MAX_GRID, grid_size))

    rule = None if eca_rule == "random" else int(eca_rule)
    seed = int(seed) if seed not in (None, "") else None

    try:
        image = _bound_input(image)
        scheme = COLOR_SCHEMES[color_scheme]()
        generator = MosaicGenerator(
            level=level,
            grid_size=grid_size,
            color_scheme=scheme,
            eca_rule=rule,
        )
        # Inject the preloaded library instead of reloading it from disk.
        generator._pattern_library = PATTERN_LIBRARIES[level]

        return generator.generate_from_pil(
            image,
            empty_tiles_cutoff=float(empty_tiles_cutoff),
            alpha_cutoff=float(alpha_cutoff),
            remove_background=False,  # the app never runs rembg
            seed=seed,
        )
    except gr.Error:
        raise
    except Exception as exc:  # surface a friendly message, keep the app alive
        raise gr.Error(f"Could not generate the mosaic: {exc}")


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
    """Build the Gradio interface."""
    with gr.Blocks(title="Game of Life Mosaics") as demo:
        gr.Markdown("# 🔬 Game of Life Mosaics")
        gr.Markdown(
            "Turn a portrait into a mosaic of Conway's Game of Life still lifes. "
            "**For best results, upload an image with the background already "
            "removed** (e.g. via [remove.bg](https://www.remove.bg)) — the subject "
            "should sit on a transparent background."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Upload image (PNG/JPG)",
                    type="pil",
                    image_mode="RGBA",  # preserve transparency of bg-free PNGs
                    sources=["upload"],
                )

                level_in = gr.Dropdown(
                    label="Detail level",
                    choices=LEVELS,
                    value=DEFAULT_LEVEL,
                    info="Higher = finer tiles. Level 5 is noticeably slower.",
                )
                color_in = gr.Dropdown(
                    label="Colour scheme",
                    choices=list(COLOR_SCHEMES.keys()),
                    value="UGent (yellow/blue)",
                )
                grid_in = gr.Slider(
                    label="Grid size (tiles across)",
                    minimum=MIN_GRID, maximum=MAX_GRID, value=DEFAULT_GRID, step=2,
                    info="Must be even; larger = more, smaller tiles.",
                )

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
                        info="Set for a reproducible result. Blank = random. "
                             "Note: the Warhol scheme stays random regardless.",
                    )

                generate_btn = gr.Button("Generate mosaic", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(
                    label="Mosaic (use the toolbar to download as PNG)",
                    type="pil",
                    format="png",
                )
                with gr.Accordion("About", open=False):
                    gr.Markdown(ABOUT)

        generate_btn.click(
            fn=generate,
            inputs=[image_in, level_in, color_in, grid_in,
                    empty_in, alpha_in, eca_in, seed_in],
            outputs=image_out,
        )

    return demo


demo = build_demo()
# Cap concurrency so simultaneous requests don't oversubscribe a small (2-vCPU)
# box; queue extra requests rather than running them all at once.
demo.queue(default_concurrency_limit=2, max_size=20)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

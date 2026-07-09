"""
Gradio web app for gol_mosaics.

Upload an image, pick a few settings, and get back a Conway's Game of Life
still-life mosaic with an Elementary Cellular Automaton (ECA) background.

Deployable as a free Hugging Face Gradio Space (`python app.py` runs it locally).

Design notes:
- Background removal runs in an upload handler, NOT inside the mosaic pipeline.
  When an upload still has its background, on_upload() removes it once (via
  rembg's u2net) and caches BOTH the with-background and background-removed copies
  in per-session state. A toggle then chooses which cached copy feeds the
  pipeline, so live tweaks (sliders etc.) never re-run the heavy removal — the
  pipeline is always called with remove_background=False.
- The pattern libraries for the exposed levels are warmed at startup;
  PatternLibrary.load() caches one shared read-only instance per level, so
  requests never re-read the (up to 19 MB) data files from disk.
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
from gol_mosaics.image_processing import ImageProcessor
from gol_mosaics.export import GollyExporter
from gol_mosaics.renderer import hex_to_rgb

# --- Safety / resource limits -------------------------------------------------

# Guard against decompression-bomb images: refuse to decode anything absurdly
# large. The mosaic pixelates to a coarse grid, so huge inputs are never needed.
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP

# The free Space is CPU-only; force the CPU provider so onnxruntime doesn't probe
# for (absent) GPU backends. Must be set before the first background removal, as
# the rembg session is cached on first use.
ImageProcessor.background_removal_providers = ["CPUExecutionProvider"]

# Inputs larger than this (longest side) are downscaled before processing. The
# pipeline resizes to the grid anyway, so this only saves memory/time.
MAX_INPUT_DIM = 1600

# Tile shapes exposed in the UI: the historical 45-degree diamond layout and
# the axis-aligned pond-frame squares. Labels map to MosaicGenerator's
# tile_shape argument; each shape ships a different pre-computed level range
# (the square level-6 census, 19.3M tiles, is too large to ship).
DIAMONDS = "Diamonds"
SQUARES = "Squares"
TILE_SHAPE_LABELS = [DIAMONDS, SQUARES]
SHAPE_ARGS = {DIAMONDS: "diamond", SQUARES: "square"}

# Compute caps exposed in the UI.
LEVELS = [3, 4, 5, 6]
SQUARE_LEVELS = [3, 4, 5]
LEVELS_BY_SHAPE = {DIAMONDS: LEVELS, SQUARES: SQUARE_LEVELS}
# Levels warmed at startup; level 6 (332k patterns, ~450 MB expanded from the
# 2.7 MB packed file) is loaded lazily on first request to keep cold start fast.
WARM_LEVELS = [3, 4, 5]
DEFAULT_LEVEL = 4
MIN_GRID, MAX_GRID, DEFAULT_GRID = 10, 200, 60

# Background (ECA) pattern size = ECA cell side in pixels. Passed straight through
# as `supersample`; ECABackground.generate upsamples-then-crops, so any positive
# value is safe regardless of the mosaic's dimensions.
MIN_BG_SIZE, MAX_BG_SIZE, DEFAULT_BG_SIZE = 1, 40, 15

# The per-session seed and the "New variation" reroll draw from this range.
SEED_MAX = 2**31

# --- One-time startup work ----------------------------------------------------

# Warm the pattern-library cache for the cheap levels. load() keeps one
# shared read-only instance per (level, shape), so requests hit the cache
# instead of re-reading the 19 MB level-5 file. The square libraries are tiny
# (packed orbit bits, ~60 KB total) and are all warmed.
for _level in WARM_LEVELS:
    PatternLibrary.load(_level)
for _level in SQUARE_LEVELS:
    PatternLibrary.load(_level, shape="square")

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
        return ColorScheme(
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

# ECA rule dropdown: a curated set of interesting rules, plus a random option and
# a "Custom…" entry that reveals a 0-255 slider. Values are ints, the sentinel
# "random" (let the generator choose), or CUSTOM (use the slider value).
CUSTOM_RULE = "custom"
ECA_CHOICES = (
    [("Random (recommended)", "random")]
    + [(f"Rule {r} — complex", r) for r in ECABackground.COMPLEX_RULES]
    + [(f"Rule {r} — chaotic", r) for r in ECABackground.CHAOTIC_RULES]
    + [("Custom rule (0–255)…", CUSTOM_RULE)]
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
    """Convert a colour-picker value to an (R, G, B, alpha) tuple."""
    return (*hex_to_rgb(_to_hex(hex_color)), alpha)


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


def _resolve_eca_rule(eca_choice, eca_custom_rule):
    """Map the ECA dropdown selection to a rule for MosaicGenerator.

    "random" -> None (the generator picks one); the "Custom…" sentinel -> the
    0-255 slider value; otherwise the curated integer rule. Rule 0 is a valid
    Wolfram rule and is now respected by MosaicGenerator.
    """
    if eca_choice == "random":
        return None
    if eca_choice == CUSTOM_RULE:
        return int(eca_custom_rule)
    return int(eca_choice)


def _prepare_generation(image, tile_shape, level, color_scheme, grid_size,
                        eca_choice, eca_custom_rule, auto_seed, manual_colors):
    """Shared setup for the PNG and .cells paths.

    Normalises the settings, seeds numpy for a stable per-session look, and
    builds a MosaicGenerator (its lazy pattern-library load hits the shared
    per-(level, shape) cache warmed at startup).
    Returns (generator, bounded_image, scheme, effective_seed).
    """
    shape_arg = SHAPE_ARGS.get(tile_shape, "diamond")
    level = int(level)
    if level not in LEVELS_BY_SHAPE.get(tile_shape, LEVELS):
        # e.g. level 6 arriving in the same tick as a switch to Squares
        level = DEFAULT_LEVEL
    grid_size = int(grid_size)
    if shape_arg == "diamond" and grid_size % 2:
        grid_size += 1  # only the diamond layout requires an even grid
    grid_size = max(MIN_GRID, min(MAX_GRID, grid_size))

    rule = _resolve_eca_rule(eca_choice, eca_custom_rule)
    effective_seed = int(auto_seed)

    image = _bound_input(image)
    scheme = _scheme_for(color_scheme, effective_seed, manual_colors)
    # Seed before constructing the generator: with a "random" ECA rule the rule is
    # drawn in __init__ (before generate_from_pil reseeds), so this keeps the
    # background stable across live tweaks.
    np.random.seed(effective_seed)
    generator = MosaicGenerator(
        level=level,
        grid_size=grid_size,
        color_scheme=scheme,
        eca_rule=rule,
        tile_shape=shape_arg,
    )
    return generator, image, scheme, effective_seed


def _generate_mosaic(image, tile_shape, level, color_scheme, grid_size,
                     empty_tiles_cutoff, alpha_cutoff, eca_choice,
                     eca_custom_rule, bg_pattern_size, auto_seed,
                     manual_colors, return_arrays=False):
    """Prepare the generator and run the pipeline once (shared by the PNG and
    .cells paths).

    Returns (result, scheme, bounded_image): result is generate_from_pil's
    return value — the mosaic image, or (image, gol_mosaic, mask) when
    return_arrays is True.
    """
    generator, image, scheme, effective_seed = _prepare_generation(
        image, tile_shape, level, color_scheme, grid_size, eca_choice,
        eca_custom_rule, auto_seed, manual_colors,
    )
    result = generator.generate_from_pil(
        image,
        empty_tiles_cutoff=float(empty_tiles_cutoff),
        alpha_cutoff=float(alpha_cutoff),
        supersample=int(bg_pattern_size),
        remove_background=False,  # the app never runs rembg inside the pipeline
        seed=effective_seed,
        return_arrays=return_arrays,
    )
    return result, scheme, image


def render_mosaic(image, tile_shape, level, color_scheme, grid_size,
                  empty_tiles_cutoff, alpha_cutoff, eca_choice, eca_custom_rule,
                  bg_pattern_size, auto_seed,
                  gol_background, gol_pixel, eca_background, eca_pixel
                  ) -> Optional[Image.Image]:
    """Core generation: returns the RGBA mosaic (or None if there's no image).

    With no image yet (e.g. a live trigger fired before upload) it returns None
    silently rather than erroring, so adjusting controls is smooth. Genuine
    failures raise gr.Error.

    Variation is driven solely by the session's auto_seed (rerolled by the "New
    variation" button), so tweaking one control changes only that aspect.
    """
    if image is None:
        return None

    manual_colors = (gol_background, gol_pixel, eca_background, eca_pixel)
    try:
        mosaic, scheme, bounded = _generate_mosaic(
            image, tile_shape, level, color_scheme, grid_size,
            empty_tiles_cutoff, alpha_cutoff, eca_choice, eca_custom_rule,
            bg_pattern_size, auto_seed, manual_colors,
        )
        # Fit to the original aspect ratio on a solid ECA-background backdrop
        # (fills the aspect padding around the opaque mosaic).
        target_ratio = bounded.width / bounded.height
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


def on_shape_change(tile_shape, level):
    """Re-range the level dropdown when the tile shape changes.

    Diamonds ship levels 3-6, squares 3-5. A selected level that doesn't
    exist for the new shape falls back to the default; otherwise it is kept.
    """
    choices = LEVELS_BY_SHAPE[tile_shape]
    level = int(level)
    return gr.update(choices=choices,
                     value=level if level in choices else DEFAULT_LEVEL)


# --- Background-removal state & selection -------------------------------------

def on_upload(img: Optional[Image.Image]):
    """Handle a new upload: remove the background once and cache both copies.

    Returns (inputs_state, bg_toggle, input_preview):
      - inputs_state: {"with_bg", "without_bg", "has_bg"} PIL copies kept in the
        session so later toggling / live tweaks never re-run the heavy removal.
      - bg_toggle: a gr.update for the "Remove background" checkbox — interactive
        only when there is a background-removed copy to switch to.
      - input_preview: the image currently feeding the pipeline.

    Removal runs at most once, and only when has_background() is true. The upload
    is bounded first so both cached copies share the same dimensions (identical
    aspect ratio) and rembg does less work.
    """
    if img is None:
        return None, gr.update(value=False, interactive=False), None

    img = _bound_input(img)

    if not ImageProcessor.has_background(img):
        # Already background-free: nothing to remove, so the toggle is moot.
        state = {"with_bg": img, "without_bg": None, "has_bg": False}
        toggle = gr.update(value=False, interactive=False,
                           info="No background detected — using image as uploaded.")
        return state, toggle, img

    try:
        without_bg = ImageProcessor.remove_background(img)
    except Exception:
        # rembg missing or removal failed: keep the original, disable the toggle.
        state = {"with_bg": img, "without_bg": None, "has_bg": True}
        toggle = gr.update(value=False, interactive=False,
                           info="Background removal unavailable — using original.")
        return state, toggle, img

    state = {"with_bg": img, "without_bg": without_bg, "has_bg": True}
    # Default to the background-removed image (the intended artistic result).
    toggle = gr.update(value=True, interactive=True,
                       info="On: subject only. Off: keep the original background.")
    return state, toggle, without_bg


def _selected_input(state: Optional[dict], remove_bg: bool) -> Optional[Image.Image]:
    """Pick the cached image the toggle selects (also used to drive the preview).

    Falls back to the with-background copy whenever there is no removed copy (a
    background-free upload, or a failed removal), even if remove_bg reads True.
    """
    if not state:
        return None
    if remove_bg and state.get("without_bg") is not None:
        return state["without_bg"]
    return state["with_bg"]


def generate_ui(state, remove_bg, *rest) -> Optional[str]:
    """Generation entry point for the UI: resolve the toggled input, then render.

    Keeps render_mosaic/generate unaware of the caching layer — they still take a
    plain PIL image as their first argument.
    """
    return generate(_selected_input(state, remove_bg), *rest)


def _binary_bbox(mosaic: np.ndarray) -> np.ndarray:
    """Binarise a GoL mosaic (0/1) and trim to the bounding box of live cells.

    The interlocking diagonal grids never overlap, so the mosaic is already
    0/1-valued; the `> 0` threshold is kept as cheap defence. Trimming the
    empty margin keeps the Golly pattern compact.
    """
    binary = (np.asarray(mosaic) > 0).astype(np.uint8)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return binary[:0, :0]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return binary[r0:r1 + 1, c0:c1 + 1]


def export_cells_ui(state, remove_bg, tile_shape, level, color_scheme,
                    grid_size, empty_tiles_cutoff, alpha_cutoff, eca_choice,
                    eca_custom_rule, bg_pattern_size, auto_seed,
                    gol_background, gol_pixel, eca_background, eca_pixel
                    ) -> Optional[str]:
    """Build a Golly .cells file of the still-life mosaic for the current settings.

    Re-runs the pipeline (deterministic via auto_seed, so it matches the displayed
    mosaic) to recover the binary GoL array, then exports just the still-life
    portrait — the ECA background is decorative and not stable in Golly.
    """
    image = _selected_input(state, remove_bg)
    if image is None:
        raise gr.Error("Upload an image first, then download its .cells file.")

    manual_colors = (gol_background, gol_pixel, eca_background, eca_pixel)
    try:
        (_, gol_mosaic, _), _, _ = _generate_mosaic(
            image, tile_shape, level, color_scheme, grid_size,
            empty_tiles_cutoff, alpha_cutoff, eca_choice, eca_custom_rule,
            bg_pattern_size, auto_seed, manual_colors, return_arrays=True,
        )
    except Exception as exc:
        raise gr.Error(f"Could not build the .cells file: {exc}")

    cells = _binary_bbox(gol_mosaic)
    if cells.size == 0:
        raise gr.Error("The mosaic has no live cells to export.")

    out_path = os.path.join(tempfile.mkdtemp(prefix="gol_"), "gol-mosaic.cells")
    GollyExporter.export_to_cells(cells, out_path)
    return out_path


ABOUT = """
### About these mosaics

Each tile is a **Game of Life still life** — a pattern that, under Conway's Game
of Life rules, never changes from one generation to the next. Darker areas of
your image are filled with denser still lifes, lighter areas with sparser ones,
so the portrait emerges from the arrangement of stable patterns.

**Tile shape:** *Diamonds* is the classic layout — diamond-shaped tiles on two
interlocking 45° grids, glued by shared pond patterns. *Squares* uses
axis-aligned square tiles bordered by a ring of ponds; adjacent tiles share
their border ponds. Both are mathematically guaranteed to stay a still life,
whatever mix of tiles the image asks for.

The coloured backdrop behind the subject is an **Elementary Cellular Automaton
(ECA)** — a one-dimensional automaton (Wolfram's rules) evolved row by row.
"Complex" rules (e.g. 110) make structured patterns; "chaotic" rules (e.g. 30)
make noisier ones.

**Background removal:** if your upload still has its background, it is removed
automatically so the subject stands out. Use the **Remove background** toggle to
switch between the subject-only and original versions — the *Input preview* shows
which one is feeding the mosaic. The first removal after the app starts is slow
(the segmentation model downloads once), then it's cached.

**Download for Golly:** use *Download .cells* to get the still-life pattern as a
[Golly](https://golly.sourceforge.io) `.cells` file (the ECA backdrop is
decorative and not exported, since it isn't made of stable Life patterns).

### Credits
Source code: [github.com/mrollier/game-of-life-mosaics](https://github.com/mrollier/game-of-life-mosaics).

The idea of rendering images as mosaics of Game of Life still lifes comes from
**Robert Bosch**, *Opt Art: From Mathematical Optimization to Visual Design*
(Princeton University Press, 2019).
"""


# Force light mode regardless of the visitor's OS/browser dark-mode preference.
# Gradio reads the theme from the `__theme` query param at load, so a small head
# script sets it to "light" and reloads once (the guard prevents a redirect loop).
# Injected via gr.Blocks(head=...) rather than js=... because Gradio 6 dropped the
# Blocks `js` param, and this runs no matter how the Space launches the app.
FORCE_LIGHT_THEME_HEAD = """
<script>
  (function () {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
      url.searchParams.set('__theme', 'light');
      window.location.replace(url.href);
    }
  })();
</script>
"""


def build_demo() -> gr.Blocks:
    """Build the Gradio interface.

    Layout: a compact controls column on the left and the large mosaic on the
    right, so the controls and result are visible together without scrolling.
    The mosaic regenerates live whenever a control changes.
    """
    with gr.Blocks(title="Game of Life Mosaics") as demo:
        gr.Markdown(
            "# 🔬 Game of Life Mosaics\n"
            "Turn a portrait into a mosaic of Conway's Game of Life still lifes. "
            "Upload any photo — if it still has a background it's **removed "
            "automatically**, and the *Remove background* toggle lets you keep the "
            "original if you prefer. **Note:** the *first* upload takes a while as "
            "the background-removal model loads in the background; later uploads "
            "are quick. You can skip this by uploading an image whose background is "
            "already removed (e.g. via [remove.bg](https://www.remove.bg))."
        )

        # Per-session seed that keeps live tweaks stable; rerolled by "New
        # variation". (There is no manual seed field: New variation is the only
        # source of variation, so it can never be overridden.)
        auto_seed = gr.State(random.randrange(SEED_MAX))

        # Cached upload copies for this session: {"with_bg", "without_bg",
        # "has_bg"}. Populated by on_upload so background removal runs only once.
        inputs_state = gr.State(None)

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
                # Background-removal controls. The toggle is greyed out until an
                # upload with a removable background arrives; the preview shows
                # exactly which copy currently feeds the mosaic.
                bg_toggle = gr.Checkbox(
                    label="Remove background",
                    value=False,
                    interactive=False,
                    info="Auto-detected on upload.",
                )
                input_preview = gr.Image(
                    label="Input preview (fed to the mosaic)",
                    type="pil",
                    interactive=False,
                    height=170,
                )
                shape_in = gr.Radio(
                    label="Tile shape",
                    choices=TILE_SHAPE_LABELS,
                    value=DIAMONDS,
                    info="Diamonds: the classic 45° layout. Squares: "
                         "axis-aligned tiles (levels 3-5).",
                )
                level_in = gr.Dropdown(
                    label="Detail level",
                    choices=LEVELS,
                    value=DEFAULT_LEVEL,
                    info="Higher = finer tiles. Levels 5-6 are noticeably "
                         "slower; diamond level 6 (332k tiles) loads on "
                         "first use.",
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
                    info="Even number; larger = more, smaller tiles. "
                         "Levels 5-6 at 200 are slow.",
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
                    # 0-255 slider, shown only when "Custom rule…" is selected.
                    eca_custom_in = gr.Slider(
                        label="Custom ECA rule (0–255)",
                        minimum=0, maximum=255, value=110, step=1,
                        visible=False,
                        info="Wolfram rule number. 0 gives an almost-empty backdrop.",
                    )
                    bg_size_in = gr.Slider(
                        label="Background pattern size",
                        minimum=MIN_BG_SIZE, maximum=MAX_BG_SIZE,
                        value=DEFAULT_BG_SIZE, step=1,
                        info="ECA cell size in pixels; larger = chunkier backdrop.",
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
                cells_btn = gr.DownloadButton(
                    "⬇ Download .cells (for Golly)",
                )
                with gr.Accordion("About", open=False):
                    gr.Markdown(ABOUT)

        _wire_events(
            inputs_state=inputs_state, auto_seed=auto_seed,
            image_in=image_in, bg_toggle=bg_toggle, input_preview=input_preview,
            shape_in=shape_in,
            level_in=level_in, color_in=color_in, manual_group=manual_group,
            gol_bg_in=gol_bg_in, gol_px_in=gol_px_in,
            eca_bg_in=eca_bg_in, eca_px_in=eca_px_in,
            grid_in=grid_in, reroll_btn=reroll_btn,
            empty_in=empty_in, alpha_in=alpha_in,
            eca_in=eca_in, eca_custom_in=eca_custom_in, bg_size_in=bg_size_in,
            image_out=image_out, cells_btn=cells_btn,
        )

    return demo


def _wire_events(*, inputs_state, auto_seed, image_in, bg_toggle, input_preview,
                 shape_in, level_in, color_in, manual_group, gol_bg_in,
                 gol_px_in, eca_bg_in, eca_px_in, grid_in, reroll_btn,
                 empty_in, alpha_in, eca_in, eca_custom_in, bg_size_in,
                 image_out, cells_btn):
    """Attach all event handlers to the components built by build_demo.

    Must be called inside the gr.Blocks context (build_demo does so), since
    Gradio registers listeners against the active Blocks.
    """
    # Inputs passed to every generation call. The first two (cached upload
    # state + toggle) are resolved to a PIL image by generate_ui; the rest
    # match render_mosaic's remaining arguments in order.
    gen_inputs = [inputs_state, bg_toggle, shape_in, level_in, color_in,
                  grid_in, empty_in, alpha_in, eca_in, eca_custom_in,
                  bg_size_in, auto_seed, gol_bg_in, gol_px_in, eca_bg_in,
                  eca_px_in]

    # Tile shape: re-range the level dropdown, then regenerate once. The level
    # dropdown's live trigger is `.input` (user edits only), so the
    # programmatic re-range here doesn't fire a second generation.
    shape_in.change(
        fn=on_shape_change, inputs=[shape_in, level_in], outputs=level_in,
    ).then(fn=generate_ui, inputs=gen_inputs, outputs=image_out)

    # Show the manual colour pickers only for the "Manual" scheme.
    color_in.change(
        fn=lambda label: gr.update(visible=(label == MANUAL)),
        inputs=color_in, outputs=manual_group,
    )

    # Show the custom-rule slider only when "Custom rule…" is selected.
    eca_in.change(
        fn=lambda choice: gr.update(visible=(choice == CUSTOM_RULE)),
        inputs=eca_in, outputs=eca_custom_in,
    )

    # Upload: remove the background once, cache both copies, update the toggle
    # and preview, then generate.
    image_in.change(
        fn=on_upload, inputs=image_in,
        outputs=[inputs_state, bg_toggle, input_preview],
    ).then(fn=generate_ui, inputs=gen_inputs, outputs=image_out)

    # Toggle: switch the previewed/used copy (no re-removal), then regenerate.
    bg_toggle.change(
        fn=_selected_input, inputs=[inputs_state, bg_toggle],
        outputs=input_preview,
    ).then(fn=generate_ui, inputs=gen_inputs, outputs=image_out)

    # Live regeneration: sliders fire on release (not every pixel of drag),
    # dropdowns / colour pickers fire on change. (Upload, toggle and tile
    # shape have their own chains above; the level dropdown listens to .input
    # so the shape chain's programmatic update doesn't double-generate.)
    live_triggers = [
        level_in.input, color_in.change,
        grid_in.release, empty_in.release, alpha_in.release,
        eca_in.change, eca_custom_in.release, bg_size_in.release,
        gol_bg_in.change, gol_px_in.change, eca_bg_in.change, eca_px_in.change,
    ]
    gr.on(triggers=live_triggers, fn=generate_ui,
          inputs=gen_inputs, outputs=image_out)

    # "New variation": pick a fresh session seed, then regenerate.
    reroll_btn.click(
        fn=lambda: random.randrange(SEED_MAX), outputs=auto_seed
    ).then(
        fn=generate_ui, inputs=gen_inputs, outputs=image_out
    )

    # Build the .cells file for the current settings on demand.
    cells_btn.click(fn=export_cells_ui, inputs=gen_inputs, outputs=cells_btn)


demo = build_demo()
# Cap concurrency so simultaneous requests don't oversubscribe a small (2-vCPU)
# box; queue extra requests rather than running them all at once.
demo.queue(default_concurrency_limit=2, max_size=20)

if __name__ == "__main__":
    # head=... injects the force-light-mode script (Gradio 6 takes head/js/theme
    # on launch(), not on the Blocks constructor).
    demo.launch(theme=gr.themes.Soft(), head=FORCE_LIGHT_THEME_HEAD)

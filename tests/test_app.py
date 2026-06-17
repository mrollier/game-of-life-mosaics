"""Smoke tests for the Gradio app's generation logic.

Skipped automatically where gradio isn't installed (e.g. a library-only CI),
since gradio is an app-only dependency, not a dependency of gol_mosaics.
"""

import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("gradio")  # app-only dependency

import app  # noqa: E402  (import after the skip guard)


def _subject_on_transparent(size=80):
    """A small subject on a transparent background, like a bg-removed upload."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    img.paste(Image.new('RGBA', (size // 2, size // 2), (20, 20, 20, 255)),
              (size // 4, size // 4))
    return img


# Default manual colours (UGent palette) for the four colour-picker args.
_MANUAL = (app.DEFAULT_MANUAL.gol_background, app.DEFAULT_MANUAL.gol_pixel,
           app.DEFAULT_MANUAL.eca_background, app.DEFAULT_MANUAL.eca_pixel)


def _render(image, color_scheme=app.UGENT, seed=1, auto_seed=0, manual=_MANUAL):
    """Call render_mosaic with sensible defaults for the fixed settings."""
    return app.render_mosaic(image, 3, color_scheme, 40, 0.65, 0.5,
                             "random", seed, auto_seed, *manual)


def test_render_returns_rgba():
    """The core renderer returns a valid RGBA image for a normal request."""
    result = _render(_subject_on_transparent())
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


def test_render_no_image_returns_none():
    """With no upload (e.g. a live trigger before upload) the renderer returns
    None silently rather than erroring, so adjusting controls stays smooth."""
    assert _render(None) is None


def test_blank_seed_uses_auto_seed_for_stability():
    """A blank seed falls back to auto_seed, so the same auto_seed reproduces
    the same mosaic (what keeps live tweaking stable)."""
    img = _subject_on_transparent()
    first = _render(img, seed=None, auto_seed=99)
    second = _render(img, seed=None, auto_seed=99)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_manual_colors_are_applied():
    """The Manual scheme renders with the chosen colours.

    The transparent background becomes the ECA overlay, so the two distinctive
    manual ECA colours must appear in the output — proving the manual palette is
    wired through (rather than a built-in scheme)."""
    manual = ("#ffffff", "#ff0000", "#00ff00", "#0000ff")  # bg, px, eca_bg, eca_px
    result = _render(_subject_on_transparent(), color_scheme=app.MANUAL,
                     manual=manual)
    colors = {tuple(int(v) for v in c) for c in np.asarray(result).reshape(-1, 4)}
    assert (0, 255, 0, 255) in colors  # eca_background green
    assert (0, 0, 255, 255) in colors  # eca_pixel blue


def test_output_matches_original_aspect_ratio():
    """The output is fitted to (almost) the original upload's aspect ratio on a
    solid ECA-coloured backdrop: ratio matches, no transparent pixels remain,
    and all four edges are the ECA background colour."""
    img = Image.new('RGBA', (60, 100), (0, 0, 0, 0))  # 0.6 portrait
    img.paste(Image.new('RGBA', (30, 50), (20, 20, 20, 255)), (15, 25))
    result = _render(img)  # UGENT -> ECA background #FFD200

    target = 60 / 100
    assert abs(result.size[0] / result.size[1] - target) < 0.02

    arr = np.asarray(result)
    assert not (arr[:, :, 3] == 0).any()  # solid backdrop, nothing transparent
    eca = app._hex_to_rgba(app.DEFAULT_MANUAL.eca_background)  # UGent yellow
    for edge in (arr[0], arr[-1], arr[:, 0], arr[:, -1]):
        assert (edge == eca).all()


def test_generate_saves_named_png():
    """The UI wrapper returns a real PNG path named gol-mosaic.png."""
    path = app.generate(_subject_on_transparent(), 3, app.UGENT, 40, 0.65, 0.5,
                        "random", 1, 0, *_MANUAL)
    assert os.path.basename(path) == "gol-mosaic.png"
    Image.open(path).load()  # opens without error => valid image


def test_generate_no_image_returns_none():
    """The UI wrapper returns None (not a path) when there's no image."""
    assert app.generate(None, 3, app.UGENT, 40, 0.65, 0.5, "random", 1, 0,
                        *_MANUAL) is None

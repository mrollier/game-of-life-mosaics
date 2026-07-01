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
    and the padding bars are the ECA background colour. The opaque mosaic (whose
    rotation rim is the GoL background colour) sits centred on that backdrop."""
    img = Image.new('RGBA', (60, 100), (0, 0, 0, 0))  # 0.6 portrait
    img.paste(Image.new('RGBA', (30, 50), (20, 20, 20, 255)), (15, 25))
    result = _render(img)  # UGENT -> ECA background #FFD200

    target = 60 / 100
    assert abs(result.size[0] / result.size[1] - target) < 0.02

    arr = np.asarray(result)
    assert not (arr[:, :, 3] == 0).any()  # solid backdrop, nothing transparent
    # A portrait is fitted by padding rows, so the top and bottom edges are the
    # solid ECA-background backdrop.
    eca = app._hex_to_rgba(app.DEFAULT_MANUAL.eca_background)  # UGent yellow
    for edge in (arr[0], arr[-1]):
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


# --- Background-removal caching / selection -----------------------------------

def _with_bg(size=80):
    """An opaque image (no alpha channel removed yet) — has_background() is True."""
    return Image.new('RGBA', (size, size), (30, 60, 90, 255))


def test_selected_input_prefers_removed_when_toggled_on():
    """With a removed copy cached and the toggle on, that copy is selected."""
    with_bg, without_bg = _with_bg(), _subject_on_transparent()
    state = {"with_bg": with_bg, "without_bg": without_bg, "has_bg": True}
    assert app._selected_input(state, True) is without_bg
    assert app._selected_input(state, False) is with_bg


def test_selected_input_falls_back_to_with_bg():
    """No removed copy (bg-free upload or failed removal) -> the with_bg copy,
    even if the toggle reads True. No state -> None."""
    state = {"with_bg": _with_bg(), "without_bg": None, "has_bg": False}
    assert app._selected_input(state, True) is state["with_bg"]
    assert app._selected_input(None, True) is None


def test_on_upload_none_disables_toggle():
    """No upload clears the state, greys out the toggle, and clears the preview."""
    state, toggle, preview = app.on_upload(None)
    assert state is None and preview is None
    assert toggle["interactive"] is False


def test_on_upload_removes_background_once(monkeypatch):
    """An image with a background is removed once, both copies cached, toggle on.

    remove_background is monkeypatched so the test never needs rembg/onnxruntime;
    a counter proves it runs exactly once per upload."""
    calls = {"n": 0}

    def fake_remove(cls, img, *a, **k):
        calls["n"] += 1
        return _subject_on_transparent()

    monkeypatch.setattr(app.ImageProcessor, "has_background",
                        staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(app.ImageProcessor, "remove_background",
                        classmethod(fake_remove))

    state, toggle, preview = app.on_upload(_with_bg())
    assert calls["n"] == 1
    assert state["without_bg"] is not None and state["has_bg"] is True
    assert toggle["interactive"] is True and toggle["value"] is True
    assert preview is state["without_bg"]


def test_on_upload_transparent_disables_toggle(monkeypatch):
    """A background-free upload caches only the original and greys the toggle."""
    monkeypatch.setattr(app.ImageProcessor, "has_background",
                        staticmethod(lambda *a, **k: False))
    state, toggle, preview = app.on_upload(_subject_on_transparent())
    assert state["without_bg"] is None and state["has_bg"] is False
    assert toggle["interactive"] is False
    assert preview is state["with_bg"]


def test_on_upload_removal_failure_falls_back(monkeypatch):
    """If removal raises (e.g. rembg missing), keep the original, disable toggle."""
    monkeypatch.setattr(app.ImageProcessor, "has_background",
                        staticmethod(lambda *a, **k: True))

    def boom(cls, img, *a, **k):
        raise ImportError("rembg not installed")

    monkeypatch.setattr(app.ImageProcessor, "remove_background", classmethod(boom))
    state, toggle, preview = app.on_upload(_with_bg())
    assert state["without_bg"] is None and state["has_bg"] is True
    assert toggle["interactive"] is False
    assert preview is state["with_bg"]

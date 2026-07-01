"""Smoke tests for the Gradio app's generation logic.

Skipped automatically where gradio isn't installed (e.g. a library-only CI),
since gradio is an app-only dependency, not a dependency of gol_mosaics.
"""

import os
import time

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


def _last(gen):
    """Consume a generator handler and return its final yielded value."""
    result = None
    for result in gen:
        pass
    return result


# Default manual colours (UGent palette) for the four colour-picker args.
_MANUAL = (app.DEFAULT_MANUAL.gol_background, app.DEFAULT_MANUAL.gol_pixel,
           app.DEFAULT_MANUAL.eca_background, app.DEFAULT_MANUAL.eca_pixel)


def _render(image, color_scheme=app.UGENT, auto_seed=0, manual=_MANUAL,
            eca_choice="random", eca_custom_rule=110,
            bg_pattern_size=app.DEFAULT_BG_SIZE):
    """Call render_mosaic with sensible defaults for the fixed settings."""
    return app.render_mosaic(image, 3, color_scheme, 40, 0.65, 0.5,
                             eca_choice, eca_custom_rule, bg_pattern_size,
                             auto_seed, *manual)


def test_render_returns_rgba():
    """The core renderer returns a valid RGBA image for a normal request."""
    result = _render(_subject_on_transparent())
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


def test_render_no_image_returns_none():
    """With no upload (e.g. a live trigger before upload) the renderer returns
    None silently rather than erroring, so adjusting controls stays smooth."""
    assert _render(None) is None


def test_auto_seed_is_stable():
    """The same auto_seed reproduces the same mosaic (what keeps live tweaking
    stable). auto_seed is now the only source of variation (no seed field)."""
    img = _subject_on_transparent()
    first = _render(img, auto_seed=99)
    second = _render(img, auto_seed=99)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_custom_rule_zero_renders():
    """A custom ECA rule of 0 renders without being silently randomised."""
    result = _render(_subject_on_transparent(),
                     eca_choice=app.CUSTOM_RULE, eca_custom_rule=0)
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


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
                        "random", 110, app.DEFAULT_BG_SIZE, 0, *_MANUAL)
    assert os.path.basename(path) == "gol-mosaic.png"
    Image.open(path).load()  # opens without error => valid image


def test_generate_no_image_returns_none():
    """The UI wrapper returns None (not a path) when there's no image."""
    assert app.generate(None, 3, app.UGENT, 40, 0.65, 0.5, "random", 110,
                        app.DEFAULT_BG_SIZE, 0, *_MANUAL) is None


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
    """No upload clears the state, greys out the toggle, and clears the preview.

    on_upload is a generator; _last() takes its final yielded value."""
    state, toggle, preview = _last(app.on_upload(None))
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

    state, toggle, preview = _last(app.on_upload(_with_bg()))
    assert calls["n"] == 1
    assert state["without_bg"] is not None and state["has_bg"] is True
    assert toggle["interactive"] is True and toggle["value"] is True
    assert preview is state["without_bg"]


def test_on_upload_transparent_disables_toggle(monkeypatch):
    """A background-free upload caches only the original and greys the toggle."""
    monkeypatch.setattr(app.ImageProcessor, "has_background",
                        staticmethod(lambda *a, **k: False))
    state, toggle, preview = _last(app.on_upload(_subject_on_transparent()))
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
    state, toggle, preview = _last(app.on_upload(_with_bg()))
    assert state["without_bg"] is None and state["has_bg"] is True
    assert toggle["interactive"] is False
    assert preview is state["with_bg"]


def test_on_upload_warns_when_removal_is_slow(monkeypatch):
    """A slow removal (past the threshold) emits a toast but still succeeds."""
    warnings = []
    monkeypatch.setattr(app.gr, "Warning", lambda msg, *a, **k: warnings.append(msg))
    monkeypatch.setattr(app, "SLOW_REMOVAL_WARN_SECONDS", 0)
    monkeypatch.setattr(app.ImageProcessor, "has_background",
                        staticmethod(lambda *a, **k: True))

    def slow_remove(cls, img, *a, **k):
        time.sleep(1.2)  # long enough for one poll past the (0 s) threshold
        return _subject_on_transparent()

    monkeypatch.setattr(app.ImageProcessor, "remove_background",
                        classmethod(slow_remove))

    state, toggle, preview = _last(app.on_upload(_with_bg()))
    assert warnings  # the slow-removal toast fired
    assert state["without_bg"] is not None
    assert toggle["interactive"] is True


# --- Golly .cells export ------------------------------------------------------

def test_binary_bbox_trims_and_binarises():
    """Overlaps (value 2) binarise to 1; the array trims to live-cell bounds."""
    arr = np.zeros((5, 5), dtype=int)
    arr[1, 2] = 2  # overlapping-tile value
    arr[3, 3] = 1
    out = app._binary_bbox(arr)
    assert out.shape == (3, 2)  # rows 1..3, cols 2..3
    assert set(np.unique(out)) <= {0, 1}
    assert out[0, 0] == 1 and out[-1, -1] == 1


def test_export_cells_writes_valid_golly_file():
    """export_cells_ui writes a named .cells file containing only Golly glyphs."""
    state = {"with_bg": _subject_on_transparent(), "without_bg": None,
             "has_bg": False}
    path = app.export_cells_ui(state, False, 3, app.UGENT, 40, 0.65, 0.5,
                               "random", 110, app.DEFAULT_BG_SIZE, 0, *_MANUAL)
    assert os.path.basename(path) == "gol-mosaic.cells"
    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("!")  # Golly header comment
    body = [ln for ln in lines if not ln.startswith("!")]
    assert body and all(set(ln) <= {".", "O"} for ln in body)

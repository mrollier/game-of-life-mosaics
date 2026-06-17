"""Smoke test for the Gradio app's generation callback.

Skipped automatically where gradio isn't installed (e.g. a library-only CI),
since gradio is an app-only dependency, not a dependency of gol_mosaics.
"""

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


def test_app_generate_returns_rgba():
    """The app callback returns a valid RGBA image for a normal request."""
    result = app.generate(
        image=_subject_on_transparent(),
        level=3,
        color_scheme="UGent (yellow/blue)",
        grid_size=40,
        empty_tiles_cutoff=0.65,
        alpha_cutoff=0.5,
        eca_rule="random",
        seed=1,
    )
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


def test_app_rejects_missing_image():
    """A missing upload raises a friendly gr.Error instead of crashing."""
    import gradio as gr
    with pytest.raises(gr.Error):
        app.generate(None, 4, "UGent (yellow/blue)", 60, 0.65, 0.5, "random", None)

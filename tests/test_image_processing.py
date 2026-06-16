"""Tests for background detection and removal in ImageProcessor."""

import sys
import numpy as np
import pytest
from PIL import Image

from src.gol_mosaics import ImageProcessor


try:
    import rembg  # noqa: F401
    _HAS_REMBG = True
except ImportError:
    _HAS_REMBG = False


# --- helpers -------------------------------------------------------------

def _opaque_rgb(size=(20, 20)):
    """A plain colour photo with no alpha channel (background still present)."""
    return Image.new('RGB', size, (120, 90, 60))


def _opaque_rgba(size=(20, 20)):
    """RGBA image that is fully opaque (alpha all 255)."""
    return Image.new('RGBA', size, (120, 90, 60, 255))


def _transparent_rgba(size=(20, 20)):
    """RGBA image whose surround is transparent (a 'cleaned' subject)."""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    subject = Image.new('RGBA', (size[0] // 2, size[1] // 2), (200, 200, 200, 255))
    img.paste(subject, (size[0] // 4, size[1] // 4))
    return img


def _spy_removal(monkeypatch):
    """Replace remove_background with a spy that records calls and returns a
    transparent stand-in, so the rest of load_image keeps working."""
    calls = []

    def fake(img):
        calls.append(img)
        out = img.convert('RGBA')
        out.putalpha(0)
        return out

    monkeypatch.setattr(ImageProcessor, 'remove_background', fake)
    return calls


@pytest.fixture
def opaque_image_path(tmp_path):
    p = tmp_path / "opaque.png"
    _opaque_rgb((40, 40)).save(p)
    return str(p)


@pytest.fixture
def transparent_image_path(tmp_path):
    p = tmp_path / "transparent.png"
    _transparent_rgba((40, 40)).save(p)
    return str(p)


# --- has_background ------------------------------------------------------

def test_has_background_true_for_rgb_photo():
    assert ImageProcessor.has_background(_opaque_rgb()) is True


def test_has_background_true_for_fully_opaque_rgba():
    assert ImageProcessor.has_background(_opaque_rgba()) is True


def test_has_background_false_for_transparent_subject():
    assert ImageProcessor.has_background(_transparent_rgba()) is False


def test_has_background_true_for_tiny_transparent_fraction():
    # Fewer than ~1% transparent pixels still counts as 'background present'.
    img = Image.new('RGBA', (100, 100), (10, 10, 10, 255))
    img.putpixel((0, 0), (0, 0, 0, 0))  # 1 / 10000 = 0.01% transparent
    assert ImageProcessor.has_background(img) is True


# --- remove_background ---------------------------------------------------

def test_remove_background_raises_clear_error_without_rembg(monkeypatch):
    """A helpful ImportError when the optional dependency is missing."""
    monkeypatch.setitem(sys.modules, 'rembg', None)
    with pytest.raises(ImportError, match='rembg'):
        ImageProcessor.remove_background(_opaque_rgba())


@pytest.mark.skipif(not _HAS_REMBG, reason="rembg not installed")
def test_remove_background_real_returns_rgba():
    cleaned = ImageProcessor.remove_background(_opaque_rgba((64, 64)))
    assert cleaned.mode == 'RGBA'
    assert cleaned.size == (64, 64)


# --- load_image dispatch -------------------------------------------------

def test_load_image_auto_removes_when_background_present(opaque_image_path, monkeypatch):
    calls = _spy_removal(monkeypatch)
    ImageProcessor.load_image(opaque_image_path, remove_background='auto')
    assert len(calls) == 1


def test_load_image_auto_skips_when_already_transparent(transparent_image_path, monkeypatch):
    calls = _spy_removal(monkeypatch)
    ImageProcessor.load_image(transparent_image_path, remove_background='auto')
    assert calls == []


def test_load_image_false_never_removes(opaque_image_path, monkeypatch):
    calls = _spy_removal(monkeypatch)
    ImageProcessor.load_image(opaque_image_path, remove_background=False)
    assert calls == []


def test_load_image_true_always_removes(transparent_image_path, monkeypatch):
    calls = _spy_removal(monkeypatch)
    ImageProcessor.load_image(transparent_image_path, remove_background=True)
    assert len(calls) == 1


def test_load_image_rejects_invalid_remove_background(opaque_image_path):
    with pytest.raises(ValueError):
        ImageProcessor.load_image(opaque_image_path, remove_background='yes')

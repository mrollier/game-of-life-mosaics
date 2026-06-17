"""Tests for background detection and removal in ImageProcessor."""

import sys
import numpy as np
import pytest
from PIL import Image

from src.gol_mosaics import ImageProcessor


try:
    import rembg  # noqa: F401
    _HAS_REMBG = True
except (ImportError, SystemExit):
    # rembg sys.exit()s on import when its onnxruntime backend is missing.
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


# --- enhance_contrast ----------------------------------------------------

def _gradient(width=256):
    arr = np.tile(np.linspace(0, 255, width).astype(np.uint8), (4, 1))
    return Image.fromarray(arr, mode='L')


def test_enhance_contrast_identity_when_zero():
    img = _gradient()
    out = ImageProcessor.enhance_contrast(img, contrast=0)
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_enhance_contrast_preserves_black_and_white():
    out = np.asarray(ImageProcessor.enhance_contrast(_gradient(), contrast=6))
    assert out[0, 0] == 0
    assert out[0, -1] == 255


def test_enhance_contrast_darkens_lows_and_lightens_highs():
    dark = Image.new('L', (1, 1), 64)
    light = Image.new('L', (1, 1), 192)
    d = int(np.asarray(ImageProcessor.enhance_contrast(dark, contrast=6))[0, 0])
    l = int(np.asarray(ImageProcessor.enhance_contrast(light, contrast=6))[0, 0])
    assert d < 64
    assert l > 192


def test_enhance_contrast_returns_grayscale_same_size():
    out = ImageProcessor.enhance_contrast(_gradient(128), contrast=5)
    assert out.mode == 'L'
    assert out.size == (128, 4)


# --- provider selection --------------------------------------------------

def test_preferred_providers_prefers_cuda():
    avail = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    assert ImageProcessor._preferred_providers(avail) == [
        'CUDAExecutionProvider', 'CPUExecutionProvider'
    ]


def test_preferred_providers_uses_coreml_on_apple_silicon():
    avail = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    assert ImageProcessor._preferred_providers(avail) == [
        'CoreMLExecutionProvider', 'CPUExecutionProvider'
    ]


def test_preferred_providers_cpu_only():
    assert ImageProcessor._preferred_providers(['CPUExecutionProvider']) == [
        'CPUExecutionProvider'
    ]


def test_preferred_providers_always_ends_with_cpu_fallback():
    avail = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
    result = ImageProcessor._preferred_providers(avail)
    assert result[0] == 'CUDAExecutionProvider'
    assert result[-1] == 'CPUExecutionProvider'


# --- remove_background ---------------------------------------------------

def test_remove_background_raises_clear_error_without_rembg(monkeypatch):
    """A helpful ImportError when the optional dependency is missing."""
    monkeypatch.setitem(sys.modules, 'rembg', None)
    with pytest.raises(ImportError, match='rembg'):
        ImageProcessor.remove_background(_opaque_rgba())


def test_remove_background_handles_missing_backend(monkeypatch):
    """rembg calls sys.exit() when its onnxruntime backend is absent; surface
    that as a clear ImportError instead of letting it kill the process."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split('.')[0] == 'rembg':
            raise SystemExit("No onnxruntime backend found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError, match='rembg'):
        ImageProcessor.remove_background(_opaque_rgba())


@pytest.mark.skipif(not _HAS_REMBG, reason="rembg not installed")
def test_remove_background_real_returns_rgba():
    cleaned = ImageProcessor.remove_background(_opaque_rgba((64, 64)))
    assert cleaned.mode == 'RGBA'
    assert cleaned.size == (64, 64)


@pytest.mark.skipif(not _HAS_REMBG, reason="rembg not installed")
def test_rembg_session_cached_per_model(monkeypatch):
    import rembg
    created = []
    monkeypatch.setattr(ImageProcessor, '_REMBG_SESSIONS', {}, raising=False)
    monkeypatch.setattr(rembg, 'new_session',
                        lambda model, providers=None: created.append(model) or object())
    a1 = ImageProcessor._rembg_session('u2net')
    a2 = ImageProcessor._rembg_session('u2net')
    b = ImageProcessor._rembg_session('silueta')
    assert a1 is a2          # same model reuses the cached session
    assert a1 is not b       # different model gets its own session
    assert created == ['u2net', 'silueta']


@pytest.mark.skipif(not _HAS_REMBG, reason="rembg not installed")
def test_remove_background_forwards_model_and_alpha_matting(monkeypatch):
    import rembg
    monkeypatch.setattr(ImageProcessor, '_REMBG_SESSIONS', {}, raising=False)
    monkeypatch.setattr(rembg, 'new_session', lambda model, providers=None: f'SESS:{model}')
    captured = {}

    def fake_remove(img, session=None, alpha_matting=False, **kwargs):
        captured.update(session=session, alpha_matting=alpha_matting, kwargs=kwargs)
        return img

    monkeypatch.setattr(rembg, 'remove', fake_remove)
    ImageProcessor.remove_background(
        _opaque_rgba(), model='isnet-general-use',
        alpha_matting=True, alpha_matting_erode_size=5,
    )
    assert captured['session'] == 'SESS:isnet-general-use'
    assert captured['alpha_matting'] is True
    assert captured['kwargs'].get('alpha_matting_erode_size') == 5


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

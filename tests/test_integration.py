"""Integration tests for complete pipeline."""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from src.gol_mosaics import MosaicGenerator, ColorScheme


@pytest.fixture
def test_image_path():
    """Create a temporary test image."""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')

    # Add some pattern
    pixels = img.load()
    for i in range(30, 70):
        for j in range(30, 70):
            pixels[i, j] = (0, 0, 0)  # Black square in middle

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
        img.save(tmp, format='PNG')
        path = tmp.name

    yield path

    # Cleanup
    os.unlink(path)


@pytest.fixture
def transparent_image_path():
    """A subject on a transparent background (background already removed)."""
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    subject = Image.new('RGBA', (50, 50), (30, 30, 30, 255))
    img.paste(subject, (25, 25))

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
        img.save(tmp, format='PNG')
        path = tmp.name

    yield path

    os.unlink(path)


def test_auto_default_skips_removal_for_transparent_image(transparent_image_path):
    """The 'auto' default leaves an already-transparent image untouched, so the
    full pipeline runs end-to-end without needing the optional rembg package."""
    generator = MosaicGenerator(level=3, grid_size=10)
    result = generator.generate_from_image(transparent_image_path, supersample=12)

    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


def test_outer_rim_is_transparent(transparent_image_path):
    """The rotation/padding rim outside the photo footprint is transparent,
    while the subject near the centre stays opaque."""
    generator = MosaicGenerator(level=3, grid_size=10)
    result = generator.generate_from_image(transparent_image_path, supersample=12)
    arr = np.asarray(result)

    # The four corners are rotation-fill rim -> fully transparent
    assert arr[0, 0, 3] == 0
    assert arr[0, -1, 3] == 0
    assert arr[-1, 0, 3] == 0
    assert arr[-1, -1, 3] == 0

    # The subject near the centre stays opaque
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    assert arr[cy, cx, 3] == 255


def test_rim_color_fills_rim_when_given(transparent_image_path):
    """A given rim_color fills the rim opaquely instead of making it transparent."""
    generator = MosaicGenerator(level=3, grid_size=10)
    result = generator.generate_from_image(
        transparent_image_path, supersample=12, rim_color=(255, 255, 255))
    arr = np.asarray(result)
    assert tuple(int(v) for v in arr[0, 0]) == (255, 255, 255, 255)


def test_full_pipeline(test_image_path):
    """Test complete mosaic generation pipeline."""
    generator = MosaicGenerator(level=3, grid_size=10)
    # Use supersample=12 which divides the mosaic width (144) evenly.
    # remove_background=False: this fixture is an opaque test pattern, not a
    # photo needing background removal (and avoids the optional rembg dep).
    result = generator.generate_from_image(test_image_path, supersample=12, remove_background=False)

    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'
    assert result.size[0] > 0
    assert result.size[1] > 0


def test_custom_colors(test_image_path):
    """Test mosaic generation with custom colors."""
    colors = ColorScheme.monochrome()
    generator = MosaicGenerator(level=3, grid_size=10, color_scheme=colors)
    # Use supersample=12 which divides the mosaic width (144) evenly.
    # remove_background=False: this fixture is an opaque test pattern, not a
    # photo needing background removal (and avoids the optional rembg dep).
    result = generator.generate_from_image(test_image_path, supersample=12, remove_background=False)

    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


def test_different_parameters(test_image_path):
    """Test mosaic with different parameters."""
    generator = MosaicGenerator(level=3, grid_size=10)
    result = generator.generate_from_image(
        test_image_path,
        empty_tiles_cutoff=0.5,
        alpha_cutoff=0.5,
        supersample=6,  # Use 6 which divides 144 evenly
        remove_background=False
    )

    assert isinstance(result, Image.Image)


def test_invalid_grid_size():
    """Test error for odd grid size."""
    with pytest.raises(ValueError):
        MosaicGenerator(level=3, grid_size=11)  # Odd number


def test_generator_repr():
    """Test string representation."""
    generator = MosaicGenerator(level=4, grid_size=30)
    repr_str = repr(generator)
    assert 'level=4' in repr_str
    assert 'grid_size=30' in repr_str

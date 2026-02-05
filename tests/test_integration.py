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


def test_full_pipeline(test_image_path):
    """Test complete mosaic generation pipeline."""
    generator = MosaicGenerator(level=3, grid_size=10)
    # Use supersample=12 which divides the mosaic width (144) evenly
    result = generator.generate_from_image(test_image_path, supersample=12)

    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'
    assert result.size[0] > 0
    assert result.size[1] > 0


def test_custom_colors(test_image_path):
    """Test mosaic generation with custom colors."""
    colors = ColorScheme.monochrome()
    generator = MosaicGenerator(level=3, grid_size=10, color_scheme=colors)
    # Use supersample=12 which divides the mosaic width (144) evenly
    result = generator.generate_from_image(test_image_path, supersample=12)

    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


def test_different_parameters(test_image_path):
    """Test mosaic with different parameters."""
    generator = MosaicGenerator(level=3, grid_size=10)
    result = generator.generate_from_image(
        test_image_path,
        empty_tiles_cutoff=0.5,
        alpha_cutoff=0.5,
        supersample=6  # Use 6 which divides 144 evenly
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

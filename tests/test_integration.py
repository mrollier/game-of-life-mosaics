"""Integration tests for complete pipeline."""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from src.gol_mosaics import MosaicGenerator, ColourScheme


@pytest.fixture
def test_image_path():
    """Create a temporary test image for pipeline testing.

    Opaque white background with a black cross that reaches the edges.
    The cross ensures the subject is never completely enclosed by the
    transparent area — avoiding the binary_fill_holes trap where an isolated
    subject gets merged into the ECA background mask.
    """
    img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 0))  # transparent bg
    pixels = img.load()
    # Vertical bar from top to bottom (subject reaches border → not enclosed)
    for i in range(100):
        for j in range(40, 60):
            pixels[i, j] = (0, 0, 0, 255)

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
        img.save(tmp, format='PNG')
        path = tmp.name

    yield path

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


def test_custom_colours(test_image_path):
    """Test mosaic generation with a custom colour scheme."""
    colours = ColourScheme.ugent()
    generator = MosaicGenerator(level=3, grid_size=10, colour_scheme=colours)
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


def test_gol_tiles_are_visible(test_image_path):
    """Regression test: GoL tile pattern must survive rendering.

    The output must contain the gol_pixel colour in the subject region.
    If the rim-painting logic incorrectly covers the entire subject area,
    gol_pixel will be absent and this test fails.
    """
    colours = ColourScheme.ugent()  # gol_pixel=#000000, gol_background=#FFFFFF
    generator = MosaicGenerator(level=3, grid_size=10, colour_scheme=colours)
    result = generator.generate_from_image(test_image_path, supersample=12, no_eca=True)

    arr = np.array(result)
    gol_pixel_rgb = [0, 0, 0]  # UGent gol_pixel is black

    matches = np.all(arr[:, :, :3] == gol_pixel_rgb, axis=-1)
    assert matches.any(), (
        "gol_pixel colour not found in rendered image — "
        "GoL tiles are being overwritten during compositing"
    )


def test_rim_colour_applied(test_image_path):
    """Regression test: rim_colour must appear in the outer corners."""
    rim_colour = '#FF0000'  # bright red — impossible to appear by accident
    colours = ColourScheme.custom(rim_colour=rim_colour)
    generator = MosaicGenerator(level=3, grid_size=10, colour_scheme=colours)
    result = generator.generate_from_image(test_image_path, supersample=12)

    arr = np.array(result)
    # Corners of the output image should contain rim_colour
    corner_size = 5
    corners = np.concatenate([
        arr[:corner_size, :corner_size].reshape(-1, 4),
        arr[:corner_size, -corner_size:].reshape(-1, 4),
        arr[-corner_size:, :corner_size].reshape(-1, 4),
        arr[-corner_size:, -corner_size:].reshape(-1, 4),
    ])
    red_pixels = np.all(corners[:, :3] == [255, 0, 0], axis=-1)
    assert red_pixels.any(), (
        "rim_colour not found in image corners — rim painting is not working"
    )


def test_generator_repr():
    """Test string representation."""
    generator = MosaicGenerator(level=4, grid_size=30)
    repr_str = repr(generator)
    assert 'level=4' in repr_str
    assert 'grid_size=30' in repr_str

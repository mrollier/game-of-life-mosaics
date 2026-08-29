"""Tests for MosaicRenderer class."""

import pytest
import numpy as np
from PIL import Image
from gol_mosaics import ColorScheme, MosaicRenderer
from gol_mosaics.renderer import hex_to_rgb


@pytest.fixture
def renderer():
    """Renderer with the default UGent scheme (white/black GoL, yellow/blue ECA)."""
    return MosaicRenderer(ColorScheme.ugent())


def test_render_gol_mosaic_colors(renderer):
    """0 maps to the background colour, 1 to the pixel colour, fully opaque."""
    img = renderer.render_gol_mosaic(np.array([[0, 1], [1, 0]]))

    assert img.mode == 'RGBA'
    assert img.size == (2, 2)
    arr = np.array(img)
    assert tuple(arr[0, 0]) == (255, 255, 255, 255)  # background: white
    assert tuple(arr[0, 1]) == (0, 0, 0, 255)        # pixel: black
    assert tuple(arr[1, 0]) == (0, 0, 0, 255)
    assert tuple(arr[1, 1]) == (255, 255, 255, 255)


def test_render_gol_mosaic_rejects_non_2d(renderer):
    """Non-2D input raises ValueError."""
    with pytest.raises(ValueError):
        renderer.render_gol_mosaic(np.zeros(4))
    with pytest.raises(ValueError):
        renderer.render_gol_mosaic(np.zeros((2, 2, 2)))


def test_render_eca_overlay_values(renderer):
    """0 stays transparent; 1 and 2 map to the opaque ECA colours."""
    overlay = renderer.render_eca_overlay(np.array([[0, 1, 2]]))

    assert overlay.mode == 'RGBA'
    arr = np.array(overlay)
    assert tuple(arr[0, 0]) == (0, 0, 0, 0)          # transparent
    assert tuple(arr[0, 1]) == (255, 210, 0, 255)    # eca_background #FFD200
    assert tuple(arr[0, 2]) == (30, 100, 200, 255)   # eca_pixel #1E64C8


def test_render_eca_overlay_rejects_non_2d(renderer):
    """Non-2D input raises ValueError."""
    with pytest.raises(ValueError):
        renderer.render_eca_overlay(np.zeros(4))


def test_composite_respects_overlay_transparency(renderer):
    """Composite shows the base through transparent overlay pixels and the
    overlay colour where it is opaque."""
    base = renderer.render_gol_mosaic(np.zeros((2, 2), dtype=int))  # all white
    overlay = renderer.render_eca_overlay(np.array([[0, 1], [2, 0]]))

    result = np.array(renderer.composite(base, overlay))
    assert tuple(result[0, 0]) == (255, 255, 255, 255)  # base shows through
    assert tuple(result[0, 1]) == (255, 210, 0, 255)    # overlay background
    assert tuple(result[1, 0]) == (30, 100, 200, 255)   # overlay pixel
    assert tuple(result[1, 1]) == (255, 255, 255, 255)


def test_composite_rejects_size_and_mode_mismatch(renderer):
    """Different sizes or non-RGBA modes raise ValueError."""
    base = Image.new('RGBA', (4, 4))
    with pytest.raises(ValueError):
        renderer.composite(base, Image.new('RGBA', (2, 2)))
    with pytest.raises(ValueError):
        renderer.composite(base.convert('RGB'), Image.new('RGBA', (4, 4)))


def test_hex_to_rgb():
    """Hex parsing handles '#' prefix and both cases."""
    assert MosaicRenderer._hex_to_rgb('#FFFFFF') == (255, 255, 255)
    assert MosaicRenderer._hex_to_rgb('#1E64C8') == (30, 100, 200)
    assert MosaicRenderer._hex_to_rgb('#ff0099') == (255, 0, 153)
    assert MosaicRenderer._hex_to_rgb('000000') == (0, 0, 0)


def test_render_full_mosaic_matches_manual_pipeline(renderer):
    """render_full_mosaic is exactly render_gol_mosaic + render_eca_overlay
    + composite (the pipeline uses it as the single rendering entry point)."""
    gol = np.array([[0, 1, 0], [1, 0, 1]])
    eca = np.array([[0, 0, 2], [1, 0, 0]])

    combined = renderer.render_full_mosaic(gol, eca)
    manual = renderer.composite(
        renderer.render_gol_mosaic(gol),
        renderer.render_eca_overlay(eca)
    )
    assert np.array_equal(np.array(combined), np.array(manual))


def test_render_eca_overlay_paints_the_fill_state(renderer):
    """A lone filler state lands on the fill colour, ramp or no ramp."""
    overlay = np.asarray(renderer.render_eca_overlay(np.array([[2, 3]])))
    assert tuple(overlay[0, 1])[:3] == hex_to_rgb(renderer.color_scheme.fill)

    tinted = MosaicRenderer(ColorScheme(eca_pixel='#1E64C8',
                                        fill_pixel='#FF0000'))
    painted = np.asarray(tinted.render_eca_overlay(np.array([[2, 3]])))
    assert tuple(painted[0, 0]) == (30, 100, 200, 255)
    assert tuple(painted[0, 1]) == (255, 0, 0, 255)


def test_render_eca_overlay_ramps_the_filler_levels():
    """Filler states walk from the ECA pixel colour to the fill colour."""
    tinted = MosaicRenderer(ColorScheme(eca_pixel='#000000',
                                        fill_pixel='#FFFFFF'))
    row = np.arange(2, 8).reshape(1, 6)
    painted = np.asarray(tinted.render_eca_overlay(row, layers=6))[0, :, 0]
    assert list(painted) == [0, 51, 102, 153, 204, 255]
    assert (np.diff(painted.astype(int)) > 0).all(), "the ramp must be monotone"


def test_render_eca_overlay_ramp_ignores_an_empty_top_layer():
    """`layers` is trusted over the mask, so a missing layer cannot shorten it."""
    tinted = MosaicRenderer(ColorScheme(eca_pixel='#000000',
                                        fill_pixel='#FFFFFF'))
    full = np.asarray(tinted.render_eca_overlay(
        np.array([[3, 4, 5]]), layers=4))[0, :, 0]
    short = np.asarray(tinted.render_eca_overlay(
        np.array([[3, 4]]), layers=4))[0, :, 0]
    assert list(short) == list(full[:2])

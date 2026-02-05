"""Tests for ColorScheme class."""

import pytest
from src.gol_mosaics.colors import ColorScheme


def test_colorscheme_ugent():
    """Test UGent color preset."""
    colors = ColorScheme.ugent()
    assert colors.gol_background == '#FFFFFF'
    assert colors.gol_pixel == '#000000'
    assert colors.eca_background == '#FFD200'
    assert colors.eca_pixel == '#1E64C8'


def test_colorscheme_monochrome():
    """Test monochrome preset."""
    colors = ColorScheme.monochrome()
    assert colors.gol_background == '#FFFFFF'
    assert colors.gol_pixel == '#000000'
    assert colors.eca_background == '#FFFFFF'
    assert colors.eca_pixel == '#000000'

    # Custom colors
    colors = ColorScheme.monochrome(foreground='#FF0000', background='#00FF00')
    assert colors.gol_pixel == '#FF0000'
    assert colors.gol_background == '#00FF00'


def test_colorscheme_inverted():
    """Test inverted preset."""
    colors = ColorScheme.inverted()
    assert colors.gol_background == '#000000'
    assert colors.gol_pixel == '#FFFFFF'


def test_colorscheme_to_dict():
    """Test conversion to dictionary."""
    colors = ColorScheme.ugent()
    d = colors.to_dict()
    assert isinstance(d, dict)
    assert 'gol_background' in d
    assert 'gol_pixel' in d
    assert 'eca_background' in d
    assert 'eca_pixel' in d


def test_colorscheme_immutable():
    """Test that ColorScheme is immutable."""
    colors = ColorScheme.ugent()
    with pytest.raises(Exception):  # FrozenInstanceError
        colors.gol_background = '#000000'

"""Tests for ColorScheme class."""

import pytest
from gol_mosaics.colors import ColorScheme


def test_colorscheme_ugent():
    """Test UGent colour preset."""
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

    # Custom colours
    colors = ColorScheme.monochrome(foreground='#FF0000', background='#00FF00')
    assert colors.gol_pixel == '#FF0000'
    assert colors.gol_background == '#00FF00'


def test_colorscheme_inverted():
    """Test inverted preset."""
    colors = ColorScheme.inverted()
    assert colors.gol_background == '#000000'
    assert colors.gol_pixel == '#FFFFFF'


def _luminance(hex_color):
    """Perceived luminance of a '#rrggbb' colour in [0, 1]."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def test_colorscheme_warhol_returns_valid_hex():
    """All four colours are '#rrggbb' strings."""
    import re
    colors = ColorScheme.warhol()
    for value in colors.to_dict().values():
        assert re.fullmatch(r'#[0-9a-fA-F]{6}', value), value


def test_colorscheme_warhol_dark_on_light():
    """dark_on_light picks a GoL pixel darker than its background (the dark
    and light palettes are disjoint in luminance)."""
    for _ in range(20):
        colors = ColorScheme.warhol(dark_on_light=True)
        assert _luminance(colors.gol_pixel) < _luminance(colors.gol_background)


def test_colorscheme_warhol_force_white():
    """force_white pins the GoL background to pure white."""
    for _ in range(5):
        assert ColorScheme.warhol(force_white=True).gol_background == '#FFFFFF'


def test_colorscheme_warhol_eca_colors_distinct():
    """The two ECA colours are always distinct (drawn without replacement)."""
    for _ in range(10):
        colors = ColorScheme.warhol()
        assert colors.eca_background != colors.eca_pixel


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

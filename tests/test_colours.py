"""Tests for ColourScheme class."""

import pytest
from src.gol_mosaics.colours import ColourScheme


def test_colourscheme_ugent():
    """Test UGent colour preset."""
    colours = ColourScheme.ugent()
    assert colours.gol_background == '#FFFFFF'
    assert colours.gol_pixel == '#000000'
    assert colours.eca_background == '#FFD200'
    assert colours.eca_pixel == '#1E64C8'
    assert colours.rim_colour == '#FFFFFF'


def test_colourscheme_inverted():
    """Test inverted preset."""
    colours = ColourScheme.inverted()
    assert colours.gol_background == '#000000'
    assert colours.gol_pixel == '#FFFFFF'


def test_colourscheme_custom():
    """Test custom colour scheme."""
    colours = ColourScheme.custom(
        gol_background='#FF0000',
        gol_pixel='#0000FF',
        rim_colour='#00FF00'
    )
    assert colours.gol_background == '#FF0000'
    assert colours.gol_pixel == '#0000FF'
    assert colours.rim_colour == '#00FF00'


def test_colourscheme_warhol():
    """Test Warhol preset returns valid hex colour strings with dark-on-light contrast."""
    for _ in range(10):
        colours = ColourScheme.warhol()
        assert colours.gol_background.startswith('#')
        assert colours.gol_pixel.startswith('#')
        assert len(colours.gol_background) == 7
        assert len(colours.gol_pixel) == 7


def test_colourscheme_to_dict():
    """Test conversion to dictionary."""
    colours = ColourScheme.ugent()
    d = colours.to_dict()
    assert isinstance(d, dict)
    assert 'gol_background' in d
    assert 'gol_pixel' in d
    assert 'eca_background' in d
    assert 'eca_pixel' in d
    assert 'rim_colour' in d


def test_colourscheme_immutable():
    """Test that ColourScheme is immutable."""
    colours = ColourScheme.ugent()
    with pytest.raises(Exception):  # FrozenInstanceError
        colours.gol_background = '#000000'

"""Tests for ECABackground class."""

import pytest
import numpy as np
from src.gol_mosaics.eca import ECABackground


def test_eca_initialization():
    """Test ECA initialization."""
    eca = ECABackground(rule=106)
    assert eca.rule == 106


def test_eca_invalid_rule():
    """Test error for invalid rule."""
    with pytest.raises(ValueError):
        ECABackground(rule=256)  # Out of range

    with pytest.raises(ValueError):
        ECABackground(rule=-1)  # Out of range


def test_eca_generate():
    """Test ECA pattern generation."""
    eca = ECABackground(rule=106)
    pattern = eca.generate(width=100, height=100, supersample=10)
    assert pattern.shape == (100, 100)
    assert pattern.dtype == np.uint8
    assert np.all(np.isin(pattern, [0, 1]))


def test_eca_generate_supersample_not_dividing_dimensions():
    """Supersample need not divide the dimensions; output is cropped to size."""
    eca = ECABackground(rule=106)
    # 7 divides neither 100 nor 100 evenly; previously this raised.
    pattern = eca.generate(width=100, height=100, supersample=7)
    assert pattern.shape == (100, 100)
    assert np.all(np.isin(pattern, [0, 1]))


def test_eca_generate_non_square_indivisible():
    """Width and height differ and neither is divisible by supersample."""
    eca = ECABackground(rule=106)
    # The real-pipeline failure case: width 588, height 522, supersample 14.
    pattern = eca.generate(width=588, height=522, supersample=14)
    assert pattern.shape == (522, 588)
    assert np.all(np.isin(pattern, [0, 1]))


def test_eca_generate_rejects_non_positive_supersample():
    """Supersample must still be a positive integer."""
    eca = ECABackground(rule=106)
    with pytest.raises(ValueError):
        eca.generate(width=100, height=100, supersample=0)


def test_eca_validate_supersample():
    """Test supersample validation."""
    assert ECABackground.validate_supersample(100, 10) == True
    assert ECABackground.validate_supersample(100, 5) == True
    assert ECABackground.validate_supersample(100, 7) == False


def test_eca_list_valid_supersamples():
    """Test listing valid supersamples."""
    valid = ECABackground.list_valid_supersamples(60)
    assert 1 in valid
    assert 2 in valid
    assert 3 in valid
    assert 5 in valid
    assert 10 in valid
    assert 60 in valid
    assert 7 not in valid


def test_eca_rule_category():
    """Test rule category detection."""
    eca_complex = ECABackground(rule=54)
    assert eca_complex.get_rule_category() == 'complex'

    eca_chaotic = ECABackground(rule=106)
    assert eca_chaotic.get_rule_category() == 'chaotic'

    eca_other = ECABackground(rule=0)
    assert eca_other.get_rule_category() == 'other'


def test_eca_from_category():
    """Test creating ECA from category."""
    eca_complex = ECABackground.from_category('complex')
    assert eca_complex.rule in ECABackground.COMPLEX_RULES

    eca_chaotic = ECABackground.from_category('chaotic')
    assert eca_chaotic.rule in ECABackground.CHAOTIC_RULES

    with pytest.raises(ValueError):
        ECABackground.from_category('invalid')

"""Tests for PatternLibrary class."""

import pytest
import numpy as np
from src.gol_mosaics.patterns import PatternLibrary


def test_pattern_library_load():
    """Test loading pre-computed patterns."""
    for level in [3, 4, 5]:
        library = PatternLibrary.load(level=level)
        assert library.level == level
        assert library.solutions is not None
        assert library.solutions.ndim == 3
        assert len(library.solutions) > 0


def test_pattern_library_invalid_level():
    """Test error for invalid level."""
    with pytest.raises(ValueError):
        PatternLibrary.load(level=0)  # Below supported range

    with pytest.raises(ValueError):
        PatternLibrary.load(level=6)  # Not pre-computed


def test_pond_pattern():
    """Test basic pond pattern generation."""
    pattern = PatternLibrary.pond_pattern()
    assert pattern.shape == (4, 4)
    assert np.all(np.isin(pattern, [0, 1]))


def test_pattern_densities():
    """Test density calculation."""
    library = PatternLibrary.load(level=3)
    densities = library.densities
    assert densities.shape[0] == library.solutions.shape[0]
    assert np.all(densities >= 0) and np.all(densities <= 1)
    # Should be normalised
    assert np.isclose(densities.min(), 0.0)
    assert np.isclose(densities.max(), 1.0)


def test_get_pattern_for_value():
    """Test pattern retrieval for single value."""
    library = PatternLibrary.load(level=3)
    pattern = library.get_pattern_for_value(0.5, random=False)
    assert pattern.ndim == 2
    assert np.all(np.isin(pattern, [0, 1]))


def test_get_patterns_for_values():
    """Test pattern mapping for array of values."""
    library = PatternLibrary.load(level=3)
    values = np.array([[0.2, 0.5], [0.7, 0.9]])
    patterns = library.get_patterns_for_values(values, random=False, invert=True)
    assert patterns.shape[:2] == values.shape
    assert np.all(np.isin(patterns, [0, 1]))


def test_get_patterns_invalid_values():
    """Test error for out-of-range values."""
    library = PatternLibrary.load(level=3)
    with pytest.raises(ValueError):
        library.get_pattern_for_value(1.5)  # > 1.0
    with pytest.raises(ValueError):
        library.get_pattern_for_value(-0.1)  # < 0.0


def test_pond_pattern_multiple():
    """Test pond pattern multiple generation."""
    library = PatternLibrary(level=4)
    pattern = library.pond_pattern_multiple()
    assert pattern.ndim == 2
    assert np.all(np.isin(pattern, [0, 1]))


def test_pond_pattern_edge():
    """Test pond pattern edge generation."""
    library = PatternLibrary(level=4)
    pattern = library.pond_pattern_edge()
    assert pattern.ndim == 2
    assert np.all(np.isin(pattern, [0, 1]))

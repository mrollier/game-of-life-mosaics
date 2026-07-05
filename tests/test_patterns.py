"""Tests for PatternLibrary class."""

import pytest
import numpy as np
from scipy.ndimage import binary_fill_holes
from gol_mosaics.patterns import PatternLibrary


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


def test_pattern_library_construction_allows_generation_levels():
    """Constructing beyond level 5 is allowed (generate() supports level 6);
    only loading is restricted to the pre-computed range."""
    assert PatternLibrary(level=6).level == 6
    with pytest.raises(ValueError):
        PatternLibrary(level=0)


def test_pattern_library_load_is_cached():
    """load() returns one shared, read-only instance per level."""
    assert PatternLibrary.load(3) is PatternLibrary.load(3)
    assert PatternLibrary.load(3) is not PatternLibrary.load(4)


def test_tile_geometry_properties():
    """tile_shape matches the edge pattern; tile_pad_size matches the
    interlock formula previously inlined in the mosaic builder."""
    for level in [2, 3, 4, 5]:
        library = PatternLibrary(level=level)
        assert library.tile_shape == library.pond_pattern_edge().shape
        pond_width = library.pond_width
        expected = ((pond_width - 3) * (2 * level - 1) + 1 + 2) // 2
        assert library.tile_pad_size == expected


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


def test_batch_values_match_single_value_mapping():
    """With the identity cutoff (1.0), the batch mapper is elementwise
    equivalent to the single-value mapper (pins the vectorized refactor)."""
    library = PatternLibrary.load(level=3)
    values = np.array([[0.0, 0.2, 0.35], [0.5, 0.75, 1.0]])

    for invert in (True, False):
        batch = library.get_patterns_for_values(
            values, random=False, invert=invert, empty_tiles_cutoff=1.0)
        flat = batch.reshape(-1, *library.solutions.shape[1:])
        for idx, val in enumerate(values.ravel()):
            single = library.get_pattern_for_value(
                val, random=False, invert=invert)
            assert np.array_equal(flat[idx], single)


def test_empty_tiles_cutoff_boundary_is_exclusive():
    """A value strictly above the cutoff becomes an empty tile; a value AT the
    cutoff still maps to a pattern (pins '>' rather than '>=')."""
    library = PatternLibrary.load(level=3)
    cutoff = 0.65

    above = library.get_patterns_for_values(
        np.array([cutoff + 1e-6]), random=False, empty_tiles_cutoff=cutoff)
    assert not above.any()

    # At the cutoff the value is rescaled to 1.0, i.e. the sparsest pattern
    # under inversion -- which is a real still life, not an empty tile.
    at = library.get_patterns_for_values(
        np.array([cutoff]), random=False, invert=True,
        empty_tiles_cutoff=cutoff)
    expected = library.get_pattern_for_value(1.0, random=False, invert=True)
    assert np.array_equal(at[0], expected)
    assert at.any()


def test_get_patterns_for_values_validates_array_range():
    """Out-of-range values anywhere in the array raise ValueError."""
    library = PatternLibrary.load(level=3)
    with pytest.raises(ValueError):
        library.get_patterns_for_values(np.array([0.5, 1.5]), random=False)
    with pytest.raises(ValueError):
        library.get_patterns_for_values(np.array([-0.1, 0.5]), random=False)


def test_random_tie_break_picks_only_nearest_patterns():
    """When several patterns share the nearest density, random selection stays
    within that tie set and (given enough draws) uses more than one member."""
    library = PatternLibrary.load(level=3)
    densities = library.densities

    # Pick a density that occurs more than once (level 3 has duplicates).
    values, counts = np.unique(densities, return_counts=True)
    target = values[counts > 1][0]
    value = 1.0 - target  # invert=True maps this back onto `target`

    # The tie set exactly as the mapper sees it.
    diffs = np.abs(densities - (1.0 - value))
    tie_set = np.where(diffs == diffs.min())[0]
    assert len(tie_set) > 1

    np.random.seed(0)
    seen = set()
    for _ in range(30):
        pattern = library.get_pattern_for_value(value, random=True,
                                                invert=True)
        matches = [i for i in tie_set
                   if np.array_equal(pattern, library.solutions[i])]
        assert matches, "random pick fell outside the nearest-density tie set"
        seen.update(matches)
    assert len(seen) > 1, "30 draws never varied within the tie set"


def test_get_patterns_for_mask_exact_tiles():
    """Mask values >= alpha_cutoff yield empty tiles; values below yield the
    hole-filled densest pattern (pins '>=' and the exact filled tile)."""
    library = PatternLibrary.load(level=3)
    mask = np.array([[0.0, 0.5], [0.49999, 1.0]])

    tiles = library.get_patterns_for_mask(mask, alpha_cutoff=0.5)

    filled = binary_fill_holes(library.solutions[-1]).astype(int)
    flat = tiles.reshape(-1, *filled.shape)
    assert np.array_equal(flat[0], filled)   # 0.0 < 0.5 -> opaque tile
    assert not flat[1].any()                 # 0.5 >= 0.5 -> empty (pins >=)
    assert np.array_equal(flat[2], filled)
    assert not flat[3].any()


def test_get_patterns_for_mask_validates_array_range():
    """Out-of-range mask values raise ValueError."""
    library = PatternLibrary.load(level=3)
    with pytest.raises(ValueError):
        library.get_patterns_for_mask(np.array([0.5, 1.5]))
    with pytest.raises(ValueError):
        library.get_patterns_for_mask(np.array([-0.1]))


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

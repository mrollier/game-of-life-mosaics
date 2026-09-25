"""Tests for TileLibrary class."""

import pytest
import numpy as np
from scipy.ndimage import binary_fill_holes
from gol_mosaics.tile_library import TileLibrary


def test_tile_library_load():
    """Test loading pre-computed patterns."""
    for level in [3, 4, 5]:
        library = TileLibrary.load(level=level)
        assert library.level == level
        assert library.tiles is not None
        assert library.tiles.ndim == 3
        assert len(library.tiles) > 0


def test_tile_library_invalid_level():
    """Test error for invalid level."""
    with pytest.raises(ValueError):
        TileLibrary.load(level=0)  # Below supported range

    with pytest.raises(ValueError):
        TileLibrary.load(level=7)  # Not pre-computed


def test_library_construction_allows_any_level():
    """Any level can be wrapped with from_tiles (e.g. a level-7 SAT
    enumeration); only load() is restricted to the shipped levels."""
    assert TileLibrary(level=6).level == 6
    tiles = np.zeros((2, 42, 42), dtype=np.uint8)
    assert TileLibrary.from_tiles(tiles, 7).tiles.shape == (2, 42, 42)
    with pytest.raises(ValueError):
        TileLibrary(level=0)


def test_tile_library_load_is_cached():
    """load() returns one shared, read-only instance per level."""
    assert TileLibrary.load(3) is TileLibrary.load(3)
    assert TileLibrary.load(3) is not TileLibrary.load(4)


def test_tile_geometry_properties():
    """tile_size matches the pond frame; lattice_offset matches the
    interlock formula previously inlined in the mosaic builder."""
    from gol_mosaics.tile_domain import POND_WIDTH, pond_frame

    for level in [2, 3, 4, 5]:
        library = TileLibrary(level=level)
        assert library.tile_size == pond_frame(level).shape
        expected = ((POND_WIDTH - 3) * (2 * level - 1) + 1 + 2) // 2
        assert library.lattice_offset == expected
        assert np.array_equal(library.pond, pond_frame(level) > 0)


def test_pond():
    """The pond is a read-only 4x4 still life."""
    from gol_mosaics.life import is_still_life
    from gol_mosaics.tile_domain import POND

    assert POND.shape == (4, 4)
    assert np.all(np.isin(POND, [0, 1]))
    assert is_still_life(np.pad(POND, 1))
    with pytest.raises(ValueError):
        POND[0, 0] = 1


def test_pattern_densities():
    """Test density calculation."""
    library = TileLibrary.load(level=3)
    densities = library.densities
    assert densities.shape[0] == library.tiles.shape[0]
    assert np.all(densities >= 0) and np.all(densities <= 1)
    # Should be normalised
    assert np.isclose(densities.min(), 0.0)
    assert np.isclose(densities.max(), 1.0)


def test_tile_for_value():
    """Test pattern retrieval for single value."""
    library = TileLibrary.load(level=3)
    pattern = library.tile_for_value(0.5, random=False)
    assert pattern.ndim == 2
    assert np.all(np.isin(pattern, [0, 1]))


def test_tiles_for_values():
    """Test pattern mapping for array of values."""
    library = TileLibrary.load(level=3)
    values = np.array([[0.2, 0.5], [0.7, 0.9]])
    patterns = library.tiles_for_values(values, random=False, invert=True)
    assert patterns.shape[:2] == values.shape
    assert np.all(np.isin(patterns, [0, 1]))


def test_tiles_for_values_invalid_values():
    """Test error for out-of-range values."""
    library = TileLibrary.load(level=3)
    with pytest.raises(ValueError):
        library.tile_for_value(1.5)  # > 1.0
    with pytest.raises(ValueError):
        library.tile_for_value(-0.1)  # < 0.0


def test_batch_values_match_single_value_mapping():
    """With the identity cutoff (1.0), the batch mapper is elementwise
    equivalent to the single-value mapper (pins the vectorized refactor)."""
    library = TileLibrary.load(level=3)
    values = np.array([[0.0, 0.2, 0.35], [0.5, 0.75, 1.0]])

    for invert in (True, False):
        batch = library.tiles_for_values(
            values, random=False, invert=invert, empty_tiles_cutoff=1.0)
        flat = batch.reshape(-1, *library.tiles.shape[1:])
        for idx, val in enumerate(values.ravel()):
            single = library.tile_for_value(
                val, random=False, invert=invert)
            assert np.array_equal(flat[idx], single)


def test_empty_tiles_cutoff_boundary_is_exclusive():
    """A value strictly above the cutoff becomes an empty tile; a value AT the
    cutoff still maps to a pattern (pins '>' rather than '>=')."""
    library = TileLibrary.load(level=3)
    cutoff = 0.65

    above = library.tiles_for_values(
        np.array([cutoff + 1e-6]), random=False, empty_tiles_cutoff=cutoff)
    assert not above.any()

    # At the cutoff the value is rescaled to 1.0, i.e. the sparsest pattern
    # under inversion -- which is a real still life, not an empty tile.
    at = library.tiles_for_values(
        np.array([cutoff]), random=False, invert=True,
        empty_tiles_cutoff=cutoff)
    expected = library.tile_for_value(1.0, random=False, invert=True)
    assert np.array_equal(at[0], expected)
    assert at.any()


def test_tiles_for_values_validates_array_range():
    """Out-of-range values anywhere in the array raise ValueError."""
    library = TileLibrary.load(level=3)
    with pytest.raises(ValueError):
        library.tiles_for_values(np.array([0.5, 1.5]), random=False)
    with pytest.raises(ValueError):
        library.tiles_for_values(np.array([-0.1, 0.5]), random=False)


def test_random_tie_break_picks_only_nearest_patterns():
    """When several patterns share the nearest density, random selection stays
    within that tie set and (given enough draws) uses more than one member."""
    library = TileLibrary.load(level=3)
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
        pattern = library.tile_for_value(value, random=True,
                                                invert=True)
        matches = [i for i in tie_set
                   if np.array_equal(pattern, library.tiles[i])]
        assert matches, "random pick fell outside the nearest-density tie set"
        seen.update(matches)
    assert len(seen) > 1, "30 draws never varied within the tie set"


def test_tiles_for_mask_exact_tiles():
    """Mask values >= alpha_cutoff yield empty tiles; values below yield the
    hole-filled densest pattern (pins '>=' and the exact filled tile)."""
    library = TileLibrary.load(level=3)
    mask = np.array([[0.0, 0.5], [0.49999, 1.0]])

    tiles = library.tiles_for_mask(mask, alpha_cutoff=0.5)

    filled = binary_fill_holes(library.tiles[-1]).astype(int)
    flat = tiles.reshape(-1, *filled.shape)
    assert np.array_equal(flat[0], filled)   # 0.0 < 0.5 -> opaque tile
    assert not flat[1].any()                 # 0.5 >= 0.5 -> empty (pins >=)
    assert np.array_equal(flat[2], filled)
    assert not flat[3].any()


def test_tiles_for_mask_validates_array_range():
    """Out-of-range mask values raise ValueError."""
    library = TileLibrary.load(level=3)
    with pytest.raises(ValueError):
        library.tiles_for_mask(np.array([0.5, 1.5]))
    with pytest.raises(ValueError):
        library.tiles_for_mask(np.array([-0.1]))


def test_pond_lattice():
    from gol_mosaics.tile_domain import pond_lattice

    pattern = pond_lattice(4)
    assert pattern.shape == (24, 24)
    assert np.all(np.isin(pattern, [0, 1]))


def test_pond_frame():
    from gol_mosaics.tile_domain import pond_frame, pond_lattice

    frame = pond_frame(4)
    assert frame.shape == (24, 24)
    assert np.all(np.isin(frame, [0, 1]))
    assert np.all(frame <= pond_lattice(4))  # the frame is part of the lattice


# --- Dead-edge derivation, level-6 data, packed format --------------------

HISTORICAL_DEAD_EDGES = {
    2: [(2, 6), (3, 6)],
    3: [(2, 9), (3, 9), (5, 11), (5, 12)],
    4: [(2, 11), (3, 11), (5, 14), (5, 15), (6, 15)],
    5: [(2, 15), (3, 15), (5, 17), (5, 18), (6, 18), (8, 20), (8, 21)],
    6: [(2, 18), (3, 18), (5, 20), (5, 21), (6, 21), (8, 23), (8, 24), (9, 24)],
}


def test_derived_dead_edges_match_historical_lists():
    """The geometric derivation reproduces the historically hard-coded
    dead-edge lists exactly, up to D4 orbit closure (the solver propagates
    forcings orbit-wide, so orbit equality is the semantic contract)."""
    from gol_mosaics.tile_domain import derive_dead_edges_full, symmetric_coords

    for level, literals in HISTORICAL_DEAD_EDGES.items():
        n = 6 * level
        closure = {img for cell in literals
                   for img in symmetric_coords(*cell, n)}
        assert derive_dead_edges_full(level) == closure, level
        assert len(closure) == 12 * level - 8


def test_dead_edges_empty_for_level_1():
    from gol_mosaics.tile_domain import derive_dead_edges

    assert derive_dead_edges(1) == []


def test_load_level_6():
    """Level 6 loads from the packed orbit-bit file and expands correctly."""
    library = TileLibrary.load(level=6)
    assert library.tiles.shape == (332321, 36, 36)
    assert library.tiles.dtype == np.uint8
    assert 0.0 <= library.densities.min() <= library.densities.max() <= 1.0


def test_packed_roundtrip():
    """Domain.pack/unpack are inverse on shipped data."""
    from gol_mosaics.tile_domain import build_domain

    for level in (3, 4):
        reference = TileLibrary.load(level).tiles
        packed = build_domain(level).pack(reference)
        assert np.array_equal(build_domain(level).unpack(packed),
                              reference.astype(np.uint8))


# --- Square-shape libraries ------------------------------------------------

SQUARE_COUNTS = {3: 3, 4: 65, 5: 10398}


def test_load_square_levels():
    """Square libraries load from the packed data files with the known
    censuses and expose their scheme."""
    from gol_mosaics.tile_scheme import square_scheme

    for level, count in SQUARE_COUNTS.items():
        library = TileLibrary.load(level=level, layout="square")
        assert library.layout == "square"
        assert library.level == level
        n = 6 * level
        assert library.tiles.shape == (count, n, n)
        assert library.tiles.dtype == np.uint8
        assert 0.0 <= library.densities.min() <= library.densities.max() <= 1.0
        assert library.scheme.name == square_scheme(level).name


def test_load_square_invalid_levels_and_shapes():
    """Squares ship for levels 3-5 only; unknown shapes are rejected."""
    for level in (1, 2, 6):
        with pytest.raises(ValueError):
            TileLibrary.load(level=level, layout="square")
    with pytest.raises(ValueError):
        TileLibrary.load(level=4, layout="hexagon")


def test_square_and_diamond_libraries_cached_separately():
    assert (TileLibrary.load(4, layout="square")
            is TileLibrary.load(4, layout="square"))
    assert (TileLibrary.load(4, layout="square")
            is not TileLibrary.load(4))
    # default shape is the historical diamond
    assert TileLibrary.load(4) is TileLibrary.load(4, layout="diamond")


def test_square_library_rejects_diamond_lattice_arithmetic():
    """The diamond grid offset is meaningless for squares and must fail
    loudly rather than produce a wrong mosaic; the pond frame and scheme
    come from the square scheme instead."""
    from gol_mosaics.tile_scheme import square_scheme

    library = TileLibrary.load(level=4, layout="square")
    with pytest.raises(ValueError):
        library.lattice_offset
    assert np.array_equal(library.pond, square_scheme(4).frame)


def test_square_tile_size_from_scheme():
    """tile_size is the n x n bounding box for squares too."""
    assert TileLibrary.load(4, layout="square").tile_size == (24, 24)


def test_square_solutions_match_fresh_enumeration():
    """The shipped packed files must equal a fresh SAT enumeration."""
    pytest.importorskip("pysat")
    from gol_mosaics.tile_scheme import (enumerate_scheme_tiles,
                                         square_scheme)

    for level in (3, 4):
        shipped = TileLibrary.load(level, layout="square").tiles
        fresh = enumerate_scheme_tiles(square_scheme(level))
        assert np.array_equal(shipped, fresh)


def test_indices_for_values_agrees_with_tiles_for_values():
    """The index mapper is the selection half of tiles_for_values:
    tiles[idx] (with -1 meaning empty) must reproduce the tile grids."""
    for shape, level in (("diamond", 3), ("square", 4)):
        library = TileLibrary.load(level, layout=shape)
        values = np.array([[0.0, 0.3, 0.6], [0.7, 0.9, 1.0]])
        idx = library.indices_for_values(
            values, random=False, empty_tiles_cutoff=0.65)
        assert idx.shape == values.shape
        assert idx.dtype.kind == "i"
        assert (idx[values > 0.65] == -1).all()
        assert (idx[values <= 0.65] >= 0).all()
        patterns = library.tiles_for_values(
            values, random=False, empty_tiles_cutoff=0.65)
        expected = np.where((idx >= 0)[..., None, None],
                            library.tiles[np.clip(idx, 0, None)], 0)
        assert np.array_equal(patterns, expected)


def test_small_levels_by_exhaustive_expansion():
    """Levels 1-3 re-derived SAT-free: every free-orbit assignment is
    expanded and tested (2^1, 2^3, 2^10 candidates); the survivors must
    equal the shipped censuses byte for byte after the same canonical
    sort (the historical small-level files predate the canonical order)."""
    from gol_mosaics.sat_search import bruteforce_tiles

    def canonical(grids):
        order = sorted(range(len(grids)),
                       key=lambda i: (int(grids[i].sum()),
                                      grids[i].tobytes()))
        return grids[order]

    for level, expected in ((1, 1), (2, 2), (3, 7)):
        tiles = bruteforce_tiles(level)  # already canonically sorted
        assert len(tiles) == expected
        reference = TileLibrary.load(level).tiles.astype(np.uint8)
        assert np.array_equal(tiles, canonical(reference))


def test_nearest_density_indices_matches_dense_argmin():
    """The deterministic path equals argmin over the full difference matrix,
    including exact ties between two density values (0.5 between 0.25 and
    0.75 here), where the lowest index across both groups wins."""
    from gol_mosaics.tile_library import nearest_density_indices

    rng = np.random.default_rng(0)
    densities = rng.choice([0.0, 0.25, 0.75, 1.0], size=200)
    wanted = np.concatenate([rng.random(500), [0.5, 0.125, 0.875, 0.0, 1.0]])
    dense = np.abs(densities[None, :] - wanted[:, None]).argmin(axis=1)
    got = nearest_density_indices(densities, wanted, random=False)
    assert np.array_equal(got, dense)


def test_nearest_density_indices_random_stays_in_nearest_group():
    from gol_mosaics.tile_library import nearest_density_indices

    rng = np.random.default_rng(1)
    densities = rng.choice(np.linspace(0, 1, 9), size=300)
    wanted = rng.random(2000)
    got = nearest_density_indices(densities, wanted, rng=rng)
    gaps = np.abs(densities[got] - wanted)
    best = np.abs(densities[None, :] - wanted[:, None]).min(axis=1)
    assert np.allclose(gaps, best)
    # every member of a large tie group gets drawn
    group = np.flatnonzero(densities == densities[got[0]])
    hits = nearest_density_indices(densities,
                                   np.full(4000, densities[got[0]]), rng=rng)
    assert set(hits) == set(group)


def test_level_6_selection_does_not_allocate_a_dense_matrix():
    """120 x 120 tiles against 332,321 level-6 densities: the old dense
    difference matrix alone was 115 GB. The search allocates O(tiles)."""
    import tracemalloc

    library = TileLibrary.load(level=6)
    library.densities  # computed and cached outside the measurement
    values = np.random.default_rng(2).random((120, 120))
    tracemalloc.start()
    try:
        indices = library.indices_for_values(values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert indices.shape == (120, 120)
    assert peak < 64 * 2 ** 20, f"peak {peak / 2 ** 20:.0f} MB"


def test_package_import_does_not_touch_gurobi():
    """gurobipy is imported only by the legacy ILP, and only when it runs."""
    import subprocess
    import sys

    code = ("import sys; sys.modules['gurobipy'] = None; "
            "import gol_mosaics, gol_mosaics.legacy_ilp; "
            "from gol_mosaics.tile_library import TileLibrary; "
            "TileLibrary.load(3); print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_legacy_ilp_reports_missing_gurobi(monkeypatch):
    import sys

    from gol_mosaics.legacy_ilp import generate_tiles

    monkeypatch.setitem(sys.modules, "gurobipy", None)
    with pytest.raises(ImportError, match="sat_search"):
        generate_tiles(3)


@pytest.mark.parametrize("n", [3, 6, 11])
def test_canonical_order_matches_sorted_key(n):
    """lexsort on packed bits equals the (population, bytes) sort it
    replaced, duplicates included (both are stable)."""
    from gol_mosaics.tile_domain import canonical_order

    rng = np.random.default_rng(n)
    grids = rng.integers(0, 2, (400, n, n)).astype(np.uint8)
    grids[200:220] = grids[:20]  # duplicates keep their relative order
    expected = sorted(range(len(grids)),
                      key=lambda i: (int(grids[i].sum()), grids[i].tobytes()))
    assert list(canonical_order(grids)) == expected

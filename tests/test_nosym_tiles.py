"""
Tests for the no-symmetry (identity-domain) diamond tile census
(gol_mosaics.nosym_tiles).

The nosym variant drops the D4 symmetry requirement while keeping every
other tile constraint: still life under the configured rule, pond frame
forced alive, and the full orbit-closed dead-edge set forced dead.

Measured censuses (levels 1-3, validated against brute force at levels
1-2 and by Burnside consistency at level 3): 1, 2, 1061 raw grids,
forming 1, 2, 181 D4 equivalence classes. The 7 singleton classes at
level 3 are exactly the historical symmetric census.
"""

import numpy as np
import pytest

from gol_mosaics.tile_domain import derive_dead_edges_full
from gol_mosaics.nosym_tiles import (
    EXPECTED_FREE_CELLS,
    build_nosym_domain,
    d4_classes,
    d4_fixed_counts,
    enumerate_nosym_tiles,
    load_nosym_tiles,
    pack_nosym_solutions,
    unpack_nosym_solutions,
)

# Raw censuses without the symmetry requirement, and their partition into
# D4 equivalence classes (size -> number of classes).
NOSYM_CENSUS = {1: 1, 2: 2, 3: 1061}
NOSYM_CLASSES = {1: 1, 2: 2, 3: 181}
NOSYM_CLASS_SIZES_L3 = {1: 7, 2: 9, 4: 71, 8: 94}

# Pinned in REPRODUCE.md; guards the _domain_clauses extraction against
# perturbing the verified symmetric encoding.
SYMMETRIC_CNF_SHA256_LEVEL_3 = (
    "d4a1e80102921a3f379619582a576421692fdefb19ee12a5b6d360499709fbde"
)


def _byte_set(grids):
    return {np.ascontiguousarray(g).tobytes() for g in grids}


def test_symmetric_fingerprint_unchanged():
    pytest.importorskip("pysat")
    from gol_mosaics.sat_search import build_cnf
    assert build_cnf(3).sha256 == SYMMETRIC_CNF_SHA256_LEVEL_3


@pytest.mark.parametrize("level,expected",
                         sorted(EXPECTED_FREE_CELLS.items()))
def test_free_cell_counts(level, expected):
    domain = build_nosym_domain(level)
    assert len(domain.free_reps) == expected
    # identity orbits: every free rep is its own single-cell orbit
    assert (domain.orbit_sizes() == 1).all()


def test_forcings_spot_checks():
    domain = build_nosym_domain(3)
    for cell in derive_dead_edges_full(3):
        assert domain.constants[cell] == 0
    # border ring fully forced dead (keeps toroidal wraparound inert)
    n = domain.n
    for k in range(n):
        for cell in ((0, k), (n - 1, k), (k, 0), (k, n - 1)):
            assert domain.constants[cell] == 0
    # the pond frame contributes live forcings
    assert sum(domain.constants.values()) > 0


@pytest.mark.parametrize("level,expected", sorted(NOSYM_CENSUS.items()))
def test_raw_census(level, expected):
    pytest.importorskip("pysat")
    assert len(enumerate_nosym_tiles(level)) == expected


def test_all_tiles_stable():
    pytest.importorskip("pysat")
    from gol_mosaics.sat_search import rule_violations
    for level in (1, 2, 3):
        tiles = enumerate_nosym_tiles(level)
        assert not rule_violations(tiles).any()


@pytest.mark.parametrize("level", [1, 2, 3])
def test_symmetric_tiles_are_subset(level):
    pytest.importorskip("pysat")
    from gol_mosaics.sat_search import enumerate_tiles
    symmetric = enumerate_tiles(level)
    nosym = _byte_set(enumerate_nosym_tiles(level))
    assert _byte_set(symmetric) <= nosym


def test_d4_class_census():
    pytest.importorskip("pysat")
    for level in (1, 2, 3):
        tiles = enumerate_nosym_tiles(level)
        classes = d4_classes(tiles)
        assert len(classes) == NOSYM_CLASSES[level]
    sizes = {}
    for members in d4_classes(enumerate_nosym_tiles(3)).values():
        sizes[len(members)] = sizes.get(len(members), 0) + 1
    assert sizes == NOSYM_CLASS_SIZES_L3


def test_size1_classes_equal_symmetric_census():
    pytest.importorskip("pysat")
    from gol_mosaics.sat_search import enumerate_tiles
    tiles = enumerate_nosym_tiles(3)
    singletons = {
        np.ascontiguousarray(tiles[members[0]]).tobytes()
        for members in d4_classes(tiles).values()
        if len(members) == 1
    }
    assert singletons == _byte_set(enumerate_tiles(3))


def test_burnside_identity():
    pytest.importorskip("pysat")
    tiles = enumerate_nosym_tiles(3)
    fixed = d4_fixed_counts(tiles)
    assert len(fixed) == 8
    assert sum(fixed) == 8 * NOSYM_CLASSES[3] == 1448


@pytest.mark.parametrize("level", [1, 2])
def test_bruteforce_cross_check(level):
    """SAT-free ground truth: expand every free-cell assignment and keep
    the stable ones; must match the SAT enumeration byte-exactly."""
    pytest.importorskip("pysat")
    from gol_mosaics.sat_search import rule_violations
    domain = build_nosym_domain(level)
    k = len(domain.free_reps)
    bits = ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1)
    grids = domain.expand_many(bits.astype(np.uint8))
    stable = grids[~rule_violations(grids)]
    assert _byte_set(stable) == _byte_set(enumerate_nosym_tiles(level))


def test_pack_unpack_roundtrip():
    pytest.importorskip("pysat")
    tiles = enumerate_nosym_tiles(3)
    packed = pack_nosym_solutions(tiles, 3)
    assert packed.shape == (NOSYM_CENSUS[3], 9)  # ceil(68 / 8) bytes
    assert np.array_equal(unpack_nosym_solutions(packed, 3), tiles)


def test_pack_rejects_foreign_grids():
    pytest.importorskip("pysat")
    tiles = enumerate_nosym_tiles(3).copy()
    tiles[0, 0, 0] = 1  # flip a border cell that must be forced dead
    with pytest.raises(AssertionError):
        pack_nosym_solutions(tiles, 3)


def test_shipped_data_matches_enumeration():
    pytest.importorskip("pysat")
    assert np.array_equal(load_nosym_tiles(3), enumerate_nosym_tiles(3))

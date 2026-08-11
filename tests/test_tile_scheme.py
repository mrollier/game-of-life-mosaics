"""
Tests for the generalised tile-scheme abstraction (gol_mosaics.tile_scheme).

Two pillars:
  1. The diamond scheme expressed as a TileScheme must reproduce the
     paper-pinned geometry exactly (interlock cells, free-orbit counts).
  2. The new pond-frame square scheme must satisfy the composition
     conditions ((H), overlap consistency), its SAT censuses must match
     brute force where feasible, and assembled mosaics must be global
     still lifes — with a negative control proving the check has teeth.
"""

import numpy as np
import pytest

from gol_mosaics.life import is_still_life
from gol_mosaics.tile_domain import EXPECTED_FREE_ORBITS, derive_dead_edges_full
from gol_mosaics.tile_scheme import (
    assemble,
    build_scheme_domain,
    check_frame_mosaic,
    dead_moat_square_scheme,
    derive_interlock,
    diamond_scheme,
    enumerate_scheme_tiles,
    pack_scheme_solutions,
    pond_square_scheme,
    unpack_scheme_solutions,
)

# Known censuses of the pond-frame square scheme (established in the
# walk-through notebook; levels 2-5).
SQUARE_CENSUS = {2: 1, 3: 3, 4: 65, 5: 10398}


@pytest.mark.parametrize("level", [2, 3, 4, 5, 6])
def test_diamond_scheme_reproduces_pinned_geometry(level):
    """The generic derivation must equal derive_dead_edges_full exactly and
    yield the design-time free-orbit counts."""
    scheme = diamond_scheme(level)
    assert derive_interlock(scheme) == derive_dead_edges_full(level)
    domain = build_scheme_domain(scheme)
    expected = EXPECTED_FREE_ORBITS.get(level)
    if expected is not None:
        assert len(domain.free_reps) == expected
    assert check_frame_mosaic(scheme), "diamond frame-only mosaic unstable"


@pytest.mark.parametrize("level", [2, 3, 4, 5])
def test_square_scheme_hypothesis_h(level):
    """(H): the frame-only mosaic of the square scheme is a still life."""
    assert check_frame_mosaic(pond_square_scheme(level), extent=5)


@pytest.mark.parametrize("level,expected", sorted(SQUARE_CENSUS.items()))
def test_square_scheme_census(level, expected):
    pytest.importorskip("pysat")
    tiles = enumerate_scheme_tiles(pond_square_scheme(level))
    assert len(tiles) == expected


@pytest.mark.parametrize("level", [3, 4])
def test_square_scheme_census_matches_bruteforce(level):
    """Independent ground truth: try all 2^k free-orbit assignments."""
    pytest.importorskip("pysat")
    from itertools import product

    from gol_mosaics.sat_search import rule_violations

    scheme = pond_square_scheme(level)
    domain = build_scheme_domain(scheme)
    k = len(domain.free_reps)
    assert k <= 15, "brute force only feasible for small levels"
    bits = np.array(list(product((0, 1), repeat=k)), dtype=np.uint8)
    grids = domain.expand_many(bits)
    n_bruteforce = int((rule_violations(grids) == 0).sum())
    assert n_bruteforce == SQUARE_CENSUS[level]


@pytest.mark.parametrize("level", [3, 4, 5])
def test_square_mosaic_is_global_still_life(level):
    pytest.importorskip("pysat")
    scheme = pond_square_scheme(level)
    tiles = enumerate_scheme_tiles(scheme)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        index_grid = rng.integers(0, len(tiles), size=(5, 5))
        mosaic = assemble(scheme, index_grid, tiles)
        assert set(np.unique(mosaic)) <= {0, 1}
        assert is_still_life(mosaic), (
            f"square level-{level} mosaic (seed {seed}) not a still life"
        )


def test_square_scheme_stability_check_has_teeth():
    """Without the derived interlock cells, tiles are still lifes in
    isolation but some mosaic of them must destabilise."""
    pytest.importorskip("pysat")
    level = 4
    scheme = pond_square_scheme(level)
    tiles = enumerate_scheme_tiles(scheme, interlock=set(), limit=2000)
    n = scheme.n
    for tile in tiles[:20]:
        assert is_still_life(np.pad(tile, n)), "tile unstable in isolation"

    rng = np.random.default_rng(0)
    broke_a_mosaic = False
    for _ in range(20):
        index_grid = rng.integers(0, len(tiles), size=(3, 3))
        try:
            mosaic = assemble(scheme, index_grid, tiles)
        except AssertionError:
            # assemble may already reject inconsistent overlaps -- without
            # interlock forcings, overlap cells are free and can disagree.
            broke_a_mosaic = True
            break
        if not is_still_life(mosaic):
            broke_a_mosaic = True
            break
    assert broke_a_mosaic, (
        "no interlock-free tile mix destabilised a mosaic — check is blind "
        "or the interlock cells are unnecessary"
    )


def test_dead_moat_scheme_baseline():
    """Empty frame + dead moat: no interlock cells needed, mosaics of
    arbitrary tiles are trivially still lifes."""
    pytest.importorskip("pysat")
    scheme = dead_moat_square_scheme(2)
    assert derive_interlock(scheme) == set()
    assert check_frame_mosaic(scheme)  # vacuous: empty grid is a still life
    tiles = enumerate_scheme_tiles(scheme)
    assert len(tiles) == 15
    rng = np.random.default_rng(0)
    mosaic = assemble(scheme, rng.integers(0, len(tiles), (4, 4)), tiles)
    assert is_still_life(mosaic)


@pytest.mark.parametrize("level", [3, 4, 5])
def test_pack_unpack_scheme_solutions_roundtrip(level):
    """Packed free-orbit bits must reconstruct the tile set byte-identically
    (this is the storage format shipped in the package data)."""
    pytest.importorskip("pysat")
    scheme = pond_square_scheme(level)
    tiles = enumerate_scheme_tiles(scheme)
    packed = pack_scheme_solutions(scheme, tiles)
    n_free = len(build_scheme_domain(scheme).free_reps)
    assert packed.shape == (len(tiles), (n_free + 7) // 8)
    assert np.array_equal(unpack_scheme_solutions(scheme, packed), tiles)


def test_pack_scheme_solutions_rejects_foreign_grids():
    """Grids that are not free-orbit assignments of the scheme (here: a
    forced frame cell flipped dead) must be rejected, not silently mangled."""
    scheme = pond_square_scheme(3)
    bad = scheme.frame.astype(np.uint8)[None].copy()
    i, j = np.argwhere(scheme.frame)[0]
    bad[0, i, j] = 0
    with pytest.raises(AssertionError):
        pack_scheme_solutions(scheme, bad)


@pytest.mark.parametrize("level", [3, 4])
def test_assemble_with_holes_is_still_life(level):
    """Index -1 omits the tile (an empty region, used for empty_tiles_cutoff).
    Because every subset of the separated-pond frame lattice is a still life
    and absent tiles only remove live cells outside the remaining tiles'
    influence, hole-punched mosaics must stay globally stable."""
    pytest.importorskip("pysat")
    scheme = pond_square_scheme(level)
    tiles = enumerate_scheme_tiles(scheme)
    rng = np.random.default_rng(0)
    for _ in range(5):
        index_grid = rng.integers(0, len(tiles), size=(5, 5))
        holes = rng.random((5, 5)) < 0.4
        index_grid[holes] = -1
        mosaic = assemble(scheme, index_grid, tiles)
        n = scheme.n
        pitch = scheme.u[0]
        for (a, b) in np.argwhere(holes):
            # the hole's private band (not shared with neighbours) is empty
            i0, j0 = 2 + a * pitch + (n - pitch), 2 + b * pitch + (n - pitch)
            assert not mosaic[i0:i0 + 2 * pitch - n,
                              j0:j0 + 2 * pitch - n].any()
        assert is_still_life(mosaic), "hole-punched mosaic destabilised"


def test_assemble_rejects_inconsistent_overlap():
    """Tiles disagreeing on shared (forced) cells must be rejected."""
    scheme = pond_square_scheme(3)
    good = scheme.frame.astype(np.uint8)
    bad = good.copy()
    # flip one frame cell: overlapping placements now disagree there
    i, j = np.argwhere(scheme.frame)[0]
    bad[i, j] = 0
    with pytest.raises(AssertionError):
        assemble(scheme, np.array([[0, 1], [1, 0]]), np.stack([good, bad]))

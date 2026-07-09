"""
End-to-end tiling tests: assembled mosaics must be global still lifes.

The per-tile constraints (still-life rules + forced dead edges) are only a
sufficient condition for tiles to interlock; these tests verify the actual
claim by assembling real mosaics through MosaicGenerator._build_mosaic and
stepping them once with an independent Game of Life implementation.
"""

import numpy as np
import pytest

from gol_mosaics import MosaicGenerator
from gol_mosaics.life import is_still_life, life_step
from gol_mosaics.tile_domain import derive_dead_edges_full


def build_random_mosaic(level: int, grid: int, seed: int) -> np.ndarray:
    """Assemble a mosaic from random greyscale input at the given level."""
    rng = np.random.default_rng(seed)
    generator = MosaicGenerator(level=level, grid_size=grid)
    # Shapes as produced by preprocess_for_mosaic for a grid of `grid` tiles.
    lowres_first = rng.integers(0, 256, (grid // 2 + 1, grid // 2)).astype(float)
    lowres_second = rng.integers(0, 256, (grid // 2, grid // 2 + 1)).astype(float)
    return generator._build_mosaic(lowres_first, lowres_second,
                                   empty_tiles_cutoff=1.0)


@pytest.mark.parametrize("level", [2, 3, 4, 5, 6])
def test_mosaic_is_global_still_life(level):
    """One GoL step must leave an assembled mosaic unchanged (all levels)."""
    for seed in (0, 1, 2):
        mosaic = build_random_mosaic(level, grid=10, seed=seed)
        assert set(np.unique(mosaic)) <= {0, 1}, "tiles overlap"
        assert mosaic.sum() > 1000, "mosaic unexpectedly (near-)empty"
        assert is_still_life(mosaic), (
            f"level-{level} mosaic (seed {seed}) is not a global still life"
        )


def test_stability_check_has_teeth():
    """Tiles enumerated WITHOUT dead-edge constraints are still lifes in
    isolation, yet some must break an assembled mosaic — proving both that
    the dead edges are necessary and that this test can detect interaction."""
    pytest.importorskip("pysat")
    from gol_mosaics import PatternLibrary
    from gol_mosaics.sat_search import enumerate_tiles

    level = 4
    unconstrained = enumerate_tiles(level, dead_edges=[])
    dead_cells = sorted(derive_dead_edges_full(level))
    ii = [c[0] for c in dead_cells]
    jj = [c[1] for c in dead_cells]
    violators = unconstrained[unconstrained[:, ii, jj].any(axis=1)]
    assert len(violators) > 0, "expected tiles that violate the dead edges"

    # Each violator on its own is a perfectly valid still life...
    n = violators.shape[1]
    for tile in violators[:20]:
        assert is_still_life(np.pad(tile, n)), "violator unstable in isolation"

    # ...but violators interact across tile boundaries: every level-4
    # violator breaks an assembled mosaic, even repeated uniformly and even
    # in contact with only the bare pond frame (verified exhaustively for
    # all 267 violators — see paper/ars/revision_data/evidence_core.json).
    library = PatternLibrary(level=level)
    library._solutions = violators
    generator = MosaicGenerator(level=level, grid_size=10)
    generator._pattern_library = library

    rng = np.random.default_rng(0)
    broke_a_mosaic = False
    for _ in range(3):
        mosaic = generator._build_mosaic(
            rng.integers(0, 256, (6, 5)).astype(float),
            rng.integers(0, 256, (5, 6)).astype(float),
            empty_tiles_cutoff=1.0)
        mosaic = np.clip(mosaic, 0, 1)
        if not np.array_equal(life_step(mosaic), mosaic):
            broke_a_mosaic = True
            break
    assert broke_a_mosaic, (
        "no mix of dead-edge-violating tiles destabilised a mosaic — "
        "either the check is blind or the dead edges are unnecessary"
    )

"""
Minimal Life-like cellular-automaton stepping on bounded grids.

Used by the tiling tests and the SAT-search notebook to verify that
assembled mosaics are global still lifes. Cells outside the grid are dead
(no wraparound) — matching how a finished mosaic sits in an image.
"""

from typing import Tuple

import numpy as np


def neighbour_counts(grid: np.ndarray) -> np.ndarray:
    """Live-neighbour count per cell; outside the grid counts as dead."""
    g = np.pad(np.asarray(grid, dtype=np.int64), 1)
    total = np.zeros_like(g)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            total += np.roll(np.roll(g, di, axis=0), dj, axis=1)
    return total[1:-1, 1:-1]


def life_step(grid: np.ndarray,
              birth: Tuple[int, ...] = (3,),
              survival: Tuple[int, ...] = (2, 3)) -> np.ndarray:
    """One synchronous step of a Life-like CA B{birth}/S{survival}."""
    g = np.asarray(grid)
    counts = neighbour_counts(g)
    alive = g == 1
    return (
        (alive & np.isin(counts, survival))
        | (~alive & np.isin(counts, birth))
    ).astype(np.uint8)


def is_still_life(grid: np.ndarray,
                  birth: Tuple[int, ...] = (3,),
                  survival: Tuple[int, ...] = (2, 3)) -> bool:
    """True when one step leaves the grid unchanged."""
    g = np.asarray(grid, dtype=np.uint8)
    return np.array_equal(life_step(g, birth, survival), g)

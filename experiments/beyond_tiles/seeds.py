"""Constructive still-life seeds for warm starts.

The construction is a block agar with per-window occupancy: 2x2 blocks on
a global lattice with period 3 in rows and 4 in columns. That asymmetric
pitch is the point — for the square period-3 lattice, a corner where
exactly three of four diagonal blocks are filled gives the centre dead
cell exactly three live neighbours (a birth), so partial occupancies need
repair passes that cascade. With the 3x4 pitch no dead cell can ever see
three live neighbours regardless of which subset of sites is filled:
edge-adjacent cells between vertical neighbours see 2 or 4, horizontal
gaps see 2, and diagonal contributions never exceed 1+1. Every subset is
a still life, so masks (which drop whole blocks) and arbitrary per-window
occupancy are safe by construction — no repair, no seams.

The price is a density ceiling of 1/3 (asymptotically; up to 24 cells in
an 8x8 window) instead of the square lattice's 4/9. The darkest windows
therefore keep a deficit of a few cells — exactly the regions the solver
or the LNS polisher works on first.
"""

from typing import List, Optional, Tuple

import numpy as np

from beyond_tiles.targets import Window

PERIOD_ROWS = 3  # 2x2 block + 1 dead row
PERIOD_COLS = 4  # 2x2 block + 2 dead columns


def build_seed(
    free_mask: np.ndarray,
    windows: List[Window],
    targets: np.ndarray,
    phase: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Density-matched still life: filled blocks track the window targets.

    Sites are visited in raster order; a site is filled while its owning
    window (the window containing the block's top-left cell) still wants
    at least half a block. Blocks touching masked cells or the border are
    skipped. The result is a valid still life for every input.
    """
    h, w = free_mask.shape
    seed = np.zeros((h, w), dtype=np.uint8)

    owner = np.full((h, w), -1, dtype=np.int64)
    for idx, (si, sj) in enumerate(windows):
        owner[si, sj] = idx
    remaining = np.asarray(targets, dtype=np.float64).copy()

    for i in range(phase[0] % PERIOD_ROWS, h - 1, PERIOD_ROWS):
        for j in range(phase[1] % PERIOD_COLS, w - 1, PERIOD_COLS):
            if not free_mask[i : i + 2, j : j + 2].all():
                continue
            idx = owner[i, j]
            if idx < 0 or remaining[idx] < 2.0:
                continue
            seed[i : i + 2, j : j + 2] = 1
            remaining[idx] -= 4.0
    return seed


def seed_objective(
    seed: np.ndarray,
    free_mask: np.ndarray,
    windows: List[Window],
    targets: np.ndarray,
    slack: int = 0,
) -> int:
    """The seed's own value of the solver objective."""
    total = 0
    for t, (si, sj) in zip(targets, windows):
        live = int(seed[si, sj][free_mask[si, sj]].sum())
        total += max(0, abs(live - int(t)) - slack)
    return total


def best_seed(
    free_mask: np.ndarray,
    windows: List[Window],
    targets: np.ndarray,
    slack: int = 0,
    phases: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, int]:
    """Try every lattice phase, keep the seed with the lowest objective."""
    if phases is None:
        phases = [
            (a, b) for a in range(PERIOD_ROWS) for b in range(PERIOD_COLS)
        ]
    best: Optional[np.ndarray] = None
    best_obj = -1
    for phase in phases:
        seed = build_seed(free_mask, windows, targets, phase)
        obj = seed_objective(seed, free_mask, windows, targets, slack)
        if best is None or obj < best_obj:
            best, best_obj = seed, obj
    return best, best_obj

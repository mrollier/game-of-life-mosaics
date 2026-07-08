"""
Tile geometry, symmetry-orbit domain, and dead-edge derivation.

This module is the rule-independent foundation of the SAT tile search
(:mod:`gol_mosaics.sat_search`) and of the compact on-disk format for large
tile databases: instead of full (N, n, n) grids, a level's solutions can be
stored as one bit per *free symmetry orbit* (59 bits per level-6 tile) and
expanded on load.

The geometry functions mirror :class:`gol_mosaics.patterns.PatternLibrary`
(``pond_pattern`` .. ``pond_pattern_edge``) as standalone functions so the
search bundle in ``workstation/level6_search`` can vendor them without the
package installed. Divergence is guarded by tests that re-derive the shipped
level 3-5 databases byte-exactly.

Dead edges — the cells that must be forced dead so tiles interlock into a
global still life — were historically hard-coded per level (through level 6).
``derive_dead_edges_full`` derives them from the mosaic geometry instead:

    A       = the tile's possibly-alive diamond (filled pond edge)
    B       = union of A at the 12 neighbouring-tile offsets
              ((+-n/2, +-n/2) for the interlocking grid, (0, +-n), (+-n, 0),
              (+-n, +-n) for the same grid; the grid offset tile_pad_size
              equals n/2 exactly)
    shared  = dilate1(A) & dilate1(B)      # cells influenced by two tiles
    dead    = free(A) & dilate1(shared)    # free cells influencing them

where ``dilate1`` is the Chebyshev-1 dilation (3x3 structuring element).
Forcing these cells dead leaves every shared cell's neighbourhood fully
determined by forced cells, whose contact configuration is a still life by
construction. The derivation reproduces the historical lists for levels 2-6
exactly (orbit-closed, 12*level - 8 cells) and extends to level 7 and beyond.

Constraint semantics replicated from the original ILP
(patterns.py, ``_find_all_symmetric_gol_mosaics``):
  alive cell => 2 <= live neighbours <= 3
  dead cell  => live neighbours != 3 (no births)
Neighbourhoods use ``% n`` wraparound exactly like the ILP; the wraparound is
inert (the border ring is forced dead for level >= 2) but kept verbatim.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes

POND_WIDTH = 6

# Free-orbit counts per level, asserted in build_domain: a mismatch means the
# geometry has diverged from patterns.py.
EXPECTED_FREE_ORBITS = {3: 10, 4: 22, 5: 38, 6: 59, 7: 84}

_DILATE1 = np.ones((3, 3), dtype=bool)


def pond_pattern() -> np.ndarray:
    """The base 4x4 pond still life."""
    return np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])


def pond_pattern_multiple(level: int) -> np.ndarray:
    """Stacked pond pattern with masked corners; grid size (6*level)^2."""
    width = POND_WIDTH * level
    pp = pond_pattern()

    if level > 1:
        pp_multiple = np.vstack((
            np.vstack([pp[:-1]] * (level - 1)),
            pp,
            np.vstack([pp[1:]] * (level - 1))
        ))
        pp_multiple = np.hstack((
            np.hstack([pp_multiple[:, :-1]] * (level - 1)),
            pp_multiple,
            np.hstack([pp_multiple[:, 1:]] * (level - 1))
        ))
    else:
        pp_multiple = pp.copy()

    pp_multiple = np.pad(pp_multiple, pad_width=1, constant_values=0)

    mask_even = np.array([
        [(i + j) < POND_WIDTH * level / 2 for j in range(width)]
        for i in range(width)
    ])
    mask_even = mask_even + mask_even[::-1, ::-1]

    mask_odd = np.array([
        [(i - j) >= POND_WIDTH * level / 2 for j in range(width)]
        for i in range(width)
    ])
    mask_odd = mask_odd + mask_odd.T

    mask = mask_even + mask_odd
    return np.where(1 - mask, pp_multiple, 0)


def pond_pattern_edge(level: int) -> np.ndarray:
    """Border ponds only (the forced-alive tile edge)."""
    width = POND_WIDTH * level
    pp_multiple = pond_pattern_multiple(level)

    mask_corner = np.array([
        [(i + j) < POND_WIDTH * level / 2 + 3 for j in range(width)]
        for i in range(width)
    ])
    mask = (mask_corner + mask_corner[::-1, ::-1] +
            mask_corner[::-1] + mask_corner[:, ::-1])

    return np.where(mask, pp_multiple, 0)


def _base_masks(level: int) -> Tuple[np.ndarray, np.ndarray]:
    """(edge_alive, outside_dead) uint8 masks, without dead edges."""
    pp_edge = pond_pattern_edge(level)
    edge_alive = (pp_edge > 0).astype(np.uint8)
    outside_dead = 1 - binary_fill_holes(edge_alive).astype(np.uint8)
    return edge_alive, outside_dead


def derive_dead_edges_full(level: int) -> Set[Tuple[int, int]]:
    """
    All forced-dead interlock cells for a level (orbit-closed by symmetry
    of the construction). Empty for level 1, whose single-pond tiles have
    no free cell near another tile (historical convention, covered by the
    whole-mosaic stability tests).
    """
    if level < 2:
        return set()

    n = POND_WIDTH * level
    half = n // 2  # == tile_pad_size: ((6-3)*(2L-1)+3)//2 == 3L
    edge_alive, outside_dead = _base_masks(level)
    diamond = binary_fill_holes(edge_alive)

    # 3n x 3n canvas with the tile at the centre block
    tile = np.zeros((3 * n, 3 * n), dtype=bool)
    tile[n:2 * n, n:2 * n] = diamond

    offsets = (
        [(si * half, sj * half) for si in (-1, 1) for sj in (-1, 1)]
        + [(0, n), (0, -n), (n, 0), (-n, 0)]
        + [(si * n, sj * n) for si in (-1, 1) for sj in (-1, 1)]
    )
    others = np.zeros_like(tile)
    for di, dj in offsets:
        others[n + di:2 * n + di, n + dj:2 * n + dj] |= diamond

    shared = binary_dilation(tile, _DILATE1) & binary_dilation(others, _DILATE1)
    influencers = binary_dilation(shared, _DILATE1)

    free = tile.copy()
    free[n:2 * n, n:2 * n] &= edge_alive == 0
    dead = free & influencers

    block = dead[n:2 * n, n:2 * n]
    return {(int(i), int(j)) for i, j in zip(*np.where(block))}


def derive_dead_edges(level: int) -> List[Tuple[int, int]]:
    """
    One octant representative per dead-edge orbit, in the convention of the
    historical hard-coded lists: j >= n//2, i <= n-1-j, lexicographic min.
    """
    n = POND_WIDTH * level
    full = derive_dead_edges_full(level)
    reps = set()
    for (i, j) in full:
        candidates = [
            (ii, jj) for (ii, jj) in symmetric_coords(i, j, n)
            if jj >= n // 2 and ii <= n - 1 - jj
        ]
        reps.add(min(candidates))
    return sorted(reps)


def forced_masks(level: int, dead_edges=None) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Per-cell forcings for one level.

    Args:
        level: tile level
        dead_edges: override for the forced-dead interlock cells; None
            derives them from the mosaic geometry (the normal case). Pass
            [] to study what happens without them (such tiles are still
            lifes in isolation but can interact in a mosaic).

    Returns:
        edge_alive: (n, n) uint8, 1 where the pond edge forces the cell alive
        outside_dead: (n, n) uint8, 1 where outside the diamond (forced dead)
        dead_edges: forced-dead interlock cells (orbit-closed when derived)
    """
    edge_alive, outside_dead = _base_masks(level)
    if dead_edges is None:
        dead_edges = sorted(derive_dead_edges_full(level))
    return edge_alive, outside_dead, list(dead_edges)


def symmetric_coords(i: int, j: int, n: int) -> List[Tuple[int, int]]:
    """The 8 D4 images of (i, j), including the identity."""
    return [
        (i, j),
        (n - 1 - i, j),
        (i, n - 1 - j),
        (n - 1 - i, n - 1 - j),
        (j, i),
        (j, n - 1 - i),
        (n - 1 - j, i),
        (n - 1 - j, n - 1 - i),
    ]


def neighbors(i: int, j: int, n: int) -> List[Tuple[int, int]]:
    """The 8 Moore neighbours with % n wraparound (verbatim ILP semantics)."""
    return [
        ((i + di) % n, (j + dj) % n)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
    ]


@dataclass(frozen=True)
class Domain:
    """Symmetry-reduced search domain for one level."""
    level: int
    n: int
    # (n, n) arrays mapping each cell to its orbit representative
    rep_i: np.ndarray
    rep_j: np.ndarray
    # forced orbit values, keyed by representative cell
    constants: Dict[Tuple[int, int], int]
    # free orbit representatives, sorted lexicographically
    free_reps: List[Tuple[int, int]]

    def rep_of(self, i: int, j: int) -> Tuple[int, int]:
        return (int(self.rep_i[i, j]), int(self.rep_j[i, j]))

    @cached_property
    def _cell_index_maps(self):
        """Precomputed (cell_index into free_reps, is_const, const_value)."""
        n = self.n
        free_index = {rep: idx for idx, rep in enumerate(self.free_reps)}
        cell_index = np.zeros((n, n), dtype=np.int64)
        const_value = np.zeros((n, n), dtype=np.uint8)
        is_const = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                rep = self.rep_of(i, j)
                if rep in self.constants:
                    is_const[i, j] = True
                    const_value[i, j] = self.constants[rep]
                else:
                    cell_index[i, j] = free_index[rep]
        return cell_index, is_const, const_value

    def expand(self, free_values: np.ndarray) -> np.ndarray:
        """Reconstruct one (n, n) grid from one free-orbit assignment."""
        return self.expand_many(np.asarray(free_values, dtype=np.uint8)[None])[0]

    def expand_many(self, free_matrix: np.ndarray) -> np.ndarray:
        """Vectorised expansion of a (m, len(free_reps)) bit matrix."""
        m = free_matrix.shape[0]
        cell_index, is_const, const_value = self._cell_index_maps
        grids = free_matrix.astype(np.uint8)[:, cell_index.ravel()]
        grids = grids.reshape(m, self.n, self.n)
        grids[:, is_const] = const_value[is_const]
        return grids

    def extract_bits(self, grids: np.ndarray) -> np.ndarray:
        """Inverse of expand_many: (m, n, n) grids -> (m, n_free) bits."""
        ii = np.array([r[0] for r in self.free_reps])
        jj = np.array([r[1] for r in self.free_reps])
        return np.ascontiguousarray(grids[:, ii, jj]).astype(np.uint8)

    def orbit_sizes(self) -> np.ndarray:
        """Number of cells in each free orbit (aligned with free_reps)."""
        sizes = np.zeros(len(self.free_reps), dtype=np.int64)
        index = {rep: k for k, rep in enumerate(self.free_reps)}
        for i in range(self.n):
            for j in range(self.n):
                k = index.get(self.rep_of(i, j))
                if k is not None:
                    sizes[k] += 1
        return sizes

    def forced_alive_count(self) -> int:
        """Live cells contributed by forced orbits (same for every tile)."""
        _, is_const, const_value = self._cell_index_maps
        return int(const_value[is_const].sum())


def build_domain(level: int, dead_edges=None) -> Domain:
    """
    Compute D4 orbits and propagate all forcings orbit-wide.

    Raises AssertionError if any orbit receives conflicting forcings or the
    free-orbit count deviates from the design-time expectation (only checked
    for the standard, derived dead edges).
    """
    n = POND_WIDTH * level
    custom_dead_edges = dead_edges is not None
    edge_alive, outside_dead, dead_edges = forced_masks(level, dead_edges)
    dead_edge_set = set(dead_edges)

    rep_i = np.empty((n, n), dtype=np.int64)
    rep_j = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            ri, rj = min(symmetric_coords(i, j, n))
            rep_i[i, j] = ri
            rep_j[i, j] = rj

    constants: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j in range(n):
            forced = None
            if edge_alive[i, j]:
                forced = 1
            if outside_dead[i, j] or (i, j) in dead_edge_set:
                assert forced is None, f"cell ({i},{j}) forced both alive and dead"
                forced = 0
            if forced is not None:
                rep = (int(rep_i[i, j]), int(rep_j[i, j]))
                assert constants.get(rep, forced) == forced, (
                    f"orbit {rep} receives conflicting forcings at cell ({i},{j})"
                )
                constants[rep] = forced

    all_reps = {(int(rep_i[i, j]), int(rep_j[i, j]))
                for i in range(n) for j in range(n)}
    free_reps = sorted(all_reps - set(constants))

    # Border ring must be entirely forced dead, which is what makes the % n
    # wraparound inert.
    if level >= 2:
        border = np.zeros((n, n), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        assert all(
            constants.get((int(rep_i[i, j]), int(rep_j[i, j]))) == 0
            for i, j in zip(*np.where(border))
        ), "border ring not fully forced dead"

    expected = EXPECTED_FREE_ORBITS.get(level)
    if expected is not None and not custom_dead_edges:
        assert len(free_reps) == expected, (
            f"level {level}: {len(free_reps)} free orbits, expected {expected}"
        )

    return Domain(level=level, n=n, rep_i=rep_i, rep_j=rep_j,
                  constants=constants, free_reps=free_reps)


def pack_solutions(grids: np.ndarray, level: int) -> np.ndarray:
    """
    Compress (m, n, n) tile grids to packed free-orbit bits:
    (m, ceil(n_free/8)) uint8. ~162x smaller than uint8 grids at level 6.
    """
    domain = build_domain(level)
    bits = domain.extract_bits(np.asarray(grids))
    assert np.array_equal(domain.expand_many(bits),
                          np.asarray(grids, dtype=np.uint8)), (
        "grids are not expressible as free-orbit assignments of this level"
    )
    return np.packbits(bits, axis=1)


def unpack_solutions(packed: np.ndarray, level: int) -> np.ndarray:
    """Inverse of pack_solutions: packed bits -> (m, n, n) uint8 grids."""
    domain = build_domain(level)
    bits = np.unpackbits(packed, axis=1, count=len(domain.free_reps))
    return domain.expand_many(bits)

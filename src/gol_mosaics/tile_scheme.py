"""
Generalised tile schemes: pluggable geometry for composable still-life tiles.

The composition theorem behind the mosaic pipeline does not depend on the
diamond shape: it needs a *tile scheme* — a support shape placed on a
translation lattice, a forced-alive frame, and derived forced-dead interlock
cells — satisfying

  (S1) tiles are placed at a*u + b*v for integer (a, b) and lattice basis
       vectors u, v;
  (S2) every tile forces its frame cells alive and its interlock cells dead,
       where the interlock cells are  free(A) & dilate1(shared)  with
       shared = dilate1(A) & dilate1(B), A the support and B the union of A
       at all neighbouring lattice offsets (the same morphological recipe as
       :func:`gol_mosaics.tile_domain.derive_dead_edges_full`, applied to an
       arbitrary scheme);
  (S3) wherever two tile supports overlap, both tiles force the overlapped
       cells to the same value (checked by :func:`derive_interlock`: forced
       frames must agree on overlaps, and every *free* overlap cell is
       swept into the interlock set, so overlaps end up fully forced);
  (H)  the frame-only mosaic (every tile just its frame) is a still life of
       the rule (checked by :func:`check_frame_mosaic`).

Under (S1)-(S3) and (H), every assignment of scheme tiles to lattice sites
yields a global still life, by the same case analysis as the diamond
composition theorem: a cell whose neighbourhood meets at most one tile is
covered by that tile's (toroidally inert) still-life constraints, and a cell
whose neighbourhood meets several tiles sees only forced cells, whose
configuration equals the frame-only mosaic — a still life by (H).

Two instances are provided:

* :func:`diamond_scheme` — the historical pond-diamond geometry, used as a
  validation anchor: its derived interlock cells and free-orbit counts must
  reproduce ``tile_domain`` exactly (they do; see tests/test_tile_scheme.py).
* :func:`pond_square_scheme` — a new axis-aligned family: square tiles of
  size ``6*level`` bordered by a ring of *separated* ponds (period 6),
  placed at pitch ``6*(level-1)`` so adjacent tiles share their border pond
  band. Separation matters: a straight chain of edge-sharing ponds is not a
  still life (a dead cell between two shared ponds sees exactly three live
  neighbours), which is why the diamond frame's shared-edge pond chains only
  ever run diagonally, and why this scheme spaces its ponds instead.

Tiles of any scheme are enumerated with the unchanged SAT machinery: the
scheme is reduced to a :class:`gol_mosaics.tile_domain.Domain` (one boolean
per free D4 orbit) and encoded exactly like the diamond CNF. The paper-pinned
diamond pipeline (``tile_domain``, ``sat_search``) is deliberately left
untouched; this module builds on it without modifying it.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

import hashlib

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes

from .life import is_still_life
from .sat_search import CONWAY, Encoding, _simplify, enumerate_all
from .tile_domain import (
    Domain,
    POND_WIDTH,
    _base_masks,
    neighbors,
    pond_pattern,
    symmetric_coords,
)

_DILATE1 = np.ones((3, 3), dtype=bool)

Offset = Tuple[int, int]


@dataclass(frozen=True)
class TileScheme:
    """One composable tile family: support + frame on a translation lattice.

    Attributes:
        name: identifier, folded into CNF fingerprints.
        level: size parameter (metadata; carried into the Domain).
        support: (n, n) bool, cells that may be alive (A). Must leave the
            outermost ring of the tile grid dead so the toroidal wraparound
            in the SAT encoding stays inert.
        frame: (n, n) bool, forced-alive cells (a subset of support).
        u, v: lattice basis vectors (di, dj); tiles sit at a*u + b*v.

    D4 symmetry of the tile is assumed throughout (as for the diamonds);
    both support and frame must be D4-invariant.
    """
    name: str
    level: int
    support: np.ndarray
    frame: np.ndarray
    u: Offset
    v: Offset

    @property
    def n(self) -> int:
        return self.support.shape[0]

    def __post_init__(self):
        assert self.support.shape == self.frame.shape
        assert self.support.shape[0] == self.support.shape[1]
        assert not np.any(self.frame & ~self.support), "frame outside support"
        n = self.n
        border = np.zeros((n, n), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        assert not np.any(self.support & border), (
            "support touches the tile border; the outermost ring must stay "
            "dead for the toroidal SAT encoding to be inert"
        )
        for mask in (self.support, self.frame):
            for k in (0, 1):
                flipped = np.flip(mask, axis=k)
                assert np.array_equal(mask, flipped), "masks must be D4-symmetric"
            assert np.array_equal(mask, mask.T), "masks must be D4-symmetric"


def neighbor_offsets(scheme: TileScheme, reach: int = 2) -> List[Offset]:
    """Lattice offsets a*u + b*v whose translated support can influence the
    home tile (bounding boxes within Chebyshev distance 1)."""
    n = scheme.n
    offsets = []
    for a in range(-reach, reach + 1):
        for b in range(-reach, reach + 1):
            if (a, b) == (0, 0):
                continue
            di = a * scheme.u[0] + b * scheme.v[0]
            dj = a * scheme.u[1] + b * scheme.v[1]
            if abs(di) <= n + 1 and abs(dj) <= n + 1:
                offsets.append((di, dj))
    return offsets


def derive_interlock(scheme: TileScheme) -> Set[Offset]:
    """Forced-dead interlock cells of a scheme (generic (S2) derivation).

    Also enforces the (S3) precondition: wherever the home support overlaps
    a translated support, the forced frames must agree. Free overlap cells
    need no separate check — they land in ``shared`` and hence in the
    returned set, so both tiles force them dead.
    """
    n = scheme.n
    offsets = neighbor_offsets(scheme)
    pad = max(max(abs(di), abs(dj)) for di, dj in offsets) + 2
    size = n + 2 * pad
    home = np.zeros((size, size), dtype=bool)
    home[pad:pad + n, pad:pad + n] = scheme.support
    home_frame = np.zeros_like(home)
    home_frame[pad:pad + n, pad:pad + n] = scheme.frame
    others = np.zeros_like(home)
    others_frame = np.zeros_like(home)
    for di, dj in offsets:
        others[pad + di:pad + di + n, pad + dj:pad + dj + n] |= scheme.support
        others_frame[pad + di:pad + di + n, pad + dj:pad + dj + n] |= scheme.frame

    overlap = home & others
    assert np.array_equal(home_frame & overlap, others_frame & overlap), (
        "frame is not lattice-consistent: overlapping tiles disagree on "
        "forced-alive cells"
    )

    shared = binary_dilation(home, _DILATE1) & binary_dilation(others, _DILATE1)
    influencers = binary_dilation(shared, _DILATE1)
    dead = (home & ~home_frame) & influencers
    block = dead[pad:pad + n, pad:pad + n]
    return {(int(i), int(j)) for i, j in zip(*np.where(block))}


def build_scheme_domain(scheme: TileScheme,
                        interlock: Optional[Set[Offset]] = None) -> Domain:
    """Reduce a scheme to a symmetry-orbit :class:`Domain` (as build_domain
    does for diamonds): frame orbits forced alive, outside-support and
    interlock orbits forced dead, the rest free.

    Args:
        interlock: override for the derived interlock cells; pass set() to
            study tiles without them (stable in isolation, unstable in
            mosaics — the negative control).
    """
    if interlock is None:
        interlock = derive_interlock(scheme)
    n = scheme.n
    rep_i = np.empty((n, n), dtype=np.int64)
    rep_j = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            rep_i[i, j], rep_j[i, j] = min(symmetric_coords(i, j, n))

    constants: Dict[Offset, int] = {}
    for i in range(n):
        for j in range(n):
            forced = None
            if scheme.frame[i, j]:
                forced = 1
            if not scheme.support[i, j] or (i, j) in interlock:
                assert forced is None, (
                    f"cell ({i},{j}) forced both alive and dead"
                )
                forced = 0
            if forced is not None:
                rep = (int(rep_i[i, j]), int(rep_j[i, j]))
                assert constants.get(rep, forced) == forced, (
                    f"orbit {rep} receives conflicting forcings at ({i},{j})"
                )
                constants[rep] = forced

    all_reps = {(int(rep_i[i, j]), int(rep_j[i, j]))
                for i in range(n) for j in range(n)}
    free_reps = sorted(all_reps - set(constants))
    return Domain(level=scheme.level, n=n, rep_i=rep_i, rep_j=rep_j,
                  constants=constants, free_reps=free_reps)


def build_scheme_cnf(scheme: TileScheme,
                     birth: Sequence[int] = CONWAY[0],
                     survival: Sequence[int] = CONWAY[1],
                     interlock: Optional[Set[Offset]] = None) -> Encoding:
    """Still-life CNF for a scheme (one variable per free D4 orbit).

    Mirrors :func:`gol_mosaics.sat_search.build_cnf`, which stays untouched
    because its per-level fingerprints are pinned by the search manifests.
    The fingerprint here is salted with the scheme name instead of a level.
    """
    birth = tuple(sorted(birth))
    survival = tuple(sorted(survival))
    domain = build_scheme_domain(scheme, interlock=interlock)
    n = domain.n
    var_of_rep = {rep: idx + 1 for idx, rep in enumerate(domain.free_reps)}

    def lit(i, j):
        rep = domain.rep_of(i, j)
        if rep in domain.constants:
            return bool(domain.constants[rep])
        return var_of_rep[rep]

    def neg(x):
        return (not x) if isinstance(x, bool) else -x

    alive_forbidden = [c for c in range(9) if c not in survival]
    dead_forbidden = list(birth)

    clause_set = set()
    positions = range(8)
    for i in range(n):
        for j in range(n):
            centre = lit(i, j)
            nbrs = [lit(ii, jj) for (ii, jj) in neighbors(i, j, n)]
            for centre_lit, forbidden in ((neg(centre), alive_forbidden),
                                          (centre, dead_forbidden)):
                for c in forbidden:
                    for subset in combinations(positions, c):
                        clause = _simplify(
                            [centre_lit]
                            + [neg(nbrs[p]) for p in subset]
                            + [nbrs[p] for p in positions if p not in subset]
                        )
                        if clause:
                            clause_set.add(clause)

    clauses = sorted(clause_set)
    digest = hashlib.sha256()
    digest.update(f"scheme={scheme.name};n={n};vars={len(var_of_rep)};"
                  f"birth={birth};survival={survival};".encode())
    digest.update(repr(domain.free_reps).encode())
    digest.update(repr(clauses).encode())
    return Encoding(domain=domain, birth=birth, survival=survival,
                    n_vars=len(var_of_rep), clauses=clauses,
                    sha256=digest.hexdigest())


def enumerate_scheme_tiles(scheme: TileScheme,
                           birth: Sequence[int] = CONWAY[0],
                           survival: Sequence[int] = CONWAY[1],
                           interlock: Optional[Set[Offset]] = None,
                           limit: Optional[int] = None) -> np.ndarray:
    """All tiles of a scheme as (m, n, n) uint8 grids, sorted by
    (population, bytes) like :func:`gol_mosaics.sat_search.enumerate_tiles`."""
    enc = build_scheme_cnf(scheme, birth, survival, interlock=interlock)
    if enc.n_vars == 0:
        # fully forced scheme (e.g. pond_square_scheme(2)): the frame tile
        return scheme.frame.astype(np.uint8)[None]
    bits = enumerate_all(enc, limit=limit)
    grids = enc.domain.expand_many(bits)
    order = sorted(range(len(grids)),
                   key=lambda i: (int(grids[i].sum()), grids[i].tobytes()))
    return grids[order]


def pack_scheme_solutions(scheme: TileScheme, grids: np.ndarray) -> np.ndarray:
    """
    Compress (m, n, n) tile grids of a scheme to packed free-orbit bits:
    (m, ceil(n_free/8)) uint8 — the scheme analogue of
    :func:`gol_mosaics.tile_domain.pack_solutions`.
    """
    domain = build_scheme_domain(scheme)
    bits = domain.extract_bits(np.asarray(grids))
    assert np.array_equal(domain.expand_many(bits),
                          np.asarray(grids, dtype=np.uint8)), (
        "grids are not expressible as free-orbit assignments of this scheme"
    )
    return np.packbits(bits, axis=1)


def unpack_scheme_solutions(scheme: TileScheme, packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_scheme_solutions: packed bits -> (m, n, n) uint8 grids."""
    domain = build_scheme_domain(scheme)
    bits = np.unpackbits(packed, axis=1, count=len(domain.free_reps))
    return domain.expand_many(bits)


# ------------------------------------------------------------- assembly

def assemble(scheme: TileScheme,
             index_grid: np.ndarray,
             tile_grids: np.ndarray,
             pad: int = 2) -> np.ndarray:
    """Paste tiles onto the scheme lattice: tile (a, b) of the index grid
    goes to a*u + b*v. Works for any basis (square or diamond).

    A negative index leaves a hole (no tile at that site) — stable because
    every subset of the frame lattice is a still life and absent tiles only
    remove live cells outside the remaining tiles' influence.

    Overlapping supports are checked for consistency (they only ever
    contain forced cells, so any disagreement means the tiles are not all
    from the same scheme).
    """
    index_grid = np.asarray(index_grid)
    H, W = index_grid.shape
    n = scheme.n
    corners = [(a * scheme.u[0] + b * scheme.v[0],
                a * scheme.u[1] + b * scheme.v[1])
               for a in range(H) for b in range(W)]
    min_i = min(c[0] for c in corners)
    min_j = min(c[1] for c in corners)
    max_i = max(c[0] for c in corners)
    max_j = max(c[1] for c in corners)
    G = np.zeros((max_i - min_i + n + 2 * pad,
                  max_j - min_j + n + 2 * pad), dtype=np.uint8)
    written = np.zeros_like(G, dtype=bool)
    for (a, b), (ci, cj) in zip(
            ((a, b) for a in range(H) for b in range(W)), corners):
        if index_grid[a, b] < 0:
            continue
        tile = np.asarray(tile_grids[index_grid[a, b]], dtype=np.uint8)
        i0 = ci - min_i + pad
        j0 = cj - min_j + pad
        sup = scheme.support | scheme.frame
        block = G[i0:i0 + n, j0:j0 + n]
        seen = written[i0:i0 + n, j0:j0 + n] & sup
        assert np.array_equal(block[seen], tile[seen]), (
            "inconsistent overlap while assembling: tiles disagree on a "
            "shared (forced) cell"
        )
        block[sup] = tile[sup]
        written[i0:i0 + n, j0:j0 + n] |= sup
    return G


def frame_mosaic(scheme: TileScheme, extent: int = 4, pad: int = 2) -> np.ndarray:
    """The frame-only mosaic over extent x extent tiles (hypothesis (H))."""
    frame_tile = scheme.frame.astype(np.uint8)[None]
    return assemble(scheme, np.zeros((extent, extent), dtype=int),
                    frame_tile, pad=pad)


def check_frame_mosaic(scheme: TileScheme,
                       birth: Sequence[int] = CONWAY[0],
                       survival: Sequence[int] = CONWAY[1],
                       extent: int = 4) -> bool:
    """Hypothesis (H): is the frame-only mosaic a still life of the rule?"""
    return is_still_life(frame_mosaic(scheme, extent),
                         birth=tuple(birth), survival=tuple(survival))


# ------------------------------------------------------------- instances

def diamond_scheme(level: int) -> TileScheme:
    """The historical pond-diamond geometry as a TileScheme.

    Validation anchor: :func:`derive_interlock` reproduces
    :func:`gol_mosaics.tile_domain.derive_dead_edges_full` exactly and the
    free-orbit counts match ``EXPECTED_FREE_ORBITS`` (levels 2-6 covered by
    tests). Mosaic assembly of diamonds remains in
    :mod:`gol_mosaics.mosaic`; this instance exists to ground the
    abstraction, and :func:`assemble` handles its lattice too.
    """
    edge_alive, _ = _base_masks(level)
    support = binary_fill_holes(edge_alive.astype(bool))
    frame = edge_alive.astype(bool)
    half = (POND_WIDTH * level) // 2
    return TileScheme(name=f"diamond_L{level}", level=level,
                      support=support, frame=frame,
                      u=(half, half), v=(half, -half))


def pond_square_scheme(level: int) -> TileScheme:
    """Axis-aligned square tiles bordered by a ring of separated ponds.

    Tile size n = 6*level with a dead outer ring; the frame is every pond
    of the period-6 pond grid whose row or column band touches the tile
    border. Lattice pitch 6*(level-1): adjacent tiles share their border
    pond band, the axis-aligned analogue of diamonds sharing edge ponds.
    Level 2 is fully forced (a single tile); free interiors start at
    level 3.
    """
    assert level >= 2, "pond_square_scheme needs level >= 2"
    n = POND_WIDTH * level
    support = np.zeros((n, n), dtype=bool)
    support[1:n - 1, 1:n - 1] = True
    pond = pond_pattern().astype(bool)
    frame = np.zeros((n, n), dtype=bool)
    for a in range(level):
        for b in range(level):
            if a in (0, level - 1) or b in (0, level - 1):
                frame[6 * a + 1:6 * a + 5, 6 * b + 1:6 * b + 5] = pond
    return TileScheme(name=f"pond_square_L{level}", level=level,
                      support=support, frame=frame,
                      u=(POND_WIDTH * (level - 1), 0),
                      v=(0, POND_WIDTH * (level - 1)))


def dead_moat_square_scheme(level: int, moat: int = 2) -> TileScheme:
    """Degenerate baseline: square tiles with an empty frame and a forced
    dead moat. Hypothesis (H) holds vacuously; tiles never interact, so any
    still life fits in the interior — maximal freedom, no visible linking.
    """
    n = POND_WIDTH * level
    support = np.zeros((n, n), dtype=bool)
    support[moat:n - moat, moat:n - moat] = True
    frame = np.zeros((n, n), dtype=bool)
    return TileScheme(name=f"dead_moat_square_L{level}", level=level,
                      support=support, frame=frame, u=(n, 0), v=(0, n))

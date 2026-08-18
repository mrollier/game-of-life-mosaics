"""
No-symmetry (identity-domain) diamond tile enumeration.

The historical censuses (1, 2, 7, 85, 2632, 332321, ... for levels 1-7)
impose D4 symmetry on every tile. This module drops that requirement and
keeps everything else: still life under the configured rule, pond frame
forced alive, and the full orbit-closed dead-edge set forced dead so
tiles remain mosaic-compatible.

Measured censuses without the symmetry requirement:

    level  raw grids   D4 classes   class sizes {size: count}
    1      1           1            {1: 1}
    2      2           2            {1: 2}
    3      1061        181          {1: 7, 2: 9, 4: 71, 8: 94}

Levels 1-2 are unchanged by dropping symmetry — every valid tile there
happens to be D4-symmetric. The singleton classes at each level are
exactly the historical symmetric census, and the counts satisfy the
Burnside identity sum|Fix(g)| = 8 * #classes.

Mechanics: `Domain` (tile_domain) is symmetry-agnostic — D4 enters the
verified pipeline only through `build_domain`'s orbit-representative
arrays. `build_nosym_domain` constructs an identity-orbit Domain (each
cell its own orbit) over the same forced masks, and the shared clause
builder `sat_search._domain_clauses` plus `enumerate_all` do the rest.
The full orbit-closed dead-edge set (`forced_masks` default) is
essential here: octant representatives alone would no longer propagate
without the symmetry constraint.

The symmetric pipeline (`tile_domain.build_domain`, `sat_search
.build_cnf` defaults and their pinned fingerprints) is untouched.
"""

import hashlib
from importlib.resources import files
from typing import Dict, List, Optional, Sequence

import numpy as np

from .tile_domain import POND_WIDTH, Domain, forced_masks
from .sat_search import (
    CONWAY,
    Encoding,
    _domain_clauses,
    enumerate_all,
)

# Free-cell counts of the identity domain (design-time expectation,
# asserted in build_nosym_domain for the standard dead edges).
EXPECTED_FREE_CELLS = {1: 4, 2: 16, 3: 68, 4: 156}


def build_nosym_domain(level: int, dead_edges=None) -> Domain:
    """
    Identity-orbit Domain: every cell is its own orbit, so no symmetry
    is imposed. Constants carry the per-cell forcings from forced_masks
    (pond frame alive; outside-diamond and dead-edge cells dead).
    """
    n = POND_WIDTH * level
    custom_dead_edges = dead_edges is not None
    edge_alive, outside_dead, dead_edges = forced_masks(level, dead_edges)
    dead_edge_set = set(dead_edges)

    rep_i, rep_j = np.indices((n, n))

    constants = {}
    for i in range(n):
        for j in range(n):
            forced = None
            if edge_alive[i, j]:
                forced = 1
            if outside_dead[i, j] or (i, j) in dead_edge_set:
                assert forced is None, \
                    f"cell ({i},{j}) forced both alive and dead"
                forced = 0
            if forced is not None:
                constants[(i, j)] = forced

    free_reps = sorted({(i, j) for i in range(n) for j in range(n)}
                       - set(constants))

    # Border ring must be entirely forced dead, which is what makes the
    # % n wraparound inert (mirrors build_domain).
    if level >= 2:
        border = ([(0, k) for k in range(n)] + [(n - 1, k) for k in range(n)]
                  + [(k, 0) for k in range(n)] + [(k, n - 1) for k in range(n)])
        assert all(constants.get(cell) == 0 for cell in border), \
            "border ring not fully forced dead"

    expected = EXPECTED_FREE_CELLS.get(level)
    if expected is not None and not custom_dead_edges:
        assert len(free_reps) == expected, (
            f"level {level}: {len(free_reps)} free cells, expected {expected}"
        )

    return Domain(level=level, n=n,
                  rep_i=rep_i.astype(np.int64), rep_j=rep_j.astype(np.int64),
                  constants=constants, free_reps=free_reps)


def build_nosym_cnf(level: int,
                    birth: Sequence[int] = CONWAY[0],
                    survival: Sequence[int] = CONWAY[1],
                    dead_edges=None) -> Encoding:
    """
    Deduplicated still-life CNF over the identity domain. The fingerprint
    is salted with "nosym" so it can never collide with the symmetric
    encodings pinned in REPRODUCE.md.
    """
    birth = tuple(sorted(birth))
    survival = tuple(sorted(survival))
    domain = build_nosym_domain(level, dead_edges=dead_edges)
    clauses = _domain_clauses(domain, birth, survival)
    digest = hashlib.sha256()
    digest.update(f"nosym;level={level};n={domain.n};"
                  f"vars={len(domain.free_reps)};"
                  f"birth={birth};survival={survival};".encode())
    digest.update(repr(domain.free_reps).encode())
    digest.update(repr(clauses).encode())
    return Encoding(domain=domain, birth=birth, survival=survival,
                    n_vars=len(domain.free_reps), clauses=clauses,
                    sha256=digest.hexdigest())


def enumerate_nosym_tiles(level: int,
                          birth: Sequence[int] = CONWAY[0],
                          survival: Sequence[int] = CONWAY[1],
                          solver_name: str = "cadical195",
                          dead_edges=None,
                          limit: Optional[int] = None) -> np.ndarray:
    """
    All tiles without the symmetry requirement, as (m, n, n) uint8 grids
    in canonical order (live-cell count, then raw grid bytes) — the same
    order as sat_search.enumerate_tiles.
    """
    enc = build_nosym_cnf(level, birth, survival, dead_edges=dead_edges)
    rows = enumerate_all(enc, solver_name=solver_name, limit=limit)
    grids = enc.domain.expand_many(rows)
    order = sorted(range(len(grids)),
                   key=lambda i: (int(grids[i].sum()), grids[i].tobytes()))
    return grids[order]


# ------------------------------------------------------------- packing

def pack_nosym_solutions(grids: np.ndarray, level: int) -> np.ndarray:
    """
    Compress (m, n, n) grids to packed free-cell bits:
    (m, ceil(n_free/8)) uint8. Mirrors tile_domain.pack_solutions but
    over the identity domain (one bit per free cell, MSB-first).
    """
    domain = build_nosym_domain(level)
    bits = domain.extract_bits(np.asarray(grids))
    assert np.array_equal(domain.expand_many(bits),
                          np.asarray(grids, dtype=np.uint8)), (
        "grids are not expressible as free-cell assignments of this level"
    )
    return np.packbits(bits, axis=1)


def unpack_nosym_solutions(packed: np.ndarray, level: int) -> np.ndarray:
    """Inverse of pack_nosym_solutions: packed bits -> (m, n, n) uint8."""
    domain = build_nosym_domain(level)
    bits = np.unpackbits(packed, axis=1, count=len(domain.free_reps))
    return domain.expand_many(bits)


def load_nosym_tiles(level: int = 3) -> np.ndarray:
    """
    Load the shipped no-symmetry census for a level (currently level 3
    only; larger levels are local-only artifacts, see REPRODUCE.md).
    """
    resource = files(__package__).joinpath(
        f"data/solutions_pattern_nosym_level_{level}_cells.npy")
    if not resource.is_file():
        raise FileNotFoundError(
            f"No shipped nosym census for level {level}: {resource}")
    with resource.open("rb") as f:
        packed = np.load(f)
    return unpack_nosym_solutions(packed, level)


# ------------------------------------------------------- D4 analysis

def d4_images(grid: np.ndarray) -> np.ndarray:
    """The 8 D4 images of one grid (rotations of it and of its mirror)."""
    images = []
    for k in range(4):
        rotated = np.rot90(grid, k)
        images.append(np.ascontiguousarray(rotated))
        images.append(np.ascontiguousarray(rotated[:, ::-1]))
    return np.stack(images)


def d4_canonical_bytes(grid: np.ndarray) -> bytes:
    """Canonical representative of a grid's D4 class, as raw bytes."""
    return min(image.tobytes() for image in d4_images(grid))


def d4_classes(grids: np.ndarray) -> Dict[bytes, List[int]]:
    """Partition grids into D4 classes: canonical bytes -> row indices."""
    classes: Dict[bytes, List[int]] = {}
    for idx in range(len(grids)):
        classes.setdefault(d4_canonical_bytes(grids[idx]), []).append(idx)
    return classes


def d4_fixed_counts(grids: np.ndarray) -> List[int]:
    """
    Number of grids fixed by each of the 8 D4 transforms (identity
    first). If the grid set is closed under D4, Burnside's lemma gives
    sum(d4_fixed_counts(grids)) == 8 * len(d4_classes(grids)).
    """
    counts = []
    for t in range(8):
        k, mirror = divmod(t, 2)
        fixed = 0
        for grid in grids:
            image = np.rot90(grid, k)
            if mirror:
                image = image[:, ::-1]
            if np.array_equal(image, grid):
                fixed += 1
        counts.append(fixed)
    return counts

"""
SAT-based exhaustive enumeration of symmetric still-life tiles, for Conway's
Game of Life and any other Life-like cellular automaton B{birth}/S{survival}.

A tile of level L is stable iff every cell satisfies, with c = live Moore
neighbours (toroidal, matching the original ILP):

    alive cell:  c in survival     (the cell survives)
    dead cell:   c not in birth    (no birth occurs)

One boolean variable per free D4 symmetry orbit (see tile_domain.build_domain);
forced orbits are folded in as constants. The rules become uniform
"forbid exact neighbour count c" binomial clauses — no cardinality networks,
no auxiliary variables — so a blocking clause over the decision variables
removes exactly one assignment and AllSAT enumeration is exact and
duplicate-free. Positions are treated as a multiset (near a symmetry axis
several neighbour positions can map to one orbit variable; duplicate literals
collapse and tautologies drop, reproducing the ILP's positional counting).

Solving requires the optional dependency python-sat (`pip install
gol-mosaics[sat]`); building CNFs and the brute-force validator need only
numpy/scipy.

CLI (validation utility):
    python -m gol_mosaics.sat_search bruteforce --level 4
enumerates tiles by exhaustively checking all 2^n_free orbit assignments with
vectorised numpy — no SAT solver involved — and compares against the SAT
enumeration. Feasible through level 4 (2^22 assignments, ~10 s).
"""

import argparse
import hashlib
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .tile_domain import Domain, build_domain, neighbors

# A position's value is either a constant bool (forced orbit) or a DIMACS
# literal (positive int for the orbit variable).
Lit = Union[bool, int]
Clause = Tuple[int, ...]

CONWAY = ((3,), (2, 3))
HIGHLIFE = ((3, 6), (2, 3))


@dataclass(frozen=True)
class Encoding:
    domain: Domain
    birth: Tuple[int, ...]
    survival: Tuple[int, ...]
    n_vars: int                       # == len(domain.free_reps)
    clauses: List[Clause]             # sorted, deduplicated
    sha256: str

    @property
    def decision_vars(self) -> List[int]:
        return list(range(1, self.n_vars + 1))


def _simplify(raw: Iterable[Lit]) -> Optional[Clause]:
    """
    Simplify one clause: fold constants, collapse duplicates, drop
    tautologies. Returns None when the clause is satisfied/tautological.
    """
    lits = set()
    for lit in raw:
        if lit is True:
            return None
        if lit is False:
            continue
        lits.add(lit)
    for lit in lits:
        if -lit in lits:
            return None
    assert lits, "empty clause: constraints are unsatisfiable"
    return tuple(sorted(lits, key=lambda l: (abs(l), l)))


def _domain_clauses(domain,
                    birth: Tuple[int, ...],
                    survival: Tuple[int, ...]) -> List[Clause]:
    """
    Sorted, deduplicated binomial still-life clauses over one Domain.

    One variable per free rep; forced reps fold in as boolean constants.
    Works for any Domain regardless of how its orbits were constructed
    (D4-reduced or identity).
    """
    n = domain.n
    var_of_rep = {rep: idx + 1 for idx, rep in enumerate(domain.free_reps)}

    def lit(i: int, j: int) -> Lit:
        rep = domain.rep_of(i, j)
        if rep in domain.constants:
            return bool(domain.constants[rep])
        return var_of_rep[rep]

    def neg(x: Lit) -> Lit:
        return (not x) if isinstance(x, bool) else -x

    alive_forbidden = [c for c in range(9) if c not in survival]
    dead_forbidden = list(birth)

    clause_set = set()
    positions = range(8)
    for i in range(n):
        for j in range(n):
            centre = lit(i, j)
            nbrs = [lit(ii, jj) for (ii, jj) in neighbors(i, j, n)]

            # Forbid "centre in given state with exactly c live neighbours"
            # for every forbidden count c: one clause per c-subset of the 8
            # positions, asserting the subset is not exactly the live set.
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

    return sorted(clause_set)


def build_cnf(level: int,
              birth: Sequence[int] = CONWAY[0],
              survival: Sequence[int] = CONWAY[1],
              dead_edges=None) -> Encoding:
    """
    Build the deduplicated still-life CNF for one level and rule.

    dead_edges overrides the derived interlock forcings (see
    tile_domain.forced_masks); pass [] to enumerate tiles that are stable
    in isolation but may interact when mosaicked.
    """
    birth = tuple(sorted(birth))
    survival = tuple(sorted(survival))
    domain = build_domain(level, dead_edges=dead_edges)
    n = domain.n
    clauses = _domain_clauses(domain, birth, survival)
    digest = hashlib.sha256()
    digest.update(f"level={level};n={n};vars={len(domain.free_reps)};"
                  f"birth={birth};survival={survival};".encode())
    digest.update(repr(domain.free_reps).encode())
    digest.update(repr(clauses).encode())
    return Encoding(domain=domain, birth=birth, survival=survival,
                    n_vars=len(domain.free_reps), clauses=clauses,
                    sha256=digest.hexdigest())


# ---------------------------------------------------------------- solving

def enumerate_all(enc: Encoding, solver_name: str = "cadical195",
                  extra_units: Sequence[int] = (),
                  limit: Optional[int] = None) -> np.ndarray:
    """
    AllSAT over the decision variables: solve, record, block, repeat until
    UNSAT (or until `limit` solutions, for bounded probes).
    Returns a (m, n_vars) uint8 bit matrix.
    """
    try:
        from pysat.solvers import Solver
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "python-sat is required for SAT enumeration. "
            "Install with: pip install gol-mosaics[sat] (or pip install python-sat)"
        ) from exc

    rows = []
    with Solver(name=solver_name, bootstrap_with=enc.clauses) as solver:
        for unit in extra_units:
            solver.add_clause([unit])
        while (limit is None or len(rows) < limit) and solver.solve():
            model = solver.get_model()
            rows.append([1 if model[v - 1] > 0 else 0 for v in enc.decision_vars])
            solver.add_clause(
                [-v if model[v - 1] > 0 else v for v in enc.decision_vars]
            )
    return np.array(rows, dtype=np.uint8).reshape(len(rows), enc.n_vars)


def cube_units(cube: int, cube_bits: int) -> List[int]:
    """Unit clauses fixing the first cube_bits decision vars to cube's bits."""
    return [
        (b + 1) if (cube >> b) & 1 else -(b + 1)
        for b in range(cube_bits)
    ]


def enumerate_tiles(level: int,
                    birth: Sequence[int] = CONWAY[0],
                    survival: Sequence[int] = CONWAY[1],
                    solver_name: str = "cadical195",
                    dead_edges=None) -> np.ndarray:
    """Convenience: enumerate all tiles of a level as (m, n, n) uint8 grids."""
    enc = build_cnf(level, birth, survival, dead_edges=dead_edges)
    bits = enumerate_all(enc, solver_name)
    grids = enc.domain.expand_many(bits)
    order = sorted(range(len(grids)),
                   key=lambda i: (int(grids[i].sum()), grids[i].tobytes()))
    return grids[order]


# ---------------------------------------------------------------- bruteforce

def rule_violations(grids: np.ndarray,
                    birth: Sequence[int] = CONWAY[0],
                    survival: Sequence[int] = CONWAY[1]) -> np.ndarray:
    """
    Vectorised stability check on a (m, n, n) stack with toroidal
    neighbourhoods (the ILP's semantics). Returns a (m,) bool array,
    True where the grid VIOLATES the rule.
    """
    g = grids.astype(np.uint8)
    counts = np.zeros(g.shape, dtype=np.uint8)  # max 8 fits in uint8
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            counts += np.roll(np.roll(g, di, axis=1), dj, axis=2)
    alive = g == 1
    ok_alive = np.zeros(9, dtype=bool)
    ok_alive[list(survival)] = True
    ok_dead = np.ones(9, dtype=bool)
    ok_dead[list(birth)] = False
    bad = np.where(alive, ~ok_alive[counts], ~ok_dead[counts])
    return bad.any(axis=(1, 2))


def bruteforce_tiles(level: int,
                     birth: Sequence[int] = CONWAY[0],
                     survival: Sequence[int] = CONWAY[1],
                     chunk_bits: int = 18,
                     log=None) -> np.ndarray:
    """
    Enumerate all tiles of a level by exhaustively testing every one of the
    2^n_free free-orbit assignments — no SAT solver involved. This is the
    independent ground truth the SAT path is validated against; only
    feasible for small levels (level 4: 2^22 assignments, ~10 s).
    """
    domain = build_domain(level)
    k = len(domain.free_reps)
    total = 1 << k
    chunk = 1 << min(chunk_bits, k)
    weights = (1 << np.arange(k, dtype=np.uint64))

    keep = []
    for start in range(0, total, chunk):
        codes = np.arange(start, min(start + chunk, total), dtype=np.uint64)
        bits = ((codes[:, None] & weights[None, :]) > 0).astype(np.uint8)
        grids = domain.expand_many(bits)
        good = ~rule_violations(grids, birth, survival)
        if good.any():
            keep.append(grids[good])
        if log:
            log(f"\r  brute force: {min(start + chunk, total)}/{total} "
                f"assignments, {sum(len(x) for x in keep)} tiles", end="")
    if log:
        log("")
    grids = (np.concatenate(keep) if keep
             else np.empty((0, domain.n, domain.n), dtype=np.uint8))
    order = sorted(range(len(grids)),
                   key=lambda i: (int(grids[i].sum()), grids[i].tobytes()))
    return grids[order]


# ---------------------------------------------------------------- CLI

def _cmd_bruteforce(args) -> int:
    import time

    def log(msg, end="\n"):
        print(msg, end=end, flush=True)

    rule = f"B{''.join(map(str, args.birth))}/S{''.join(map(str, args.survival))}"
    t0 = time.time()
    brute = bruteforce_tiles(args.level, args.birth, args.survival, log=log)
    t1 = time.time()
    print(f"brute force ({rule}, level {args.level}): {len(brute)} tiles "
          f"in {t1 - t0:.1f}s")

    sat = enumerate_tiles(args.level, args.birth, args.survival)
    match = (sat.shape == brute.shape
             and sat.tobytes() == brute.astype(np.uint8).tobytes())
    print(f"SAT enumeration: {len(sat)} tiles in {time.time() - t1:.1f}s — "
          f"{'EXACT MATCH with brute force' if match else 'MISMATCH'}")
    return 0 if match else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "bruteforce",
        help="exhaustive 2^n_free enumeration (no SAT) vs SAT enumeration")
    p.add_argument("--level", type=int, default=4,
                   help="tile level (default 4; >4 is infeasible)")
    p.add_argument("--birth", type=int, nargs="+", default=list(CONWAY[0]))
    p.add_argument("--survival", type=int, nargs="+", default=list(CONWAY[1]))
    p.set_defaults(func=_cmd_bruteforce)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

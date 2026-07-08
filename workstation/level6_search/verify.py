"""
Independent solution checker and reference comparison.

Deliberately does NOT reuse the SAT encoding or the orbit machinery: rules,
symmetry, and forcings are re-derived here from first principles with
vectorised numpy, so an encoding bug cannot hide from the check. Every
solution the search produces is passed through ``check_solution`` before it
reaches the output file.
"""

from typing import List, Tuple

import numpy as np

from geometry import forced_masks


def neighbour_counts(grid: np.ndarray) -> np.ndarray:
    """Live-neighbour count per cell on a torus (matches the ILP's % n)."""
    g = grid.astype(np.int64)
    total = np.zeros_like(g)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            total += np.roll(np.roll(g, di, axis=0), dj, axis=1)
    return total


def check_solution(grid: np.ndarray, level: int) -> List[str]:
    """
    Return a list of violation descriptions (empty = valid level tile).

    Checks: binary values, torus still-life rules, D4 symmetry, pond-edge
    forced-alive, outside-diamond forced-dead, dead-edge forced-dead.
    """
    problems = []
    g = np.asarray(grid)
    n = g.shape[0]
    if g.shape != (n, n) or not np.isin(g, (0, 1)).all():
        return [f"not a binary {n}x{n} grid"]

    counts = neighbour_counts(g)
    alive = g == 1
    if (counts[alive] < 2).any() or (counts[alive] > 3).any():
        problems.append("alive cell with neighbour count outside 2..3")
    if (counts[~alive] == 3).any():
        problems.append("dead cell with exactly 3 neighbours (birth)")

    if not (np.array_equal(g, g[::-1]) and np.array_equal(g, g[:, ::-1])
            and np.array_equal(g, g.T)):
        problems.append("not D4 symmetric")

    edge_alive, outside_dead, dead_edges = forced_masks(level)
    if not (g[edge_alive == 1] == 1).all():
        problems.append("pond-edge cell not alive")
    if not (g[outside_dead == 1] == 0).all():
        problems.append("cell outside diamond not dead")
    for (i, j) in dead_edges:
        if g[i, j] != 0:
            problems.append(f"dead-edge cell ({i},{j}) not dead")

    return problems


def check_batch(solutions: np.ndarray, level: int) -> np.ndarray:
    """
    Vectorised validity check of a (m, n, n) stack; returns (m,) bool,
    True where the solution is a valid tile. Same semantics as
    check_solution, fast enough for 10^8-scale chunked verification.
    """
    g = np.asarray(solutions)
    m, n = g.shape[0], g.shape[1]
    valid = np.isin(g, (0, 1)).all(axis=(1, 2))

    counts = np.zeros(g.shape, dtype=np.int64)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            counts += np.roll(np.roll(g, di, axis=1), dj, axis=2)
    alive = g == 1
    valid &= ~((alive & ((counts < 2) | (counts > 3)))
               | (~alive & (counts == 3))).any(axis=(1, 2))

    valid &= (g == g[:, ::-1]).all(axis=(1, 2))
    valid &= (g == g[:, :, ::-1]).all(axis=(1, 2))
    valid &= (g == g.transpose(0, 2, 1)).all(axis=(1, 2))

    edge_alive, outside_dead, dead_edges = forced_masks(level)
    valid &= (g[:, edge_alive == 1] == 1).all(axis=1)
    valid &= (g[:, outside_dead == 1] == 0).all(axis=1)
    for (i, j) in dead_edges:
        valid &= g[:, i, j] == 0
    return valid


def check_all(solutions: np.ndarray, level: int) -> List[Tuple[int, List[str]]]:
    """Check every solution; return [(index, problems)] for failures only."""
    failures = []
    for idx in np.where(~check_batch(solutions, level))[0]:
        failures.append((int(idx), check_solution(solutions[idx], level)))
    return failures


def canonical_set(solutions: np.ndarray) -> set:
    """Order- and dtype-insensitive representation of a solution array."""
    return {np.asarray(g, dtype=np.uint8).tobytes() for g in solutions}


def compare_sets(found: np.ndarray, reference: np.ndarray) -> Tuple[bool, str]:
    """Compare two solution arrays as sets. Returns (equal, description)."""
    a, b = canonical_set(found), canonical_set(reference)
    if a == b:
        return True, f"exact match ({len(a)} unique solutions)"
    return False, (
        f"MISMATCH: found {len(a)} unique ({len(a - b)} not in reference), "
        f"reference {len(b)} unique ({len(b - a)} missing)"
    )

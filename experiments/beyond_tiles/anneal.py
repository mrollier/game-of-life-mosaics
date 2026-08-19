"""Massively parallel annealing for density-matched still lifes.

Energy: E = lam * (#unstable cells, bounded-plane semantics, including
no-birth in the dead border ring) + sum_w max(0, |live_w - t_w| - slack).
A still life meeting every window target within the slack has E = 0.

Correctness of the parallel sweep: a flip at u changes u's state, the
neighbour counts of the 8 cells around u, and u's window count. A
proposal at v computes its acceptance from states within Chebyshev
distance 2 and counts within distance 1 of v, so two simultaneous
proposals are independent iff their distance is >= 3. Cells are swept in
nine colour classes (i mod 3, j mod 3) — same-class cells are >= 3 apart
— and, within a phase, the parallel loop runs over *windows* with the
window's class cells handled serially, which also serialises updates to
that window's live count (a cell belongs to exactly one window in the
disjoint geometry this module requires).

Single-cell flips alone cannot nucleate structure at low temperature (a
lone live cell costs lam before it earns anything), so sweeps alternate
with 2x2 block moves: filling an isolated block is stability-neutral and
pays 4 cells of window deviation at once. Block origins are swept in
sixteen classes (i mod 4, j mod 4) — at distance 4 the 4x4 regions a
block move writes and the regions a concurrent proposal reads are
disjoint — and only blocks wholly inside one window are proposed, so the
window-serial loop keeps live counts race-free.

CPU backend via numba prange; the same kernel structure ports to an MLX
Metal kernel if CPU throughput turns out to be the limit.
"""

import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from numba import njit, prange

from beyond_tiles.targets import Window


@dataclass
class AnnealConfig:
    sweeps: int = 4000
    lam: float = 4.0  # weight of one unstable cell, in window-deviation units
    beta0: float = 0.3  # inverse temperature at the hot end
    beta1: float = 8.0  # at the cold end
    replicas: int = 4  # parallel-tempering ladder size
    swap_every: int = 50  # sweeps between replica swap attempts
    slack: int = 0
    seed: int = 0
    report_every: int = 500


@njit(cache=True)
def _unstable(state: np.uint8, count: np.int16) -> np.int64:
    if state == 1:
        return 1 if (count < 2 or count > 3) else 0
    return 1 if count == 3 else 0


@njit(parallel=True, cache=True)
def _phase(
    grid, counts, live, targets, slack, rows, cols, ptr, rand, beta, lam
):
    """One colour-class phase; returns the accepted energy change."""
    delta_total = 0.0
    for w in prange(ptr.shape[0] - 1):
        for idx in range(ptr[w], ptr[w + 1]):
            i, j = rows[idx], cols[idx]
            state = grid[i, j]
            flip_to = 1 - state
            d_count = 1 if flip_to == 1 else -1

            d_stab = -_unstable(state, counts[i, j]) + _unstable(
                np.uint8(flip_to), counts[i, j]
            )
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    c = counts[ii, jj]
                    s = grid[ii, jj]
                    d_stab += -_unstable(s, c) + _unstable(
                        s, np.int16(c + d_count)
                    )

            t = targets[w]
            l_now = live[w]
            dev_now = l_now - t if l_now > t else t - l_now
            dev_now = dev_now - slack if dev_now > slack else 0
            l_new = l_now + d_count
            dev_new = l_new - t if l_new > t else t - l_new
            dev_new = dev_new - slack if dev_new > slack else 0

            d_e = lam * d_stab + (dev_new - dev_now)
            if d_e <= 0.0 or rand[idx] < np.exp(-beta * d_e):
                grid[i, j] = flip_to
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di != 0 or dj != 0:
                            counts[i + di, j + dj] += d_count
                live[w] = l_new
                delta_total += d_e
    return delta_total


@njit(parallel=True, cache=True)
def _block_phase(
    grid, counts, live, targets, slack, rows, cols, ptr, rand, beta, lam
):
    """One block-move phase: toggle whole 2x2 blocks (fill unless full)."""
    # Per-window accumulator instead of a scalar prange reduction, which
    # numba's parfor pass rejects for this loop shape.
    deltas = np.zeros(ptr.shape[0] - 1, dtype=np.float64)
    for w in prange(ptr.shape[0] - 1):
        for idx in range(ptr[w], ptr[w + 1]):
            i, j = rows[idx], cols[idx]
            filled = (
                int(grid[i, j])
                + int(grid[i, j + 1])
                + int(grid[i + 1, j])
                + int(grid[i + 1, j + 1])
            )
            new_state = 0 if filled == 4 else 1

            # Stability delta over the 4x4 neighbourhood, by recompute:
            # states change only inside the block, counts inside the ring.
            d_stab = 0
            for di in range(-1, 3):
                for dj in range(-1, 3):
                    ii, jj = i + di, j + dj
                    s_old = grid[ii, jj]
                    in_block = 0 <= di <= 1 and 0 <= dj <= 1
                    s_new = new_state if in_block else s_old
                    c_old = counts[ii, jj]
                    # new count: recount live neighbours under the move
                    c_new = 0
                    for ai in range(-1, 2):
                        for aj in range(-1, 2):
                            if ai == 0 and aj == 0:
                                continue
                            bi, bj = ii + ai, jj + aj
                            if not (
                                0 <= bi < grid.shape[0]
                                and 0 <= bj < grid.shape[1]
                            ):
                                continue
                            nb = grid[bi, bj]
                            if 0 <= bi - i <= 1 and 0 <= bj - j <= 1:
                                nb = new_state
                            c_new += nb
                    d_stab += -_unstable(s_old, c_old) + _unstable(
                        np.uint8(s_new), np.int16(c_new)
                    )

            t = targets[w]
            l_now = live[w]
            dev_now = l_now - t if l_now > t else t - l_now
            dev_now = dev_now - slack if dev_now > slack else 0
            l_new = l_now + (4 - filled if new_state == 1 else -filled)
            dev_new = l_new - t if l_new > t else t - l_new
            dev_new = dev_new - slack if dev_new > slack else 0

            d_e = lam * d_stab + (dev_new - dev_now)
            if d_e <= 0.0 or rand[idx] < np.exp(-beta * d_e):
                for di in range(0, 2):
                    for dj in range(0, 2):
                        ii, jj = i + di, j + dj
                        if grid[ii, jj] != new_state:
                            d = 1 if new_state == 1 else -1
                            grid[ii, jj] = new_state
                            for ai in range(-1, 2):
                                for aj in range(-1, 2):
                                    if ai != 0 or aj != 0:
                                        counts[ii + ai, jj + aj] += d
                live[w] = l_new
                deltas[w] += d_e
    return deltas.sum()


def neighbour_counts(grid: np.ndarray) -> np.ndarray:
    """Padded live-neighbour counts (same shape as the padded grid)."""
    counts = np.zeros_like(grid, dtype=np.int16)
    h, w = grid.shape
    g = grid.astype(np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) == (0, 0):
                continue
            counts[
                max(0, di) : h + min(0, di), max(0, dj) : w + min(0, dj)
            ] += g[max(0, -di) : h + min(0, -di), max(0, -dj) : w + min(0, -dj)]
    return counts


def instability(grid: np.ndarray) -> int:
    """Unstable cells of the padded grid, bounded-plane semantics."""
    counts = neighbour_counts(grid)
    alive = grid == 1
    return int(
        (alive & ((counts < 2) | (counts > 3))).sum()
        + (~alive & (counts == 3)).sum()
    )


def energy(
    grid_padded: np.ndarray,
    free_mask: np.ndarray,
    windows: Sequence[Window],
    targets: np.ndarray,
    lam: float,
    slack: int,
) -> float:
    """Full recompute; the kernels track the same quantity incrementally."""
    interior = grid_padded[1:-1, 1:-1]
    dev = 0
    for t, (si, sj) in zip(targets, windows):
        live = int(interior[si, sj][free_mask[si, sj]].sum())
        dev += max(0, abs(live - int(t)) - int(slack))
    return lam * instability(grid_padded) + dev


def _class_csr(
    free_mask: np.ndarray, win_id: np.ndarray, n_windows: int
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per colour class: (rows, cols, window CSR ptr) in padded coords."""
    h, w = free_mask.shape
    out = []
    for ci in range(3):
        for cj in range(3):
            per_win: List[List[Tuple[int, int]]] = [[] for _ in range(n_windows)]
            for i in range(ci, h, 3):
                for j in range(cj, w, 3):
                    widx = win_id[i, j]
                    if widx >= 0 and free_mask[i, j]:
                        per_win[widx].append((i + 1, j + 1))
            rows, cols, ptr = [], [], [0]
            for cells in per_win:
                for i, j in cells:
                    rows.append(i)
                    cols.append(j)
                ptr.append(len(rows))
            out.append(
                (
                    np.asarray(rows, dtype=np.int64),
                    np.asarray(cols, dtype=np.int64),
                    np.asarray(ptr, dtype=np.int64),
                )
            )
    return out


def _block_csr(
    free_mask: np.ndarray, win_id: np.ndarray, n_windows: int
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per (i mod 4, j mod 4) class: 2x2 block origins wholly inside one
    free window, as (rows, cols, window CSR ptr) in padded coords."""
    h, w = free_mask.shape
    out = []
    for ci in range(4):
        for cj in range(4):
            per_win: List[List[Tuple[int, int]]] = [[] for _ in range(n_windows)]
            for i in range(ci, h - 1, 4):
                for j in range(cj, w - 1, 4):
                    widx = win_id[i, j]
                    if widx < 0:
                        continue
                    if (
                        win_id[i + 1, j] == widx
                        and win_id[i, j + 1] == widx
                        and win_id[i + 1, j + 1] == widx
                        and free_mask[i : i + 2, j : j + 2].all()
                    ):
                        per_win[widx].append((i + 1, j + 1))
            rows, cols, ptr = [], [], [0]
            for cells in per_win:
                for i, j in cells:
                    rows.append(i)
                    cols.append(j)
                ptr.append(len(rows))
            out.append(
                (
                    np.asarray(rows, dtype=np.int64),
                    np.asarray(cols, dtype=np.int64),
                    np.asarray(ptr, dtype=np.int64),
                )
            )
    return out


@dataclass
class _Replica:
    grid: np.ndarray  # padded (H+2, W+2)
    counts: np.ndarray
    live: np.ndarray
    beta: float
    lam: float
    energy: float


def anneal(
    seed_pattern: np.ndarray,
    free_mask: np.ndarray,
    windows: Sequence[Window],
    targets: np.ndarray,
    cfg: AnnealConfig,
    log=print,
) -> Tuple[np.ndarray, dict]:
    """Parallel tempering over a (beta, lam) ladder; returns the best grid
    (by energy at the campaign lam) and a stats dict."""
    h, w = seed_pattern.shape
    win_id = np.full((h, w), -1, dtype=np.int64)
    for idx, (si, sj) in enumerate(windows):
        win_id[si, sj] = idx
    csr = _class_csr(free_mask, win_id, len(windows))
    block_csr = _block_csr(free_mask, win_id, len(windows))
    targets64 = np.asarray(targets, dtype=np.int64)

    rng = np.random.default_rng(cfg.seed)
    # Ladder: geometric in beta, lam relaxed toward the hot end so hot
    # replicas can cross stability barriers (the 2-D tempering idea).
    betas = np.geomspace(cfg.beta0, cfg.beta1, cfg.replicas)
    lams = np.linspace(max(1.0, cfg.lam / 2), cfg.lam, cfg.replicas)

    replicas: List[_Replica] = []
    for r in range(cfg.replicas):
        grid = np.pad(seed_pattern.astype(np.uint8), 1)
        counts = neighbour_counts(grid)
        live = np.array(
            [
                int(seed_pattern[si, sj][free_mask[si, sj]].sum())
                for si, sj in windows
            ],
            dtype=np.int64,
        )
        e = energy(grid, free_mask, windows, targets64, lams[r], cfg.slack)
        replicas.append(_Replica(grid, counts, live, betas[r], lams[r], e))

    t0 = time.perf_counter()
    updates = 0
    best_grid, best_e = replicas[-1].grid.copy(), replicas[-1].energy
    history = []
    for sweep in range(cfg.sweeps):
        for rep in replicas:
            for rows, cols, ptr in csr:
                rand = rng.random(len(rows))
                rep.energy += _phase(
                    rep.grid,
                    rep.counts,
                    rep.live,
                    targets64,
                    cfg.slack,
                    rows,
                    cols,
                    ptr,
                    rand,
                    rep.beta,
                    rep.lam,
                )
                updates += len(rows)
            for rows, cols, ptr in block_csr:
                rand = rng.random(len(rows))
                rep.energy += _block_phase(
                    rep.grid,
                    rep.counts,
                    rep.live,
                    targets64,
                    cfg.slack,
                    rows,
                    cols,
                    ptr,
                    rand,
                    rep.beta,
                    rep.lam,
                )
                updates += len(rows)
        cold = replicas[-1]
        if cold.energy < best_e:
            best_e = cold.energy
            best_grid = cold.grid.copy()
            if best_e <= 1e-9:
                break
        if cfg.swap_every and (sweep + 1) % cfg.swap_every == 0:
            for r in range(cfg.replicas - 1):
                a, b = replicas[r], replicas[r + 1]
                # Exact swap acceptance for a ladder where both beta and
                # lam vary: evaluate each configuration at both lams.
                e_a_at_a = a.energy
                e_b_at_b = b.energy
                e_b_at_a = energy(
                    b.grid, free_mask, windows, targets64, a.lam, cfg.slack
                )
                e_a_at_b = energy(
                    a.grid, free_mask, windows, targets64, b.lam, cfg.slack
                )
                d = a.beta * (e_b_at_a - e_a_at_a) + b.beta * (e_a_at_b - e_b_at_b)
                if d <= 0 or rng.random() < np.exp(-d):
                    a.grid, b.grid = b.grid, a.grid
                    a.counts, b.counts = b.counts, a.counts
                    a.live, b.live = b.live, a.live
                    a.energy, b.energy = e_b_at_a, e_a_at_b
        if cfg.report_every and (sweep + 1) % cfg.report_every == 0:
            log(
                f"sweep {sweep + 1}: cold E={replicas[-1].energy:.0f} "
                f"best={best_e:.0f} "
                f"unstable={instability(replicas[-1].grid)} "
                f"({updates / max(time.perf_counter() - t0, 1e-9):.2e} updates/s)"
            )
        history.append((time.perf_counter() - t0, float(replicas[-1].energy)))

    wall = time.perf_counter() - t0
    pattern = best_grid[1:-1, 1:-1].astype(np.uint8)
    info = {
        "wall_time_s": wall,
        "updates": updates,
        "updates_per_s": updates / max(wall, 1e-9),
        "best_energy": float(best_e),
        "unstable_cells": instability(best_grid),
        "history": history,
    }
    return pattern, info


def kill_repair(pattern: np.ndarray) -> np.ndarray:
    """Greedy exact repair: kill live cells until no violation remains.

    Death violations vanish with the dying cell; a birth violation (dead
    cell with exactly three live neighbours) is resolved by killing one
    of the three. Monotone, so it always terminates; the cost is a little
    density, which the LNS polisher can win back.
    """
    grid = np.pad(pattern.astype(np.uint8), 1)
    for _ in range(grid.size):
        counts = neighbour_counts(grid)
        alive = grid == 1
        dying = alive & ((counts < 2) | (counts > 3))
        if dying.any():
            grid[dying] = 0
            continue
        births = np.argwhere(~alive & (counts == 3))
        if len(births) == 0:
            return grid[1:-1, 1:-1]
        i, j = births[0]
        neighbourhood = grid[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
        live_at = np.argwhere(neighbourhood == 1)
        di, dj = live_at[0]
        grid[max(i - 1, 0) + di, max(j - 1, 0) + dj] = 0
    raise AssertionError("kill_repair did not converge")

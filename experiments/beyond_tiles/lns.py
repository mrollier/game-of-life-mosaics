"""Rectangular large-neighbourhood search over a solved still life.

CP-SAT's generic LNS neighbourhoods cannot see that the variables live on
a 2-D grid. This module can: it frees a window-aligned rectangular patch,
keeps everything else frozen at the incumbent, and re-solves the patch to
(near-)optimality in a second or two. Because stability constraints have
radius 1, a sub-solve that also enforces the constraints of the frozen
1-ring around the patch is exactly a sub-problem of the global model:
every accepted patch is a valid still life again and the global objective
changes by exactly the sub-objective delta. Patches are chosen where the
deviation actually lives (the darkest windows), and non-interacting
patches (>= 2 frozen cells apart) are solved in parallel.

Requires pairwise-disjoint windows (stride == k, with edge="partial" when
the canvas is not a multiple of k).
"""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from beyond_tiles.targets import Window

_MARGIN = 2  # frozen context shipped around a patch; radius-1 constraints
# read one ring, and the ring's own constraints read a second one.


@dataclass
class LnsConfig:
    patch_windows: int = 5  # patch side length, in windows (5 x 8 = 40 cells)
    patch_time_s: float = 2.0
    budget_s: float = 600.0
    n_procs: int = 4  # the M4 has 4 performance cores
    seed: int = 0
    slack: int = 0
    max_diag_run: Optional[int] = 5  # match SpikeConfig, or repairs re-draw lines


@dataclass
class LnsResult:
    pattern: np.ndarray
    objective: int
    obj_history: List[Tuple[float, int]] = field(default_factory=list)
    rounds: int = 0
    patches_solved: int = 0
    patches_improved: int = 0


def window_devs(
    pattern: np.ndarray,
    free_mask: np.ndarray,
    windows: Sequence[Window],
    targets: np.ndarray,
    slack: int = 0,
) -> np.ndarray:
    """Per-window objective terms max(0, |live - t| - slack)."""
    devs = np.zeros(len(windows), dtype=np.int64)
    for idx, (t, (si, sj)) in enumerate(zip(targets, windows)):
        live = int(pattern[si, sj][free_mask[si, sj]].sum())
        devs[idx] = max(0, abs(live - int(t)) - int(slack))
    return devs


def _window_grid(windows: Sequence[Window]) -> Tuple[list, list, np.ndarray]:
    """Row starts, col starts and the (R, C) -> window-index lattice."""
    rows = sorted({w[0].start for w in windows})
    cols = sorted({w[1].start for w in windows})
    index = np.full((len(rows), len(cols)), -1, dtype=np.int64)
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: j for j, c in enumerate(cols)}
    for idx, (si, sj) in enumerate(windows):
        index[ri[si.start], ci[sj.start]] = idx
    return rows, cols, index


def _assert_disjoint(shape: Tuple[int, int], windows: Sequence[Window]) -> None:
    owner = np.zeros(shape, dtype=np.int8)
    for si, sj in windows:
        owner[si, sj] += 1
    if (owner > 1).any():
        raise ValueError(
            "LNS needs pairwise-disjoint windows (stride == k; use "
            "edge_windows='partial' when the canvas size is not a "
            "multiple of k)"
        )


def _patch_boxes(
    n_rows: int, n_cols: int, patch: int
) -> List[Tuple[int, int, int, int]]:
    """Window-grid boxes (r0, r1, c0, c1), overlapping by half a patch."""
    step = max(1, patch // 2)
    r_starts = sorted({min(r, max(0, n_rows - patch)) for r in range(0, n_rows, step)})
    c_starts = sorted({min(c, max(0, n_cols - patch)) for c in range(0, n_cols, step)})
    return [
        (r, min(r + patch, n_rows), c, min(c + patch, n_cols))
        for r in r_starts
        for c in c_starts
    ]


def _cell_box(
    box: Tuple[int, int, int, int], windows: Sequence[Window], index: np.ndarray
) -> Tuple[int, int, int, int]:
    """Cell-coordinate bounding box of a window-grid box."""
    r0, r1, c0, c1 = box
    members = index[r0:r1, c0:c1].ravel()
    members = members[members >= 0]
    i0 = min(windows[m][0].start for m in members)
    i1 = max(windows[m][0].stop for m in members)
    j0 = min(windows[m][1].start for m in members)
    j1 = max(windows[m][1].stop for m in members)
    return i0, i1, j0, j1


def _select_disjoint(
    scored: List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int]]],
    limit: int,
) -> List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """Greedy top-score patches whose freed cells stay >= 2 cells apart."""
    chosen: List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
    for item in sorted(scored, key=lambda s: -s[0]):
        if len(chosen) >= limit:
            break
        i0, i1, j0, j1 = item[2]
        clash = False
        for _, _, (a0, a1, b0, b1) in chosen:
            if i0 < a1 + _MARGIN and a0 < i1 + _MARGIN and j0 < b1 + _MARGIN and b0 < j1 + _MARGIN:
                clash = True
                break
        if not clash:
            chosen.append(item)
    return chosen


def _solve_patch_task(payload: dict) -> Optional[np.ndarray]:
    """Re-solve one patch to (near-)optimality; runs in a worker process.

    `region` is the frozen incumbent around the patch (patch + 2-cell
    margin, already embedded in the global dead border where relevant).
    Every cell within one ring of a freed variable keeps its stability
    constraint, with frozen neighbours folded in as constants — so any
    solution drops back into the global pattern as a still life.
    """
    from ortools.sat.python import cp_model

    region = payload["region"]
    free = payload["free"]  # same shape; freed-variable mask (patch & free)
    win_boxes = payload["windows"]  # [(i0, i1, j0, j1, target), ...] region coords
    slack = payload["slack"]

    h, w = region.shape
    model = cp_model.CpModel()
    x = {}
    for i in range(h):
        for j in range(w):
            if free[i, j]:
                x[(i, j)] = model.NewBoolVar("")

    constrained = np.zeros((h, w), dtype=bool)
    for (i, j) in x:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < h and 0 <= jj < w:
                    constrained[ii, jj] = True

    for i in range(h):
        for j in range(w):
            if not constrained[i, j]:
                continue
            var_nbrs, const_live = [], 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di, dj) == (0, 0):
                        continue
                    ii, jj = i + di, j + dj
                    if not (0 <= ii < h and 0 <= jj < w):
                        continue  # beyond the margin: frozen dead plane
                    if (ii, jj) in x:
                        var_nbrs.append(x[(ii, jj)])
                    else:
                        const_live += int(region[ii, jj])
            s = cp_model.LinearExpr.Sum(var_nbrs)
            if (i, j) in x:
                model.AddLinearConstraint(s, 2 - const_live, 3 - const_live).OnlyEnforceIf(x[(i, j)])
                shifted = cp_model.Domain.FromIntervals(
                    [[-const_live, 2 - const_live], [4 - const_live, 8 - const_live]]
                )
                model.AddLinearExpressionInDomain(s, shifted).OnlyEnforceIf(
                    x[(i, j)].Not()
                )
            elif region[i, j]:
                model.AddLinearConstraint(s, 2 - const_live, 3 - const_live)
            else:
                shifted = cp_model.Domain.FromIntervals(
                    [[-const_live, 2 - const_live], [4 - const_live, 8 - const_live]]
                )
                model.AddLinearExpressionInDomain(s, shifted)

    devs = []
    for i0, i1, j0, j1, target in win_boxes:
        cells = [x[(i, j)] for i in range(i0, i1) for j in range(j0, j1) if (i, j) in x]
        live = cp_model.LinearExpr.Sum(cells)
        dev = model.NewIntVar(0, max(1, len(cells)), "")
        model.Add(dev >= live - int(target) - slack)
        model.Add(dev >= int(target) - live - slack)
        devs.append(dev)
    model.Minimize(cp_model.LinearExpr.Sum(devs))

    if payload["max_diag_run"] is not None:
        from beyond_tiles.still_image import forbid_diagonal_runs

        # Frozen live cells become `True` so a run straddling the patch
        # boundary is still broken by the free cells it does contain.
        forbid_diagonal_runs(
            model,
            lambda i, j: x[(i, j)] if (i, j) in x else (True if region[i, j] else None),
            region.shape,
            payload["max_diag_run"],
        )

    for (i, j), var in x.items():
        model.AddHint(var, int(region[i, j]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = payload["time_s"]
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = payload["seed"]
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    out = region.copy()
    for (i, j), var in x.items():
        out[i, j] = solver.Value(var)
    return out


def improve(
    pattern: np.ndarray,
    free_mask: np.ndarray,
    windows: Sequence[Window],
    targets: np.ndarray,
    lcfg: LnsConfig,
    log=print,
) -> LnsResult:
    """Round-based destroy-and-repair until the budget or a fixpoint."""
    _assert_disjoint(pattern.shape, windows)
    rows, cols, index = _window_grid(windows)
    n_rows, n_cols = len(rows), len(cols)
    boxes = _patch_boxes(n_rows, n_cols, lcfg.patch_windows)

    pattern = pattern.astype(np.uint8).copy()
    t0 = time.perf_counter()
    result = LnsResult(pattern=pattern, objective=0)
    devs = window_devs(pattern, free_mask, windows, targets, lcfg.slack)
    result.obj_history.append((0.0, int(devs.sum())))

    # Boxes that failed to improve go stale and are skipped until an
    # accepted patch nearby invalidates their context. Without this, a
    # few high-deviation but locally-optimal regions (dark windows at
    # the density ceiling) monopolize every round and the loop gives up
    # while plenty of improvable patches never got a turn.
    stale: set = set()

    while time.perf_counter() - t0 < lcfg.budget_s:
        scored = []
        for box in boxes:
            if box in stale:
                continue
            members = index[box[0] : box[1], box[2] : box[3]].ravel()
            members = members[members >= 0]
            score = int(devs[members].sum())
            if score > 0:
                scored.append((score, box, _cell_box(box, windows, index)))
        if not scored:
            break
        chosen = _select_disjoint(scored, limit=max(1, lcfg.n_procs) * 2)

        pad = np.pad(pattern, _MARGIN + 1)  # dead plane beyond the canvas
        free_pad = np.pad(free_mask, _MARGIN + 1)
        payloads = []
        for _, box, (i0, i1, j0, j1) in chosen:
            o = _MARGIN + 1
            g0, g1 = i0 + o - _MARGIN, i1 + o + _MARGIN
            h0, h1 = j0 + o - _MARGIN, j1 + o + _MARGIN
            region = pad[g0:g1, h0:h1].copy()
            rfree = np.zeros_like(region, dtype=bool)
            rfree[_MARGIN : _MARGIN + (i1 - i0), _MARGIN : _MARGIN + (j1 - j0)] = (
                free_pad[g0 + _MARGIN : g1 - _MARGIN, h0 + _MARGIN : h1 - _MARGIN]
            )
            members = index[box[0] : box[1], box[2] : box[3]].ravel()
            members = members[members >= 0]
            win_boxes = [
                (
                    windows[m][0].start - i0 + _MARGIN,
                    windows[m][0].stop - i0 + _MARGIN,
                    windows[m][1].start - j0 + _MARGIN,
                    windows[m][1].stop - j0 + _MARGIN,
                    int(targets[m]),
                )
                for m in members
            ]
            payloads.append(
                dict(
                    region=region,
                    free=rfree,
                    windows=win_boxes,
                    slack=lcfg.slack,
                    max_diag_run=lcfg.max_diag_run,
                    time_s=lcfg.patch_time_s,
                    seed=lcfg.seed,
                    box=(i0, i1, j0, j1),
                )
            )

        if lcfg.n_procs > 1 and len(payloads) > 1:
            with ProcessPoolExecutor(max_workers=lcfg.n_procs) as pool:
                outs = list(pool.map(_solve_patch_task, payloads))
        else:
            outs = [_solve_patch_task(p) for p in payloads]

        accepted_boxes = []
        for (_, wbox, cbox), payload, out in zip(chosen, payloads, outs):
            result.patches_solved += 1
            improved = False
            if out is not None:
                i0, i1, j0, j1 = payload["box"]
                new_patch = out[
                    _MARGIN : _MARGIN + (i1 - i0), _MARGIN : _MARGIN + (j1 - j0)
                ]
                candidate = pattern.copy()
                candidate[i0:i1, j0:j1] = new_patch
                new_devs = window_devs(
                    candidate, free_mask, windows, targets, lcfg.slack
                )
                if new_devs.sum() < devs.sum():
                    pattern, devs = candidate, new_devs
                    result.patches_improved += 1
                    improved = True
                    accepted_boxes.append(cbox)
            if not improved:
                stale.add(wbox)
        # A change of context wakes up nearby stale boxes.
        if accepted_boxes:
            for box in list(stale):
                b0, b1, c0, c1 = _cell_box(box, windows, index)
                for a0, a1, d0, d1 in accepted_boxes:
                    if b0 < a1 + _MARGIN and a0 < b1 + _MARGIN and \
                            c0 < d1 + _MARGIN and d0 < c1 + _MARGIN:
                        stale.discard(box)
                        break
        result.rounds += 1
        result.obj_history.append(
            (time.perf_counter() - t0, int(devs.sum()))
        )
        log(
            f"lns round {result.rounds}: objective {int(devs.sum())} "
            f"({result.patches_improved}/{result.patches_solved} patches improved)"
        )

    result.pattern = pattern
    result.objective = int(devs.sum())
    return result

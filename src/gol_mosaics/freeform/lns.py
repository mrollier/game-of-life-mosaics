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

from .targets import Window, box_sums, live_table, window_bounds

_MARGIN = 2  # frozen context shipped around a patch; radius-1 constraints
# read one ring, and the ring's own constraints read a second one.


def _margin(lcfg: "LnsConfig") -> int:
    """Frozen context shipped around a patch, in cells.

    Two for stability, and `max_diag_run` for the run clauses: a chain that
    continues past the shipped margin is invisible to the sub-model, so the
    patch may lengthen it without ever seeing a clause. That is how the Lam
    Gods polish (2026-09-16) left twenty chains of 6-7 cells behind a cap of
    5. The same distance keeps concurrently solved patches apart.
    """
    return max(_MARGIN, lcfg.max_diag_run or 0)


@dataclass
class LnsConfig:
    patch_windows: int = 5  # patch side length, in windows (5 x 8 = 40 cells)
    patch_time_s: float = 2.0
    budget_s: float = 600.0
    n_procs: int = 4  # the M4 has 4 performance cores
    seed: int = 0
    slack: int = 0
    max_diag_run: Optional[int] = 5  # match SolveConfig, or repairs re-draw lines
    # Seam sub-targets. A window total is blind to how its cells are spread,
    # so the dead separator lines of a strip or block decomposition stay
    # half-empty once their windows are satisfied: the Lam Gods separators
    # held 53-75 % of their neighbours' density after a polish to objective
    # 469, a visible line across every dense band. With the first row (or
    # column) of each `seam_width`-wide separator listed here, every window
    # crossing it gets a sub-box with its proportional share as target,
    # weighted `seam_weight` in the patch solver and, one-sided (deficit
    # only), in selection and acceptance. The flyer's seam pass, generalised.
    seam_rows: Tuple[int, ...] = ()
    seam_cols: Tuple[int, ...] = ()
    seam_width: int = 2
    seam_weight: int = 3


@dataclass
class LnsResult:
    pattern: np.ndarray
    objective: int
    obj_history: List[Tuple[float, int]] = field(default_factory=list)
    rounds: int = 0
    patches_solved: int = 0
    patches_improved: int = 0


def seam_boxes(
    window: Window, target: int, lcfg: "LnsConfig"
) -> List[Tuple[int, int, int, int, int]]:
    """Separator sub-boxes (i0, i1, j0, j1, target) of one window, global coords."""
    si, sj = window
    out = []
    w = lcfg.seam_width
    for s in lcfg.seam_rows:
        if si.start <= s and s + w <= si.stop:
            share = int(round(target * w / (si.stop - si.start)))
            out.append((s, s + w, sj.start, sj.stop, share))
    for s in lcfg.seam_cols:
        if sj.start <= s and s + w <= sj.stop:
            share = int(round(target * w / (sj.stop - sj.start)))
            out.append((si.start, si.stop, s, s + w, share))
    return out


def window_devs(
    pattern: np.ndarray,
    free_mask: np.ndarray,
    windows: Sequence[Window],
    targets: np.ndarray,
    slack: int = 0,
    lcfg: Optional["LnsConfig"] = None,
) -> np.ndarray:
    """Per-window objective terms max(0, |live - t| - slack).

    With seam lines configured on `lcfg`, a window crossing one also pays
    `seam_weight` per cell its separator sub-box falls short of its share
    (deficit only: a full separator is never penalised).
    """
    targets = np.asarray(targets, dtype=np.int64)
    bounds = window_bounds(windows)
    table, local = live_table(pattern, free_mask, bounds)
    live = box_sums(None, local, table)
    devs = np.maximum(0, np.abs(live - targets) - int(slack))
    if lcfg is None or not (lcfg.seam_rows or lcfg.seam_cols):
        return devs

    origin = bounds[:, [0, 0, 2, 2]].min(axis=0) if len(bounds) else 0
    w = lcfg.seam_width
    # The same sub-boxes as seam_boxes(), for every window at once
    for axis, seams in ((0, lcfg.seam_rows), (1, lcfg.seam_cols)):
        lo, hi = bounds[:, 2 * axis], bounds[:, 2 * axis + 1]
        for s in seams:
            hit = (lo <= s) & (s + w <= hi)
            if not hit.any():
                continue
            share = np.round(targets[hit] * w / (hi[hit] - lo[hit])).astype(np.int64)
            boxes = bounds[hit].copy()
            boxes[:, 2 * axis], boxes[:, 2 * axis + 1] = s, s + w
            got = box_sums(None, boxes - origin, table)
            devs[hit] += lcfg.seam_weight * np.maximum(0, share - got)
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
    margin: int = _MARGIN,
) -> List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """Greedy top-score patches whose freed cells stay >= `margin` cells apart."""
    chosen: List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
    for item in sorted(scored, key=lambda s: -s[0]):
        if len(chosen) >= limit:
            break
        i0, i1, j0, j1 = item[2]
        clash = False
        for _, _, (a0, a1, b0, b1) in chosen:
            if i0 < a1 + margin and a0 < i1 + margin and j0 < b1 + margin and b0 < j1 + margin:
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
        from .solver import forbid_diagonal_runs

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
    # One read of the solution vector instead of a Value() call per cell
    cells = np.array(list(x), dtype=np.int64).reshape(-1, 2)
    index = np.array([var.Index() for var in x.values()], dtype=np.int64)
    solution = np.asarray(solver.ResponseProto().solution, dtype=np.int64)
    out = region.copy()
    out[cells[:, 0], cells[:, 1]] = solution[index]
    return out


def _patch_payload(pattern, free_mask, windows, targets, index, box, cbox, lcfg):
    """The sub-problem for one window-grid box: patch, frozen margin, targets."""
    m = _margin(lcfg)
    i0, i1, j0, j1 = cbox
    pad = np.pad(pattern, m + 1)  # dead plane beyond the canvas
    free_pad = np.pad(free_mask, m + 1)
    o = m + 1
    g0, g1 = i0 + o - m, i1 + o + m
    h0, h1 = j0 + o - m, j1 + o + m
    region = pad[g0:g1, h0:h1].copy()
    rfree = np.zeros_like(region, dtype=bool)
    rfree[m : m + (i1 - i0), m : m + (j1 - j0)] = free_pad[g0 + m : g1 - m, h0 + m : h1 - m]
    members = index[box[0] : box[1], box[2] : box[3]].ravel()
    members = members[members >= 0]
    win_boxes = [
        (
            windows[k][0].start - i0 + m,
            windows[k][0].stop - i0 + m,
            windows[k][1].start - j0 + m,
            windows[k][1].stop - j0 + m,
            int(targets[k]),
        )
        for k in members
    ]
    # Seam sub-boxes, repeated seam_weight times: the patch solver weighs
    # every box the same, so the weight is carried by repetition.
    for k in members:
        for a0, a1, b0, b1, share in seam_boxes(windows[k], int(targets[k]), lcfg):
            win_boxes += [(a0 - i0 + m, a1 - i0 + m, b0 - j0 + m, b1 - j0 + m, share)] * lcfg.seam_weight
    return dict(
        region=region,
        free=rfree,
        windows=win_boxes,
        slack=lcfg.slack,
        max_diag_run=lcfg.max_diag_run,
        time_s=lcfg.patch_time_s,
        seed=lcfg.seed,
        box=(i0, i1, j0, j1),
        margin=m,
    )


def _solve_patches(payloads, lcfg, pool=None):
    """Solve patch payloads, in `pool` when there is one and it pays."""
    if pool is not None and len(payloads) > 1:
        return list(pool.map(_solve_patch_task, payloads))
    return [_solve_patch_task(p) for p in payloads]


class _PatchPool:
    """One worker pool for a whole improve() or repair call.

    Spawning processes costs a re-import of numpy and OR-Tools per worker,
    which on Windows (spawn, no fork) is paid every round if the pool is
    rebuilt per round. None when a single process is asked for.
    """

    def __init__(self, lcfg):
        self.pool = (ProcessPoolExecutor(max_workers=lcfg.n_procs)
                     if lcfg.n_procs > 1 else None)

    def __enter__(self):
        return self.pool

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        return False


def _members(index: np.ndarray, box) -> np.ndarray:
    """Window indices inside a window-grid box."""
    members = index[box[0]: box[1], box[2]: box[3]].ravel()
    return members[members >= 0]


def _updated_devs(devs, candidate, free_mask, windows, targets, members,
                  slack, lcfg):
    """`devs` after a patch whose changes lie inside the `members` windows.

    Windows are disjoint and a patch only rewrites cells of its own member
    windows (their bounding box), so every other window keeps its term.
    """
    new = devs.copy()
    new[members] = window_devs(candidate, free_mask,
                               [windows[k] for k in members],
                               np.asarray(targets)[members], slack, lcfg)
    return new


def _splice(pattern, payload, out):
    """The global pattern with one solved patch dropped back in."""
    m = payload["margin"]
    i0, i1, j0, j1 = payload["box"]
    candidate = pattern.copy()
    candidate[i0:i1, j0:j1] = out[m : m + (i1 - i0), m : m + (j1 - j0)]
    return candidate


def seam_occupancy(
    pattern: np.ndarray, lcfg: "LnsConfig", free_mask: Optional[np.ndarray] = None,
    span: int = 4,
) -> dict:
    """Separator density relative to the `span` lines on either side, per seam.

    1.0 means a seam is indistinguishable from its surroundings; seams whose
    neighbourhood is nearly empty (< 2 % live) are skipped, nothing to see.
    Returns {"rows": {first_row: ratio}, "cols": {...}, "worst": min}.
    """
    p = np.asarray(pattern, dtype=np.int64)
    free = np.ones(p.shape, bool) if free_mask is None else free_mask
    w = lcfg.seam_width
    out = {"rows": {}, "cols": {}}
    for axis, seams, key in ((0, lcfg.seam_rows, "rows"), (1, lcfg.seam_cols, "cols")):
        q, f = (p, free) if axis == 0 else (p.T, free.T)
        for s in seams:
            nb = np.r_[max(0, s - span):s, s + w:s + w + span]
            nb_free, sep_free = int(f[nb].sum()), int(f[s:s + w].sum())
            if nb_free == 0 or sep_free == 0 or q[nb].sum() < 0.02 * nb_free:
                continue
            out[key][int(s)] = float((q[s:s + w].sum() / sep_free) / (q[nb].sum() / nb_free))
    ratios = list(out["rows"].values()) + list(out["cols"].values())
    out["worst"] = min(ratios) if ratios else 1.0
    return out


def _long_run_centres(pattern: np.ndarray, max_run: int):
    """Middle cell of every diagonal chain longer than `max_run`."""
    live = np.asarray(pattern, dtype=bool)
    h, w = live.shape
    centres = []
    for dj in (1, -1):
        chain = live.copy()
        for n in range(1, max_run + 1):  # starts of runs of max_run + 1
            shifted = np.zeros_like(live)
            if dj == 1:
                shifted[: h - n, : w - n] = live[n:, n:]
            else:
                shifted[: h - n, n:] = live[n:, : w - n]
            chain &= shifted
        for i, j in zip(*np.nonzero(chain)):
            t = (max_run + 1) // 2
            centres.append((int(i + t), int(j + t * dj)))
    return centres


def repair_diagonal_runs(
    pattern: np.ndarray,
    free_mask: np.ndarray,
    windows: Sequence[Window],
    targets: np.ndarray,
    lcfg: LnsConfig,
    passes: int = 3,
    max_cost: int = 2,
    log=print,
) -> LnsResult:
    """Break every diagonal chain longer than `lcfg.max_diag_run`.

    Each chain gets a patch centred on it, so the whole chain is free and
    the run clauses see it; the patch solver then minimises deviation
    subject to the cap, and the patch is accepted when it costs at most
    `max_cost` cells of deviation. The incumbent hint violates the new
    clause, so the solver is effectively starting over: keep the patches
    small (3 windows) and the time generous (30 s) — 40x40 at 10 s came
    back ~150 cells worse per patch on the Lam Gods pattern.
    """
    _assert_disjoint(pattern.shape, windows)
    assert lcfg.max_diag_run, "nothing to repair without a cap"
    rows, cols, index = _window_grid(windows)
    n_rows, n_cols = len(rows), len(cols)
    p = lcfg.patch_windows
    pattern = pattern.astype(np.uint8).copy()
    devs = window_devs(pattern, free_mask, windows, targets, lcfg.slack, lcfg)
    result = LnsResult(pattern=pattern, objective=int(devs.sum()))
    m = _margin(lcfg)

    for _ in range(passes):
        centres = _long_run_centres(pattern, lcfg.max_diag_run)
        if not centres:
            break
        scored = []
        for i, j in centres:
            r = int(np.searchsorted(rows, i, side="right") - 1)
            c = int(np.searchsorted(cols, j, side="right") - 1)
            r0 = min(max(0, r - p // 2), max(0, n_rows - p))
            c0 = min(max(0, c - p // 2), max(0, n_cols - p))
            box = (r0, min(r0 + p, n_rows), c0, min(c0 + p, n_cols))
            scored.append((1, box, _cell_box(box, windows, index)))
        chosen = _select_disjoint(scored, limit=len(scored), margin=m)
        payloads = [
            _patch_payload(pattern, free_mask, windows, targets, index, box, cbox, lcfg)
            for _, box, cbox in chosen
        ]
        with _PatchPool(lcfg) as pool:
            outs = _solve_patches(payloads, lcfg, pool)
        for (_, box, _), payload, out in zip(chosen, payloads, outs):
            result.patches_solved += 1
            if out is None:
                continue
            candidate = _splice(pattern, payload, out)
            new_devs = _updated_devs(devs, candidate, free_mask, windows,
                                     targets, _members(index, box),
                                     lcfg.slack, lcfg)
            if new_devs.sum() - devs.sum() <= max_cost:
                pattern, devs = candidate, new_devs
                result.patches_improved += 1
        result.rounds += 1
        log(f"diagonal repair pass {result.rounds}: {len(centres)} chains, "
            f"{len(chosen)} patches, objective {int(devs.sum())}")

    result.pattern = pattern
    result.objective = int(devs.sum())
    return result


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
    m = _margin(lcfg)
    t0 = time.perf_counter()
    result = LnsResult(pattern=pattern, objective=0)
    devs = window_devs(pattern, free_mask, windows, targets, lcfg.slack, lcfg)
    result.obj_history.append((0.0, int(devs.sum())))

    # Patch geometry is fixed for the whole call: compute each box's member
    # windows and cell bounding box once, not every round.
    # The grid holds only windows with free cells, so on a canvas with
    # large fixed regions some boxes hold none: drop them.
    members = {box: _members(index, box) for box in boxes}
    boxes = [box for box in boxes if len(members[box])]
    cell_boxes = {box: _cell_box(box, windows, index) for box in boxes}

    # Boxes that failed to improve go stale and are skipped until an
    # accepted patch nearby invalidates their context. Without this, a
    # few high-deviation but locally-optimal regions (dark windows at
    # the density ceiling) monopolize every round and the loop gives up
    # while plenty of improvable patches never got a turn.
    stale: set = set()

    with _PatchPool(lcfg) as pool:
        while time.perf_counter() - t0 < lcfg.budget_s:
            scored = []
            for box in boxes:
                if box in stale:
                    continue
                score = int(devs[members[box]].sum())
                if score > 0:
                    scored.append((score, box, cell_boxes[box]))
            if not scored:
                break
            chosen = _select_disjoint(scored, limit=max(1, lcfg.n_procs) * 2,
                                      margin=m)

            payloads = [
                _patch_payload(pattern, free_mask, windows, targets, index,
                               box, cbox, lcfg)
                for _, box, cbox in chosen
            ]
            outs = _solve_patches(payloads, lcfg, pool)

            accepted_boxes = []
            for (_, wbox, cbox), payload, out in zip(chosen, payloads, outs):
                result.patches_solved += 1
                improved = False
                if out is not None:
                    candidate = _splice(pattern, payload, out)
                    new_devs = _updated_devs(devs, candidate, free_mask,
                                             windows, targets, members[wbox],
                                             lcfg.slack, lcfg)
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
                    b0, b1, c0, c1 = cell_boxes[box]
                    for a0, a1, d0, d1 in accepted_boxes:
                        if (b0 < a1 + m and a0 < b1 + m and c0 < d1 + m
                                and d0 < c1 + m):
                            stale.discard(box)
                            break
            result.rounds += 1
            result.obj_history.append(
                (time.perf_counter() - t0, int(devs.sum()))
            )
            log(
                f"lns round {result.rounds}: objective {int(devs.sum())} "
                f"({result.patches_improved}/{result.patches_solved} patches "
                f"improved)"
            )

    result.pattern = pattern
    result.objective = int(devs.sum())
    return result

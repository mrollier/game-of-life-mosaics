"""CP-SAT encoder for free-form, density-matched still lifes.

Stability is hard; window densities are soft (minimize total absolute
deviation from the greyscale-derived targets). Variables live on an
(H+2) x (W+2) grid whose border ring is forced dead — the ring cells
still carry the no-birth constraint, so a solution is a genuine still
life embedded in a dead plane, not just internally consistent.
"""

import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ortools.sat.python import cp_model

from beyond_tiles.targets import Window, cell_targets, window_slices, window_targets

# Live-neighbour counts a DEAD cell may have (everything except birth on 3).
_DEAD_OK = cp_model.Domain.FromIntervals([[0, 2], [4, 8]])


@dataclass
class SpikeConfig:
    k: int = 8
    stride: int = 4
    d_max: float = 0.45
    time_limit_s: float = 600.0
    workers: int = 10
    seed: int = 0
    mask_mode: str = "force_dead"  # "force_dead" | "soft_zero" | "none"
    snapshot_gap_s: float = 0.0  # 0 = don't keep incumbent patterns
    max_snapshots: int = 400
    log_to: Optional[str] = None  # write the CP-SAT search log to this file
    edge_windows: str = "clamp"  # "clamp" (historic) | "partial" (disjoint)
    slack: int = 0  # per-window deviation tolerance, in cells
    dither: str = "round"  # "round" | "fs" (error-diffused window targets)
    add_lb_subsolvers: bool = False  # schedule the bound-improving subsolvers
    num_violation_ls: int = 0  # Feasibility-Jump local-search workers
    symmetry_level: Optional[int] = None  # None = CP-SAT default
    max_det_time: Optional[float] = None  # deterministic-time cap (repro A/Bs)
    hint_mode: str = "none"  # "none" | "agar" (constructive block-lattice seed)
    repair_hint: bool = True
    hint_conflict_limit: int = 100_000  # CP-SAT default of 10 is useless here


# One captured incumbent: (wall time, objective, interior pattern).
Snapshot = Tuple[float, int, np.ndarray]


@dataclass
class SpikeResult:
    pattern: np.ndarray  # (H, W) uint8, interior only
    status: str
    objective: int
    best_bound: int
    wall_time_s: float
    obj_history: List[Tuple[float, int]]
    max_rss_mb: float
    config: SpikeConfig
    snapshots: List[Snapshot] = field(default_factory=list)
    build_time_s: float = 0.0
    # Window geometry and integer targets of the solved model, so downstream
    # consumers (save_run, LNS) never re-derive them from the greyscale.
    windows: Optional[List[Window]] = None
    targets: Optional[np.ndarray] = None
    seed_objective: Optional[int] = None  # objective of the warm-start seed


@dataclass
class ModelBundle:
    model: cp_model.CpModel
    x: list  # (H+2) x (W+2) nested list of BoolVar
    shape: Tuple[int, int]  # interior (H, W)
    windows: List[Window]
    targets: np.ndarray
    build_time_s: float = 0.0


class _ObjectiveLogger(cp_model.CpSolverSolutionCallback):
    """Logs (time, objective) for every incumbent; optionally keeps patterns.

    Keeping every incumbent pattern is wasteful (a long 400^2 run improves
    thousands of times), so snapshots are throttled to one per
    `gap_s` of wall time. If that still overflows `max_snapshots`, every
    second frame is dropped and the gap doubles — the surviving frames stay
    evenly spread over the run.
    """

    def __init__(
        self,
        index_grid: Optional[np.ndarray] = None,
        gap_s: float = 0.0,
        max_snapshots: int = 400,
    ) -> None:
        super().__init__()
        self.history: List[Tuple[float, int]] = []
        self.snapshots: List[Snapshot] = []
        self._index_grid = index_grid
        self._gap_s = gap_s
        self._max_snapshots = max_snapshots
        self._next_at = 0.0

    def _pattern(self) -> np.ndarray:
        solution = np.asarray(self.response_proto.solution, dtype=np.int8)
        return solution[self._index_grid].astype(np.uint8)

    def on_solution_callback(self) -> None:
        t, obj = self.WallTime(), int(self.ObjectiveValue())
        self.history.append((t, obj))
        if self._index_grid is None or self._gap_s <= 0 or t < self._next_at:
            return
        self.snapshots.append((t, obj, self._pattern()))
        self._next_at = t + self._gap_s
        if len(self.snapshots) > self._max_snapshots:
            self.snapshots = self.snapshots[::2]
            self._gap_s *= 2


def build_model(
    cell_t: np.ndarray,
    free_mask: np.ndarray,
    cfg: SpikeConfig,
    relax_top: bool = False,
    relax_bottom: bool = False,
) -> ModelBundle:
    """Encode stability (hard) + window-density deviation (objective).

    `relax_top` / `relax_bottom` drop every stability constraint touching
    the first/last interior row (including the adjacent ring row). That
    is the strip *relaxation* used for decomposition lower bounds: the
    remaining constraints are a subset of the global model's, so the sum
    of strip optima is a valid lower bound on the global optimum.
    """
    t_build = time.perf_counter()
    h, w = cell_t.shape
    model = cp_model.CpModel()
    # Unnamed variables: names would be serialized into the proto, and at
    # 400^2 that is 160k+ strings of pure ballast.
    x = [[model.NewBoolVar("") for _ in range(w + 2)] for _ in range(h + 2)]

    fixed_dead = np.ones((h + 2, w + 2), dtype=bool)
    fixed_dead[1:-1, 1:-1] = ~free_mask if cfg.mask_mode == "force_dead" else False

    # A fixed-dead cell can never satisfy the alive branch, and only needs
    # the no-birth branch when it has a neighbour that could be alive at
    # all. Skipping the rest is exactly what presolve would conclude —
    # minus the cost of building and shipping those constraints.
    open_grid = ~fixed_dead
    padded = np.pad(open_grid, 1)
    open_nbrs = np.zeros((h + 2, w + 2), dtype=np.int8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0):
                open_nbrs += padded[1 + di : h + 3 + di, 1 + dj : w + 3 + dj]

    fixed_rows = fixed_dead.tolist()  # plain bools: no numpy boxing in the loop
    open_rows = open_nbrs.tolist()
    for i in range(h + 2):
        skip_stability = (relax_top and i <= 1) or (relax_bottom and i >= h)
        for j in range(w + 2):
            if fixed_rows[i][j]:
                model.Add(x[i][j] == 0)
                if skip_stability or not open_rows[i][j]:
                    continue
            elif skip_stability:
                continue
            neighbours = [
                x[i + di][j + dj]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)
                and 0 <= i + di < h + 2
                and 0 <= j + dj < w + 2
                and not fixed_rows[i + di][j + dj]
            ]
            s = cp_model.LinearExpr.Sum(neighbours)
            if fixed_rows[i][j]:
                # x is pinned to 0, so the no-birth branch holds outright.
                model.AddLinearExpressionInDomain(s, _DEAD_OK)
                continue
            model.AddLinearConstraint(s, 2, 3).OnlyEnforceIf(x[i][j])
            model.AddLinearExpressionInDomain(s, _DEAD_OK).OnlyEnforceIf(
                x[i][j].Not()
            )

    windows = window_slices((h, w), cfg.k, cfg.stride, edge=cfg.edge_windows)
    targets, kept = window_targets(cell_t, free_mask, windows, dither=cfg.dither)
    free_rows = free_mask.tolist()
    devs = []
    for t, (si, sj) in zip(targets, kept):
        cells = [
            x[i + 1][j + 1]
            for i in range(si.start, si.stop)
            for j in range(sj.start, sj.stop)
            if free_rows[i][j]
        ]
        live = cp_model.LinearExpr.Sum(cells)
        dev = model.NewIntVar(0, len(cells), "")
        # `slack` cells of deviation per window are free; with slack > 0
        # an objective of 0 becomes reachable, which lets CP-SAT prove
        # optimality instead of chasing a vacuous lower bound.
        model.Add(dev >= live - int(t) - cfg.slack)
        model.Add(dev >= int(t) - live - cfg.slack)
        devs.append(dev)
    model.Minimize(cp_model.LinearExpr.Sum(devs))

    return ModelBundle(
        model=model,
        x=x,
        shape=(h, w),
        windows=kept,
        targets=targets,
        build_time_s=time.perf_counter() - t_build,
    )


def _apply_solver_params(solver: cp_model.CpSolver, cfg: SpikeConfig) -> None:
    p = solver.parameters
    p.max_time_in_seconds = cfg.time_limit_s
    p.num_workers = cfg.workers
    p.random_seed = cfg.seed
    if cfg.max_det_time is not None:
        p.max_deterministic_time = cfg.max_det_time
    if cfg.add_lb_subsolvers:
        # At num_workers <= 10 the default portfolio schedules only 7 full
        # subsolvers and cuts exactly the bound-improving ones.
        p.extra_subsolvers.extend(["lb_tree_search", "objective_lb_search"])
    if cfg.num_violation_ls:
        p.num_violation_ls = cfg.num_violation_ls
    if cfg.symmetry_level is not None:
        p.symmetry_level = cfg.symmetry_level


def solve(
    bundle: ModelBundle,
    cfg: SpikeConfig,
    hint: Optional[np.ndarray] = None,
    allow_unknown: bool = False,
) -> SpikeResult:
    h, w = bundle.shape
    if hint is not None:
        for i in range(h):
            for j in range(w):
                bundle.model.AddHint(bundle.x[i + 1][j + 1], int(hint[i, j]))

    solver = cp_model.CpSolver()
    _apply_solver_params(solver, cfg)
    if hint is not None:
        solver.parameters.repair_hint = cfg.repair_hint
        solver.parameters.hint_conflict_limit = cfg.hint_conflict_limit
    log_lines: List[str] = []
    if cfg.log_to:
        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = False
        solver.log_callback = log_lines.append
    # Variable indices of the interior cells, so a callback can slice a whole
    # incumbent out of the response proto in one go.
    index_grid = np.array(
        [[bundle.x[i + 1][j + 1].Index() for j in range(w)] for i in range(h)]
    )
    logger = _ObjectiveLogger(index_grid, cfg.snapshot_gap_s, cfg.max_snapshots)

    t0 = time.perf_counter()
    status = solver.Solve(bundle.model, logger)
    wall = time.perf_counter() - t0
    if cfg.log_to:
        Path(cfg.log_to).write_text("\n".join(log_lines) + "\n")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # All-dead is always feasible, so this is only reachable with an
        # extreme time limit. Callers that just need the proven bound
        # (e.g. the strip relaxation) can opt in to an empty pattern.
        assert allow_unknown, solver.StatusName(status)
        return SpikeResult(
            pattern=np.zeros((h, w), dtype=np.uint8),
            status=solver.StatusName(status),
            objective=int(sum(int(t) for t in bundle.targets)),
            best_bound=int(solver.BestObjectiveBound()),
            wall_time_s=wall,
            obj_history=logger.history,
            max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            config=cfg,
            build_time_s=bundle.build_time_s,
            windows=bundle.windows,
            targets=bundle.targets,
        )
    # Bulk read: slice the whole solution vector once instead of 160k+
    # per-variable Value() calls at 400^2.
    solution = np.asarray(solver.ResponseProto().solution, dtype=np.int8)
    pattern = solution[index_grid].astype(np.uint8)
    snapshots = logger.snapshots
    if snapshots and not np.array_equal(snapshots[-1][2], pattern):
        # The reported solution is always the last frame of the movie.
        snapshots.append(
            (solver.WallTime(), int(solver.ObjectiveValue()), pattern.copy())
        )
    return SpikeResult(
        pattern=pattern,
        status=solver.StatusName(status),
        objective=int(solver.ObjectiveValue()),
        best_bound=int(solver.BestObjectiveBound()),
        wall_time_s=wall,
        obj_history=logger.history,
        max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        config=cfg,
        snapshots=snapshots,
        build_time_s=bundle.build_time_s,
        windows=bundle.windows,
        targets=bundle.targets,
    )


def solve_image(
    grey: np.ndarray,
    free_mask: np.ndarray,
    cfg: SpikeConfig,
    hint: Optional[np.ndarray] = None,
) -> SpikeResult:
    """Greyscale (uint8) + mask -> solved still-life pattern."""
    cell_t = cell_targets(grey, cfg.d_max)
    if cfg.mask_mode == "none":
        free_mask = np.ones_like(free_mask, dtype=bool)
    elif cfg.mask_mode == "soft_zero":
        cell_t = np.where(free_mask, cell_t, 0.0)
        free_mask = np.ones_like(free_mask, dtype=bool)
    bundle = build_model(cell_t, free_mask, cfg)
    seed_obj: Optional[int] = None
    if hint is None and cfg.hint_mode == "agar":
        from beyond_tiles.seeds import best_seed

        hint, seed_obj = best_seed(
            free_mask, bundle.windows, bundle.targets, slack=cfg.slack
        )
    result = solve(bundle, cfg, hint)
    result.seed_objective = seed_obj
    return result


def verify_still_life(pattern: np.ndarray) -> Dict[str, bool]:
    """Two independent stability checks; both must hold."""
    from gol_mosaics.life import is_still_life
    from gol_mosaics.sat_search import rule_violations

    padded = np.pad(np.asarray(pattern, dtype=np.uint8), 1)
    return {
        "bounded": bool(is_still_life(padded)),
        "toroidal": bool(not rule_violations(padded[None])[0]),
    }

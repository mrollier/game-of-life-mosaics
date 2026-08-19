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


@dataclass
class ModelBundle:
    model: cp_model.CpModel
    x: list  # (H+2) x (W+2) nested list of BoolVar
    shape: Tuple[int, int]  # interior (H, W)
    windows: List[Window]
    targets: np.ndarray


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
    cell_t: np.ndarray, free_mask: np.ndarray, cfg: SpikeConfig
) -> ModelBundle:
    """Encode stability (hard) + window-density deviation (objective)."""
    h, w = cell_t.shape
    model = cp_model.CpModel()
    x = [
        [model.NewBoolVar(f"x_{i}_{j}") for j in range(w + 2)]
        for i in range(h + 2)
    ]

    fixed_dead = np.ones((h + 2, w + 2), dtype=bool)
    fixed_dead[1:-1, 1:-1] = ~free_mask if cfg.mask_mode == "force_dead" else False

    for i in range(h + 2):
        for j in range(w + 2):
            if fixed_dead[i, j]:
                model.Add(x[i][j] == 0)
            neighbours = [
                x[i + di][j + dj]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)
                and 0 <= i + di < h + 2
                and 0 <= j + dj < w + 2
            ]
            s = cp_model.LinearExpr.Sum(neighbours)
            model.AddLinearConstraint(s, 2, 3).OnlyEnforceIf(x[i][j])
            model.AddLinearExpressionInDomain(s, _DEAD_OK).OnlyEnforceIf(
                x[i][j].Not()
            )

    windows = window_slices((h, w), cfg.k, cfg.stride)
    targets, kept = window_targets(cell_t, free_mask, windows)
    devs = []
    for t, (si, sj) in zip(targets, kept):
        cells = [
            x[i + 1][j + 1]
            for i in range(si.start, si.stop)
            for j in range(sj.start, sj.stop)
            if free_mask[i, j]
        ]
        live = cp_model.LinearExpr.Sum(cells)
        dev = model.NewIntVar(0, len(cells), f"dev_{si.start}_{sj.start}")
        model.Add(dev >= live - int(t))
        model.Add(dev >= int(t) - live)
        devs.append(dev)
    model.Minimize(cp_model.LinearExpr.Sum(devs))

    return ModelBundle(model=model, x=x, shape=(h, w), windows=kept, targets=targets)


def solve(
    bundle: ModelBundle, cfg: SpikeConfig, hint: Optional[np.ndarray] = None
) -> SpikeResult:
    h, w = bundle.shape
    if hint is not None:
        for i in range(h):
            for j in range(w):
                bundle.model.AddHint(bundle.x[i + 1][j + 1], int(hint[i, j]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.time_limit_s
    solver.parameters.num_workers = cfg.workers
    solver.parameters.random_seed = cfg.seed
    # Variable indices of the interior cells, so a callback can slice a whole
    # incumbent out of the response proto in one go.
    index_grid = np.array(
        [[bundle.x[i + 1][j + 1].Index() for j in range(w)] for i in range(h)]
    )
    logger = _ObjectiveLogger(index_grid, cfg.snapshot_gap_s, cfg.max_snapshots)

    t0 = time.perf_counter()
    status = solver.Solve(bundle.model, logger)
    wall = time.perf_counter() - t0

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), solver.StatusName(status)
    pattern = np.array(
        [
            [solver.Value(bundle.x[i + 1][j + 1]) for j in range(w)]
            for i in range(h)
        ],
        dtype=np.uint8,
    )
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
    return solve(bundle, cfg, hint)


def verify_still_life(pattern: np.ndarray) -> Dict[str, bool]:
    """Two independent stability checks; both must hold."""
    from gol_mosaics.life import is_still_life
    from gol_mosaics.sat_search import rule_violations

    padded = np.pad(np.asarray(pattern, dtype=np.uint8), 1)
    return {
        "bounded": bool(is_still_life(padded)),
        "toroidal": bool(not rule_violations(padded[None])[0]),
    }

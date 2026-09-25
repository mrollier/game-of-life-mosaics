"""
The production pipeline: a grey image to one verified still life, any size.

    strips or blocks  ->  polish rounds  ->  seam rounds  ->  diagonal repair

Each stage is its own module; this one wires them in the order and with the
settings that the large solves settled on (the John Conway poster, the Lam
Gods, the defence flyer; experiments/beyond_tiles/REPORT.md sections 5-9):

* The canvas is cut into strips (or, when wide, blocks) separated by two
  dead lines, which solve independently and stitch into a still life.
* LNS rounds re-solve the worst patches with 10 s budgets while the
  objective keeps falling by more than 2 % a round.
* A polished decomposition still shows every separator (window totals
  cannot see where inside a window the cells sit), so seam rounds give each
  separator box a proportional sub-target.
* Patching can lengthen diagonal chains, so a last pass repairs any chain
  longer than the cap.

Example:
    >>> from gol_mosaics.freeform.targets import load_rect_target
    >>> grey, free = load_rect_target('input/images/john.png', 400, 560)
    >>> result = solve_poster(grey, free, PosterConfig(), out='runs/john')
    >>> result.pattern.shape
    (560, 400)
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

K = 8  # window side; strips, blocks and canvas sides are multiples of it


@dataclass
class PosterConfig:
    """Settings of :func:`solve_poster`; the defaults are the poster's."""
    d_max: float = 0.40          # target density of pure black
    seed: int = 0
    dither: str = "round"        # "fs" for slow fades (no dotted lines)
    edge_windows: str = "clamp"  # "partial" when a side is not a multiple of 8
    max_diag_run: Optional[int] = 5  # None lifts the cap
    # strips (block_cols == 0) or blocks
    strip_rows: int = 48
    block_cols: int = 0
    strip_time: float = 300.0    # CP-SAT seconds per strip or block
    strip_procs: int = 5         # strips solved at once
    strip_workers: int = 2       # CP-SAT workers per strip
    isolate: bool = False        # one process per strip, retried on abort
    # polish
    polish_rounds: int = 3
    polish_budget: float = 1800.0
    polish_stop: float = 0.98    # stop once a round keeps more than this
    patch_time: float = 10.0
    patch_windows: int = 5
    polish_procs: int = 4
    # seams
    seam_rounds: int = 2
    seam_budget: float = 900.0
    # final diagonal repair
    repair_diagonals: bool = True


@dataclass
class PosterResult:
    pattern: np.ndarray          # the verified still life, uint8
    objective: int               # total window deviation, plain objective
    cell_targets: np.ndarray
    targets: np.ndarray          # per-window live-cell targets
    windows: list                # the windows those targets belong to
    report: Dict = field(default_factory=dict)


def _quiet(*args, **kwargs) -> None:
    pass


def solve_poster(grey: np.ndarray,
                 free: np.ndarray,
                 cfg: Optional[PosterConfig] = None,
                 out=None,
                 resume: bool = False,
                 log: Optional[Callable[[str], None]] = None) -> PosterResult:
    """
    Solve a grey image into one still life with strips, LNS and seam rounds.

    Args:
        grey: (H, W) uint8 image, 0 black; H and W multiples of 8 (or use
            ``edge_windows="partial"`` and strips only)
        free: (H, W) bool, True where cells may live (the subject)
        cfg: Pipeline settings (default :class:`PosterConfig`)
        out: Directory for checkpoints (strips_pattern.npy, pattern.npy,
            polish_rounds.json, blocks/); None keeps everything in memory
        resume: Continue from what `out` already holds: the polished
            pattern, else the stitched strips, else the block checkpoints
        log: Progress callback taking one string (default: silent)

    Returns:
        A :class:`PosterResult`; its report dict holds the plan, per-stage
        statistics, the deviation, the verification and the diagonal run

    Raises:
        AssertionError: If the final pattern fails still-life verification
    """
    from .decompose import plan_blocks, plan_strips, solve_blocks, solve_strips
    from .lns import (LnsConfig, improve, repair_diagonal_runs, seam_occupancy,
                      window_devs)
    from .metrics import deviation_stats, max_diagonal_run
    from .solver import SolveConfig, verify_still_life
    from .targets import cell_targets, window_slices, window_targets

    cfg = cfg or PosterConfig()
    log = log or _quiet
    out = Path(out) if out is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    def save(name, array):
        if out is not None:
            np.save(out / name, array)

    height, width = grey.shape
    cell_t = cell_targets(grey, cfg.d_max)
    windows = window_slices(cell_t.shape, k=K, stride=K, edge=cfg.edge_windows)
    targets, kept = window_targets(cell_t, free, windows, dither=cfg.dither)
    log(f"{len(kept):,} windows, total live-cell target {int(targets.sum()):,} "
        f"({cfg.dither} dither)")

    max_diag_run = cfg.max_diag_run or None
    scfg = SolveConfig(k=K, stride=K, d_max=cfg.d_max, seed=cfg.seed,
                       time_limit_s=cfg.strip_time, workers=cfg.strip_workers,
                       max_diag_run=max_diag_run, dither=cfg.dither,
                       edge_windows=cfg.edge_windows)
    if cfg.block_cols:
        plan = plan_blocks(height, width, K, block_rows=cfg.strip_rows,
                           block_cols=cfg.block_cols, gap=2)
        row_spans, col_spans = plan.row_spans, plan.col_spans
        plan_json = {"row_spans": [list(s) for s in row_spans],
                     "col_spans": [list(s) for s in col_spans],
                     "gap": plan.gap}
    else:
        plan = plan_strips(height, K, strip_rows=cfg.strip_rows, gap=2)
        row_spans, col_spans = plan.spans, [(0, width)]
        plan_json = {"spans": [list(s) for s in plan.spans], "gap": plan.gap}
    n_pieces = len(row_spans) * len(col_spans)
    kind = "blocks" if cfg.block_cols else "strips"

    rounds: List[dict] = []
    pattern = None
    if resume and out is not None:
        if (out / "pattern.npy").exists():  # mid-polish
            pattern = np.load(out / "pattern.npy")
            if (out / "polish_rounds.json").exists():
                rounds = json.loads((out / "polish_rounds.json").read_text())
            log(f"resumed from pattern.npy after {len(rounds)} polish round(s)")
        elif (out / "strips_pattern.npy").exists():  # stitched, unpolished
            pattern = np.load(out / "strips_pattern.npy")
            log("resumed from strips_pattern.npy")
        # otherwise solve_blocks restores whatever block checkpoints exist

    if pattern is not None:
        pieces = dict(statuses=["resumed"] * n_pieces, total_cpu_s=0.0)
        pieces_wall, n_opt = 0.0, 0
    else:
        log(f"solving {n_pieces} {kind} ({cfg.strip_procs} procs x "
            f"{cfg.strip_workers} workers)...")
        t0 = time.perf_counter()
        if cfg.block_cols:
            pieces = solve_blocks(
                grey, free, scfg, plan, n_procs=cfg.strip_procs,
                checkpoint_dir=out / "blocks" if out is not None else None,
                log=log)
        else:
            pieces = solve_strips(grey, free, scfg, plan,
                                  n_procs=cfg.strip_procs,
                                  isolate=cfg.isolate, log=log)
        pieces_wall = time.perf_counter() - t0
        n_opt = sum(s == "OPTIMAL" for s in pieces["statuses"])
        pattern = pieces["pattern"]
        save("strips_pattern.npy", pattern)
    obj = int(window_devs(pattern, free, kept, targets).sum())
    log(f"{kind.upper()} DONE in {pieces_wall:.0f}s: {n_opt}/{n_pieces} "
        f"OPTIMAL, full-mask objective {obj:,}")

    for rnd in range(len(rounds) + 1, cfg.polish_rounds + 1):
        if obj == 0:
            break
        lcfg = LnsConfig(patch_windows=cfg.patch_windows,
                         patch_time_s=cfg.patch_time,
                         budget_s=cfg.polish_budget,
                         n_procs=cfg.polish_procs, seed=cfg.seed + rnd,
                         max_diag_run=max_diag_run)
        t0 = time.perf_counter()
        res = improve(pattern, free, kept, targets, lcfg, log=_quiet)
        wall = time.perf_counter() - t0
        log(f"POLISH {rnd} DONE in {wall:.0f}s: objective {obj:,} -> "
            f"{res.objective:,} ({res.patches_improved}/{res.patches_solved} "
            f"patches improved)")
        pattern, prev, obj = res.pattern, obj, int(res.objective)
        rounds.append({
            "objective": obj, "rounds": res.rounds, "wall_time_s": wall,
            "patches_improved": res.patches_improved,
            "patches_solved": res.patches_solved,
        })
        save("pattern.npy", pattern)
        if out is not None:
            (out / "polish_rounds.json").write_text(json.dumps(rounds, indent=2))
        if obj == 0 or obj > cfg.polish_stop * prev:
            log("converged (or stalled) - stopping polish early")
            break

    seams: Dict = {}
    if cfg.seam_rounds and n_pieces > 1:
        seam_cfg = LnsConfig(
            patch_windows=cfg.patch_windows, patch_time_s=cfg.patch_time,
            budget_s=cfg.seam_budget, n_procs=cfg.polish_procs,
            seed=cfg.seed + 50, max_diag_run=max_diag_run,
            seam_rows=tuple(r1 - plan.gap for _, r1 in row_spans[:-1]),
            seam_cols=tuple(c1 - plan.gap for _, c1 in col_spans[:-1]))
        seams["before"] = seam_occupancy(pattern, seam_cfg, free)
        seam_obj = int(window_devs(pattern, free, kept, targets, 0,
                                   seam_cfg).sum())
        log(f"seams before: worst separator at {seams['before']['worst']:.2f} "
            f"of its neighbours, seam objective {seam_obj:,}")
        for rnd in range(1, cfg.seam_rounds + 1):
            if seam_obj == 0:
                break
            t0 = time.perf_counter()
            res = improve(pattern, free, kept, targets,
                          dataclasses.replace(seam_cfg, seed=seam_cfg.seed + rnd),
                          log=_quiet)
            wall = time.perf_counter() - t0
            pattern, prev, seam_obj = res.pattern, seam_obj, int(res.objective)
            obj = int(window_devs(pattern, free, kept, targets).sum())
            log(f"SEAMS {rnd} DONE in {wall:.0f}s: seam objective {prev:,} -> "
                f"{seam_obj:,}, plain objective {obj:,} "
                f"({res.patches_improved}/{res.patches_solved} patches improved)")
            seams.setdefault("rounds", []).append(dict(
                seam_objective=seam_obj, objective=obj, wall_time_s=wall,
                patches_improved=res.patches_improved,
                patches_solved=res.patches_solved))
            save("pattern.npy", pattern)
            if seam_obj > cfg.polish_stop * prev:
                break
        seams["after"] = seam_occupancy(pattern, seam_cfg, free)
        log(f"seams after: worst separator at {seams['after']['worst']:.2f}")

    if max_diag_run and cfg.repair_diagonals:
        rcfg = LnsConfig(patch_windows=3, patch_time_s=30.0,
                         n_procs=cfg.polish_procs, seed=cfg.seed + 99,
                         max_diag_run=max_diag_run)
        res = repair_diagonal_runs(pattern, free, kept, targets, rcfg, log=log)
        pattern, obj = res.pattern, res.objective
        save("pattern.npy", pattern)

    ver = verify_still_life(pattern)
    assert ver["bounded"] and ver["toroidal"], f"verification failed: {ver}"
    report = {
        "grid": [height, width],
        "plan": plan_json,
        "solve": {"n_optimal": n_opt, "wall_time_s": pieces_wall,
                  "total_cpu_s": pieces["total_cpu_s"],
                  "statuses": pieces["statuses"]},
        "polish_rounds": rounds,
        "seams": seams,
        "objective": int(obj),
        "deviation": deviation_stats(pattern, cell_t, free, kept),
        "verify": ver,
        "max_diagonal_run": max_diagonal_run(pattern),
        "live_cells": int(pattern.sum()),
    }
    return PosterResult(pattern=pattern, objective=int(obj),
                        cell_targets=cell_t, targets=targets, windows=kept,
                        report=report)

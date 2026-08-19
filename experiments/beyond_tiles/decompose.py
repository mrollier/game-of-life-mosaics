"""Strip decomposition: scale past 400² and the first real lower bound.

Two distinct uses of horizontal strips, easy to conflate:

* **Restriction** (`solve_strips`): force two dead rows between strips.
  Two is exactly enough — a stability constraint has radius 1, so with
  rows r and r+1 dead, constraints centred at r depend only on the strip
  above and those at r+1 only on the strip below (each strip's model
  already enforces the no-birth constraint of its own dead ring). Every
  strip solves to proven optimality in seconds, in parallel, and the
  stitched pattern is a genuine still life. The price is the forced-dead
  gap rows: their windows lose 2 of 8 rows of achievable density.

* **Relaxation** (`lower_bound_strips`): cut WITHOUT gaps, dropping the
  stability constraints that straddle each cut (a pure constraint
  deletion, with cuts aligned to window boundaries so the objective
  partitions exactly). The sum of strip optima is then a valid lower
  bound on the global optimum — where CP-SAT's own bound stays stuck
  near zero at 400².
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from beyond_tiles.still_image import SpikeConfig, build_model, solve


@dataclass
class StripPlan:
    spans: List[Tuple[int, int]]  # interior row spans, k-aligned
    gap: int  # dead rows folded into the bottom of each non-final strip


def plan_strips(h: int, k: int, strip_rows: int = 48, gap: int = 2) -> StripPlan:
    """Row spans covering h, each a multiple of k (the last takes the rest)."""
    assert strip_rows % k == 0 and h % k == 0 and gap < k
    spans = []
    row = 0
    while row < h:
        stop = min(row + strip_rows, h)
        if h - stop < k:  # avoid a runt strip shorter than one window
            stop = h
        spans.append((row, stop))
        row = stop
    return StripPlan(spans=spans, gap=gap)


def _solve_strip_task(payload: dict):
    from beyond_tiles.still_image import build_model as bm
    from beyond_tiles.still_image import solve as sv
    from beyond_tiles.targets import cell_targets

    cfg: SpikeConfig = payload["cfg"]
    cell_t = cell_targets(payload["grey"], cfg.d_max)
    bundle = bm(
        cell_t,
        payload["free"],
        cfg,
        relax_top=payload.get("relax_top", False),
        relax_bottom=payload.get("relax_bottom", False),
    )
    result = sv(bundle, cfg)
    return {
        "pattern": result.pattern,
        "status": result.status,
        "objective": result.objective,
        "best_bound": result.best_bound,
        "wall_time_s": result.wall_time_s,
    }


def _run_strips(payloads: List[dict], n_procs: int) -> List[dict]:
    if n_procs > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=n_procs) as pool:
            return list(pool.map(_solve_strip_task, payloads))
    return [_solve_strip_task(p) for p in payloads]


def solve_strips(
    grey: np.ndarray,
    free_mask: np.ndarray,
    cfg: SpikeConfig,
    plan: Optional[StripPlan] = None,
    n_procs: int = 4,
) -> dict:
    """Restriction form: independent strips separated by dead gap rows.

    Returns the stitched pattern plus per-strip stats. The stitched
    pattern is a valid still life by construction (each strip is one,
    embedded in a dead plane, and gaps keep them out of reach of each
    other); callers should still run verify_still_life on it.
    """
    h, w = grey.shape
    plan = plan or plan_strips(h, cfg.k)
    payloads = []
    for idx, (r0, r1) in enumerate(plan.spans):
        free = free_mask[r0:r1].copy()
        if idx < len(plan.spans) - 1 and plan.gap:
            free[-plan.gap :] = False  # the dead separator, inside this strip
        payloads.append(dict(grey=grey[r0:r1], free=free, cfg=cfg))

    outs = _run_strips(payloads, n_procs)
    pattern = np.vstack([o["pattern"] for o in outs]).astype(np.uint8)
    return {
        "pattern": pattern,
        "objective": int(sum(o["objective"] for o in outs)),
        "statuses": [o["status"] for o in outs],
        "wall_time_s": max(o["wall_time_s"] for o in outs),
        "total_cpu_s": sum(o["wall_time_s"] for o in outs),
        "per_strip": [
            {k: v for k, v in o.items() if k != "pattern"} for o in outs
        ],
        "spans": plan.spans,
    }


def lower_bound_strips(
    grey: np.ndarray,
    free_mask: np.ndarray,
    cfg: SpikeConfig,
    plan: Optional[StripPlan] = None,
    n_procs: int = 4,
) -> dict:
    """Relaxation form: a valid global lower bound from strip optima.

    Every strip must reach OPTIMAL for the bound to be valid; strips that
    time out contribute their proven best_bound instead (still valid).
    """
    h, w = grey.shape
    plan = plan or plan_strips(h, cfg.k)
    payloads = []
    for idx, (r0, r1) in enumerate(plan.spans):
        payloads.append(
            dict(
                grey=grey[r0:r1],
                free=free_mask[r0:r1].copy(),
                cfg=cfg,
                relax_top=idx > 0,
                relax_bottom=idx < len(plan.spans) - 1,
            )
        )
    outs = _run_strips(payloads, n_procs)
    bound = sum(
        o["objective"] if o["status"] == "OPTIMAL" else o["best_bound"]
        for o in outs
    )
    return {
        "lower_bound": int(bound),
        "all_optimal": all(o["status"] == "OPTIMAL" for o in outs),
        "wall_time_s": max(o["wall_time_s"] for o in outs),
        "total_cpu_s": sum(o["wall_time_s"] for o in outs),
        "per_strip": [
            {k: v for k, v in o.items() if k != "pattern"} for o in outs
        ],
        "spans": plan.spans,
    }

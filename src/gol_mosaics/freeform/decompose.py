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

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .solver import SpikeConfig, build_model, solve


def _check_cfg(cfg: SpikeConfig) -> SpikeConfig:
    """Strip solving supports exactly the geometry its proofs assume.

    - force_dead only: the dead separator is implemented through the free
      mask, which pins cells only in that mode.
    - stride == k: cuts must align with every window boundary, or windows
      straddling a cut silently vanish from both strips' objectives.
    - dither forced to "round": per-window rounding is local, so strip
      targets equal global targets; Floyd-Steinberg error diffusion is
      not local and would make the strip objective (and therefore the
      lower bound) refer to different targets than the global model's.
    """
    if cfg.mask_mode != "force_dead":
        raise ValueError("strip decomposition requires mask_mode='force_dead'")
    if cfg.stride != cfg.k:
        raise ValueError("strip decomposition requires stride == k")
    if cfg.dither != "round":
        cfg = dataclasses.replace(cfg, dither="round")
    return cfg


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
    from .solver import build_model as bm
    from .solver import solve as sv
    from .targets import cell_targets

    cfg: SpikeConfig = payload["cfg"]
    cell_t = cell_targets(payload["grey"], cfg.d_max)
    bundle = bm(
        cell_t,
        payload["free"],
        cfg,
        relax_top=payload.get("relax_top", False),
        relax_bottom=payload.get("relax_bottom", False),
    )
    result = sv(bundle, cfg, allow_unknown=True)
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
    cfg = _check_cfg(cfg)
    h, w = grey.shape
    plan = plan or plan_strips(h, cfg.k)
    if plan.gap < 2 and len(plan.spans) > 1:
        raise ValueError(
            "gap must be >= 2 dead rows: one row is checked against only "
            "one side by either strip's model, so 2+1 live neighbours "
            "across the cut could still give birth"
        )
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
    cfg = _check_cfg(cfg)
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


# --- 2-D blocks ---------------------------------------------------------------
#
# Strips stop paying once the canvas is wide: a 48x2480 strip (119k cells) is
# six times the largest model that closes in minutes, and at 5 GB apiece a
# handful of them fill the machine (frietjes, 2026-09-15: FEASIBLE at
# objective 18,627 after 300 s, versus OPTIMAL 0 in 140 s for a 48x416 block).
# Cutting along both axes keeps every block a 200²-class instance. The
# separator argument is the same per axis: two dead columns decouple the
# blocks left and right exactly as two dead rows do above and below, and the
# 2x2 dead corner where four blocks meet has at most one live neighbour (the
# corner cell of the diagonal block), so it can never be born.


@dataclass
class BlockPlan:
    row_spans: List[Tuple[int, int]]
    col_spans: List[Tuple[int, int]]
    gap: int  # dead rows/cols folded into the bottom/right of non-final blocks


def plan_blocks(
    h: int, w: int, k: int, block_rows: int = 64, block_cols: int = 416, gap: int = 2
) -> BlockPlan:
    """k-aligned row and column spans; the last of each takes the remainder."""
    return BlockPlan(
        row_spans=plan_strips(h, k, block_rows, gap).spans,
        col_spans=plan_strips(w, k, block_cols, gap).spans,
        gap=gap,
    )


def solve_blocks(
    grey: np.ndarray,
    free_mask: np.ndarray,
    cfg: SpikeConfig,
    plan: BlockPlan,
    n_procs: int = 4,
    checkpoint_dir=None,
    log=None,
    retries: int = 2,
) -> dict:
    """Restriction form on a 2-D grid of blocks separated by dead gap lines.

    Each block runs in a process pool of its own (an aborting CP-SAT worker
    then costs one retry with another seed instead of the whole batch, the
    lesson of the first flyer solve), and with `checkpoint_dir` every solved
    block is saved as it lands, so a killed run resumes where it stopped.
    The stitched pattern is a still life by construction; callers should
    still run verify_still_life on it.
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from pathlib import Path

    cfg = _check_cfg(cfg)
    if plan.gap < 2:
        raise ValueError("gap must be >= 2 dead rows/cols (see solve_strips)")
    n_rows, n_cols = len(plan.row_spans), len(plan.col_spans)
    tasks = [(ri, ci) for ri in range(n_rows) for ci in range(n_cols)]
    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)
    say = log or (lambda msg: None)

    def payload_for(ri, ci):
        r0, r1 = plan.row_spans[ri]
        c0, c1 = plan.col_spans[ci]
        free = free_mask[r0:r1, c0:c1].copy()
        if ri < n_rows - 1:
            free[-plan.gap :, :] = False
        if ci < n_cols - 1:
            free[:, -plan.gap :] = False
        return dict(grey=grey[r0:r1, c0:c1], free=free, cfg=cfg)

    done = [0]

    def run(task):
        ri, ci = task
        path = ckpt / f"block_{ri:02d}_{ci:02d}.npz" if ckpt else None
        if path and path.exists():
            with np.load(path) as z:
                out = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
            done[0] += 1
            say(f"block ({ri},{ci}) restored [{done[0]}/{len(tasks)}]")
            return task, out
        payload = payload_for(ri, ci)
        for attempt in range(retries + 1):
            try:
                with ProcessPoolExecutor(max_workers=1) as pool:
                    out = pool.submit(_solve_strip_task, payload).result()
                break
            except BrokenProcessPool:
                say(f"block ({ri},{ci}) aborted on attempt {attempt + 1}; "
                    f"retrying with another seed")
                payload = dict(
                    payload, cfg=dataclasses.replace(cfg, seed=cfg.seed + 100 * (attempt + 1))
                )
        else:
            raise RuntimeError(f"block ({ri},{ci}) aborted {retries + 1} times")
        if path:
            np.savez_compressed(path, **out)
        done[0] += 1
        say(f"block ({ri},{ci}) {out['status']} obj {out['objective']} "
            f"in {out['wall_time_s']:.0f}s [{done[0]}/{len(tasks)}]")
        return task, out

    with ThreadPoolExecutor(max_workers=n_procs) as tp:
        outs = dict(tp.map(run, tasks))

    pattern = np.block(
        [[outs[(ri, ci)]["pattern"] for ci in range(n_cols)] for ri in range(n_rows)]
    ).astype(np.uint8)
    per_block = {
        f"{ri},{ci}": {k: v for k, v in outs[(ri, ci)].items() if k != "pattern"}
        for ri, ci in tasks
    }
    return {
        "pattern": pattern,
        "objective": int(sum(o["objective"] for o in outs.values())),
        "statuses": [outs[t]["status"] for t in tasks],
        "wall_time_s": max(o["wall_time_s"] for o in outs.values()),
        "total_cpu_s": sum(o["wall_time_s"] for o in outs.values()),
        "per_block": per_block,
        "row_spans": plan.row_spans,
        "col_spans": plan.col_spans,
    }

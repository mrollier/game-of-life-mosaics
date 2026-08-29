"""Poster pipeline: any image, any rectangular size, strips + LNS rounds.

The `e9` subcommand is wired to the square Marilyn demo; this script is the
general form, distilled from the John Conway 1416x2000 run (2026-08-20,
REPORT.md section 5, C8):

    python experiments/beyond_tiles/poster.py input/images/john.png \
        --width 1416 --height 2000 --out results/john_d40

Differences from the square demo pipeline, all learned on that run:

* **Rectangular canvas.** The image is resized straight to (width, height)
  instead of being padded square first, so an A-format poster stays an
  A-format poster. Both axes must be multiples of the window size 8 (strip
  cuts and disjoint windows both need it).
* **d_max defaults to 0.40, not 0.45.** On dark, low-key photos histogram
  equalization sends a quarter of the windows to the very top of the tone
  range; at 0.45 those targets sit on the still-life packing ceiling where
  strip solves cap out and the polisher cannot help. At 0.40 the dense
  targets are achievable and the whole pipeline converges.
* **Several polish rounds with 10 s patch solves.** Seam windows need a
  handful of cells inserted between two dense slabs — a local rebuild that
  a 2 s patch budget cannot do at density 0.40 (2 s rounds stalled at
  objective 44,793; three 10 s rounds went 44,793 -> 26,875 -> 4,511 with
  every patch improving). One round is far from converged on a large
  canvas: keep going while the improvement rate is high.

Each stage's pattern is saved, the final pattern is verified (bounded and
toroidal), and a UGent-colored render lands next to the report JSON.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rect(src: str, width: int, height: int, tone: str, contrast: float):
    """Image -> (grey, free_mask) at (height, width), no square padding."""
    from PIL import Image

    from beyond_tiles.targets import equalize_grey, normalize_grey
    from gol_mosaics.image_processing import ImageProcessor

    img, mask = ImageProcessor.load_image(
        src, return_alpha=True, remove_background="auto", contrast=contrast
    )
    grey = np.asarray(
        img.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    free = np.asarray(mask.resize((width, height), Image.Resampling.LANCZOS)) >= 128
    if tone == "eq":
        grey = equalize_grey(grey, free)
    elif tone == "norm":
        grey = normalize_grey(grey, free)
    return grey, free


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image", help="path to the source image (alpha-cut subject)")
    ap.add_argument("--width", type=int, required=True, help="cells, multiple of 8")
    ap.add_argument("--height", type=int, required=True, help="cells, multiple of 8")
    ap.add_argument("--out", required=True,
                    help="output directory (relative paths land under "
                         "experiments/beyond_tiles/)")
    ap.add_argument("--dmax", type=float, default=0.40)
    ap.add_argument("--tone", default="eq", choices=["eq", "norm", "raw"])
    ap.add_argument("--contrast", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-rows", type=int, default=48)
    ap.add_argument("--strip-time", type=float, default=300.0,
                    help="per-strip CP-SAT cap in seconds")
    ap.add_argument("--strip-procs", type=int, default=5)
    ap.add_argument("--strip-workers", type=int, default=2)
    ap.add_argument("--polish-rounds", type=int, default=3)
    ap.add_argument("--polish-budget", type=float, default=1800.0,
                    help="seconds of LNS per round")
    ap.add_argument("--patch-time", type=float, default=10.0,
                    help="seconds per LNS patch solve (10 s for dense images; "
                         "2 s is enough for light ones)")
    ap.add_argument("--patch-windows", type=int, default=5)
    ap.add_argument("--polish-procs", type=int, default=4)
    ap.add_argument("--max-diag-run", type=int, default=5,
                    help="longest solid diagonal chain of live cells; 0 lifts "
                         "the cap and lets the solver draw pencil lines again")
    args = ap.parse_args()

    if args.width % 8 or args.height % 8:
        ap.error("--width and --height must be multiples of 8")

    from beyond_tiles.decompose import plan_strips, solve_strips
    from beyond_tiles.lns import LnsConfig, improve, window_devs
    from beyond_tiles.metrics import deviation_stats, max_diagonal_run
    from beyond_tiles.still_image import SpikeConfig, verify_still_life
    from beyond_tiles.targets import cell_targets, window_slices, window_targets

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.mkdir(parents=True, exist_ok=True)

    grey, free = load_rect(args.image, args.width, args.height,
                           args.tone, args.contrast)
    log(f"loaded: {grey.shape} grid, {int(free.sum()):,} free cells")

    cell_t = cell_targets(grey, args.dmax)
    windows = window_slices(cell_t.shape, k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows)
    log(f"{len(kept):,} windows, total live-cell target {int(targets.sum()):,}")

    plan = plan_strips(args.height, 8, strip_rows=args.strip_rows, gap=2)
    max_diag_run = args.max_diag_run or None
    cfg = SpikeConfig(k=8, stride=8, d_max=args.dmax, seed=args.seed,
                      time_limit_s=args.strip_time, workers=args.strip_workers,
                      max_diag_run=max_diag_run)
    log(f"solving {len(plan.spans)} strips "
        f"({args.strip_procs} procs x {args.strip_workers} workers)...")
    t0 = time.perf_counter()
    strips = solve_strips(grey, free, cfg, plan, n_procs=args.strip_procs)
    strips_wall = time.perf_counter() - t0
    n_opt = sum(s == "OPTIMAL" for s in strips["statuses"])
    np.save(out / "strips_pattern.npy", strips["pattern"])
    pattern = strips["pattern"]
    obj = int(window_devs(pattern, free, kept, targets).sum())
    log(f"STRIPS DONE in {strips_wall:.0f}s: {n_opt}/{len(plan.spans)} OPTIMAL, "
        f"full-mask objective {obj:,}")

    rounds = []
    for rnd in range(1, args.polish_rounds + 1):
        lcfg = LnsConfig(patch_windows=args.patch_windows,
                         patch_time_s=args.patch_time,
                         budget_s=args.polish_budget,
                         n_procs=args.polish_procs, seed=args.seed + rnd,
                         max_diag_run=max_diag_run)
        t0 = time.perf_counter()
        res = improve(pattern, free, kept, targets, lcfg,
                      log=lambda *a, **k: None)
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
        np.save(out / "pattern.npy", pattern)
        if obj == 0 or obj > 0.98 * prev:
            log("converged (or stalled) - stopping polish early")
            break

    ver = verify_still_life(pattern)
    assert ver["bounded"] and ver["toroidal"], f"verification failed: {ver}"
    stats = deviation_stats(pattern, cell_t, free, kept)
    longest_diag = max_diagonal_run(pattern)
    report = {
        "image": args.image,
        "grid": [args.height, args.width],
        "config": vars(args),
        "plan": {"spans": [list(s) for s in plan.spans], "gap": plan.gap},
        "solve": {"n_optimal": n_opt, "wall_time_s": strips_wall,
                  "total_cpu_s": strips["total_cpu_s"],
                  "statuses": strips["statuses"]},
        "polish_rounds": rounds,
        "objective": obj,
        "deviation": stats,
        "verify": ver,
        "max_diagonal_run": longest_diag,
        "live_cells": int(pattern.sum()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    from gol_mosaics import ColorScheme, MosaicRenderer

    MosaicRenderer(ColorScheme.ugent()).render_gol_mosaic(pattern).save(
        out / "render.png")
    log(f"ALL DONE: verify {ver}, {int(pattern.sum()):,} live cells, "
        f"longest diagonal chain {longest_diag}, "
        f"MAD {stats['mad']:.4f} (darkest quartile "
        f"{stats['mad_darkest_quartile']:.4f}, Pearson {stats['pearson']:.4f}); "
        f"saved to {out}")


if __name__ == "__main__":
    main()

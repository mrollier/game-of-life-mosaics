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

Lessons from the Lam Gods run (2480x1656, the whole painting as one still
life, frietjes 2026-09-15), all switches below:

* **Wide canvases need blocks, not strips.** A 48x2480 strip (119k cells) is
  FEASIBLE at objective 18,627 after 300 s and 5.4 GB; a 48x416 block closes
  to OPTIMAL 0 in 140 s at 1 GB. `--block-cols` cuts the columns too
  (`decompose.solve_blocks`), so every block stays a 200²-class instance and
  the process count is bounded by memory (~1.3 GB per 64x416 block) instead
  of by one strip's appetite. 0 keeps the full-width strips.
* **A painting has no background.** `--keep-background` skips rembg and
  makes the whole frame the subject; `auto` would carve a "subject" out of
  the altarpiece.
* **`--dither fs`** for slow fades (the sky), the flyer's fix for dotted
  lines; the block solves use rounded targets regardless (see
  `decompose._check_cfg`), the polish targets carry the dither.
* **`--resume`** picks up from the per-block checkpoints, the stitched
  pattern, or the last polished pattern, whichever is furthest along — a
  two-hour job on a shared box should survive being killed.
* **Seam rounds after the polish.** A polished pattern still showed every
  separator: the two dead rows held 53-75 % of their neighbours' density,
  because a window total says nothing about where inside the window the
  cells sit. `--seam-rounds` runs the LNS with `LnsConfig.seam_rows/cols`
  set from the plan (a proportional sub-target on every separator box,
  the flyer's seam pass generalised to both axes); the plain objective
  is reported alongside. A final `repair_diagonal_runs` pass keeps the
  diagonal cap honest after all the patching.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rect(src: str, width: int, height: int, tone: str, contrast: float,
              keep_background: bool = False):
    """Image -> (grey, free_mask) at (height, width), no square padding.

    `keep_background` skips the subject cut-out: the whole frame is free.
    """
    from PIL import Image

    from gol_mosaics.freeform.targets import equalize_grey, normalize_grey
    from gol_mosaics.image_processing import ImageProcessor

    img, mask = ImageProcessor.load_image(
        src, return_alpha=True,
        remove_background=False if keep_background else "auto",
        contrast=contrast,
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
    ap.add_argument("--block-cols", type=int, default=0,
                    help="also cut the columns into blocks this wide (multiple "
                         "of 8); 0 = full-width strips")
    ap.add_argument("--keep-background", action="store_true",
                    help="no rembg: the whole frame is the subject (paintings)")
    ap.add_argument("--dither", choices=["round", "fs"], default="round",
                    help="window targets: rounded, or error-diffused (slow "
                         "fades stop turning into dotted lines)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from what --out already holds")
    ap.add_argument("--render-scale", type=int, default=1,
                    help="also save render_x<N>.png at N px per cell")
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
    ap.add_argument("--seam-rounds", type=int, default=2,
                    help="LNS rounds with separator sub-targets, after the polish")
    ap.add_argument("--seam-budget", type=float, default=900.0,
                    help="seconds of seam LNS per round")
    ap.add_argument("--max-diag-run", type=int, default=5,
                    help="longest solid diagonal chain of live cells; 0 lifts "
                         "the cap and lets the solver draw pencil lines again")
    args = ap.parse_args()

    if args.width % 8 or args.height % 8:
        ap.error("--width and --height must be multiples of 8")
    if args.block_cols % 8:
        ap.error("--block-cols must be a multiple of 8")

    from gol_mosaics.freeform.decompose import (plan_blocks, plan_strips, solve_blocks,
                                        solve_strips)
    from gol_mosaics.freeform.lns import (LnsConfig, improve, repair_diagonal_runs,
                                  seam_occupancy, window_devs)
    from gol_mosaics.freeform.metrics import deviation_stats, max_diagonal_run
    from gol_mosaics.freeform.solver import SpikeConfig, verify_still_life
    from gol_mosaics.freeform.targets import cell_targets, window_slices, window_targets

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.mkdir(parents=True, exist_ok=True)

    grey, free = load_rect(args.image, args.width, args.height,
                           args.tone, args.contrast, args.keep_background)
    log(f"loaded: {grey.shape} grid, {int(free.sum()):,} free cells")
    Image.fromarray(grey).save(out / "target_grey.png")

    cell_t = cell_targets(grey, args.dmax)
    windows = window_slices(cell_t.shape, k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows, dither=args.dither)
    log(f"{len(kept):,} windows, total live-cell target {int(targets.sum()):,} "
        f"({args.dither} dither)")

    max_diag_run = args.max_diag_run or None
    cfg = SpikeConfig(k=8, stride=8, d_max=args.dmax, seed=args.seed,
                      time_limit_s=args.strip_time, workers=args.strip_workers,
                      max_diag_run=max_diag_run, dither=args.dither)
    if args.block_cols:
        plan = plan_blocks(args.height, args.width, 8, block_rows=args.strip_rows,
                           block_cols=args.block_cols, gap=2)
        n_pieces = len(plan.row_spans) * len(plan.col_spans)
        plan_json = {"row_spans": [list(s) for s in plan.row_spans],
                     "col_spans": [list(s) for s in plan.col_spans],
                     "gap": plan.gap}
    else:
        plan = plan_strips(args.height, 8, strip_rows=args.strip_rows, gap=2)
        n_pieces = len(plan.spans)
        plan_json = {"spans": [list(s) for s in plan.spans], "gap": plan.gap}

    rounds = []
    resumed_pattern = None
    if args.resume:
        if (out / "pattern.npy").exists():  # mid-polish
            resumed_pattern = np.load(out / "pattern.npy")
            if (out / "polish_rounds.json").exists():
                rounds = json.loads((out / "polish_rounds.json").read_text())
            log(f"resumed from pattern.npy after {len(rounds)} polish round(s)")
        elif (out / "strips_pattern.npy").exists():  # stitched, unpolished
            resumed_pattern = np.load(out / "strips_pattern.npy")
            log("resumed from strips_pattern.npy")
        # otherwise solve_blocks restores whatever block checkpoints exist

    if resumed_pattern is not None:
        pattern = resumed_pattern
        strips = dict(statuses=["resumed"] * n_pieces, total_cpu_s=0.0)
        strips_wall, n_opt = 0.0, 0
    else:
        log(f"solving {n_pieces} {'blocks' if args.block_cols else 'strips'} "
            f"({args.strip_procs} procs x {args.strip_workers} workers)...")
        t0 = time.perf_counter()
        if args.block_cols:
            strips = solve_blocks(grey, free, cfg, plan, n_procs=args.strip_procs,
                                  checkpoint_dir=out / "blocks", log=log)
        else:
            strips = solve_strips(grey, free, cfg, plan, n_procs=args.strip_procs)
        strips_wall = time.perf_counter() - t0
        n_opt = sum(s == "OPTIMAL" for s in strips["statuses"])
        np.save(out / "strips_pattern.npy", strips["pattern"])
        pattern = strips["pattern"]
    obj = int(window_devs(pattern, free, kept, targets).sum())
    log(f"{'BLOCKS' if args.block_cols else 'STRIPS'} DONE in {strips_wall:.0f}s: "
        f"{n_opt}/{n_pieces} OPTIMAL, full-mask objective {obj:,}")

    for rnd in range(len(rounds) + 1, args.polish_rounds + 1):
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
        (out / "polish_rounds.json").write_text(json.dumps(rounds, indent=2))
        if obj == 0 or obj > 0.98 * prev:
            log("converged (or stalled) - stopping polish early")
            break

    seams = {}
    if args.seam_rounds and n_pieces > 1:
        row_spans = plan.row_spans if args.block_cols else plan.spans
        scfg = LnsConfig(patch_windows=args.patch_windows,
                         patch_time_s=args.patch_time, budget_s=args.seam_budget,
                         n_procs=args.polish_procs, seed=args.seed + 50,
                         max_diag_run=max_diag_run,
                         seam_rows=tuple(r1 - plan.gap for _, r1 in row_spans[:-1]),
                         seam_cols=tuple(c1 - plan.gap for _, c1 in plan.col_spans[:-1])
                         if args.block_cols else ())
        seams["before"] = seam_occupancy(pattern, scfg, free)
        seam_obj = int(window_devs(pattern, free, kept, targets, 0, scfg).sum())
        log(f"seams before: worst separator at {seams['before']['worst']:.2f} of "
            f"its neighbours, seam objective {seam_obj:,}")
        for rnd in range(1, args.seam_rounds + 1):
            if seam_obj == 0:
                break
            t0 = time.perf_counter()
            res = improve(pattern, free, kept, targets,
                          dataclasses.replace(scfg, seed=scfg.seed + rnd),
                          log=lambda *a, **k: None)
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
            np.save(out / "pattern.npy", pattern)
            if seam_obj > 0.98 * prev:
                break
        seams["after"] = seam_occupancy(pattern, scfg, free)
        log(f"seams after: worst separator at {seams['after']['worst']:.2f}")

    if max_diag_run:
        rcfg = LnsConfig(patch_windows=3, patch_time_s=30.0, n_procs=args.polish_procs,
                         seed=args.seed + 99, max_diag_run=max_diag_run)
        res = repair_diagonal_runs(pattern, free, kept, targets, rcfg, log=log)
        pattern, obj = res.pattern, res.objective
        np.save(out / "pattern.npy", pattern)

    ver = verify_still_life(pattern)
    assert ver["bounded"] and ver["toroidal"], f"verification failed: {ver}"
    stats = deviation_stats(pattern, cell_t, free, kept)
    longest_diag = max_diagonal_run(pattern)
    report = {
        "image": args.image,
        "grid": [args.height, args.width],
        "config": vars(args),
        "plan": plan_json,
        "solve": {"n_optimal": n_opt, "wall_time_s": strips_wall,
                  "total_cpu_s": strips["total_cpu_s"],
                  "statuses": strips["statuses"]},
        "polish_rounds": rounds,
        "seams": seams,
        "objective": obj,
        "deviation": stats,
        "verify": ver,
        "max_diagonal_run": longest_diag,
        "live_cells": int(pattern.sum()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    from gol_mosaics import ColorScheme, MosaicRenderer
    from gol_mosaics.export import GollyExporter

    render = MosaicRenderer(ColorScheme.ugent()).render_gol_mosaic(pattern)
    render.save(out / "render.png")
    if args.render_scale > 1:
        big = (render.width * args.render_scale, render.height * args.render_scale)
        render.resize(big, Image.Resampling.NEAREST).save(
            out / f"render_x{args.render_scale}.png")
    GollyExporter.export_to_cells(pattern, str(out / "pattern.cells"))
    log(f"ALL DONE: verify {ver}, {int(pattern.sum()):,} live cells, "
        f"longest diagonal chain {longest_diag}, "
        f"MAD {stats['mad']:.4f} (darkest quartile "
        f"{stats['mad_darkest_quartile']:.4f}, Pearson {stats['pearson']:.4f}); "
        f"saved to {out}")


if __name__ == "__main__":
    main()

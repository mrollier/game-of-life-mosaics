"""Poster pipeline: any image, any rectangular size, strips + LNS rounds.

The pipeline itself is gol_mosaics.freeform.poster.solve_poster; this script
is its command line, plus the render and the Golly export.

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
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from PIL import Image


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rect(src: str, width: int, height: int, tone: str, contrast: float,
              keep_background: bool = False):
    """Kept for the studies' imports; see freeform.targets.load_rect_target."""
    from gol_mosaics.freeform.targets import load_rect_target

    return load_rect_target(src, width, height, tone, contrast,
                            keep_background)


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

    from gol_mosaics.freeform.poster import PosterConfig, solve_poster
    from gol_mosaics.freeform.targets import load_rect_target

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.mkdir(parents=True, exist_ok=True)

    grey, free = load_rect_target(args.image, args.width, args.height,
                                  args.tone, args.contrast, args.keep_background)
    log(f"loaded: {grey.shape} grid, {int(free.sum()):,} free cells")
    Image.fromarray(grey).save(out / "target_grey.png")

    cfg = PosterConfig(
        d_max=args.dmax, seed=args.seed, dither=args.dither,
        max_diag_run=args.max_diag_run or None,
        strip_rows=args.strip_rows, block_cols=args.block_cols,
        strip_time=args.strip_time, strip_procs=args.strip_procs,
        strip_workers=args.strip_workers,
        polish_rounds=args.polish_rounds, polish_budget=args.polish_budget,
        patch_time=args.patch_time, patch_windows=args.patch_windows,
        polish_procs=args.polish_procs,
        seam_rounds=args.seam_rounds, seam_budget=args.seam_budget)
    result = solve_poster(grey, free, cfg, out=out, resume=args.resume, log=log)
    pattern, stats = result.pattern, result.report["deviation"]
    report = {"image": args.image, "config": vars(args), **result.report}
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
    log(f"ALL DONE: verify {result.report['verify']}, "
        f"{int(pattern.sum()):,} live cells, longest diagonal chain "
        f"{result.report['max_diagonal_run']}, MAD {stats['mad']:.4f} "
        f"(darkest quartile {stats['mad_darkest_quartile']:.4f}, Pearson "
        f"{stats['pearson']:.4f}); saved to {out}")


if __name__ == "__main__":
    main()

"""Free-form still lifes of the Ghent Belfry dragon on the postcard's cell grid.

Each design is a synthetic density target built from
output/data/dragon_layout.npz (the dragon's mask and the luminance of the
gilded-metal photo), solved with the library pipeline
gol_mosaics.freeform.solve_poster (strips, LNS polish, seam pass, diagonal
repair; Floyd-Steinberg window targets, peak density 0.30):

* shaded:     the dragon's own shading, smoothed (no plate seams or rivets),
              contrast-stretched and posterised to four density steps, dark
              metal dense and highlights sparse, with a denser rim along the
              silhouette; nothing outside the dragon.
* silhouette: the dragon as one uniform dense texture in a soft glow of
              sparse grain (a blurred mask, not a distance transform, which
              throws streaks off the sword and the tail), a two-cell dead moat
              between the two; the rest of the card empty.
* negative:   the dragon as a void (two-cell dead margin) cut out of a
              card-wide grain field that thickens from top to bottom.

Only the bounding box of the free cells is solved, on the 8-cell window
lattice, and embedded in the dead canvas. The settings suit a laptop with
~2 GB of free commit charge: one strip at a time, two CP-SAT workers.

    python studies/postcards/freeform_dragon.py targets          # previews only
    python studies/postcards/freeform_dragon.py solve shaded     # CP-SAT, ~15 min
    python studies/postcards/freeform_dragon.py verify           # recheck results

Results: output/data/freeform_<name>.npz (`live`, uint8 (366, 264), 1 = live)
and freeform_<name>.png (3 px per cell); checkpoints and reports in
output/runs/<name>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", HERE.parent, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from common import log
from postcards import H, W

DATA = HERE / "output" / "data"
RUNS = HERE / "output" / "runs"
D_MAX = 0.32        # density of grey 0; every target stays below it
K = 8               # window side

SHADES = (0.30, 0.22, 0.15, 0.08)   # dark metal .. highlight
RIM_CELLS, RIM_DENSITY = 3, 0.26     # denser rim inside the silhouette edge
SOLID = 0.26                         # silhouette: the dragon (0.28 stalls)
GLOW_SIGMA, GLOW_PEAK = 12.0, 0.12   # silhouette: the halo, ~20 cells
MOAT = 2                             # dead cells between dragon and grain
FIELD = (0.10, 0.24)                 # negative: top and bottom density


def load_layout():
    d = np.load(DATA / "dragon_layout.npz")
    assert d["mask"].shape == (H, W)
    return d["mask"].astype(bool), d["lum"].astype(np.float64)


def moat(mask, cells=MOAT):
    return ndi.binary_dilation(mask, iterations=cells)


# --- Targets: density per cell (0 .. 0.30) and the free mask --------------

def target_shaded(mask, lum):
    # Fill outside with the nearest dragon value, so the blur and the
    # median filter see no white halo at the edge.
    _, (ii, jj) = ndi.distance_transform_edt(~mask, return_indices=True)
    smooth = ndi.gaussian_filter(lum[ii, jj], 2.5)
    lo, hi = np.percentile(smooth[mask], [2, 98])
    t = np.clip((smooth - lo) / (hi - lo), 0.0, 1.0)
    level = np.minimum((t * len(SHADES)).astype(int), len(SHADES) - 1)
    level = ndi.median_filter(level, size=7)  # drop islands of one tone
    dens = np.asarray(SHADES)[level]
    rim = ndi.distance_transform_edt(mask) <= RIM_CELLS
    dens = np.where(rim, np.maximum(dens, RIM_DENSITY), dens)
    return np.where(mask, dens, 0.0), mask


def target_silhouette(mask, lum):
    g = np.clip(ndi.gaussian_filter(mask.astype(float), GLOW_SIGMA) / 0.5,
                0.0, 1.0) ** 1.2
    halo = np.where(moat(mask) | (g < 0.02), 0.0, GLOW_PEAK * g)
    dens = np.where(mask, SOLID, halo)
    return dens, dens > 0


def target_negative(mask, lum):
    y = np.linspace(0.0, 1.0, H)[:, None]
    ramp = FIELD[0] + (FIELD[1] - FIELD[0]) * y * y * (3 - 2 * y)
    free = ~moat(mask)
    return np.where(free, np.broadcast_to(ramp, (H, W)), 0.0), free


DESIGNS = {"shaded": target_shaded, "silhouette": target_silhouette,
           "negative": target_negative}


def to_grey(dens):
    return np.round(255.0 * (1.0 - np.clip(dens, 0, D_MAX) / D_MAX)).astype(np.uint8)


def crop_box(free, margin=4):
    """Rows and columns to solve: the free bounding box plus `margin`, grown
    to whole windows where the canvas allows ("partial" windows otherwise)."""
    spans, edge = [], "clamp"
    for axis, n in ((1, H), (0, W)):
        idx = np.where(free.any(axis))[0]
        a, b = max(0, idx[0] - margin), min(n, idx[-1] + 1 + margin)
        size = -(-(b - a) // K) * K
        if size > n:
            size, edge = b - a, "partial"
        a = min(a, n - size)
        spans.append((a, a + size))
    return spans[0], spans[1], edge


# --- Verification and output ------------------------------------------------

def still_life_errors(live):
    """(bad live, bad dead) on the canvas plus one dead ring around it."""
    p = np.pad(live.astype(np.int8), 2)
    n = sum(np.roll(np.roll(p, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    p, n = p[1:-1, 1:-1], n[1:-1, 1:-1]
    return (int(((p == 1) & ((n < 2) | (n > 3))).sum()),
            int(((p == 0) & (n == 3)).sum()))


def save(name, live):
    np.savez_compressed(DATA / f"freeform_{name}.npz", live=live)
    img = Image.fromarray(((1 - live) * 255).astype(np.uint8))
    img.resize((W * 3, H * 3), Image.NEAREST).save(DATA / f"freeform_{name}.png")


def cmd_targets(names):
    mask, lum = load_layout()
    RUNS.mkdir(parents=True, exist_ok=True)
    panels = []
    for name in names:
        dens, free = DESIGNS[name](mask, lum)
        (r0, r1), (c0, c1), edge = crop_box(free)
        vals = sorted({round(float(v), 3) for v in np.unique(dens[free])})
        log(f"{name}: {int(free.sum()):,} free cells, mean density "
            f"{dens[free].mean():.3f}, crop rows {r0}-{r1} cols {c0}-{c1} "
            f"({edge}), levels {vals[:6]}{' ...' if len(vals) > 6 else ''}")
        panels.append(to_grey(dens))
        panels.append(np.full((H, 8), 128, np.uint8))
    sheet = Image.fromarray(np.hstack(panels[:-1]))
    sheet.resize((sheet.width * 2, H * 2), Image.NEAREST).save(RUNS / "targets.png")
    log(f"wrote {RUNS / 'targets.png'}")


def patch_lns():
    """lns._cell_box fails on a patch box without any kept window, which a
    diagonal mask leaves in the empty corners of its lattice. Such a box
    scores 0 and is never solved, so any cell box will do."""
    from gol_mosaics.freeform import lns

    orig = lns._cell_box

    def cell_box(box, windows, index):
        r0, r1, c0, c1 = box
        if not (index[r0:r1, c0:c1] >= 0).any():
            return (0, 0, 0, 0)
        return orig(box, windows, index)

    lns._cell_box = cell_box


def cmd_solve(name, args):
    from gol_mosaics.freeform.poster import PosterConfig, solve_poster

    patch_lns()
    mask, lum = load_layout()
    dens, free = DESIGNS[name](mask, lum)
    (r0, r1), (c0, c1), edge = crop_box(free)
    grey, sub = to_grey(dens[r0:r1, c0:c1]), free[r0:r1, c0:c1]
    log(f"{name}: crop {r1 - r0}x{c1 - c0} at ({r0}, {c0}), "
        f"{int(sub.sum()):,} free cells, edge windows {edge}")
    cfg = PosterConfig(
        d_max=D_MAX, seed=args.seed, dither="fs", edge_windows=edge,
        max_diag_run=5, strip_rows=48, block_cols=args.block_cols,
        strip_time=args.strip_time, strip_procs=args.strip_procs, strip_workers=args.workers, isolate=True,
        polish_rounds=args.polish_rounds, polish_budget=args.polish_budget,
        patch_time=10.0, polish_procs=args.polish_procs, seam_rounds=2,
        seam_budget=args.seam_budget)
    t0 = time.perf_counter()
    res = solve_poster(grey, sub, cfg, out=RUNS / name, resume=args.resume,
                       log=log)
    wall = time.perf_counter() - t0
    live = np.zeros((H, W), np.uint8)
    live[r0:r1, c0:c1] = res.pattern
    assert not live[~free].any(), "live cells outside the free mask"
    bad = still_life_errors(live)
    save(name, live)
    report = dict(design=name, wall_time_s=round(wall), crop=[r0, r1, c0, c1],
                  still_life_errors=bad, live_cells=int(live.sum()),
                  density_on_free=float(live[free].mean()),
                  target_on_free=float(dens[free].mean()),
                  objective=res.objective, deviation=res.report["deviation"],
                  polish=res.report["polish_rounds"], seams=res.report["seams"],
                  config=vars(args))
    (RUNS / name / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"{name} DONE in {wall:.0f}s: {int(live.sum()):,} live cells, "
        f"objective {res.objective:,}, still-life errors (live, dead) {bad}")


def cmd_verify():
    for path in sorted(DATA.glob("freeform_*.npz")):
        live = np.load(path)["live"]
        log(f"{path.name}: {live.shape}, {int(live.sum()):,} live, "
            f"errors (live, dead) {still_life_errors(live)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["targets", "solve", "verify"])
    ap.add_argument("names", nargs="*", default=list(DESIGNS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-time", type=float, default=90.0)
    ap.add_argument("--block-cols", type=int, default=0,
                    help="solve 48 x N blocks instead of strips (dense designs)")
    ap.add_argument("--strip-procs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--polish-procs", type=int, default=2)
    ap.add_argument("--polish-rounds", type=int, default=2)
    ap.add_argument("--polish-budget", type=float, default=300.0)
    ap.add_argument("--seam-budget", type=float, default=180.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.cmd == "targets":
        cmd_targets(args.names)
    elif args.cmd == "verify":
        cmd_verify()
    else:
        for name in args.names:
            cmd_solve(name, args)


if __name__ == "__main__":
    main()

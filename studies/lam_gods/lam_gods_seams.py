"""Before/after study of the seam rounds on the Lam Gods still life.

v1 is the pattern after blocks + polish + diagonal repair (objective 469),
v2 the same after `poster.py --seam-rounds` (LNS with separator sub-targets,
REPORT.md C9). The question is whether v2 is more seamless without being a
worse rendering of the painting, so the study measures both: the separator
occupancy of every seam relative to its surroundings (1.0 = invisible), the
mean occupancy profile across all seams aligned on the separator, and the
plain window objective / MAD, which the seam pass is not allowed to spend.

    python studies/lam_gods/lam_gods_seams.py

Reads v1 and v2 from the committed bit-packed assets (a local
`results/pattern_v1_before_seams.npy` / `results/pattern.npy` from a fresh
poster.py run take precedence), writes `figures/lam_gods_seams.png`
(profile, crops) and `output/seam_study.json`, and prints the table.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gol_mosaics.freeform.lns import LnsConfig, seam_occupancy, window_devs
from gol_mosaics.freeform.metrics import deviation_stats, max_diagonal_run
from gol_mosaics.freeform.io import load_packed
from gol_mosaics.freeform.solver import verify_still_life
from gol_mosaics.freeform.targets import (cell_targets, load_rect_target,
                                          window_slices, window_targets)

RES = HERE / "results"
ASSETS = HERE / "assets"
W, H, BLOCK_ROWS, BLOCK_COLS, GAP = 2480, 1656, 64, 416, 2
SEAM_ROWS = tuple(r - GAP for r in range(BLOCK_ROWS, H, BLOCK_ROWS))
SEAM_COLS = tuple(c - GAP for c in range(BLOCK_COLS, W, BLOCK_COLS))


def profile(pattern: np.ndarray, seams, axis: int, span: int = 8) -> np.ndarray:
    """Mean occupancy at offsets -span..span+GAP-1 from the separator, all seams."""
    occ = pattern.mean(axis=1 - axis)
    rows = [occ[s - span : s + GAP + span] for s in seams if s - span >= 0]
    return np.mean(rows, axis=0)


def crop(pattern, r0, c0, h=96, w=160, scale=3):
    tile = pattern[r0 : r0 + h, c0 : c0 + w]
    return np.kron(1 - tile, np.ones((scale, scale), dtype=np.uint8)) * 255


def load(run: str, asset: str) -> np.ndarray:
    """A local run if present, else the committed bit-packed asset."""
    path = RES / run
    return np.load(path) if path.exists() else load_packed(ASSETS / asset)


def main() -> None:
    v1 = load("pattern_v1_before_seams.npy", "lam_gods_2480x1656_v1_before_seams.npz")
    v2 = load("pattern.npy", "lam_gods_2480x1656_pipeline.npz")
    grey, free = load_rect_target(str(REPO / "input/images/lam-gods-classic.png"), W, H,
                           "eq", 5.0, keep_background=True)
    cell_t = cell_targets(grey, 0.40)
    windows = window_slices(cell_t.shape, k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows, dither="fs")
    lcfg = LnsConfig(seam_rows=SEAM_ROWS, seam_cols=SEAM_COLS)

    table = {}
    for name, p in (("v1", v1), ("v2", v2)):
        occ = seam_occupancy(p, lcfg, free)
        stats = deviation_stats(p, cell_t, free, kept)
        table[name] = dict(
            objective=int(window_devs(p, free, kept, targets).sum()),
            seam_objective=int(window_devs(p, free, kept, targets, 0, lcfg).sum()),
            mad=stats["mad"], pearson=stats["pearson"],
            live=int(p.sum()), longest_diagonal=max_diagonal_run(p),
            verify=verify_still_life(p),
            seam_rows_mean=float(np.mean(list(occ["rows"].values()))),
            seam_rows_worst=min(occ["rows"].values()),
            seam_cols_mean=float(np.mean(list(occ["cols"].values()))),
            seam_cols_worst=min(occ["cols"].values()),
            rows=occ["rows"], cols=occ["cols"],
        )
    (HERE / "output").mkdir(exist_ok=True)
    (HERE / "output" / "seam_study.json").write_text(json.dumps(table, indent=2))
    for name, t in table.items():
        print(f"{name}: objective {t['objective']:,}, seam objective "
              f"{t['seam_objective']:,}, MAD {t['mad']:.4f}, Pearson "
              f"{t['pearson']:.4f}, live {t['live']:,}, longest diagonal "
              f"{t['longest_diagonal']}, verify {t['verify']}; separators at "
              f"{t['seam_rows_mean']:.2f} (worst {t['seam_rows_worst']:.2f}) of "
              f"neighbours across rows, {t['seam_cols_mean']:.2f} (worst "
              f"{t['seam_cols_worst']:.2f}) across columns")

    fig = Figure(figsize=(12, 9), dpi=120)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    for k, (axis, seams, label) in enumerate(((0, SEAM_ROWS, "rows"), (1, SEAM_COLS, "columns"))):
        ax = fig.add_subplot(gs[0, k])
        x = np.arange(-8, 8 + GAP)
        for p, name, style in ((v1, "v1 (before)", "--"), (v2, "v2 (seam rounds)", "-")):
            ax.plot(x, profile(p, seams, axis), style, label=name)
        ax.axvspan(-0.5, GAP - 0.5, color="0.85", label="separator")
        ax.set_title(f"mean occupancy across all {len(seams)} {label} seams")
        ax.set_xlabel(f"offset from separator ({label[:-1]})")
        ax.set_ylabel("live fraction")
        ax.legend(fontsize=8)
    # Two dense spots: a horizontal seam through the left crowd, a vertical
    # seam through the right crowd, before above and after below.
    spots = ((SEAM_ROWS[13] - 40, 300), (SEAM_ROWS[16] - 40, SEAM_COLS[3] - 80))
    for k, (r0, c0) in enumerate(spots):
        for j, (p, name) in enumerate(((v1, "v1"), (v2, "v2"))):
            ax = fig.add_subplot(gs[1 + j, k])
            ax.imshow(crop(p, r0, c0), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{name}: rows {r0}-{r0 + 96}, cols {c0}-{c0 + 160}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = HERE / "figures/lam_gods_seams.png"
    fig.savefig(out)
    print("figure:", out)


if __name__ == "__main__":
    main()

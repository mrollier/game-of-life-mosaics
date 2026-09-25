"""Persist a spike run: pattern, renders, convergence log, metrics, Golly file.

The bit-packed pattern and snapshot formats live in gol_mosaics.freeform.io;
they are re-exported here for the harness scripts.
"""

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from gol_mosaics.freeform.io import (load_pattern_asset, load_snapshots,  # noqa: F401
                                     save_pattern_asset, save_snapshots)
from gol_mosaics.freeform.metrics import deviation_stats
from gol_mosaics.freeform.targets import (box_sums, cell_targets, window_bounds,
                                         window_live_counts, window_slices,
                                         window_targets)

# Committed extracts of the headline runs (the full results/ tree is
# gitignored). Small enough to version: bit-packed patterns, ~16 kB total.
ASSETS = Path(__file__).resolve().parent / "assets"


def _window_field(values, windows, shape) -> np.ndarray:
    """Scatter per-window values onto the window-grid for plotting."""
    rows = sorted({w[0].start for w in windows})
    cols = sorted({w[1].start for w in windows})
    field = np.full((len(rows), len(cols)), np.nan)
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: j for j, c in enumerate(cols)}
    for v, (si, sj) in zip(values, windows):
        field[ri[si.start], ci[sj.start]] = v
    return field


def save_run(outdir, result, grey: np.ndarray, free_mask: np.ndarray) -> dict:
    """Write all artifacts for one solve; returns the metrics dict."""
    from gol_mosaics.colors import ColorScheme
    from gol_mosaics.export import GollyExporter
    from gol_mosaics.renderer import MosaicRenderer

    from gol_mosaics.freeform.solver import verify_still_life

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    pattern = result.pattern

    np.save(outdir / "pattern.npy", pattern)
    MosaicRenderer(ColorScheme.ugent()).render_gol_mosaic(pattern).save(
        outdir / "render.png"
    )
    GollyExporter.export_to_cells(pattern, str(outdir / "pattern.cells"))

    if result.snapshots:
        save_snapshots(outdir / "snapshots.npz", result.snapshots)

    with open(outdir / "convergence.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wall_time_s", "objective"])
        writer.writerows(result.obj_history)

    cell_t = cell_targets(grey, cfg.d_max)
    if getattr(result, "windows", None) is not None:
        # The geometry and targets the model was actually solved against.
        kept, targets = result.windows, result.targets
    else:
        windows = window_slices(
            grey.shape, cfg.k, cfg.stride, edge=getattr(cfg, "edge_windows", "clamp")
        )
        targets, kept = window_targets(cell_t, free_mask, windows)
    n_free = box_sums(np.asarray(free_mask, dtype=bool), window_bounds(kept))
    live = window_live_counts(pattern, free_mask, kept)
    # A window can hold no free cell when result.windows came from a
    # soft_zero/none solve (all-ones model mask) but the caller's mask has
    # a fully-masked window.
    some = n_free > 0
    achieved = live[some] / n_free[some]
    wanted = np.asarray(targets)[some] / n_free[some]
    plotted = [w for w, keep in zip(kept, some) if keep]
    # Pure object-oriented matplotlib: no pyplot import, so importing this
    # module never hijacks a notebook's inline backend.
    fig = Figure(figsize=(12, 4))
    axes = fig.subplots(1, 3)
    for ax, (title, vals) in zip(
        axes,
        [
            ("target density", wanted),
            ("achieved density", achieved),
            ("|difference|", np.abs(np.array(achieved) - np.array(wanted))),
        ],
    ):
        im = ax.imshow(
            _window_field(vals, plotted, grey.shape), cmap="viridis", vmin=0
        )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outdir / "density_maps.png", dpi=150)

    metrics = {
        "deviation": deviation_stats(pattern, cell_t, free_mask, kept),
        "verify": verify_still_life(pattern),
        "status": result.status,
        "objective": result.objective,
        "best_bound": result.best_bound,
        "wall_time_s": result.wall_time_s,
        "build_time_s": getattr(result, "build_time_s", 0.0),
        "seed_objective": getattr(result, "seed_objective", None),
        "max_rss_mb": result.max_rss_mb,
        "live_cells": int(pattern.sum()),
        "config": dataclasses.asdict(cfg),
    }
    with open(outdir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics

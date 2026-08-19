"""Persist a spike run: pattern, renders, convergence log, metrics, Golly file."""

import csv
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beyond_tiles.metrics import deviation_stats
from beyond_tiles.targets import cell_targets, window_slices, window_targets


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

    from beyond_tiles.still_image import verify_still_life

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    pattern = result.pattern

    np.save(outdir / "pattern.npy", pattern)
    MosaicRenderer(ColorScheme.ugent()).render_gol_mosaic(pattern).save(
        outdir / "render.png"
    )
    GollyExporter.export_to_cells(pattern, str(outdir / "pattern.cells"))

    with open(outdir / "convergence.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wall_time_s", "objective"])
        writer.writerows(result.obj_history)

    cell_t = cell_targets(grey, cfg.d_max)
    windows = window_slices(grey.shape, cfg.k, cfg.stride)
    targets, kept = window_targets(cell_t, free_mask, windows)
    achieved, wanted = [], []
    for t, (si, sj) in zip(targets, kept):
        n_free = free_mask[si, sj].sum()
        achieved.append(pattern[si, sj][free_mask[si, sj]].sum() / n_free)
        wanted.append(t / n_free)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, vals) in zip(
        axes,
        [
            ("target density", wanted),
            ("achieved density", achieved),
            ("|difference|", np.abs(np.array(achieved) - np.array(wanted))),
        ],
    ):
        im = ax.imshow(
            _window_field(vals, kept, grey.shape), cmap="viridis", vmin=0
        )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outdir / "density_maps.png", dpi=150)
    plt.close(fig)

    metrics = {
        "deviation": deviation_stats(pattern, cell_t, free_mask, kept),
        "verify": verify_still_life(pattern),
        "status": result.status,
        "objective": result.objective,
        "best_bound": result.best_bound,
        "wall_time_s": result.wall_time_s,
        "max_rss_mb": result.max_rss_mb,
        "live_cells": int(pattern.sum()),
        "config": dataclasses.asdict(cfg),
    }
    with open(outdir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics

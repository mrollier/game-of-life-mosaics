"""Turn a run's incumbent snapshots into a convergence movie.

CP-SAT keeps improving its solution until the time limit; the objective
curve says by how much, but not whether the *picture* still changes. These
helpers render the successive incumbents — pattern next to the convergence
curve — so a cut-off time can be chosen by looking rather than guessing.

Object-oriented matplotlib throughout: importing this module must not
hijack a notebook's inline backend.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from beyond_tiles.targets import Window

Snapshot = Tuple[float, int, np.ndarray]


def select_frames(
    times: Sequence[float], max_frames: int = 80, pacing: str = "time"
) -> List[int]:
    """Indices of the snapshots to animate; first and last are always kept.

    `pacing="time"` spreads the frames evenly over wall time, so playback
    reads as (compressed) real time — the right choice when the question is
    "how long should I let this run?". `"index"` takes every n-th incumbent,
    `"log"` spreads them evenly in log time (dense early on).
    """
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    if n == 0:
        return []
    if n <= max_frames:
        return list(range(n))

    if pacing == "index":
        picks = np.linspace(0, n - 1, max_frames)
    else:
        if pacing == "log":
            scale = np.log1p(times - times[0])
        elif pacing == "time":
            scale = times
        else:
            raise ValueError(f"unknown pacing: {pacing}")
        wanted = np.linspace(scale[0], scale[-1], max_frames)
        picks = np.searchsorted(scale, wanted)

    keep = sorted({0, n - 1, *(int(min(max(p, 0), n - 1)) for p in picks)})
    return keep


def frame_image(
    snapshot: Snapshot,
    history: Sequence[Tuple[float, int]],
    label: str,
    extra: str = "",
    panel: bool = True,
    px: int = 720,
):
    """Render one frame: the incumbent, optionally beside the curve so far."""
    from PIL import Image

    t, obj, pattern = snapshot
    dpi = 100
    if panel:
        fig = Figure(figsize=(px / dpi * 1.9, px / dpi), dpi=dpi)
        ax_img, ax_curve = fig.subplots(1, 2, width_ratios=[1, 1.1])
    else:
        fig = Figure(figsize=(px / dpi, px / dpi), dpi=dpi)
        ax_img, ax_curve = fig.subplots(), None

    ax_img.imshow(1 - pattern, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax_img.axis("off")
    caption = f"{label} | t = {t:6.1f} s | objective {obj}"
    if extra:
        caption += f" | {extra}"
    ax_img.set_title(caption, fontsize=10, family="monospace")

    if ax_curve is not None:
        hist = np.asarray(history, dtype=np.float64)
        ax_curve.plot(hist[:, 0], np.maximum(hist[:, 1], 1), color="0.7", lw=1)
        shown = hist[hist[:, 0] <= t]
        ax_curve.plot(shown[:, 0], np.maximum(shown[:, 1], 1), color="tab:red", lw=1.6)
        ax_curve.scatter([t], [max(obj, 1)], color="tab:red", zorder=3)
        ax_curve.set_xscale("log")
        ax_curve.set_yscale("log")
        ax_curve.set_xlabel("wall time (s)")
        ax_curve.set_ylabel("objective (total window deviation)")
        ax_curve.grid(alpha=0.3)

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def write_gif(
    path,
    images,
    duration_ms: int = 120,
    hold_last_ms: int = 2000,
    colors: int = 64,
) -> Path:
    """Write frames as an animated GIF, holding the final frame."""
    from PIL import Image

    images = list(images)
    if not images:
        raise ValueError("no frames to write")
    quantized = [
        im.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
        for im in images
    ]
    durations = [duration_ms] * len(quantized)
    durations[-1] = hold_last_ms
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return Path(path)


def filmstrip(
    snapshots: Sequence[Snapshot],
    history: Sequence[Tuple[float, int]],
    n_frames: int = 6,
    title: str = "",
):
    """Static version of the movie: N incumbents above the convergence curve.

    Renders in any viewer (including GitHub), which an animation does not.
    """
    picks = select_frames([t for t, _, _ in snapshots], n_frames, pacing="time")
    fig = Figure(figsize=(2.3 * len(picks), 4.4), dpi=110)
    grid = fig.add_gridspec(2, len(picks), height_ratios=[2.1, 1.2], hspace=0.35)

    for column, idx in enumerate(picks):
        t, obj, pattern = snapshots[idx]
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(1 - pattern, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{t:.0f} s\nobj {obj}", fontsize=9)
        ax.axis("off")

    hist = np.asarray(history, dtype=np.float64)
    ax = fig.add_subplot(grid[1, :])
    ax.plot(hist[:, 0], np.maximum(hist[:, 1], 1), color="tab:red", lw=1.4)
    for idx in picks:
        ax.axvline(snapshots[idx][0], color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("objective")
    ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=11)
    FigureCanvasAgg(fig)  # so the figure can render itself in a notebook
    return fig


def movie_from_run(
    run_dir,
    max_frames: int = 80,
    pacing: str = "time",
    fps: float = 8.0,
    panel: bool = True,
    label: Optional[str] = None,
    out_name: str = "evolution.gif",
) -> Path:
    """Build `evolution.gif` from a saved run directory (never re-solves)."""
    import csv
    import json

    from beyond_tiles.artifacts import load_snapshots

    run_dir = Path(run_dir)
    snapshots = load_snapshots(run_dir / "snapshots.npz")
    with open(run_dir / "convergence.csv") as fh:
        history = [(float(t), int(o)) for t, o in list(csv.reader(fh))[1:]]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    n_windows = metrics["deviation"]["n_windows"]
    label = label or "x".join(str(v) for v in snapshots[0][2].shape)

    picks = select_frames([t for t, _, _ in snapshots], max_frames, pacing)
    images = [
        frame_image(
            snapshots[i],
            history,
            label,
            extra=f"{snapshots[i][1] / n_windows:5.2f} cells/window off",
            panel=panel,
        )
        for i in picks
    ]
    out = write_gif(run_dir / out_name, images, duration_ms=int(1000 / fps))
    print(
        f"{out}: {len(images)} of {len(snapshots)} frames, "
        f"{out.stat().st_size / 1024:.0f} kB"
    )
    return out

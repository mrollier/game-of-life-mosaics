"""Helpers shared by the studies: timestamped logging and contact sheets.

Each study script puts this folder on sys.path and imports it as `common`.
"""

import time

import numpy as np


def log(msg: str) -> None:
    """Print with a wall-clock timestamp, flushed (long solves are logged
    to files and tailed)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def contact_sheet(columns, path, width=16, panel_h=1.35) -> None:
    """One sheet, a column per variation set, so sets compare row by row.

    Args:
        columns: List of columns, each a list of (title, PIL image) panels
        path: Output PNG
        width: Figure width in inches
        panel_h: Height of one panel row in inches
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(len(c) for c in columns)
    fig, axes = plt.subplots(rows, len(columns),
                             figsize=(width, panel_h * rows), squeeze=False)
    for j, column in enumerate(columns):
        for i in range(rows):
            ax = axes[i, j]
            ax.axis("off")
            if i < len(column):
                title, image = column[i]
                ax.imshow(np.asarray(image.convert("RGB")))
                ax.set_title(title, fontsize=8, pad=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

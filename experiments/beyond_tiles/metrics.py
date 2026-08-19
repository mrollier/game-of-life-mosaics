"""Fidelity and texture metrics for spike results. Pure numpy."""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from beyond_tiles.targets import Window


def motif_stats(
    pattern: np.ndarray,
    block: int = 4,
    min_live: int = 1,
    stride: Optional[int] = None,
) -> Dict[str, float]:
    """Distinct-block census over non-empty blocks: how tile-like is the texture?

    A tiled mosaic reuses few motifs (low entropy); a free-form solve
    should show many distinct blocks. Blocks with fewer than `min_live`
    live cells are excluded so empty background doesn't dominate.
    """
    stride = block if stride is None else stride
    counts: Counter = Counter()
    h, w = pattern.shape
    for i in range(0, h - block + 1, stride):
        for j in range(0, w - block + 1, stride):
            b = pattern[i : i + block, j : j + block]
            if b.sum() >= min_live:
                counts[b.tobytes()] += 1
    total = sum(counts.values())
    if total == 0:
        return {"n_blocks": 0, "n_distinct": 0, "entropy_bits": 0.0}
    p = np.array(list(counts.values())) / total
    return {
        "n_blocks": int(total),
        "n_distinct": int(len(counts)),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
    }


def tile_db_overlap(pattern: np.ndarray, level: int = 1) -> float:
    """Fraction of non-empty 6L x 6L blocks that ARE a level-L tile (mod D4).

    Quantifies the "beyond tiles" claim: a tile mosaic scores near 1,
    a free-form solve should score near 0.
    """
    import gol_mosaics

    tiles = np.load(
        Path(gol_mosaics.__file__).parent
        / "data"
        / f"solutions_pattern_level_{level}.npy"
    ).astype(np.uint8)
    images = set()
    for t in tiles:
        for g in (t, np.rot90(t), np.rot90(t, 2), np.rot90(t, 3)):
            images.add(g.tobytes())
            images.add(g.T.tobytes())
    n = tiles.shape[1]
    matches = total = 0
    h, w = pattern.shape
    for i in range(0, h - n + 1, n):
        for j in range(0, w - n + 1, n):
            b = np.ascontiguousarray(pattern[i : i + n, j : j + n])
            if b.any():
                total += 1
                matches += b.tobytes() in images
    return matches / total if total else 0.0


def spectral(pattern: np.ndarray, n_bins: int = 40) -> Dict[str, list]:
    """Radially averaged power spectrum. A tile mosaic concentrates power at
    the tile-lattice frequency; free-form texture should not."""
    p = pattern.astype(np.float64)
    power = np.abs(np.fft.fftshift(np.fft.fft2(p - p.mean()))) ** 2
    h, w = power.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h // 2, xx - w // 2)
    r_max = r.max()
    bins = np.linspace(0, r_max, n_bins + 1)
    which = np.digitize(r.ravel(), bins) - 1
    prof = np.bincount(
        which.clip(0, n_bins - 1), weights=power.ravel(), minlength=n_bins
    ) / np.maximum(np.bincount(which.clip(0, n_bins - 1), minlength=n_bins), 1)
    centers = (bins[:-1] + bins[1:]) / 2
    return {"radius": centers.tolist(), "power": prof.tolist()}


def deviation_stats(
    pattern: np.ndarray,
    cell_t: np.ndarray,
    free_mask: np.ndarray,
    windows: List[Window],
) -> Dict[str, float]:
    """Achieved-vs-target window densities, in live-fraction units.

    Windows without free cells are ignored. `mad_darkest_quartile` is the
    MAD restricted to the quarter of windows with the highest targets —
    the regime closest to the Elkies density ceiling.
    """
    achieved, target = [], []
    for si, sj in windows:
        free = free_mask[si, sj]
        n_free = int(free.sum())
        if n_free == 0:
            continue
        achieved.append(pattern[si, sj][free].sum() / n_free)
        target.append(cell_t[si, sj][free].sum() / n_free)
    a = np.asarray(achieved)
    t = np.asarray(target)
    dev = np.abs(a - t)

    darkest = np.argsort(t)[::-1][: max(1, int(np.ceil(len(t) / 4)))]
    if len(t) >= 2 and a.std() > 0 and t.std() > 0:
        pearson = float(np.corrcoef(a, t)[0, 1])
    else:
        pearson = float("nan")
    return {
        "mad": float(dev.mean()),
        "p95": float(np.percentile(dev, 95)),
        "max": float(dev.max()),
        "mad_darkest_quartile": float(dev[darkest].mean()),
        "pearson": pearson,
        "n_windows": int(len(t)),
    }

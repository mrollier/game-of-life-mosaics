"""Greyscale image -> per-cell density targets and window geometry.

Pure numpy; no solver dependencies. Windows are k x k boxes at the given
stride, with a clamped final window so the grid edge is always covered.
"""

from typing import List, Tuple

import numpy as np

Window = Tuple[slice, slice]


def _axis_starts(n: int, k: int, stride: int) -> List[int]:
    assert n >= k, f"axis length {n} smaller than window {k}"
    starts = list(range(0, n - k + 1, stride))
    if starts[-1] != n - k:
        starts.append(n - k)
    return starts


def window_slices(shape: Tuple[int, int], k: int, stride: int) -> List[Window]:
    """All k x k windows at `stride`, clamped so the last row/col is covered."""
    rows = _axis_starts(shape[0], k, stride)
    cols = _axis_starts(shape[1], k, stride)
    return [
        (slice(i, i + k), slice(j, j + k)) for i in rows for j in cols
    ]


def cell_targets(grey: np.ndarray, d_max: float) -> np.ndarray:
    """Per-cell target density: black (0) -> d_max, white (255) -> 0."""
    return d_max * (1.0 - np.asarray(grey, dtype=np.float64) / 255.0)


def window_targets(
    cell_t: np.ndarray, free_mask: np.ndarray, windows: List[Window]
) -> Tuple[np.ndarray, List[Window]]:
    """Integer live-cell target per window, summed over free cells only.

    Windows without any free cell are dropped.
    """
    targets: List[int] = []
    kept: List[Window] = []
    for si, sj in windows:
        free = free_mask[si, sj]
        if not free.any():
            continue
        targets.append(int(round(cell_t[si, sj][free].sum())))
        kept.append((si, sj))
    return np.asarray(targets, dtype=np.int64), kept


def grey_and_mask_from_image(
    src,
    size: int,
    contrast: float = 5.0,
    remove_background="auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Image (path or PIL) -> (grey uint8 (size,size), free_mask bool).

    Reuses the tile pipeline's load path (alpha handling, optional rembg,
    sigmoid contrast), pads to square, then Lanczos-resizes to the cell
    grid. The alpha mask goes through the identical geometry so subject
    and grey stay aligned; background padding counts as masked.
    """
    from PIL import Image

    from gol_mosaics.image_processing import ImageProcessor

    img, mask = ImageProcessor.load_image(
        src,
        return_alpha=True,
        remove_background=remove_background,
        contrast=contrast,
    )
    img = ImageProcessor.square_image(img, return_aspect=False, fill_color=255)
    mask = ImageProcessor.square_image(mask, return_aspect=False, fill_color=0)
    grey = np.asarray(
        img.resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    free = (
        np.asarray(mask.resize((size, size), Image.Resampling.LANCZOS)) >= 128
    )
    return grey, free


def normalize_grey(
    grey: np.ndarray,
    free_mask: np.ndarray,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
) -> np.ndarray:
    """Percentile-stretch grey values inside the subject mask to 0..255.

    The tile pipeline maps grey to density *relatively* (min-max normalised
    tile densities); the free-form absolute mapping needs the same courtesy,
    otherwise a light-midtone portrait lands near density zero. Background
    pixels are forced white (target 0).
    """
    vals = grey[free_mask]
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if hi <= lo:
        out = grey.copy()
    else:
        stretched = np.clip((grey.astype(np.float64) - lo) / (hi - lo), 0, 1)
        out = (stretched * 255).round().astype(np.uint8)
    out[~free_mask] = 255
    return out


def equalize_grey(grey: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    """Histogram-equalize grey values inside the subject mask.

    For high-key sources (e.g. the Marilyn photo, median grey 234) a
    faithful absolute mapping leaves most of the subject near density
    zero. Equalization spreads the subject's tones over the full range —
    the free-form analogue of the tile pipeline's perceptual remap
    (relative density stretch + empty-tile cutoff). Background stays white.
    """
    vals = grey[free_mask]
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    cdf_min = cdf[np.flatnonzero(hist)[0]]
    lut = np.clip(
        (cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255, 0, 255
    ).round().astype(np.uint8)
    out = grey.copy()
    out[free_mask] = lut[vals]
    out[~free_mask] = 255
    return out


def ramp_grey(size: int, axis: int = 1) -> np.ndarray:
    """Linear 0..255 ramp along `axis`, e.g. for density-fidelity tests."""
    line = np.linspace(0, 255, size).round().astype(np.uint8)
    ramp = np.tile(line, (size, 1))
    return ramp if axis == 1 else ramp.T


def uniform_grey(size: int, value: int) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)

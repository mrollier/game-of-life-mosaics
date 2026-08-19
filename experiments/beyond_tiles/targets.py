"""Greyscale image -> per-cell density targets and window geometry.

Pure numpy; no solver dependencies. Windows are k x k boxes at the given
stride, with a clamped final window so the grid edge is always covered.
"""

from typing import List, Tuple

import numpy as np

Window = Tuple[slice, slice]


def _axis_starts(
    n: int, k: int, stride: int, edge: str = "clamp"
) -> List[Tuple[int, int]]:
    """(start, stop) spans along one axis.

    `edge="clamp"` (historic default) shifts a final full-size window back
    to end at `n`, which overlaps its neighbour whenever `n % k != 0` —
    even in otherwise disjoint stride==k geometry. `edge="partial"` keeps
    the grid disjoint by emitting a final short window instead.
    """
    assert n >= k, f"axis length {n} smaller than window {k}"
    starts = list(range(0, n - k + 1, stride))
    spans = [(s, s + k) for s in starts]
    if edge == "clamp":
        if starts[-1] != n - k:
            spans.append((n - k, n))
    elif edge == "partial":
        last_stop = starts[-1] + k
        if last_stop < n:
            spans.append((last_stop, n))
    else:
        raise ValueError(f"unknown edge mode: {edge}")
    return spans


def window_slices(
    shape: Tuple[int, int], k: int, stride: int, edge: str = "clamp"
) -> List[Window]:
    """All k x k windows at `stride`; the edge is covered per `edge` mode."""
    rows = _axis_starts(shape[0], k, stride, edge)
    cols = _axis_starts(shape[1], k, stride, edge)
    return [
        (slice(i0, i1), slice(j0, j1)) for i0, i1 in rows for j0, j1 in cols
    ]


def cell_targets(grey: np.ndarray, d_max: float) -> np.ndarray:
    """Per-cell target density: black (0) -> d_max, white (255) -> 0."""
    return d_max * (1.0 - np.asarray(grey, dtype=np.float64) / 255.0)


def _dither_scan(
    windows: List[Window],
    exact: List[float],
    counts: List[int],
    kept_flags: List[bool],
) -> List[int]:
    """Serpentine Floyd-Steinberg on the window lattice.

    Plain per-window rounding throws away up to half a live cell per
    window, and the residual correlates with image gradients (a slow ramp
    rounds the same way for whole bands of windows). Error diffusion pushes
    each residual onto not-yet-quantized neighbouring windows, so the
    rounding error cancels in aggregate: the total live-cell mass is
    preserved to within half a cell. Weights are renormalized over the
    neighbours that exist; when none of the four targets is available
    (end of a kept segment, isolated window), the residual is carried to
    the next kept window in scan order instead of being dropped, so only
    the final window's residual is ever lost.
    """
    rows = sorted({w[0].start for w in windows})
    cols = sorted({w[1].start for w in windows})
    n_rows, n_cols = len(rows), len(cols)
    assert n_rows * n_cols == len(windows), "windows must form a full lattice"

    value = np.asarray(exact, dtype=np.float64).reshape(n_rows, n_cols)
    n_free = np.asarray(counts, dtype=np.int64).reshape(n_rows, n_cols)
    kept = np.asarray(kept_flags, dtype=bool).reshape(n_rows, n_cols)
    out = np.zeros((n_rows, n_cols), dtype=np.int64)
    err = np.zeros((n_rows, n_cols), dtype=np.float64)

    # Scan order, so a residual with no diffusion target can be handed to
    # the next kept window instead of vanishing.
    order = []
    for r in range(n_rows):
        cs = range(n_cols) if r % 2 == 0 else range(n_cols - 1, -1, -1)
        order.extend((r, c) for c in cs if kept[r, c])
    successor = {rc: order[i + 1] for i, rc in enumerate(order[:-1])}

    for r, c in order:
        d = 1 if r % 2 == 0 else -1
        v = value[r, c] + err[r, c]
        q = int(np.clip(round(v), 0, n_free[r, c]))
        out[r, c] = q
        residual = v - q
        neighbours = [
            (r, c + d, 7.0),
            (r + 1, c - d, 3.0),
            (r + 1, c, 5.0),
            (r + 1, c + d, 1.0),
        ]
        avail = [
            (rr, cc, wgt)
            for rr, cc, wgt in neighbours
            if 0 <= rr < n_rows and 0 <= cc < n_cols and kept[rr, cc]
        ]
        if avail:
            total = sum(wgt for _, _, wgt in avail)
            for rr, cc, wgt in avail:
                err[rr, cc] += residual * wgt / total
        elif (r, c) in successor:
            err[successor[(r, c)]] += residual
    return out[kept].tolist()


def window_targets(
    cell_t: np.ndarray,
    free_mask: np.ndarray,
    windows: List[Window],
    dither: str = "round",
) -> Tuple[np.ndarray, List[Window]]:
    """Integer live-cell target per window, summed over free cells only.

    Windows without any free cell are dropped. `dither="fs"` replaces the
    per-window rounding with Floyd-Steinberg error diffusion over the
    window lattice (see `_dither_scan`).
    """
    exact: List[float] = []
    counts: List[int] = []
    kept_flags: List[bool] = []
    for si, sj in windows:
        free = free_mask[si, sj]
        n = int(free.sum())
        counts.append(n)
        kept_flags.append(n > 0)
        exact.append(float(cell_t[si, sj][free].sum()) if n else 0.0)

    if dither == "round":
        targets = [
            int(round(v)) for v, keep in zip(exact, kept_flags) if keep
        ]
    elif dither == "fs":
        targets = _dither_scan(windows, exact, counts, kept_flags)
    else:
        raise ValueError(f"unknown dither mode: {dither}")
    kept = [w for w, keep in zip(windows, kept_flags) if keep]
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

"""
Post-hoc backgrounds and recolouring for finished patterns.

Rendering is separated from generation on purpose. A tile mosaic is cheap to
regenerate, but a free-form still life is solved cell-by-cell with CP-SAT and
costs minutes to hours — so its colours and its backdrop have to be choosable
afterwards, from the stored pattern alone.

The backdrop is drawn from a tri-state layer (0 = subject, 1 = field
background, 2 = field pixel) that `MosaicRenderer.render_full_mosaic` already
knows how to paint. Four styles are available:

- ``'none'``  transparent, so the piece can be placed on any canvas
- ``'flat'``  a solid fill
- ``'eca'``   an elementary cellular automaton field, as in the tile mosaics
- ``'agar'``  a still-life block agar

Only ``'agar'`` produces real Game of Life cells. The other three are paint:
they never enter a Golly export, and the pattern stays exactly the still life
that was verified. For a backdrop that is itself alive, use
`life_safe_pattern`, whose output is a still life over the whole grid.

Example:
    >>> from gol_mosaics import ColorScheme, compose
    >>> art = compose(pattern, ~free_mask, ColorScheme.warhol(seed=7),
    ...               style='eca', rule=110, scale=3)
    >>> art.save('poster.png')
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .colors import ColorScheme
from .eca import ECABackground
from .renderer import MosaicRenderer

#: Backdrop styles accepted by :func:`compose`.
STYLES = ('none', 'flat', 'eca', 'agar')

#: Side of the agar's square blocks, in cells.
BLOCK = 2

#: Smallest agar lattice pitch that is still safe (see :func:`agar_background`).
MIN_PITCH = (3, 4)

#: Smallest gap that keeps the agar and the subject from interacting.
MIN_GAP = 2


def agar_background(background_mask: np.ndarray,
                    pitch: Tuple[int, int] = (3, 4),
                    gap: int = 2) -> np.ndarray:
    """
    Fill the background region with a still-life block agar.

    The construction is the one from the beyond-tiles warm-start seeds: 2x2
    blocks on a lattice with period 3 in rows and 4 in columns. That
    asymmetric pitch is the point — no dead cell can ever see exactly three
    live neighbours regardless of which subset of sites is filled, so dropping
    whole blocks (which is all a mask ever does) is safe by construction. No
    repair pass, no seams.

    Blocks are kept only where the mask is background for `gap` cells in every
    direction, which also keeps them `gap` cells clear of the grid border. At
    the default gap of 2 every agar cell is at Chebyshev distance 3 or more
    from any non-background cell, so no dead cell's 3x3 neighbourhood can
    contain both an agar cell and a subject cell. Neither population sees the
    other, both keep the neighbour counts they were verified with, and the
    union is a still life.

    Args:
        background_mask: Boolean array, True where the background is. Note
            this is the opposite convention to the beyond-tiles `free_mask`,
            which is True on the subject — pass `~free_mask`.
        pitch: (rows, columns) lattice period in cells. Must be at least
            (3, 4); larger values are sparser and remain safe.
        gap: Cells of clearance required around each block (default 2).

    Returns:
        Binary uint8 array of the same shape, 1 on agar cells

    Raises:
        ValueError: If the mask is not 2D, the pitch is too tight, or the gap
            is below 2

    Example:
        >>> agar = agar_background(~free_mask)
        >>> is_still_life(pattern | agar)
        True
    """
    mask = np.asarray(background_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(
            f"Background mask must be 2D array, got shape {mask.shape}"
        )

    rows, cols = pitch
    if rows < MIN_PITCH[0] or cols < MIN_PITCH[1]:
        raise ValueError(
            f"Pitch must be at least {MIN_PITCH} for the agar to be a still "
            f"life, got {tuple(pitch)}"
        )
    if gap < MIN_GAP:
        raise ValueError(
            f"Gap must be at least {MIN_GAP} to keep the agar clear of the "
            f"subject, got {gap}"
        )

    height, width = mask.shape
    agar = np.zeros((height, width), dtype=np.uint8)

    # The range bounds keep every slice below in range. Negative starts must
    # not reach the indexing: numpy would wrap them silently and .all() could
    # then pass on a block that runs off the edge.
    # ponytail: per-site loop, ~240k iterations on a 1416x2000 poster (~1 s).
    # Swap in a summed-area table if that ever becomes the slow part.
    for i in range(gap, height - BLOCK - gap + 1, rows):
        for j in range(gap, width - BLOCK - gap + 1, cols):
            if mask[i - gap:i + BLOCK + gap, j - gap:j + BLOCK + gap].all():
                agar[i:i + BLOCK, j:j + BLOCK] = 1

    return agar


def life_safe_pattern(pattern: np.ndarray,
                      background_mask: np.ndarray,
                      pitch: Tuple[int, int] = (3, 4),
                      gap: int = 2) -> np.ndarray:
    """
    Merge a still-life agar background into a pattern.

    Unlike the painted backdrops, the result is one Game of Life pattern: the
    whole grid, subject and background together, is a still life and can be
    exported to Golly as a single object.

    Args:
        pattern: Binary array holding the subject still life
        background_mask: Boolean array, True where the background is
        pitch: Agar lattice period, see :func:`agar_background`
        gap: Clearance around each agar block, see :func:`agar_background`

    Returns:
        Binary uint8 array of the union

    Raises:
        ValueError: If the pattern has live cells inside the background region

    Example:
        >>> whole = life_safe_pattern(pattern, ~free_mask)
        >>> GollyExporter.export_to_cells(whole, 'art.cells')
    """
    pattern = np.asarray(pattern, dtype=np.uint8)
    mask = np.asarray(background_mask, dtype=bool)
    if mask.shape != pattern.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match pattern {pattern.shape}"
        )

    # A solve with mask_mode 'soft_zero' or 'none' leaves live cells outside
    # the subject. Growing an agar around them would change their neighbour
    # counts, and the still-life guarantee below would be a lie.
    stray = int(pattern[mask].sum())
    if stray:
        raise ValueError(
            f"Pattern has {stray} live cells inside the background region; "
            f"a life-safe background needs the background to be dead"
        )

    return pattern | agar_background(mask, pitch=pitch, gap=gap)


def compose(pattern: np.ndarray,
            background_mask: np.ndarray,
            scheme: Optional[ColorScheme] = None,
            style: str = 'eca',
            *,
            rule: Optional[int] = None,
            supersample: int = 15,
            pitch: Tuple[int, int] = (3, 4),
            gap: int = 2,
            scale: int = 1) -> Image.Image:
    """
    Render a finished pattern with a colour scheme and a backdrop.

    Args:
        pattern: Binary array (0=background, 1=pixel)
        background_mask: Boolean array, True where the background is. The
            beyond-tiles pipeline carries the opposite convention in
            `free_mask` (True on the subject) — pass `~free_mask`.
        scheme: ColorScheme to paint with (default: `ColorScheme.ugent()`).
            The backdrop uses the scheme's two ECA colours.
        style: One of 'none' (transparent), 'flat', 'eca', 'agar'
        rule: ECA rule for style='eca'. None picks an interesting rule at
            random, as the mosaic pipeline does.
        supersample: ECA cell side in pixels for style='eca' (default 15)
        pitch: Agar lattice period for style='agar', see
            :func:`agar_background`
        gap: Agar clearance for style='agar', see :func:`agar_background`
        scale: Integer nearest-neighbour upscale, applied last. The pattern
            renders at one pixel per cell, which is far below print
            resolution, so a poster wants 3 or more.

    Returns:
        RGBA PIL Image, `scale` pixels per cell

    Raises:
        ValueError: If the pattern is not 2D, the mask does not match it, the
            style is unknown, or the scale is not a positive integer

    Example:
        >>> compose(pattern, ~free_mask, ColorScheme.warhol(seed=7),
        ...         style='agar', scale=3).save('art.png')
    """
    pattern = np.asarray(pattern)
    if pattern.ndim != 2:
        raise ValueError(
            f"Pattern must be 2D array, got shape {pattern.shape}"
        )

    mask = np.asarray(background_mask, dtype=bool)
    if mask.shape != pattern.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match pattern {pattern.shape}"
        )

    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}, expected one of {STYLES}")

    if int(scale) != scale or scale < 1:
        raise ValueError(f"Scale must be a positive integer, got {scale}")

    renderer = MosaicRenderer(scheme if scheme is not None
                              else ColorScheme.ugent())

    if style == 'none':
        rgba = np.asarray(renderer.render_gol_mosaic(pattern)).copy()
        rgba[mask, 3] = 0
        image = Image.fromarray(rgba, mode='RGBA')
    else:
        height, width = pattern.shape
        if style == 'flat':
            field = np.zeros((height, width), dtype=np.uint8)
        elif style == 'eca':
            if rule is None:
                rule = int(np.random.choice(
                    ECABackground.COMPLEX_RULES + ECABackground.CHAOTIC_RULES
                ))
            field = ECABackground(rule).generate(
                width=width, height=height, supersample=supersample
            )
        else:
            field = agar_background(mask, pitch=pitch, gap=gap)

        # Tri-state layer: 0=subject (transparent), 1=field background,
        # 2=field pixel. Same construction the tile mosaics use.
        backdrop = mask.astype(np.uint8)
        image = renderer.render_full_mosaic(
            pattern, backdrop * (field + backdrop)
        )

    if scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.NEAREST
        )
    return image

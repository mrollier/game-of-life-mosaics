"""
Post-hoc backgrounds and recolouring for finished patterns.

Rendering is separated from generation on purpose. A tile mosaic is cheap to
regenerate, but a free-form still life is solved cell-by-cell with CP-SAT and
costs minutes to hours — so its colours and its backdrop have to be choosable
afterwards, from the stored pattern alone.

The backdrop is drawn from a layer stack (0 = subject, 1 = field background,
2 = field pixel, 3 and up for the filler levels) that
`MosaicRenderer.render_full_mosaic` already knows how to paint. Four styles
are available:

- ``'none'``  transparent, so the piece can be placed on any canvas
- ``'flat'``  a solid fill
- ``'eca'``   an elementary cellular automaton field, as in the tile mosaics
- ``'agar'``  a still-life block agar
- ``'mosaic'`` a field of this project's own pond tiles

A plain mosaic leaves a ragged halo: a tile is seated only where its whole
footprint fits, so the distance from the subject to the nearest tile is
whatever the lattice happens to allow — on a 800x200 banner, up to 55 cells
for level 6. ``'mosaic'`` therefore takes a ``fill`` argument that packs that
gap with progressively smaller tiles and then with loose elementary still
lifes, which cuts the worst halo to single digits. Each of those levels is
numbered separately so the renderer can grade them by size. See
:func:`filled_background`.

Only ``'agar'`` and ``'mosaic'`` produce real Game of Life cells. The other
three are paint: they never enter a Golly export, and the pattern stays
exactly the still life that was verified. For a backdrop that is itself alive,
use `life_safe_pattern`, whose output is a still life over the whole grid.

Example:
    >>> from gol_mosaics import ColorScheme, compose
    >>> art = compose(pattern, ~free_mask, ColorScheme.warhol(seed=7),
    ...               style='eca', rule=110, scale=3)
    >>> art.save('poster.png')
"""

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .colors import ColorScheme
from .eca import ECABackground
from .renderer import MosaicRenderer

#: Backdrop styles accepted by :func:`compose`.
STYLES = ('none', 'flat', 'eca', 'agar', 'mosaic')

#: Tile geometries accepted by :func:`mosaic_background`.
SHAPES = ('diamond', 'square')

#: Density gradients accepted by :func:`mosaic_background`.
TONES = (None, 'linear', 'radial', 'vignette')

#: Side of the agar's square blocks, in cells.
BLOCK = 2

#: Smallest agar lattice pitch that is still safe (see :func:`agar_background`).
MIN_PITCH = (3, 4)

#: Smallest gap that keeps the agar and the subject from interacting.
MIN_GAP = 2

#: Loose still lifes used by :func:`scatter_background` to close the last of
#: the gap. Small ones only: a filler is worth having exactly where nothing
#: bigger fits, and every one of these is a still life on its own (checked in
#: tests/test_compose.py).
ELEMENTARY = {
    'block': np.array([[1, 1],
                       [1, 1]], dtype=np.uint8),
    'tub': np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=np.uint8),
    'boat': np.array([[1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=np.uint8),
    'ship': np.array([[1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 1]], dtype=np.uint8),
    'beehive': np.array([[0, 1, 1, 0],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]], dtype=np.uint8),
    'pond': np.array([[0, 1, 1, 0],
                      [1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [0, 1, 1, 0]], dtype=np.uint8),
    'loaf': np.array([[0, 1, 1, 0],
                      [1, 0, 0, 1],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]], dtype=np.uint8),
}


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


def density_band(level: int,
                 shape: str = 'diamond',
                 density: Tuple[float, float] = (0.0, 1.0),
                 ) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Tile indices whose density falls inside a band, and what that band is.

    `PatternLibrary.densities` is min-max normalised *per level*, so the same
    band means a different absolute fill at level 3 than at level 5. The
    absolute range is returned alongside so that relative scale is never a
    hidden trap.

    Args:
        level: Tile level (diamond 1-6, square 3-5)
        shape: Tile geometry, 'diamond' or 'square'
        density: (low, high) normalised density band, both ends inclusive

    Returns:
        (indices into the library's solutions, (min, max) absolute density of
        the selected tiles as live cells per bounding-box cell)

    Raises:
        ValueError: If the band is inverted or selects no tiles

    Example:
        >>> indices, (lo, hi) = density_band(4, 'diamond', (0.3, 0.7))
        >>> f"{len(indices)} tiles, absolute fill {lo:.3f}-{hi:.3f}"
        '32 tiles, absolute fill 0.137-0.184'
    """
    from .patterns import PatternLibrary

    low, high = float(density[0]), float(density[1])
    if low > high:
        raise ValueError(
            f"Density band is inverted: {low} > {high}"
        )

    library = PatternLibrary.load(level, shape=shape)
    normalised = library.densities
    indices = np.flatnonzero((normalised >= low) & (normalised <= high))
    if indices.size == 0:
        raise ValueError(
            f"No level-{level} {shape} tile has a normalised density in "
            f"[{low}, {high}]; the bank spans "
            f"[{normalised.min():.3f}, {normalised.max():.3f}]"
        )

    absolute = _tile_fill(level, shape)[indices]
    return indices, (float(absolute.min()), float(absolute.max()))


@lru_cache(maxsize=None)
def _tile_fill(level: int, shape: str) -> np.ndarray:
    """Absolute density (live cells per bounding-box cell) of every tile.

    Summed rather than fancy-indexed: the level-6 diamond bank is 332,321
    tiles of 36x36, and a copy of a subset of it is 100s of MB.
    """
    from .patterns import PatternLibrary

    solutions = PatternLibrary.load(level, shape=shape).solutions
    return solutions.sum(axis=(1, 2)) / float(solutions[0].size)


def _tone_field(tone: str, angle: float, height: int, width: int) -> np.ndarray:
    """Scalar 0..1 field over the canvas, 1 where the background is densest."""
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]

    if tone == 'linear':
        # 0 degrees ramps left to right, 90 bottom to top (row 0 is the top).
        theta = np.radians(angle)
        field = np.cos(theta) * x - np.sin(theta) * y
    else:
        radius = np.hypot(y - (height - 1) / 2, x - (width - 1) / 2)
        field = radius if tone == 'vignette' else -radius

    span = field.max() - field.min()
    if span == 0:
        return np.zeros((height, width), dtype=np.float64)
    return (field - field.min()) / span


def mosaic_background(background_mask: np.ndarray,
                      level: int = 4,
                      shape: str = 'diamond',
                      density: Tuple[float, float] = (0.0, 1.0),
                      tone: Optional[str] = None,
                      tone_angle: float = 0.0,
                      gap: int = 2,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Fill the background region with a still-life tile mosaic.

    This is the project's own pond-tile mosaic used as a backdrop: the subject
    stays the free-form still life that CP-SAT solved, and the field behind it
    is built from the tile bank. No solving happens here.

    Tiles are `6 * level` cells wide and adjacent tiles *overlap*, so the
    safety rule cannot be about individual cells of one tile — it is about the
    tile's whole *support*, the set of cells any tile of the family may fill.
    A lattice site is used only when the support, dilated by `gap`, is
    background and inside the grid. Every mosaic cell then lies at Chebyshev
    distance `gap + 1` or more from any subject cell, so no dead cell's 3x3
    neighbourhood contains both populations: each keeps the neighbour counts
    it was verified with and the union is a still life. The same argument as
    :func:`agar_background`, one shape larger.

    Testing the support rather than the `6 * level` bounding box matters: the
    diamonds fill only 46% of their box at level 6, so a box test rejects
    sites over corners that no tile can ever reach, and leaves wedges of bare
    background between neighbouring diamonds. Square tiles fill 87% of theirs,
    so there the two rules nearly coincide.

    The tile lattice does not have to divide the grid. Sites that do not fit
    simply go unused, which is safe because every subset of the frame lattice
    is a still life.

    Args:
        background_mask: Boolean array, True where the background is. Note
            this is the opposite convention to the beyond-tiles `free_mask`,
            which is True on the subject — pass `~free_mask`.
        level: Tile level, setting tile size (`6 * level` cells) and how
            intricate each tile can be. Diamond 1-6, square 3-5.
        shape: Tile geometry, 'diamond' (the historical pond diamonds) or
            'square' (axis-aligned pond-frame squares)
        density: (low, high) normalised density band tiles are drawn from,
            see :func:`density_band`
        tone: None for a flat random field, or 'linear', 'radial' (dense at
            the centre) or 'vignette' (dense at the edges) for a gradient
        tone_angle: Degrees for tone='linear'; 0 ramps left to right, 90
            bottom to top
        gap: Cells of clearance required around each tile box (default 2)
        seed: Seed for the tile draw, so a composition is reproducible

    Returns:
        Binary uint8 array of the same shape, 1 on mosaic cells. Empty when
        no lattice site fits — the lattice is fixed rather than centred on
        the canvas, so a small or crowded background can seat nothing.

    Raises:
        ValueError: If the mask is not 2D, the shape or tone is unknown, the
            gap is below 2, or the density band selects no tiles

    Example:
        >>> field = mosaic_background(~free_mask, level=4, density=(0.3, 0.7))
        >>> is_still_life(pattern | field)
        True
    """
    from scipy.ndimage import binary_dilation

    from .patterns import PatternLibrary
    from .tile_scheme import assemble, diamond_scheme, pond_square_scheme

    mask = np.asarray(background_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(
            f"Background mask must be 2D array, got shape {mask.shape}"
        )
    if shape not in SHAPES:
        raise ValueError(f"Unknown shape {shape!r}, expected one of {SHAPES}")
    if tone not in TONES:
        raise ValueError(f"Unknown tone {tone!r}, expected one of {TONES}")
    if gap < MIN_GAP:
        raise ValueError(
            f"Gap must be at least {MIN_GAP} to keep the mosaic clear of the "
            f"subject, got {gap}"
        )

    build = diamond_scheme if shape == 'diamond' else pond_square_scheme
    scheme = build(level)
    tiles = np.asarray(PatternLibrary.load(level, shape=shape).solutions,
                       dtype=np.uint8)
    indices, _ = density_band(level, shape, density)

    height, width = mask.shape
    n = scheme.n
    (u_i, u_j), (v_i, v_j) = scheme.u, scheme.v

    # `assemble` writes only inside the support, so that — dilated by `gap` —
    # is the whole footprint a site has to keep clear. Pad before dilating:
    # the support runs to the box's interior ring, and an unpadded dilation is
    # clipped there, leaving the outermost cell of clearance untested. That
    # failure is silent in the render and shows up as a birth beside a tile
    # edge, nowhere near the line that caused it.
    support = scheme.support | scheme.frame
    padded = np.zeros((n + 2 * gap, n + 2 * gap), dtype=bool)
    padded[gap:gap + n, gap:gap + n] = support
    offsets = np.argwhere(
        binary_dilation(padded, np.ones((2 * gap + 1,) * 2, dtype=bool))) - gap
    off_i, off_j = offsets[:, 0], offsets[:, 1]

    # Everything off the grid counts as blocked, which enforces the border
    # rule for free: a footprint reaching the edge simply fails the test.
    edge = 2 * n
    blocked = np.ones((height + 2 * edge, width + 2 * edge), dtype=bool)
    blocked[edge:edge + height, edge:edge + width] = ~mask

    # Lattice sites (a, b) sit at a*u + b*v. Invert the basis and map the
    # canvas corners back to get a tight (a, b) rectangle; a loose symmetric
    # range would make `assemble` allocate a scratch grid of ~100 MB at
    # poster size.
    det = u_i * v_j - v_i * u_j
    ab = [((i * v_j - j * v_i) / det, (j * u_i - i * u_j) / det)
          for i in (-n, height) for j in (-n, width)]
    a_lo = int(np.floor(min(a for a, _ in ab)))
    a_hi = int(np.ceil(max(a for a, _ in ab)))
    b_lo = int(np.floor(min(b for _, b in ab)))
    b_hi = int(np.ceil(max(b for _, b in ab)))

    index_grid = np.full((a_hi - a_lo + 1, b_hi - b_lo + 1), -1, dtype=np.int64)
    rows, cols, corner_i, corner_j = [], [], [], []
    # Per-site loop; the cheap integer bounds test rejects the sites that lie
    # wholly off the canvas before any indexing, so a 1416x2000 poster costs
    # well under a second even at level 6.
    for a in range(a_lo, a_hi + 1):
        for b in range(b_lo, b_hi + 1):
            ci = a * u_i + b * v_i
            cj = a * u_j + b * v_j
            if not (-n <= ci <= height and -n <= cj <= width):
                continue
            if not blocked[ci + edge + off_i, cj + edge + off_j].any():
                rows.append(a - a_lo)
                cols.append(b - b_lo)
                corner_i.append(ci)
                corner_j.append(cj)

    if rows:
        rng = np.random.default_rng(seed)
        if tone is None:
            chosen = rng.choice(indices, size=len(rows))
        else:
            field = _tone_field(tone, tone_angle, height, width)
            middle = n // 2
            wanted = field[np.asarray(corner_i) + middle,
                           np.asarray(corner_j) + middle]
            low, high = float(density[0]), float(density[1])
            chosen = _draw_by_density(
                PatternLibrary.load(level, shape=shape).densities, indices,
                low + wanted * (high - low), rng)
        index_grid[rows, cols] = chosen

    origin_i = min(a_lo * u_i, a_hi * u_i) + min(b_lo * v_i, b_hi * v_i)
    origin_j = min(a_lo * u_j, a_hi * u_j) + min(b_lo * v_j, b_hi * v_j)
    pad = 2
    grid = assemble(scheme, index_grid, tiles, pad=pad)
    top, left = pad - origin_i, pad - origin_j
    field = grid[top:top + height, left:left + width]
    # Guards the offset arithmetic above; a sign error shows up here, not as
    # a subtly misplaced background.
    assert field.shape == mask.shape, (
        f"cropped mosaic {field.shape} does not match mask {mask.shape}"
    )
    return np.ascontiguousarray(field)


def _draw_by_density(normalised: np.ndarray,
                     indices: np.ndarray,
                     wanted: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Pick a random band tile of the density nearest each wanted value.

    Tile densities are quantised (live cells over a fixed box), so many tiles
    share a value: matching the nearest one and then drawing uniformly among
    its ties gives a graded field that still varies from site to site. Working
    on the sorted unique values also avoids the (n_sites, n_tiles) difference
    matrix `PatternLibrary.get_indices_for_values` builds, which is 1.6 GB
    against the 10,398-tile level-5 square bank at poster size.
    """
    order = indices[np.argsort(normalised[indices], kind='stable')]
    values, first = np.unique(normalised[order], return_index=True)
    sizes = np.diff(np.append(first, len(order)))

    right = np.clip(np.searchsorted(values, wanted), 0, len(values) - 1)
    left = np.clip(right - 1, 0, len(values) - 1)
    nearest = np.where(
        np.abs(values[left] - wanted) <= np.abs(values[right] - wanted),
        left, right)
    offset = (rng.random(len(wanted)) * sizes[nearest]).astype(np.int64)
    return order[first[nearest] + np.minimum(offset, sizes[nearest] - 1)]


def scatter_background(background_mask: np.ndarray,
                       occupied: Optional[np.ndarray] = None,
                       gap: int = 2,
                       band: Optional[int] = None,
                       fade: bool = True,
                       shapes: Optional[dict] = None,
                       seed: Optional[int] = None) -> np.ndarray:
    """
    Scatter loose elementary still lifes wherever they fit.

    Off any lattice, which is the point: :func:`mosaic_background` and
    :func:`agar_background` can only place on their own grid, so the last few
    cells next to a ragged silhouette are unreachable to both. A greedy scan
    over shuffled candidate positions reaches them.

    A shape is accepted at `(i, j)` only when its whole box plus `gap` cells
    of margin is background, inside the grid, and free of every cell already
    placed — by this call or by `occupied`. Each placed cell is then at
    Chebyshev distance `gap + 1` or more from the subject and from every other
    population, so no dead cell's 3x3 neighbourhood sees two of them: each
    keeps the neighbour counts it was verified with and the union is a still
    life. Same argument as :func:`agar_background`, with the lattice dropped.

    Args:
        background_mask: Boolean array, True where the background is. Note
            this is the opposite convention to the beyond-tiles `free_mask`,
            which is True on the subject — pass `~free_mask`.
        occupied: Cells already taken by another background layer, kept clear
            by the same `gap`. None treats the background as empty.
        gap: Cells of clearance required around each shape (default 2)
        band: Only scatter within this Chebyshev distance of the subject, so
            the filler reads as a fringe rather than as snow over the whole
            field. Measured at each shape's anchor cell, so a shape can reach
            up to its own size (at most 4 cells) past the band. None scatters
            everywhere, as does a mask holding no subject to measure from.
        fade: With a `band`, thin the scatter out towards its far edge, so the
            filler dissolves instead of stopping at a visible line. Ignored
            when there is no band — there is no scale to fade over.
        shapes: Name -> binary array of still lifes to draw from (default
            :data:`ELEMENTARY`). Every shape is used in all eight
            orientations.
        seed: Seed for the draw, so a composition is reproducible

    Returns:
        Binary uint8 array of the same shape, 1 on scattered cells

    Raises:
        ValueError: If the mask is not 2D, `occupied` does not match it, the
            gap is below 2, or `shapes` is empty

    Example:
        >>> loose = scatter_background(~free_mask, occupied=tiles, band=18)
        >>> is_still_life(pattern | tiles | loose)
        True
    """
    mask = np.asarray(background_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(
            f"Background mask must be 2D array, got shape {mask.shape}"
        )
    if gap < MIN_GAP:
        raise ValueError(
            f"Gap must be at least {MIN_GAP} to keep the filler clear of the "
            f"subject, got {gap}"
        )

    taken = (np.zeros(mask.shape, dtype=bool) if occupied is None
             else np.asarray(occupied).astype(bool))
    if taken.shape != mask.shape:
        raise ValueError(
            f"Occupied shape {taken.shape} does not match mask {mask.shape}"
        )

    shapes = ELEMENTARY if shapes is None else shapes
    if not shapes:
        raise ValueError("No shapes to scatter")
    # All eight orientations up front: the chiral shapes (boat, ship, loaf)
    # would otherwise repeat as one glyph across the whole fringe. Duplicates
    # from the symmetric shapes are harmless, they only weight the draw.
    variants = [np.ascontiguousarray(np.rot90(tile[::-1] if flip else tile, k))
                for tile in (np.asarray(t, dtype=np.uint8)
                             for t in shapes.values())
                for flip in (False, True) for k in range(4)]

    from scipy.ndimage import binary_erosion

    height, width = mask.shape
    room = mask & ~taken
    distance = _subject_distance(mask)
    # A mask with no subject has no silhouette to hug, so the band means
    # nothing there and is dropped rather than emptying the field.
    graded = band is not None and not mask.all()
    if graded:
        room &= distance <= band

    # Drop the anchors where even the smallest shape cannot fit, before
    # building the candidate list. The 2x2 block needs a clear `2 + 2 * gap`
    # box; scipy's border_value of 0 rejects a box running off the grid,
    # which is the same call the loop's own bounds tests make. Without this
    # an unbanded scatter walks every background cell — over a million of
    # them on a 1416x2000 poster, for a thousand-odd placements.
    side = BLOCK + 2 * gap
    room &= binary_erosion(room, np.ones((side, side), dtype=bool),
                           origin=(-(side // 2 - gap),) * 2)

    candidates = np.argwhere(room)
    field = np.zeros((height, width), dtype=np.uint8)
    if not candidates.size:
        return field

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    # Random draws vectorised up front: at poster size the candidate list runs
    # to a million rows, and three rng calls inside the loop dominate it.
    pick = rng.integers(len(variants), size=len(candidates))
    keep = None
    if graded and fade:
        # Acceptance falls from 1 at the silhouette to a floor at the far edge
        # of the band, so the fringe dissolves instead of ending on a line.
        far = distance[candidates[:, 0], candidates[:, 1]]
        keep = rng.random(len(candidates)) <= np.clip(
            1.0 - (far - gap) / float(band), 0.05, 1.0)

    for n, (i, j) in enumerate(candidates):
        if keep is not None and not keep[n]:
            continue
        tile = variants[pick[n]]
        h, w = tile.shape
        # Bounds before slicing. A negative start must not reach the indexing:
        # numpy would wrap it silently and .all() could then pass on a box
        # running off the edge.
        if i - gap < 0 or j - gap < 0:
            continue
        if i + h + gap > height or j + w + gap > width:
            continue
        rows = slice(i - gap, i + h + gap)
        cols = slice(j - gap, j + w + gap)
        if not mask[rows, cols].all():
            continue
        if field[rows, cols].any() or taken[rows, cols].any():
            continue
        field[i:i + h, j:j + w] = tile

    return field


def _subject_distance(mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance from each background cell to the nearest subject cell.

    An all-background mask has nothing to measure from — the transform would
    have no zero to find — so every cell is infinitely far from a subject that
    is not there.
    """
    if mask.all():
        return np.full(mask.shape, np.inf)

    from scipy.ndimage import distance_transform_cdt

    return distance_transform_cdt(mask, metric='chessboard').astype(np.float64)


def filled_background(background_mask: np.ndarray,
                      level: int = 4,
                      shape: str = 'diamond',
                      *,
                      density: Tuple[float, float] = (0.0, 1.0),
                      tone: Optional[str] = None,
                      tone_angle: float = 0.0,
                      fill: Optional[object] = 'auto',
                      fill_band: Optional[int] = None,
                      fill_fade: bool = True,
                      gap: int = 2,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    A tile mosaic with the gap around the subject packed as tight as it goes.

    :func:`mosaic_background` seats a tile only where its whole `6 * level`
    box fits, so the halo between the subject and the nearest tile is set by
    where the lattice falls rather than by the subject's outline. Measured on
    an 800x200 banner, that halo runs to 54 cells at level 4 and 73 at level
    6, and the variation is what reads as untidy — not the width.

    This runs the same generator again at smaller levels, each pass seeing
    only what the previous ones left, and finishes with
    :func:`scatter_background` for the crevices no lattice reaches. On the
    same banner that brings the worst halo to 7 cells. It cannot go below
    `gap + 1`: that clearance is what makes the union a still life.

    Filler levels always draw from the full density band and ignore `tone`.
    The graded band belongs to the tiles that carry the image; a level-1 bank
    holds a single tile, so a narrow band there would select nothing at all.

    Args:
        background_mask: Boolean array, True where the background is. Note
            this is the opposite convention to the beyond-tiles `free_mask`,
            which is True on the subject — pass `~free_mask`.
        level: Tile level of the main mosaic, see :func:`mosaic_background`
        shape: Tile geometry, 'diamond' or 'square'
        density: (low, high) normalised band for the main mosaic
        tone: Density gradient for the main mosaic: None, 'linear', 'radial'
            or 'vignette'
        tone_angle: Degrees for tone='linear'
        fill: 'auto' cascades through every smaller level of the same shape
            and then scatters; a sequence names the levels to use; `()`
            scatters only; None returns the plain mosaic
        fill_band: Scatter only within this distance of the subject, see
            :func:`scatter_background`. None, the default, scatters wherever
            a shape fits
        fill_fade: Thin the scatter towards the far edge of that band
        gap: Cells of clearance required around every box (default 2)
        seed: Seed for the draws; each layer offsets it, so a composition is
            reproducible as a whole

    Returns:
        uint8 array of the same shape, one number per layer: 0 empty, 1 a
        main-mosaic cell, 2 the first cascade level, and so on down to
        `len(levels) + 2` for the scatter — which keeps that number whether or
        not it placed anything. `field != 0` is the pattern; the numbering
        exists so a renderer can grade the layers by tile size.

    Raises:
        ValueError: Whatever :func:`mosaic_background` and
            :func:`scatter_background` raise, plus a `fill` that is neither
            'auto', None, nor a sequence of levels

    Example:
        >>> field = filled_background(~free_mask, level=4)
        >>> is_still_life(pattern | (field != 0))
        True
    """
    mask = np.asarray(background_mask, dtype=bool)
    tiles = mosaic_background(mask, level=level, shape=shape, density=density,
                              tone=tone, tone_angle=tone_angle, gap=gap,
                              seed=seed)
    if fill is None:
        return tiles

    levels = _resolve_fill_levels(level, shape, fill)

    field = tiles.astype(np.uint8)
    placed = tiles.astype(bool)
    for step, small in enumerate(levels, start=1):
        # Handing each pass `mask & ~placed` is the whole trick: the support
        # rule then demands the footprint plus `gap` be free of every cell
        # already down, which is exactly the clearance the argument needs.
        layer = mosaic_background(mask & ~placed, level=small, shape=shape,
                                  gap=gap,
                                  seed=None if seed is None else seed + step)
        fresh = layer.astype(bool) & ~placed
        field[fresh] = step + 1
        placed |= fresh

    loose = scatter_background(
        mask, occupied=placed, gap=gap, band=fill_band, fade=fill_fade,
        seed=None if seed is None else seed + len(levels) + 1)
    # The scatter keeps its number whether or not it placed anything, so the
    # renderer's ramp spans the same range every time.
    field[loose.astype(bool)] = len(levels) + 2
    return field


def _fill_levels(level: int, shape: str) -> Tuple[int, ...]:
    """Every smaller level of the same shape, largest first.

    Square tiles are only enumerated for levels 3-5, so that cascade bottoms
    out at 18 cells and the scatter does the rest.
    """
    if shape == 'square':
        return tuple(small for small in (4, 3) if small < level)
    return tuple(range(level - 1, 0, -1))


def _resolve_fill_levels(level: int,
                         shape: str,
                         fill: Optional[object]) -> Optional[Tuple[int, ...]]:
    """The cascade's levels, largest first, or None for no cascade at all.

    Shared by :func:`filled_background` and :func:`compose` so both agree on
    how many layers a field has — the renderer's colour ramp is spread over
    that count, and recovering it from the field itself would go wrong on a
    layer that happened to place nothing.
    """
    if fill is None:
        return None
    if fill == 'auto':
        return _fill_levels(level, shape)
    if isinstance(fill, str):
        raise ValueError(
            f"Unknown fill {fill!r}, expected 'auto', None or a sequence of "
            f"levels"
        )
    return tuple(int(small) for small in fill)


def fill_layer_count(level: int,
                     shape: str = 'diamond',
                     fill: Optional[object] = 'auto') -> int:
    """How many layers :func:`filled_background` numbers, cascade included.

    1 for the main mosaic alone, otherwise the main mosaic plus one per
    cascade level plus one for the scatter.

    Example:
        >>> fill_layer_count(6, 'diamond', 'auto')
        7
    """
    levels = _resolve_fill_levels(level, shape, fill)
    return 1 if levels is None else len(levels) + 2


def life_safe_pattern(pattern: np.ndarray,
                      background_mask: np.ndarray,
                      pitch: Tuple[int, int] = (3, 4),
                      gap: int = 2,
                      field: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Merge a still-life background into a pattern.

    Unlike the painted backdrops, the result is one Game of Life pattern: the
    whole grid, subject and background together, is a still life and can be
    exported to Golly as a single object.

    Args:
        pattern: Binary array holding the subject still life
        background_mask: Boolean array, True where the background is
        pitch: Agar lattice period, see :func:`agar_background`
        gap: Clearance around each background cell, see :func:`agar_background`
        field: Background cells to merge in. None grows the block agar; pass
            the output of :func:`mosaic_background`, :func:`filled_background`
            or :func:`scatter_background` for a tile field. Any non-zero value
            counts as a live cell, so `filled_background`'s layer numbering is
            accepted as-is. Any field is checked for clearance rather than
            trusted — that check is the only thing standing between a caller
            and a false still-life claim.

    Returns:
        Binary uint8 array of the union

    Raises:
        ValueError: If the pattern has live cells inside the background
            region, or the given field comes within `gap` of the subject or
            the border

    Example:
        >>> whole = life_safe_pattern(pattern, ~free_mask)
        >>> GollyExporter.export_to_cells(whole, 'art.cells')

        >>> tiles = filled_background(~free_mask, level=4)
        >>> whole = life_safe_pattern(pattern, ~free_mask, field=tiles)
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

    if field is None:
        return pattern | agar_background(mask, pitch=pitch, gap=gap)

    from scipy.ndimage import binary_dilation

    # Any non-zero value is a live cell: `filled_background` numbers its
    # layers 1 and 2, and `pattern | field` would otherwise carry those
    # numbers straight into the returned pattern.
    field = (np.asarray(field) != 0).astype(np.uint8)
    if field.shape != pattern.shape:
        raise ValueError(
            f"Field shape {field.shape} does not match pattern {pattern.shape}"
        )

    near = binary_dilation(pattern.astype(bool),
                           structure=np.ones((2 * gap + 1,) * 2, dtype=bool))
    crowding = int((field.astype(bool) & near).sum())
    if crowding:
        raise ValueError(
            f"Field has {crowding} cells within {gap} of the subject; the "
            f"still-life guarantee needs a Chebyshev clearance of {gap + 1}"
        )

    # The toroidal half of the guarantee needs the same clearance at the wrap.
    edge = np.ones(field.shape, dtype=bool)
    edge[gap:-gap, gap:-gap] = False
    if field[edge].any():
        raise ValueError(
            f"Field reaches within {gap} cells of the grid border, which "
            f"breaks stability across the toroidal wrap"
        )

    return pattern | field


def compose(pattern: np.ndarray,
            background_mask: np.ndarray,
            scheme: Optional[ColorScheme] = None,
            style: str = 'eca',
            *,
            rule: Optional[int] = None,
            supersample: int = 15,
            pitch: Tuple[int, int] = (3, 4),
            gap: int = 2,
            level: int = 4,
            shape: str = 'diamond',
            density: Tuple[float, float] = (0.0, 1.0),
            tone: Optional[str] = None,
            tone_angle: float = 0.0,
            fill: Optional[object] = None,
            fill_band: Optional[int] = None,
            fill_fade: bool = True,
            seed: Optional[int] = None,
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
        style: One of 'none' (transparent), 'flat', 'eca', 'agar', 'mosaic'
        rule: ECA rule for style='eca'. None picks an interesting rule at
            random, as the mosaic pipeline does.
        supersample: ECA cell side in pixels for style='eca' (default 15)
        pitch: Agar lattice period for style='agar', see
            :func:`agar_background`
        gap: Clearance for style='agar' and style='mosaic', see
            :func:`agar_background`
        level: Tile level for style='mosaic', see :func:`mosaic_background`
        shape: Tile geometry for style='mosaic', 'diamond' or 'square'
        density: Normalised tile density band for style='mosaic', see
            :func:`density_band`
        tone: Density gradient for style='mosaic': None, 'linear', 'radial'
            or 'vignette'
        tone_angle: Degrees for tone='linear'
        fill: Gap filler for style='mosaic'. None leaves the plain mosaic and
            its ragged halo; 'auto' packs the gap with smaller tiles and loose
            still lifes, see :func:`filled_background`. Each filler level
            is painted a step further along a ramp from the scheme's
            `eca_pixel` to its `fill` colour.
        fill_band: How far from the subject the loose still lifes reach;
            None, the default, lets them reach everywhere
        fill_fade: Thin those out towards the far edge of the band
        seed: Tile draw seed for style='mosaic', so a composition is
            reproducible
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
        elif style == 'agar':
            field = agar_background(mask, pitch=pitch, gap=gap)
        else:
            # fill=None makes this exactly mosaic_background, so one call
            # covers both.
            field = filled_background(
                mask, level=level, shape=shape, density=density, tone=tone,
                tone_angle=tone_angle, fill=fill, fill_band=fill_band,
                fill_fade=fill_fade, gap=gap, seed=seed
            )

        # Layer stack: 0=subject (transparent), 1=field background, 2=field
        # pixel. Same construction the tile mosaics use. A filled mosaic
        # numbers its layers 1..n, which land on 2..n+1 here and pick up the
        # renderer's filler ramp — no extra arithmetic needed. The layer count
        # is passed rather than read back off the field, so an empty layer
        # cannot shorten the ramp.
        layers = (fill_layer_count(level, shape, fill)
                  if style == 'mosaic' else None)
        backdrop = mask.astype(np.uint8)
        image = renderer.render_full_mosaic(
            pattern, backdrop * (field + backdrop), layers=layers
        )

    if scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.NEAREST
        )
    return image

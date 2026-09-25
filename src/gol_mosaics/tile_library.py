"""
The tile databases and density matching.

TileLibrary loads the still-life tiles shipped with the package (enumerated
with SAT, gol_mosaics.sat_search) and maps grey values to tiles of matching
live-cell density. The original Gurobi ILP generator is
gol_mosaics.legacy_ilp.

Vocabulary: a *pond* is the 4x4 still life the frames are built from, a
*tile* is one still life inside a level's pond frame, and a *mosaic* is
tiles assembled on a lattice.
"""

import logging
from functools import lru_cache
from importlib.resources import files
from typing import Optional

import numpy as np
from scipy.ndimage import binary_fill_holes

from . import tile_domain
from .tile_domain import build_domain
from .tile_scheme import build_scheme_domain, diamond_scheme, square_scheme

logger = logging.getLogger(__name__)


def nearest_density_indices(densities: np.ndarray,
                            wanted: np.ndarray,
                            rng=None,
                            random: bool = True,
                            candidates: Optional[np.ndarray] = None
                            ) -> np.ndarray:
    """Index of a tile whose density is nearest each wanted value.

    Tile densities are quantised (live cells over a fixed box), so many tiles
    share a value. Working on the sorted unique values costs O(n log n) once
    plus O(log u) per wanted value, and allocates nothing of size
    (len(wanted), len(densities)): the dense difference matrix this replaces
    ran to about 14 GB for a 60-tile grid at level 6 (332,321 tiles).

    Args:
        densities: (N,) density per tile
        wanted: Values to match, any shape
        rng: Source of uniform draws with a ``random(size)`` method, such as
            a ``np.random.Generator``. None uses numpy's global RNG, so
            ``np.random.seed`` makes the draw reproducible.
        random: Draw uniformly among the tiles sharing the nearest density.
            When two density values are equally near, the lower one wins.
            False returns the lowest-index tile among all nearest ones, as
            ``argmin`` over the full difference matrix would.
        candidates: Ascending tile indices to choose from (default: all)

    Returns:
        int64 array of tile indices, shaped like `wanted`
    """
    densities = np.asarray(densities)
    wanted = np.asarray(wanted, dtype=float)
    shape = wanted.shape
    wanted = wanted.ravel()
    if candidates is None:
        candidates = np.arange(len(densities))
    candidates = np.asarray(candidates)

    order = candidates[np.argsort(densities[candidates], kind='stable')]
    values, first = np.unique(densities[order], return_index=True)
    sizes = np.diff(np.append(first, len(order)))

    right = np.clip(np.searchsorted(values, wanted), 0, len(values) - 1)
    left = np.clip(right - 1, 0, len(values) - 1)
    left_gap = np.abs(values[left] - wanted)
    right_gap = np.abs(values[right] - wanted)

    if random:
        if rng is None:
            rng = np.random
        nearest = np.where(left_gap <= right_gap, left, right)
        offset = (rng.random(len(wanted)) * sizes[nearest]).astype(np.int64)
        chosen = order[first[nearest] + np.minimum(offset, sizes[nearest] - 1)]
    else:
        # Within a group the stable sort keeps indices ascending, so a
        # group's first entry is its lowest index. An exact tie between two
        # values takes the lower index of the two groups.
        lowest = order[first]
        chosen = np.where(left_gap < right_gap, lowest[left],
                          np.where(right_gap < left_gap, lowest[right],
                                   np.minimum(lowest[left], lowest[right])))
    return chosen.astype(np.int64).reshape(shape)


LAYOUTS = ("diamond", "square")

#: Levels shipped in gol_mosaics/data/ per layout. The square level-6 census
#: (19,287,185 tiles) is too large to ship; square level 2 is a single fully
#: forced tile.
SHIPPED_LEVELS = {"diamond": (1, 2, 3, 4, 5, 6), "square": (3, 4, 5)}


def _check_layout(layout: str) -> None:
    if layout not in LAYOUTS:
        raise ValueError(
            f"Unknown layout {layout!r}; expected 'diamond' or 'square'."
        )


class TileLibrary:
    """
    The still-life tiles of one level and layout, and density matching.

    A *tile* is one still life inside the tile's pond frame; a library holds
    every tile of a level (the census), loaded lazily from the packed data
    shipped with the package, and maps grey values to tiles of matching
    live-cell density.

    Attributes:
        level: Tile level; tiles are 6*level cells wide
        layout: 'diamond' (tiles on two interlocking 45-degree grids) or
            'square' (axis-aligned tiles that share their border ponds)

    Example:
        >>> library = TileLibrary.load(level=5)
        >>> tile = library.tile_for_value(0.5)
        >>> tiles = library.tiles_for_values(np.array([0.2, 0.5, 0.8]))
    """

    def __init__(self, level: int = 4, layout: str = "diamond"):
        """
        An empty library; fill it with :meth:`load` or :meth:`from_tiles`.

        Args:
            level: Tile level (positive integer)
            layout: 'diamond' or 'square'

        Raises:
            ValueError: If level is smaller than 1 or the layout is unknown
        """
        if level < 1:
            raise ValueError(f"Level must be a positive integer, got {level}.")
        _check_layout(layout)
        self.level = level
        self.layout = layout
        self._tiles: Optional[np.ndarray] = None
        self._densities: Optional[np.ndarray] = None
        self._filled_tile: Optional[np.ndarray] = None
        self._scheme = None

    @classmethod
    def load(cls, level: int, layout: str = "diamond") -> 'TileLibrary':
        """
        The shipped tiles of one level.

        Diamond tiles ship for levels 1-6, square tiles for levels 3-5, all
        as packed symmetry-orbit bits expanded to uint8 grids on load; level
        6 (332,321 tiles) takes about 0.5 s and 430 MB.

        Args:
            level: Tile level (diamond: 1-6, square: 3-5)
            layout: 'diamond' or 'square'

        Returns:
            The library. Instances are cached and shared per (level, layout),
            so treat the library and its arrays as read-only.

        Raises:
            ValueError: If the level is not shipped for the layout
            FileNotFoundError: If the data file is missing

        Example:
            >>> library = TileLibrary.load(level=5)
            >>> len(library.tiles)
            2632
        """
        _check_layout(layout)
        if level not in SHIPPED_LEVELS[layout]:
            shipped = SHIPPED_LEVELS[layout]
            raise ValueError(
                f"{layout.capitalize()} tiles ship for levels {shipped[0]}-"
                f"{shipped[-1]}, got level={level}. Higher levels can be "
                f"enumerated with gol_mosaics.sat_search and wrapped with "
                f"TileLibrary.from_tiles()."
            )
        return _load_tile_library(level, layout)

    @classmethod
    def from_tiles(cls, tiles: np.ndarray, level: int,
                   layout: str = "diamond") -> 'TileLibrary':
        """
        A library around tiles computed elsewhere, e.g. with
        gol_mosaics.sat_search.enumerate_tiles or legacy_ilp.generate_tiles.

        Args:
            tiles: (N, 6*level, 6*level) 0/1 array
            level: Their level
            layout: Their layout
        """
        library = cls(level=level, layout=layout)
        library._tiles = np.asarray(tiles, dtype=np.uint8)
        return library

    @property
    def tiles(self) -> np.ndarray:
        """(N, H, W) uint8 array of the library's tiles."""
        if self._tiles is None:
            raise ValueError(
                "No tiles yet: use TileLibrary.load() or TileLibrary.from_tiles()."
            )
        return self._tiles

    @property
    def densities(self) -> np.ndarray:
        """
        Live-cell density of each tile, min-max normalised to [0, 1] per
        library (computed once).
        """
        if self._densities is None:
            densities = np.mean(self.tiles, axis=(1, 2))
            low, high = densities.min(), densities.max()
            if high == low:
                # Degenerate case (e.g. the one-tile level-1 library): every
                # tile shares one density, so give each the same value.
                self._densities = np.ones(len(densities))
            else:
                self._densities = (densities - low) / (high - low)
        return self._densities

    @property
    def scheme(self):
        """The TileScheme of the layout: its lattice drives mosaic assembly."""
        if self._scheme is None:
            build = diamond_scheme if self.layout == "diamond" else square_scheme
            self._scheme = build(self.level)
        return self._scheme

    @property
    def pond(self) -> np.ndarray:
        """The pond frame every tile shares (forced-alive cells, 0/1)."""
        return self.scheme.frame.astype(np.uint8)

    @property
    def tile_size(self) -> tuple:
        """(height, width) of one tile's bounding box."""
        n = tile_domain.POND_WIDTH * self.level
        return (n, n)

    @property
    def lattice_offset(self) -> int:
        """
        Offset in cells between the two interlocking diagonal grids of the
        diamond layout (half a tile): each grid is shifted by this amount,
        one horizontally and one vertically, so the tiles of one sit in the
        gaps of the other.
        """
        if self.layout != "diamond":
            raise ValueError(
                "lattice_offset is diamond-lattice arithmetic; square "
                "libraries place tiles through their scheme."
            )
        return (tile_domain.POND_WIDTH * self.level) // 2

    def tile_for_value(self,
                       value: float,
                       random: bool = True,
                       invert: bool = True) -> np.ndarray:
        """
        One tile matching a grey value.

        Args:
            value: Grey value in [0, 1]
            random: Draw at random among the tiles of the nearest density
                (otherwise the lowest-index one)
            invert: Map 0 (black) to the densest tiles

        Raises:
            ValueError: If value is not in [0, 1]

        Example:
            >>> TileLibrary.load(level=4).tile_for_value(0.5).shape
            (24, 24)
        """
        if value < 0 or value > 1:
            raise ValueError(f"Value must be in [0, 1], got {value}")
        wanted = 1.0 - value if invert else value
        return self.tiles[int(nearest_density_indices(self.densities, wanted,
                                                      random=random))]

    def tiles_for_values(self,
                         greyscale_values: np.ndarray,
                         random: bool = True,
                         invert: bool = True,
                         empty_tiles_cutoff: float = 1.0) -> np.ndarray:
        """
        Map grey values to tiles by density.

        Args:
            greyscale_values: Array of grey values in [0, 1]
            random: Draw at random among the tiles of the nearest density
            invert: Map 0 (black) to the densest tiles
            empty_tiles_cutoff: Values above this become empty (all-dead) tiles

        Returns:
            Array of tiles with shape (*greyscale_values.shape, H, W)

        Raises:
            ValueError: If any value is outside [0, 1]

        Example:
            >>> library = TileLibrary.load(level=4)
            >>> library.tiles_for_values(np.array([[0.2, 0.5], [0.7, 0.9]])).shape
            (2, 2, 24, 24)
        """
        tiles = self.tiles
        indices = self.indices_for_values(
            greyscale_values, random=random, invert=invert,
            empty_tiles_cutoff=empty_tiles_cutoff)
        flat = indices.ravel()
        chosen = tiles[np.clip(flat, 0, None)]
        chosen[flat < 0] = 0
        return chosen.reshape(indices.shape + tiles.shape[1:])

    def indices_for_values(self,
                           greyscale_values: np.ndarray,
                           random: bool = True,
                           invert: bool = True,
                           empty_tiles_cutoff: float = 1.0) -> np.ndarray:
        """
        Map grey values to tile indices by density.

        The selection half of :meth:`tiles_for_values`: -1 marks an empty
        tile (value above the cutoff), any other entry indexes `tiles`.
        Lattice assemblers that place tiles by index (the square layout)
        consume this directly.

        Args:
            greyscale_values: Array of grey values in [0, 1]
            random: Draw at random among the tiles of the nearest density
            invert: Map 0 (black) to the densest tiles
            empty_tiles_cutoff: Values above this become empty (-1)

        Returns:
            int64 array with the shape of greyscale_values

        Raises:
            ValueError: If any value is outside [0, 1]
        """
        greyscale_values = np.asarray(greyscale_values, dtype=float)

        flat = greyscale_values.ravel()
        if flat.size and (flat.min() < 0 or flat.max() > 1):
            bad = flat[(flat < 0) | (flat > 1)][0]
            raise ValueError(f"Greyscale value must be in [0, 1], got {bad}")

        if empty_tiles_cutoff <= 0:
            # Every value sits above the cutoff: all tiles are empty.
            return np.full(greyscale_values.shape, -1, dtype=np.int64)

        empty = flat > empty_tiles_cutoff
        adjusted = flat / empty_tiles_cutoff
        # Invert mapping: black (0) -> dense (1), white (1) -> sparse (0)
        if invert:
            adjusted = 1.0 - adjusted

        indices = nearest_density_indices(self.densities, adjusted,
                                          random=random)
        indices[empty] = -1
        return indices.reshape(greyscale_values.shape)

    def tiles_for_mask(self,
                       mask: np.ndarray,
                       alpha_cutoff: float = 0.5) -> np.ndarray:
        """
        Map an alpha mask to a per-tile background mask.

        Sites whose alpha is below alpha_cutoff (transparent: background)
        get a solid diamond, 1 on every cell of the tile's support; opaque
        sites get an empty tile. The diamond layout assembles these into the
        mask of where the background is drawn. Any tile filled in gives the
        same solid diamond, because the frame encloses the whole support.

        Args:
            mask: Array of alpha values in [0, 1]
            alpha_cutoff: Threshold for transparency

        Returns:
            Array with shape (*mask.shape, H, W), 1 on background cells

        Raises:
            ValueError: If any value is outside [0, 1]
        """
        mask = np.asarray(mask, dtype=float)
        tiles = self.tiles

        flat = mask.ravel()
        if flat.size and (flat.min() < 0 or flat.max() > 1):
            bad = flat[(flat < 0) | (flat > 1)][0]
            raise ValueError(f"Mask value must be in [0, 1], got {bad}")

        if self._filled_tile is None:
            self._filled_tile = binary_fill_holes(tiles[-1]).astype(tiles.dtype)
        empty_tile = np.zeros_like(tiles[0])

        opaque = (flat >= alpha_cutoff)[:, None, None]
        chosen = np.where(opaque, empty_tile, self._filled_tile)
        return chosen.reshape(mask.shape + tiles.shape[1:]).astype(
            tiles.dtype, copy=False)

    def __repr__(self) -> str:
        count = "not loaded" if self._tiles is None else f"{len(self._tiles)} tiles"
        return f"TileLibrary(level={self.level}, layout={self.layout!r}, {count})"


@lru_cache(maxsize=None)
def _load_tile_library(level: int, layout: str = "diamond") -> TileLibrary:
    """
    Load and cache one shared TileLibrary per (level, layout).

    Every database ships inside the package (gol_mosaics/data/) as packed
    free-orbit bits, one bit per free symmetry orbit per tile, and is
    expanded to full uint8 grids here. importlib.resources finds the file
    wherever the package is installed. Cached instances are shared between
    callers and must be treated as read-only.
    """
    library = TileLibrary(level=level, layout=layout)
    name = f"tiles_{layout}_level_{level}_orbits.npy"
    resource = files(__package__).joinpath("data", name)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Tile data file not found: {resource}\n"
            f"Expected packaged resource: gol_mosaics/data/{name}"
        )
    with resource.open("rb") as f:
        packed = np.load(f)
    domain = (build_scheme_domain(library.scheme) if layout == "square"
              else build_domain(level))
    library._tiles = domain.unpack(packed)
    return library

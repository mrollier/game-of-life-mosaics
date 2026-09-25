"""
Game of Life pattern generation and management.

This module provides the PatternLibrary class, which loads the shipped
symmetric still-life tiles and maps grey values to tiles by density. The
tiles are enumerated with SAT (gol_mosaics.sat_search); the original Gurobi
ILP generator lives in gol_mosaics.legacy_ilp.
"""

import logging
import numpy as np
from functools import lru_cache
from importlib.resources import files
from typing import Optional
from scipy.ndimage import binary_fill_holes

from . import tile_domain
from .tile_domain import derive_dead_edges, unpack_solutions
from .tile_scheme import pond_square_scheme, unpack_scheme_solutions

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


class PatternLibrary:
    """
    Manages Game of Life still-life patterns.

    Loads the pre-computed symmetric patterns and maps grey values to them
    by density. Patterns are loaded lazily to minimise memory usage.

    Attributes:
        level: Pattern complexity level (1-5 pre-computed, others require generation)
        pond_width: Fixed width parameter (6) for pattern generation

    Example:
        >>> # Load pre-computed patterns
        >>> library = PatternLibrary.load(level=5)
        >>> pattern = library.get_pattern_for_value(0.5)

        >>> # Map multiple values to patterns
        >>> values = np.array([0.2, 0.5, 0.8])
        >>> patterns = library.get_patterns_for_values(values)
    """

    def __init__(self, level: int = 4, shape: str = "diamond"):
        """
        Initialise PatternLibrary.

        Levels 1-5 are pre-computed and can be loaded with load(); higher
        levels can be constructed here but must be filled via generate().

        Args:
            level: Pattern complexity level (positive integer)
            shape: Tile geometry, "diamond" (the historical pond-diamond
                tiles) or "square" (axis-aligned pond-frame squares from
                gol_mosaics.tile_scheme)

        Raises:
            ValueError: If level is smaller than 1 or shape is unknown
        """
        if level < 1:
            raise ValueError(
                f"Level must be a positive integer, got {level}. "
                "Pre-computed patterns exist for levels 1-5 (use load()); "
                "higher levels must be created with generate()."
            )
        if shape not in ("diamond", "square"):
            raise ValueError(
                f"Unknown tile shape {shape!r}; expected 'diamond' or 'square'."
            )

        self.level = level
        self.shape = shape
        self.pond_width = 6
        self._solutions: Optional[np.ndarray] = None
        self._densities: Optional[np.ndarray] = None
        self._pond_pattern_multiple: Optional[np.ndarray] = None
        self._pond_pattern_edge: Optional[np.ndarray] = None
        self._scheme = None

    @property
    def scheme(self):
        """The TileScheme behind a square library (its lattice drives mosaic
        assembly). Diamond libraries keep their geometry in the historical
        pond_pattern_* helpers instead."""
        if self.shape != "square":
            raise ValueError(
                "scheme is only defined for square libraries; diamond "
                "geometry lives in the pond_pattern_* helpers."
            )
        if self._scheme is None:
            self._scheme = pond_square_scheme(self.level)
        return self._scheme

    def _require_diamond(self, what: str) -> None:
        if self.shape != "diamond":
            raise ValueError(
                f"{what} is diamond-lattice arithmetic and is undefined for "
                f"shape={self.shape!r}; use the library's scheme instead."
            )

    @property
    def solutions(self) -> np.ndarray:
        """
        Get pattern solutions (lazy-loaded).

        Returns:
            Numpy array of shape (N, H, W) containing N patterns

        Raises:
            ValueError: If patterns haven't been loaded or generated
        """
        if self._solutions is None:
            raise ValueError(
                "Patterns not loaded. Use PatternLibrary.load() or "
                "PatternLibrary.generate() first."
            )
        return self._solutions

    @property
    def densities(self) -> np.ndarray:
        """
        Get normalised density values for each pattern (lazy-computed).

        Densities are cached after first computation.

        Returns:
            Numpy array of shape (N,) with normalised densities in [0, 1]
        """
        if self._densities is None:
            self._densities = self._calculate_densities()
        return self._densities

    def _calculate_densities(self) -> np.ndarray:
        """Calculate and normalise density values for all patterns."""
        densities = np.mean(self.solutions, axis=(1, 2))
        # Normalise to [0, 1]
        dens_max = densities.max()
        dens_min = densities.min()
        if dens_max == dens_min:
            # Degenerate case (e.g. the trivial level-1 library): every
            # pattern shares one density, so give each the same value.
            return np.ones(len(densities))
        return (densities - dens_min) / (dens_max - dens_min)

    @classmethod
    def load(cls, level: int, shape: str = "diamond") -> 'PatternLibrary':
        """
        Load pre-computed patterns from disk.

        Diamond patterns are available for levels 1-6. Levels 1-5 are
        stored as full grids; level 6 (332,321 patterns) ships as packed
        symmetry-orbit bits and is expanded on load (~0.5 s, ~450 MB).
        Square patterns (axis-aligned pond-frame squares) are available for
        levels 3-5, all shipped as packed orbit bits. For other diamond
        levels, use PatternLibrary.generate() instead.

        Args:
            level: Pattern complexity level (diamond: 1-6, square: 3-5)
            shape: Tile geometry, "diamond" or "square"

        Returns:
            PatternLibrary instance with loaded patterns. Instances are
            cached and shared per (level, shape) (the data files are large),
            so treat the returned library and its arrays as read-only.

        Raises:
            ValueError: If level is outside the shape's pre-computed range
            FileNotFoundError: If data file is missing

        Example:
            >>> library = PatternLibrary.load(level=5)
            >>> print(f"Loaded {len(library.solutions)} patterns")
        """
        if shape not in ("diamond", "square"):
            raise ValueError(
                f"Unknown tile shape {shape!r}; expected 'diamond' or 'square'."
            )
        if shape == "square":
            if level not in [3, 4, 5]:
                raise ValueError(
                    f"Pre-computed square patterns only available for levels "
                    f"3-5, got level={level}. (Level 2 is a single fully "
                    f"forced tile; the level-6 census of 19,287,185 tiles is "
                    f"too large to ship.)"
                )
        elif level not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"Pre-computed patterns only available for levels 1-6. "
                f"Got level={level}. Use PatternLibrary.generate(level={level}) "
                f"to create patterns for this level."
            )

        return _load_pattern_library(level, shape)

    @classmethod
    def generate(cls, level: int, solution_limit: int = 1000) -> 'PatternLibrary':
        """
        Generate new patterns using Gurobi optimisation.

        Uses integer linear programming to find all symmetric Game of Life
        still-life patterns that satisfy:
        - Conway's Game of Life rules (2-3 neighbours for survival)
        - 8-fold symmetry (vertical, horizontal, diagonal)
        - Pond pattern edge constraints

        Note: Requires Gurobi licence. Can be time-consuming for high levels.
        Dead-edge tiling data is defined through level 6; levels above that
        would run without forced dead edges.

        Args:
            level: Pattern complexity level (2-6)
            solution_limit: Maximum number of patterns to find (default: 1000)

        Returns:
            PatternLibrary instance with generated patterns

        Raises:
            ImportError: If gurobipy is not available
            RuntimeError: If Gurobi optimisation fails

        Example:
            >>> # Generate patterns for level 6 (not pre-computed)
            >>> library = PatternLibrary.generate(level=6, solution_limit=500)
            >>> # Save for future use
            >>> np.save('solutions_level_6.npy', library.solutions)
        """
        from .legacy_ilp import generate_tiles

        library = cls(level=level)
        library._solutions = generate_tiles(level, solution_limit)
        return library

    def get_pattern_for_value(self,
                              value: float,
                              random: bool = True,
                              invert: bool = True) -> np.ndarray:
        """
        Get a single pattern matching the given greyscale value.

        Args:
            value: Greyscale value in [0, 1]
            random: If True, randomly select from patterns with matching density
            invert: If True, invert the density mapping (1.0 -> darkest)

        Returns:
            Single pattern as 2D numpy array

        Raises:
            ValueError: If value is not in [0, 1]

        Example:
            >>> library = PatternLibrary.load(level=4)
            >>> pattern = library.get_pattern_for_value(0.5)
            >>> pattern.shape
            (24, 24)
        """
        if value < 0 or value > 1:
            raise ValueError(f"Value must be in [0, 1], got {value}")

        # Invert mapping: black (0) -> dense (1), white (1) -> sparse (0)
        adjusted_value = 1.0 - value if invert else value

        return self.solutions[self._nearest_density_index(adjusted_value,
                                                          random)]

    def _nearest_density_index(self, adjusted_value: float,
                               random: bool) -> int:
        """
        Index of the pattern whose density is closest to adjusted_value.

        Ties are broken randomly when random=True; otherwise the first
        (lowest-index) match wins.
        """
        return int(nearest_density_indices(self.densities, adjusted_value,
                                           random=random))

    def get_patterns_for_values(self,
                                greyscale_values: np.ndarray,
                                random: bool = True,
                                invert: bool = True,
                                empty_tiles_cutoff: float = 1.0) -> np.ndarray:
        """
        Map greyscale values to patterns by density matching.

        Args:
            greyscale_values: Array of greyscale values in [0, 1]
            random: If True, randomly select from patterns with matching density
            invert: If True, invert the density mapping
            empty_tiles_cutoff: Values above this threshold become empty tiles

        Returns:
            Array of patterns with shape (*greyscale_values.shape, H, W)

        Raises:
            ValueError: If any value is outside [0, 1]

        Example:
            >>> library = PatternLibrary.load(level=4)
            >>> values = np.array([[0.2, 0.5], [0.7, 0.9]])
            >>> patterns = library.get_patterns_for_values(values)
            >>> patterns.shape
            (2, 2, 24, 24)
        """
        solutions = self.solutions
        indices = self.get_indices_for_values(
            greyscale_values, random=random, invert=invert,
            empty_tiles_cutoff=empty_tiles_cutoff)
        output_shape = indices.shape + solutions.shape[1:]

        flat = indices.ravel()
        mosaics = solutions[np.clip(flat, 0, None)]
        mosaics[flat < 0] = 0
        return mosaics.reshape(output_shape)

    def get_indices_for_values(self,
                               greyscale_values: np.ndarray,
                               random: bool = True,
                               invert: bool = True,
                               empty_tiles_cutoff: float = 1.0) -> np.ndarray:
        """
        Map greyscale values to solution indices by density matching.

        The selection half of get_patterns_for_values: -1 marks an empty
        tile (value above the cutoff), any other entry indexes solutions.
        Lattice assemblers that place tiles by index (the square path)
        consume this directly.

        Args:
            greyscale_values: Array of greyscale values in [0, 1]
            random: If True, randomly select from patterns with matching density
            invert: If True, invert the density mapping
            empty_tiles_cutoff: Values above this threshold become empty (-1)

        Returns:
            Integer array with the same shape as greyscale_values

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

    def get_patterns_for_mask(self,
                             mask: np.ndarray,
                             alpha_cutoff: float = 0.5) -> np.ndarray:
        """
        Map transparency mask to patterns.

        Values below alpha_cutoff get filled patterns, values above get empty.

        Args:
            mask: Array of alpha/mask values in [0, 1]
            alpha_cutoff: Threshold for transparency

        Returns:
            Array of patterns with shape (*mask.shape, H, W)

        Raises:
            ValueError: If any value is outside [0, 1]

        Example:
            >>> library = PatternLibrary.load(level=4)
            >>> mask = np.array([[0.0, 0.3], [0.6, 1.0]])  # Transparency values
            >>> patterns = library.get_patterns_for_mask(mask, alpha_cutoff=0.5)
        """
        mask = np.asarray(mask, dtype=float)
        solutions = self.solutions

        flat = mask.ravel()
        if flat.size and (flat.min() < 0 or flat.max() > 1):
            bad = flat[(flat < 0) | (flat > 1)][0]
            raise ValueError(f"Mask value must be in [0, 1], got {bad}")

        # Two invariant tiles: transparent -> empty, opaque -> the densest
        # pattern with its interior holes filled.
        empty_tile = np.zeros_like(solutions[0])
        filled_tile = binary_fill_holes(solutions[-1]).astype(int)

        transparent = (flat >= alpha_cutoff)[:, None, None]
        mosaics = np.where(transparent, empty_tile, filled_tile)
        return mosaics.reshape(mask.shape + solutions.shape[1:]).astype(
            solutions.dtype, copy=False)

    @staticmethod
    def pond_pattern() -> np.ndarray:
        """
        The base pond pattern (4x4 still life), see tile_domain.pond_pattern.

        Example:
            >>> PatternLibrary.pond_pattern().shape
            (4, 4)
        """
        return tile_domain.pond_pattern()

    @property
    def tile_shape(self) -> tuple:
        """Shape (height, width) of one tile: the pond edge pattern for
        diamonds, the scheme's n x n bounding box for squares."""
        if self.shape == "square":
            return (self.scheme.n, self.scheme.n)
        return self.pond_pattern_edge().shape

    @property
    def tile_pad_size(self) -> int:
        """
        Grid offset (in cells) between the two interlocking diagonal grids.

        Each diagonal grid is shifted by this amount (one horizontally, one
        vertically) so that the tiles of one grid sit in the gaps of the
        other without overlapping.
        """
        self._require_diamond("tile_pad_size")
        return ((self.pond_width - 3) * (2 * self.level - 1) + 1 + 2) // 2

    def pond_pattern_multiple(self) -> np.ndarray:
        """
        Stacked pond pattern with masked corners, see
        tile_domain.pond_pattern_multiple.

        The result is cached on the instance; treat it as read-only.
        """
        self._require_diamond("pond_pattern_multiple")
        if self._pond_pattern_multiple is None:
            self._pond_pattern_multiple = tile_domain.pond_pattern_multiple(
                self.level)
        return self._pond_pattern_multiple

    def pond_pattern_edge(self) -> np.ndarray:
        """
        The tile's forced-alive pond frame, see tile_domain.pond_pattern_edge.

        The result is cached on the instance; treat it as read-only.
        """
        self._require_diamond("pond_pattern_edge")
        if self._pond_pattern_edge is None:
            self._pond_pattern_edge = tile_domain.pond_pattern_edge(self.level)
        return self._pond_pattern_edge

    def pond_pattern_eighth(self) -> np.ndarray:
        """
        The free cells of one D4 octant, see tile_domain.pond_pattern_eighth.
        """
        self._require_diamond("pond_pattern_eighth")
        return tile_domain.pond_pattern_eighth(self.level)

    @staticmethod
    def _get_dead_edges(level: int) -> list:
        """Octant representatives of the forced-dead interlock cells, see
        tile_domain.derive_dead_edges."""
        return derive_dead_edges(level)


@lru_cache(maxsize=None)
def _load_pattern_library(level: int, shape: str = "diamond") -> PatternLibrary:
    """
    Load and cache one shared PatternLibrary per (level, shape).

    The solution files are large (up to ~19 MB), so each library is read
    from disk once per process. Cached instances are shared between callers
    and must be treated as read-only.
    """
    library = PatternLibrary(level=level, shape=shape)

    # Load solutions bundled inside the package (gol_mosaics/data/), located
    # via importlib.resources so it works regardless of install location.
    # Large levels ship as packed free-orbit bits (one bit per free symmetry
    # orbit per pattern) and are expanded to full grids here.
    data = files(__package__).joinpath("data")

    if shape == "square":
        packed_resource = data.joinpath(
            f"solutions_square_level_{level}_orbits.npy")
        if not packed_resource.is_file():
            raise FileNotFoundError(
                f"Pattern data file not found: {packed_resource}\n"
                f"Expected packaged resource: "
                f"gol_mosaics/data/solutions_square_level_{level}_orbits.npy"
            )
        with packed_resource.open("rb") as f:
            packed = np.load(f)
        library._solutions = unpack_scheme_solutions(library.scheme, packed)
        return library

    resource = data.joinpath(f"solutions_pattern_level_{level}.npy")
    packed_resource = data.joinpath(f"solutions_pattern_level_{level}_orbits.npy")

    if resource.is_file():
        with resource.open("rb") as f:
            library._solutions = np.load(f)
    elif packed_resource.is_file():
        with packed_resource.open("rb") as f:
            packed = np.load(f)
        library._solutions = unpack_solutions(packed, level)
    else:
        raise FileNotFoundError(
            f"Pattern data file not found: {resource}\n"
            f"Expected packaged resource: "
            f"gol_mosaics/data/solutions_pattern_level_{level}.npy "
            f"(or its packed _orbits variant)"
        )
    return library

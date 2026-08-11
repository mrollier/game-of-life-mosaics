"""
Game of Life pattern generation and management.

This module provides the PatternLibrary class for generating and managing
symmetric Game of Life still-life patterns using integer linear programming.
"""

import logging
import numpy as np
from functools import lru_cache
from importlib.resources import files
from typing import Optional
from scipy.ndimage import binary_fill_holes

from .tile_domain import derive_dead_edges, unpack_solutions
from .tile_scheme import pond_square_scheme, unpack_scheme_solutions

# Gurobi is optional - only needed for generating new patterns
try:
    from gurobipy import Model, GRB, quicksum
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

logger = logging.getLogger(__name__)


class PatternLibrary:
    """
    Manages Game of Life still-life patterns.

    Handles both generation (using Gurobi optimisation) and loading
    of pre-computed symmetric patterns. Patterns are loaded lazily to
    minimise memory usage.

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
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi is required for pattern generation but is not installed. "
                "Install with: pip install gurobipy\n"
                "Note: Gurobi requires a licence (free academic licences available at gurobi.com).\n"
                "For pre-computed patterns (levels 1-5), use PatternLibrary.load() instead."
            )

        library = cls(level=level)
        library._solutions = library._find_all_symmetric_gol_mosaics(solution_limit)
        return library

    def _find_all_symmetric_gol_mosaics(self, solution_limit: int = 1000) -> np.ndarray:
        """
        Find all symmetric GoL still-life patterns using Gurobi ILP.

        This is the core optimisation routine that uses Gurobi's mixed-integer
        programming solver to exhaustively find patterns.

        This approach is inspired by Rob Bosch's 2019 book "Opt Art".

        Args:
            solution_limit: Maximum number of solutions to find

        Returns:
            Numpy array of shape (N, H, W) with N found patterns
        """
        # Create empty mosaic and print initial information
        pp_edge = self.pond_pattern_edge()
        n = pp_edge.shape[0]
        logger.info("Looking for pattern level %d with grid size %dx%d",
                    self.level, n, n)

        # Make mask for cells outside the tile pattern
        pp_edge_binary = (pp_edge > 0).astype(np.uint8)
        pp_outside = 1 - binary_fill_holes(pp_edge_binary).astype(np.uint8)

        # Create Gurobi model
        model = Model("still_life")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Decision variables
        alive = {}  # Alive cells
        ldead = {}  # Low dead (< 2 neighbours)
        hdead = {}  # High dead (> 3 neighbours)

        for i in range(n):
            for j in range(n):
                alive[i, j] = model.addVar(vtype=GRB.BINARY, name=f"A_{i}_{j}")
                ldead[i, j] = model.addVar(vtype=GRB.BINARY, name=f"L_{i}_{j}")
                hdead[i, j] = model.addVar(vtype=GRB.BINARY, name=f"H_{i}_{j}")

        model.update()

        # Define neighbors function
        def neighbors(i, j, n):
            return [
                ((i + di) % n, (j + dj) % n)
                for di in [-1, 0, 1]
                for dj in [-1, 0, 1]
                if not (di == 0 and dj == 0)
            ]

        def symmetric_coords(i, j, n):
            """Return all symmetric positions for (i,j)."""
            return {
                "ver": (n - 1 - i, j),
                "hor": (i, n - 1 - j),
                "ver_and_hor": (n - 1 - i, n - 1 - j),
                "diag": (j, i),
                "diag_and_ver": (j, n - 1 - i),
                "diag_and_hor": (n - 1 - j, i),
                "all_sym": (n - 1 - j, n - 1 - i)
            }

        # Get dead edges for this level
        dead_edges = self._get_dead_edges(self.level)

        # Add constraints
        for i in range(n):
            for j in range(n):
                N = neighbors(i, j, n)
                neighbor_sum = quicksum(alive[ii, jj] for (ii, jj) in N)

                # Low-dead: cells with < 2 neighbours
                model.addConstr(
                    4 * ldead[i, j] + neighbor_sum <= 6,
                    name=f"low_dead_{i}_{j}"
                )

                # High-dead: cells with > 3 neighbours
                model.addConstr(
                    4 * hdead[i, j] <= neighbor_sum,
                    name=f"high_dead_{i}_{j}"
                )

                # Stayin' alive: alive cells need 2-3 neighbours
                model.addConstr(
                    2 * alive[i, j] <= neighbor_sum,
                    name=f"stay1_{i}_{j}"
                )
                model.addConstr(
                    3 * alive[i, j] + neighbor_sum <= 6,
                    name=f"stay2_{i}_{j}"
                )

                # Exactly one of L, H, or A is true
                model.addConstr(
                    ldead[i, j] + hdead[i, j] + alive[i, j] == 1,
                    name=f"oneof_{i}_{j}"
                )

                # Symmetry constraints
                sym_coords = symmetric_coords(i, j, n)
                for (ii, jj) in sym_coords.values():
                    model.addConstr(alive[i, j] == alive[ii, jj])
                    model.addConstr(ldead[i, j] == ldead[ii, jj])
                    model.addConstr(hdead[i, j] == hdead[ii, jj])

                # Force alive along tile pattern edge
                if pp_edge_binary[i, j]:
                    model.addConstr(
                        alive[i, j] == int(pp_edge_binary[i, j]),
                        name=f"force_alive_{i}_{j}"
                    )

                # Force dead outside tile pattern
                if pp_outside[i, j]:
                    model.addConstr(
                        alive[i, j] == 0,
                        name=f"force_dead_{i}_{j}"
                    )

                # Force dead on specific edges
                if (i, j) in dead_edges:
                    model.addConstr(
                        alive[i, j] == 0,
                        name=f"force_dead_edge_{i}_{j}"
                    )

        # Objective: maximise number of living cells
        model.setObjective(
            quicksum(alive[i, j] for i in range(n) for j in range(n)),
            GRB.MAXIMIZE
        )

        # Iterative exclusion to find all solutions
        solutions = []

        while True:
            model.optimize()

            if model.status != GRB.OPTIMAL:
                logger.info("Found %d optimal solutions.", len(solutions))
                break

            # Extract current solution
            sol = np.array([
                [round(alive[i, j].X) for j in range(n)]
                for i in range(n)
            ])
            solutions.append(sol)

            # Identify alive cells
            alive_cells = [
                (i, j) for i in range(n) for j in range(n)
                if round(alive[i, j].X) == 1
            ]

            # Add exclusion constraint
            model.addConstr(
                quicksum(1 - alive[i, j] for (i, j) in alive_cells) +
                quicksum(
                    alive[i, j]
                    for i in range(n) for j in range(n)
                    if (i, j) not in alive_cells
                ) >= 1,
                name=f"exclude_solution_{len(solutions)}"
            )

            if len(solutions) >= solution_limit:
                logger.info("Reached solution limit (%d).", solution_limit)
                break

        return np.array(solutions)

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
        diffs = np.abs(self.densities - adjusted_value)
        indices = np.where(diffs == diffs.min())[0]
        if random:
            return int(np.random.choice(indices))
        return int(indices[0])

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

        # Nearest-density pattern per tile, vectorised over all tiles.
        diffs = np.abs(self.densities[None, :] - adjusted[:, None])
        if random:
            # Uniform pick among each row's ties: random scores on the tie
            # positions, -1 elsewhere, then argmax.
            ties = diffs == diffs.min(axis=1, keepdims=True)
            scores = np.where(ties, np.random.random(diffs.shape), -1.0)
            indices = scores.argmax(axis=1)
        else:
            indices = diffs.argmin(axis=1)  # first tie wins

        indices = indices.astype(np.int64)
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
        Generate the base pond pattern (4x4 still-life).

        The pond is a simple still-life in Conway's Game of Life.

        Returns:
            4x4 binary array

        Example:
            >>> pattern = PatternLibrary.pond_pattern()
            >>> pattern.shape
            (4, 4)
        """
        return np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])

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
        Generate multiple pond patterns with symmetry.

        The result is cached on the instance (level and pond_width are
        immutable); treat it as read-only.

        Returns:
            Pattern array sized according to level
        """
        self._require_diamond("pond_pattern_multiple")
        if self._pond_pattern_multiple is not None:
            return self._pond_pattern_multiple

        width = self.pond_width * self.level
        pp = self.pond_pattern()

        # Create multiple pattern by stacking
        if self.level > 1:
            pp_multiple = np.vstack((
                np.vstack([pp[:-1]] * (self.level - 1)),
                pp,
                np.vstack([pp[1:]] * (self.level - 1))
            ))
            pp_multiple = np.hstack((
                np.hstack([pp_multiple[:, :-1]] * (self.level - 1)),
                pp_multiple,
                np.hstack([pp_multiple[:, 1:]] * (self.level - 1))
            ))
        else:
            pp_multiple = pp.copy()

        # Add edge
        pp_multiple = np.pad(pp_multiple, pad_width=1, constant_values=0)

        # Mask corners
        # Diagonal corners
        mask_even = np.array([
            [(i + j) < self.pond_width * self.level / 2 for j in range(width)]
            for i in range(width)
        ])
        mask_even = mask_even + mask_even[::-1, ::-1]

        # Off-diagonal corners
        mask_odd = np.array([
            [(i - j) >= self.pond_width * self.level / 2 for j in range(width)]
            for i in range(width)
        ])
        mask_odd = mask_odd + mask_odd.T

        mask = mask_even + mask_odd
        pp_multiple = np.where(1 - mask, pp_multiple, 0)

        self._pond_pattern_multiple = pp_multiple
        return pp_multiple

    def pond_pattern_edge(self) -> np.ndarray:
        """
        Generate edge pond pattern (only the border).

        The result is cached on the instance; treat it as read-only.

        Returns:
            Pattern array with only edge tiles
        """
        self._require_diamond("pond_pattern_edge")
        if self._pond_pattern_edge is not None:
            return self._pond_pattern_edge

        width = self.pond_width * self.level
        pp_multiple = self.pond_pattern_multiple()

        # Mask the interior
        mask_corner = np.array([
            [(i + j) < self.pond_width * self.level / 2 + 3 for j in range(width)]
            for i in range(width)
        ])
        mask = (mask_corner + mask_corner[::-1, ::-1] +
                mask_corner[::-1] + mask_corner[:, ::-1])

        pp_edge = np.where(mask, pp_multiple, 0)
        self._pond_pattern_edge = pp_edge
        return pp_edge
    
    def pond_pattern_eighth(self) -> np.ndarray:
        """
        Generate the pattern corresponding to all cells whose values can become either 0 or 1 in the D4 dihedral symmetry.

        Returns:
            Pattern array with the eighth of unique cells
        """
        self._require_diamond("pond_pattern_eighth")
        width = self.pond_width * self.level
        half_width = width // 2

        # Cache edge pattern to avoid redundant calls
        pp_edge = self.pond_pattern_edge()

        # Extract first quarter (top-right) of edge pattern
        pp_edge_eighth = np.zeros_like(pp_edge)
        pp_edge_eighth[:half_width, half_width:] = pp_edge[:half_width, half_width:]

        # Create sub-diagonal through first quarter
        pp_diagonal = np.diag(np.ones(width - 1, dtype=int), k=1)[::-1]

        # Create vertical line one cell to the left of centre
        pp_vertical = np.zeros_like(pp_edge)
        pp_vertical[:, half_width - 1] = 1

        # Union of the three patterns
        pp_outer = (pp_edge_eighth | pp_diagonal | pp_vertical).astype(bool)

        # Find the pattern surrounded by 1s on all sides
        pp_eighth = binary_fill_holes(pp_outer).astype(int)
        return pp_eighth - pp_outer

    @staticmethod
    def _get_dead_edges(level: int) -> list:
        """
        Get dead edge coordinates for a given level.

        These are specific cells that must be forced dead so that adjacent
        tiles in a mosaic cannot interact. Derived from the interlocking
        geometry for any level (see tile_domain.derive_dead_edges); the
        derivation reproduces the historically hard-coded lists for levels
        2-6 exactly (a regression test guards this).

        Args:
            level: Pattern level

        Returns:
            List of (i, j) tuples for dead edges (octant representatives;
            the solver's symmetry constraints propagate them orbit-wide)
        """
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

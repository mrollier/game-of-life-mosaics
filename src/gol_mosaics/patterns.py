"""
Game of Life pattern generation and management.

This module provides the PatternLibrary class for generating and managing
symmetric Game of Life still-life patterns using integer linear programming.
"""

import numpy as np
import os
from typing import Optional
from scipy.ndimage import binary_fill_holes

# Gurobi is optional - only needed for generating new patterns
try:
    from gurobipy import Model, GRB, quicksum
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class PatternLibrary:
    """
    Manages Game of Life still-life patterns.

    Handles both generation (using Gurobi optimization) and loading
    of pre-computed symmetric patterns. Patterns are loaded lazily to
    minimize memory usage.

    Attributes:
        level: Pattern complexity level (3-5 for pre-computed, others require generation)
        pond_width: Fixed width parameter (6) for pattern generation

    Example:
        >>> # Load pre-computed patterns
        >>> library = PatternLibrary.load(level=5)
        >>> pattern = library.get_pattern_for_value(0.5)

        >>> # Map multiple values to patterns
        >>> values = np.array([0.2, 0.5, 0.8])
        >>> patterns = library.get_patterns_for_values(values)
    """

    def __init__(self, level: int = 4):
        """
        Initialize PatternLibrary.

        Args:
            level: Pattern complexity level (1-5 supported, all pre-computed)

        Raises:
            ValueError: If level is not between 1 and 5
        """
        if level < 1 or level > 5:
            raise ValueError(
                f"Level must be between 1 and 5, got {level}. \
                Patterns for levels 1 and 2 are trivial. \
                Patterns for level 6 and higher are very demanding to compute."
            )

        self.level = level
        self.pond_width = 6
        self._solutions: Optional[np.ndarray] = None
        self._densities: Optional[np.ndarray] = None

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
        Get normalized density values for each pattern (lazy-computed).

        Densities are cached after first computation.

        Returns:
            Numpy array of shape (N,) with normalized densities in [0, 1]
        """
        if self._densities is None:
            self._densities = self._calculate_densities()
        return self._densities

    def _calculate_densities(self) -> np.ndarray:
        """Calculate and normalize density values for all patterns."""
        densities = np.mean(self.solutions, axis=(1, 2))
        # Normalize to [0, 1]
        dens_max = densities.max()
        dens_min = densities.min()
        if dens_max != dens_min:
            densities = (densities - densities.min()) / (densities.max() - densities.min())
            return densities
        # trivial solution for level 1
        else:
            return np.array([1])

    @classmethod
    def load(cls, level: int) -> 'PatternLibrary':
        """
        Load pre-computed patterns from disk.

        Pre-computed patterns are available for levels 1, 2, 3, 4, and 5.
        For other levels, use PatternLibrary.generate() instead.

        Args:
            level: Pattern complexity level (must be 1, 2, 3, 4, or 5)

        Returns:
            PatternLibrary instance with loaded patterns

        Raises:
            ValueError: If level is not 1, 2, 3, 4, or 5
            FileNotFoundError: If data file is missing

        Example:
            >>> library = PatternLibrary.load(level=5)
            >>> print(f"Loaded {len(library.solutions)} patterns")
        """
        if level not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Pre-computed patterns only available for levels 1, 2, 3, 4, 5. "
                f"Got level={level}. Use PatternLibrary.generate(level={level}) "
                f"to create patterns for this level."
            )

        library = cls(level=level)

        # Load solutions from data directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            base_dir,
            "../../data",
            f"solutions_pattern_level_{level}.npy"
        )

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Pattern data file not found: {data_path}\n"
                f"Expected location: data/solutions_pattern_level_{level}.npy"
            )

        library._solutions = np.load(data_path)
        return library

    @classmethod
    def generate(cls, level: int, solution_limit: int = 1000) -> 'PatternLibrary':
        """
        Generate new patterns using Gurobi optimization.

        Uses integer linear programming to find all symmetric Game of Life
        still-life patterns that satisfy:
        - Conway's Game of Life rules (2-3 neighbors for survival)
        - 8-fold symmetry (vertical, horizontal, diagonal)
        - Pond pattern edge constraints

        Note: Requires Gurobi license. Can be time-consuming for high levels.

        Args:
            level: Pattern complexity level (2-6)
            solution_limit: Maximum number of patterns to find (default: 1000)

        Returns:
            PatternLibrary instance with generated patterns

        Raises:
            ImportError: If gurobipy is not available
            RuntimeError: If Gurobi optimization fails

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
                "Note: Gurobi requires a license (free academic licenses available at gurobi.com).\n"
                "For pre-computed patterns (levels 3-5), use PatternLibrary.load() instead."
            )

        library = cls(level=level)
        library._solutions = library._find_all_symmetric_gol_mosaics(solution_limit)
        return library

    def _find_all_symmetric_gol_mosaics(self, solution_limit: int = 1000) -> np.ndarray:
        """
        Find all symmetric GoL still-life patterns using Gurobi ILP.

        This is the core optimization routine that uses Gurobi's mixed-integer
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
        print(f"Looking for pattern level {self.level} with grid size {n}x{n}")

        # Make mask for cells outside the tile pattern
        pp_edge_binary = (pp_edge > 0).astype(np.uint8)
        pp_outside = 1 - binary_fill_holes(pp_edge_binary).astype(np.uint8)

        # Create Gurobi model
        model = Model("still_life")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Decision variables
        alive = {}  # Alive cells
        ldead = {}  # Low dead (< 2 neighbors)
        hdead = {}  # High dead (> 3 neighbors)

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

                # Low-dead: cells with < 2 neighbors
                model.addConstr(
                    4 * ldead[i, j] + neighbor_sum <= 6,
                    name=f"low_dead_{i}_{j}"
                )

                # High-dead: cells with > 3 neighbors
                model.addConstr(
                    4 * hdead[i, j] <= neighbor_sum,
                    name=f"high_dead_{i}_{j}"
                )

                # Stayin' alive: alive cells need 2-3 neighbors
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

        # Objective: maximize number of living cells
        model.setObjective(
            quicksum(alive[i, j] for i in range(n) for j in range(n)),
            GRB.MAXIMIZE
        )

        # Iterative exclusion to find all solutions
        solutions = []

        while True:
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print(f"Found {len(solutions)} optimal solutions.")
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
                print(f"Reached solution limit ({solution_limit}).")
                break

        return np.array(solutions)

    def get_pattern_for_value(self,
                              value: float,
                              random: bool = True,
                              invert: bool = True) -> np.ndarray:
        """
        Get a single pattern matching the given grayscale value.

        Args:
            value: Grayscale value in [0, 1]
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
        adjusted_value = value
        if invert:
            adjusted_value = 1.0 - value

        # Find closest density
        densities = self.densities
        diffs = np.abs(densities - adjusted_value)
        min_diff = diffs.min()
        indices = np.where(diffs == min_diff)[0]

        if random:
            index = np.random.choice(indices)
        else:
            index = indices[0]

        return self.solutions[index]

    def get_patterns_for_values(self,
                                greyscale_values: np.ndarray,
                                random: bool = True,
                                invert: bool = True,
                                empty_tiles_cutoff: float = 1.0) -> np.ndarray:
        """
        Map grayscale values to patterns by density matching.

        Args:
            greyscale_values: Array of grayscale values in [0, 1]
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
        greyscale_values = np.asarray(greyscale_values)

        # Calculate densities
        densities = self.densities.copy()
        solutions = self.solutions

        # Prepare output array
        output_shape = greyscale_values.shape + solutions.shape[1:]
        mosaics = np.empty(output_shape, dtype=solutions.dtype)

        # Map each value
        flat_grey = greyscale_values.ravel()

        for idx, val in enumerate(flat_grey):
            if val < 0 or val > 1:
                raise ValueError(
                    f"Greyscale value must be in [0, 1], got {val}"
                )

            if val > empty_tiles_cutoff:
                # Use empty pattern
                solution = np.zeros_like(solutions[0])
            else:
                # Adjust value and find matching pattern
                adjusted_val = val / empty_tiles_cutoff

                # Invert mapping: black (0) -> dense (1), white (1) -> sparse (0)
                if invert:
                    adjusted_val = 1.0 - adjusted_val

                diffs = np.abs(densities - adjusted_val)
                min_diff = diffs.min()
                indices = np.where(diffs == min_diff)[0]

                if random:
                    index = np.random.choice(indices)
                else:
                    index = indices[0]

                solution = solutions[index]

            mosaics.reshape(-1, *solutions.shape[1:])[idx, ...] = solution

        return mosaics

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
        mask = np.asarray(mask)
        solutions = self.solutions

        # Prepare output array
        output_shape = mask.shape + solutions.shape[1:]
        mosaics = np.empty(output_shape, dtype=solutions.dtype)

        # Map each value
        flat_mask = mask.ravel()

        for idx, val in enumerate(flat_mask):
            if val < 0 or val > 1:
                raise ValueError(
                    f"Mask value must be in [0, 1], got {val}"
                )

            if val >= alpha_cutoff:
                # Transparent -> empty pattern
                solution = np.zeros_like(solutions[0])
            else:
                # Opaque -> filled pattern
                solution = binary_fill_holes(solutions[-1]).astype(int)

            mosaics.reshape(-1, *solutions.shape[1:])[idx, ...] = solution

        return mosaics

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

    def pond_pattern_multiple(self) -> np.ndarray:
        """
        Generate multiple pond patterns with symmetry.

        Returns:
            Pattern array sized according to level
        """
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

        return pp_multiple

    def pond_pattern_edge(self) -> np.ndarray:
        """
        Generate edge pond pattern (only the border).

        Returns:
            Pattern array with only edge tiles
        """
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
        return pp_edge
    
    def pond_pattern_eighth(self) -> np.ndarray:
        """
        Generate the pattern corresponding to all cells whose values can become either 0 or 1 in the D4 dihedral symmetry.

        Returns:
            Pattern array with the eighth of unique cells
        """
        width = self.pond_width * self.level
        half_width = width // 2

        # Cache edge pattern to avoid redundant calls
        pp_edge = self.pond_pattern_edge()

        # Extract first quarter (top-right) of edge pattern
        pp_edge_eighth = np.zeros_like(pp_edge)
        pp_edge_eighth[:half_width, half_width:] = pp_edge[:half_width, half_width:]

        # Create sub-diagonal through first quarter
        pp_diagonal = np.diag(np.ones(width - 1, dtype=int), k=1)[::-1]

        # Create vertical line one cell to the left of center
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

        These are specific cells that must be forced dead to ensure
        proper tiling.

        Args:
            level: Pattern level

        Returns:
            List of (i, j) tuples for dead edges
        """
        dead_edges_map = {
            2: [(2, 6), (3, 6)],
            3: [(2, 9), (3, 9), (5, 11), (5, 12)],
            4: [(2, 11), (3, 11), (5, 14), (5, 15), (6, 15)],
            5: [(2, 15), (3, 15), (5, 17), (5, 18), (6, 18), (8, 20), (8, 21)],
            6: [(2, 18), (3, 18), (5, 20), (5, 21), (6, 21), (8, 23), (8, 24), (9, 24)]
        }
        return dead_edges_map.get(level, [])

"""
Elementary Cellular Automaton background generation.

This module provides the ECABackground class for generating
Elementary Cellular Automaton patterns to use as mosaic backgrounds.
"""

import numpy as np
import cellpylib as cpl
from typing import List


class ECABackground:
    """
    Generate Elementary Cellular Automaton (ECA) backgrounds.

    Elementary Cellular Automata are simple 1-dimensional cellular automata
    that evolve over time according to simple rules. They can create
    interesting patterns that serve as backgrounds for the GoL mosaics.

    Common rules:
    - Complex patterns: 54, 147, 110, 124, 137, 193
    - Chaotic patterns: 30, 45, 106, 150

    Attributes:
        rule: Wolfram rule number (0-255)

    Example:
        >>> eca = ECABackground(rule=106)
        >>> pattern = eca.generate(width=100, height=100, supersample=10)
        >>> pattern.shape
        (1000, 1000)
    """

    # Predefined rule categories
    COMPLEX_RULES = [54, 147, 110, 124, 137, 193]
    CHAOTIC_RULES = [30, 45, 106, 150]

    def __init__(self, rule: int = 106):
        """
        Initialise ECA background generator.

        Args:
            rule: Wolfram rule number (0-255). Default is 106 (chaotic)

        Raises:
            ValueError: If rule is not in range [0, 255]

        Example:
            >>> eca = ECABackground(rule=54)  # Complex pattern
            >>> eca = ECABackground(rule=106)  # Chaotic pattern
        """
        if not 0 <= rule <= 255:
            raise ValueError(
                f"Rule must be in range [0, 255], got {rule}"
            )
        self.rule = rule

    def generate(self,
                width: int,
                height: int,
                supersample: int = 15) -> np.ndarray:
        """
        Generate ECA pattern at full resolution.

        The pattern is generated at low resolution (ceil(width/supersample),
        ceil(height/supersample)), upsampled by repeating each cell into a
        supersample x supersample block, and then cropped to exactly
        (height, width). Because of the crop, supersample need not divide the
        target dimensions: any positive value works without interpolation.

        Args:
            width: Target width in pixels
            height: Target height in pixels
            supersample: Upsampling factor, i.e. the side of each ECA cell in
                pixels (default: 15). Must be a positive integer.

        Returns:
            Binary numpy array of shape (height, width)

        Raises:
            ValueError: If supersample is not a positive integer

        Example:
            >>> eca = ECABackground(rule=106)
            >>> pattern = eca.generate(width=1000, height=1000, supersample=10)
            >>> pattern.shape
            (1000, 1000)
            >>> # Dimensions need not be divisible by supersample:
            >>> eca.generate(width=588, height=522, supersample=14).shape
            (522, 588)
            >>> np.unique(pattern)
            array([0, 1])
        """
        # Validate supersample
        if not isinstance(supersample, (int, np.integer)) or supersample < 1:
            raise ValueError(
                f"supersample must be a positive integer, got {supersample!r}"
            )

        # Generate at low resolution, rounding up so the upsampled pattern is at
        # least as large as the target; the surplus is cropped off below.
        eca_width = -(-width // supersample)   # ceil division
        eca_height = -(-height // supersample)

        # Initialise with random state
        eca = cpl.init_random(eca_width)

        # Evolve the cellular automaton
        eca = cpl.evolve(
            eca,
            timesteps=eca_height,
            memoize=True,
            apply_rule=lambda n, c, t: cpl.nks_rule(n, self.rule)
        )

        # Upsample by repeating pixels, then crop to the exact target size
        eca_upsized = np.repeat(eca, supersample, axis=0)
        eca_upsized = np.repeat(eca_upsized, supersample, axis=1)
        eca_upsized = eca_upsized[:height, :width]

        return eca_upsized.astype(np.uint8)

    @staticmethod
    def validate_supersample(width: int, supersample: int) -> bool:
        """
        Check if supersample value is compatible with width.

        Args:
            width: Image width
            supersample: Proposed supersample value

        Returns:
            True if compatible, False otherwise

        Example:
            >>> ECABackground.validate_supersample(100, 10)
            True
            >>> ECABackground.validate_supersample(100, 7)
            False
        """
        return width % supersample == 0

    @staticmethod
    def list_valid_supersamples(width: int) -> List[int]:
        """
        List all valid supersample values for given width.

        Returns all divisors of width, which are the values that
        allow upsampling without interpolation.

        Args:
            width: Image width

        Returns:
            Sorted list of valid supersample values

        Example:
            >>> ECABackground.list_valid_supersamples(60)
            [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
        """
        if not isinstance(width, int) or width <= 0:
            raise ValueError("Width must be a positive integer")

        divisors = []
        for i in range(1, int(np.sqrt(width)) + 1):
            if width % i == 0:
                divisors.append(i)
                if i != width // i:
                    divisors.append(width // i)

        return sorted(divisors)

    def get_rule_category(self) -> str:
        """
        Get the category of the current rule.

        Returns:
            'complex', 'chaotic', or 'other'

        Example:
            >>> eca = ECABackground(rule=54)
            >>> eca.get_rule_category()
            'complex'
            >>> eca = ECABackground(rule=106)
            >>> eca.get_rule_category()
            'chaotic'
        """
        if self.rule in self.COMPLEX_RULES:
            return 'complex'
        elif self.rule in self.CHAOTIC_RULES:
            return 'chaotic'
        else:
            return 'other'

    @classmethod
    def from_category(cls, category: str = 'chaotic') -> 'ECABackground':
        """
        Create ECA generator with a rule from specified category.

        Args:
            category: 'complex' or 'chaotic'

        Returns:
            ECABackground instance with a rule from that category

        Raises:
            ValueError: If category is not 'complex' or 'chaotic'

        Example:
            >>> eca = ECABackground.from_category('complex')
            >>> eca.rule in ECABackground.COMPLEX_RULES
            True
        """
        if category == 'complex':
            rule = np.random.choice(cls.COMPLEX_RULES)
        elif category == 'chaotic':
            rule = np.random.choice(cls.CHAOTIC_RULES)
        else:
            raise ValueError(
                f"Category must be 'complex' or 'chaotic', got '{category}'"
            )

        return cls(rule=rule)

    def __repr__(self) -> str:
        """String representation of ECABackground."""
        category = self.get_rule_category()
        return f"ECABackground(rule={self.rule}, category='{category}')"

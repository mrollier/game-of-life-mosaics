"""
Elementary Cellular Automaton background generation.

This module provides the ECABackground class for generating
Elementary Cellular Automaton patterns to use as mosaic backgrounds.
"""

import numpy as np
import cellpylib as cpl
from typing import List

from .image_processing import ImageProcessor


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
        Initialize ECA background generator.

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

        The pattern is generated at low resolution (width/supersample,
        height/supersample) and then upsampled by repeating pixels.

        Args:
            width: Target width in pixels (must be divisible by supersample)
            height: Target height in pixels (must be divisible by supersample)
            supersample: Upsampling factor (default: 15)

        Returns:
            Binary numpy array of shape (height, width)

        Raises:
            ValueError: If width or height not divisible by supersample

        Example:
            >>> eca = ECABackground(rule=106)
            >>> pattern = eca.generate(width=1000, height=1000, supersample=10)
            >>> pattern.shape
            (1000, 1000)
            >>> np.unique(pattern)
            array([0, 1])
        """
        # Validate supersample
        if width % supersample != 0:
            raise ValueError(
                f"Width {width} must be divisible by supersample {supersample}. "
                f"Valid supersample values: {self.list_valid_supersamples(width)}"
            )

        if height % supersample != 0:
            raise ValueError(
                f"Height {height} must be divisible by supersample {supersample}. "
                f"Valid supersample values: {self.list_valid_supersamples(height)}"
            )

        # Generate at low resolution
        eca_width = width // supersample
        eca_height = height // supersample

        # Initialize with random state
        eca = cpl.init_random(eca_width)

        # Evolve the cellular automaton
        eca = cpl.evolve(
            eca,
            timesteps=eca_height,
            memoize=True,
            apply_rule=lambda n, c, t: cpl.nks_rule(n, self.rule)
        )

        # Upsample by repeating pixels
        eca_upsized = np.repeat(eca, supersample, axis=0)
        eca_upsized = np.repeat(eca_upsized, supersample, axis=1)

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
        return ImageProcessor.list_denominators(width)

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

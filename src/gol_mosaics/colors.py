"""
Color scheme management for Game of Life mosaics.

This module provides the ColorScheme dataclass for managing color configurations
used in mosaic rendering.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ColorScheme:
    """
    Immutable color configuration for mosaic rendering.

    All colors are specified as hex color strings (e.g., '#FFFFFF').

    Attributes:
        gol_background: Background color for Game of Life mosaic
        gol_pixel: Foreground/alive cell color for Game of Life mosaic
        eca_background: Background color for Elementary Cellular Automaton overlay
        eca_pixel: Foreground color for Elementary Cellular Automaton overlay

    Example:
        >>> colors = ColorScheme(
        ...     gol_background='#FFFFFF',
        ...     gol_pixel='#000000',
        ...     eca_background='#FFD200',
        ...     eca_pixel='#1E64C8'
        ... )
        >>> colors.gol_background
        '#FFFFFF'
    """

    gol_background: str = '#FFFFFF'
    gol_pixel: str = '#000000'
    eca_background: str = '#FFD200'
    eca_pixel: str = '#1E64C8'

    @classmethod
    def ugent(cls) -> 'ColorScheme':
        """
        UGent (Ghent University) brand colors.

        Uses UGent's official color palette:
        - White/black for Game of Life patterns
        - Yellow/blue for ECA background

        This is the default color scheme.

        Returns:
            ColorScheme with UGent brand colors

        Example:
            >>> colors = ColorScheme.ugent()
            >>> colors.eca_pixel  # UGent blue
            '#1E64C8'
        """
        return cls(
            gol_background='#FFFFFF',  # white
            gol_pixel='#000000',       # black
            eca_background='#FFD200',  # warm yellow
            eca_pixel='#1E64C8'        # dark blue
        )

    @classmethod
    def monochrome(cls,
                   foreground: str = '#000000',
                   background: str = '#FFFFFF') -> 'ColorScheme':
        """
        Simple two-color monochrome scheme.

        Uses the same foreground and background colors for both
        GoL patterns and ECA overlay.

        Args:
            foreground: Foreground/pixel color (default: black)
            background: Background color (default: white)

        Returns:
            ColorScheme with monochrome colors

        Example:
            >>> colors = ColorScheme.monochrome()
            >>> colors.gol_pixel
            '#000000'
            >>> # Blue on white
            >>> colors = ColorScheme.monochrome(
            ...     foreground='#0000FF',
            ...     background='#FFFFFF'
            ... )
        """
        return cls(
            gol_background=background,
            gol_pixel=foreground,
            eca_background=background,
            eca_pixel=foreground
        )

    @classmethod
    def inverted(cls) -> 'ColorScheme':
        """
        Inverted UGent colors (light on dark).

        Uses black background with white patterns and
        blue background with yellow ECA overlay.

        Returns:
            ColorScheme with inverted UGent colors

        Example:
            >>> colors = ColorScheme.inverted()
            >>> colors.gol_background  # Black instead of white
            '#000000'
        """
        return cls(
            gol_background='#000000',  # black
            gol_pixel='#FFFFFF',       # white
            eca_background='#1E64C8',  # dark blue
            eca_pixel='#FFD200'        # warm yellow
        )

    def to_dict(self) -> Dict[str, str]:
        """
        Convert color scheme to dictionary format.

        Useful for compatibility or serialization.

        Returns:
            Dictionary with color configuration

        Example:
            >>> colors = ColorScheme.ugent()
            >>> colors.to_dict()
            {'gol_background': '#FFFFFF', 'gol_pixel': '#000000', ...}
        """
        return {
            'gol_background': self.gol_background,
            'gol_pixel': self.gol_pixel,
            'eca_background': self.eca_background,
            'eca_pixel': self.eca_pixel
        }

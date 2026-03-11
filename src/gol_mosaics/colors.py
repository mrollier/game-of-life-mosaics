"""
Color scheme management for Game of Life mosaics.

This module provides the ColorScheme dataclass for managing color configurations
used in mosaic rendering.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


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
    def custom(cls,
               gol_background: str = '#FFFFFF',
               gol_pixel: str = '#000000',
               eca_background: str = '#FFD200',
               eca_pixel: str = '#1E64C8') -> 'ColorScheme':
        """
        Custom color scheme.

        Args:
            gol_background: Background color for Game of Life mosaic (default: white)
            gol_pixel: Foreground/alive cell color for Game of Life mosaic (default: black)
            eca_background: Background color for Elementary Cellular Automaton overlay (default: yellow)
            eca_pixel: Foreground color for Elementary Cellular Automaton overlay (default: blue)

        Returns:
            ColorScheme with custom colors

        Example:
            >>> colors = ColorScheme.custom()
            >>> colors.gol_pixel
            '#000000'
            >>> # Blue on white
            >>> colors = ColorScheme.custom(
            ...     gol_background='#FFFFFF',
            ...     gol_pixel='#0000FF',
            ...     eca_background='#FFD200',
            ...     eca_pixel='#1E64C8'
            ... )
        """
        return cls(
            gol_background=gol_background,
            gol_pixel=gol_pixel,
            eca_background=eca_background,
            eca_pixel=eca_pixel
        )

    @classmethod
    def warhol(cls, force_white=False) -> 'ColorScheme':
        """
        Warhol-inspired color scheme.

        Uses bright, contrasting colors reminiscent of Andy Warhol's pop art style.

        Returns:
            ColorScheme with Warhol-inspired colors
        
        Example:
            >>> colors = ColorScheme.warhol()
            >>> colors.gol_pixel  # Bright magenta
            '#FF00FF'
        """

        warhol_palette = {
            "hot_pink": "#ff2d8d",
            "magenta": "#ff00a8",
            "fuchsia": "#ff008f",
            "neon_red": "#ff2a2a",
            "crimson": "#d4002a",
            "tangerine": "#ff6a00",
            "bright_orange": "#ff7a00",
            "sunflower_yellow": "#ffd400",
            "lemon_yellow": "#fff200",
            "acid_yellow": "#eaff00",
            "lime_green": "#7fff00",
            "neon_green": "#39ff14",
            "chartreuse": "#bfff00",
            "mint_green": "#3fffbf",
            "turquoise": "#00ffd5",
            "aqua": "#00e5ff",
            "electric_blue": "#007bff",
            "cobalt_blue": "#0047ff",
            "royal_purple": "#7a00ff",
            "violet": "#b300ff"
        }

        rng = np.random.default_rng()
        picked_colors = rng.choice(list(warhol_palette.values()), size=4, replace=False)

        if force_white:
            gol_background='#FFFFFF'  # Force white background for Game of Life
        else:
            gol_background = picked_colors[0]  # Randomly pick from palette

        return cls(
            gol_background=gol_background,
            gol_pixel=picked_colors[1],
            eca_background=picked_colors[2],
            eca_pixel=picked_colors[3]
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

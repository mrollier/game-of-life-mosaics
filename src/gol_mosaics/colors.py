"""
Colour scheme management for Game of Life mosaics.

This module provides the ColorScheme dataclass for managing colour configurations
used in mosaic rendering.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class ColorScheme:
    """
    Immutable colour configuration for mosaic rendering.

    All colours are specified as hex colour strings (e.g., '#FFFFFF').

    Attributes:
        gol_background: Background colour for Game of Life mosaic
        gol_pixel: Foreground/alive cell colour for Game of Life mosaic
        eca_background: Background colour for Elementary Cellular Automaton overlay
        eca_pixel: Foreground colour for Elementary Cellular Automaton overlay

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
        UGent (Ghent University) brand colours.

        Uses UGent's official colour palette:
        - White/black for Game of Life patterns
        - Yellow/blue for ECA background

        This is the default colour scheme.

        Returns:
            ColorScheme with UGent brand colours

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
        Two-colour monochrome scheme.

        Uses a single foreground colour on a single background colour for
        both the Game of Life patterns and the ECA overlay.

        Args:
            foreground: Pixel/foreground colour (default: black)
            background: Background colour (default: white)

        Returns:
            ColorScheme with the same two colours for GoL and ECA

        Example:
            >>> colors = ColorScheme.monochrome()
            >>> colors.gol_pixel
            '#000000'
            >>> # Red on green
            >>> colors = ColorScheme.monochrome(foreground='#FF0000', background='#00FF00')
        """
        return cls(
            gol_background=background,
            gol_pixel=foreground,
            eca_background=background,
            eca_pixel=foreground
        )

    @classmethod
    def custom(cls,
               gol_background: str = '#FFFFFF',
               gol_pixel: str = '#000000',
               eca_background: str = '#FFD200',
               eca_pixel: str = '#1E64C8') -> 'ColorScheme':
        """
        Custom colour scheme.

        Args:
            gol_background: Background colour for Game of Life mosaic (default: white)
            gol_pixel: Foreground/alive cell colour for Game of Life mosaic (default: black)
            eca_background: Background colour for Elementary Cellular Automaton overlay (default: yellow)
            eca_pixel: Foreground colour for Elementary Cellular Automaton overlay (default: blue)

        Returns:
            ColorScheme with custom colours

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
    def warhol(cls, force_white=False, dark_on_light=True) -> 'ColorScheme':
        """
        Warhol-inspired colour scheme.

        Uses bright, contrasting colours reminiscent of Andy Warhol's pop art style.

        Args:
            force_white: If True, forces the Game of Life background to be white (default: False)
            dark_on_light: If True, uses dark colours on a light background; if False, just picks randomly from the palette (default: True)

        Returns:
            ColorScheme with Warhol-inspired colours
        
        Example:
            >>> colors = ColorScheme.warhol()
            >>> colors.gol_pixel  # Bright magenta
            '#FF00FF'
        """

        # Warhol-inspired colour palette with bright, contrasting colours
        warhol_light_colors = {
            "pastel_pink": "#ff9ecb",
            "bubblegum_pink": "#ff8cc6",
            "light_fuchsia": "#ff77d4",
            "rose_pink": "#ff6fa0",
            "coral_pink": "#ff7f7f",
            "peach": "#ffb07c",
            "light_orange": "#ffb347",
            "apricot": "#ffc87c",
            "sunny_yellow": "#fff44f",
            "lemon": "#fff76a",
            "butter_yellow": "#fff1a8",
            "lime": "#bfff66",
            "neon_lime": "#aaff33",
            "light_chartreuse": "#dfff4f",
            "mint": "#7fffd4",
            "seafoam": "#71f7c6",
            "light_turquoise": "#66ffe0",
            "aqua": "#66eaff",
            "sky_blue": "#7ec8ff",
            "light_cyan": "#8fe9ff",
            "lavender": "#c79cff",
            "light_violet": "#d19cff",
            "orchid": "#e29bff",
            "soft_magenta": "#ff8ae2",
            "light_plum": "#d78cff"
        }

        # Warhol-inspired colour palette with darker, more saturated colours for contrast
        warhol_dark_colors = {
            "hot_pink": "#ff2d8d",
            "magenta": "#ff0099",
            "fuchsia": "#d1007a",
            "deep_crimson": "#b00020",
            "pop_red": "#ff0033",
            "scarlet": "#e60026",
            "burnt_orange": "#d94a00",
            "tangerine": "#ff5e00",
            "deep_gold": "#cc9a00",
            "mustard": "#bfa000",
            "olive": "#7a8f00",
            "acid_green": "#66cc00",
            "kelly_green": "#00a550",
            "emerald": "#009b5e",
            "jade": "#00a86b",
            "deep_teal": "#008080",
            "turquoise_blue": "#0096a7",
            "electric_blue": "#0066ff",
            "cobalt": "#0047ab",
            "royal_blue": "#0033cc",
            "indigo": "#3f00ff",
            "deep_violet": "#6a00cc",
            "purple": "#8000ff",
            "plum": "#7b1fa2",
            "deep_magenta": "#9b0056"
        }

        # Merge dicts
        warhol_colors = {**warhol_light_colors, **warhol_dark_colors}

        # Prepare random generator
        rng = np.random.default_rng()

        # Randomly select colours based on the dark_on_light flag
        if not dark_on_light:
            gol_colors = rng.choice(list(warhol_colors.values()), size=2, replace=False)
            gol_background = gol_colors[0]
            gol_pixel = gol_colors[1]
        else:
            # Force dark pixels on light background
            gol_pixel = rng.choice(list(warhol_dark_colors.values()))
            gol_background = rng.choice(list(warhol_light_colors.values()))

        # Select two distinct ECA colours
        eca_colors = rng.choice(list(warhol_colors.values()), size=2, replace=False)
        eca_background = eca_colors[0]
        eca_pixel = eca_colors[1]

        # Force white background if requested
        if force_white:
            gol_background='#FFFFFF'

        return cls(
            gol_background=gol_background,
            gol_pixel=gol_pixel,
            eca_background=eca_background,
            eca_pixel=eca_pixel
        )


    @classmethod
    def inverted(cls) -> 'ColorScheme':
        """
        Inverted UGent colours (light on dark).

        Uses black background with white patterns and
        blue background with yellow ECA overlay.

        Returns:
            ColorScheme with inverted UGent colours

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
        Convert colour scheme to dictionary format.

        Useful for compatibility or serialisation.

        Returns:
            Dictionary with colour configuration

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

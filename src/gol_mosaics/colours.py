"""
Colour scheme management for Game of Life mosaics.

This module provides the ColourScheme dataclass for managing colour configurations
used in mosaic rendering.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class ColourScheme:
    """
    Immutable colour configuration for mosaic rendering.

    All colours are specified as hex colour strings (e.g., '#FFFFFF').

    Attributes:
        gol_background: Background colour for Game of Life mosaic
        gol_pixel: Foreground/alive cell colour for Game of Life mosaic
        eca_background: Background colour for Elementary Cellular Automaton overlay
        eca_pixel: Foreground colour for Elementary Cellular Automaton overlay
        rim_colour: Colour for the outer rim of the mosaic (the diamond corners
            outside the tile content). Defaults to white ('#FFFFFF').

    Example:
        >>> colours = ColourScheme(
        ...     gol_background='#FFFFFF',
        ...     gol_pixel='#000000',
        ...     eca_background='#FFD200',
        ...     eca_pixel='#1E64C8'
        ... )
        >>> colours.gol_background
        '#FFFFFF'
    """

    gol_background: str = '#FFFFFF'
    gol_pixel: str = '#000000'
    eca_background: str = '#FFD200'
    eca_pixel: str = '#1E64C8'
    rim_colour: str = '#FFFFFF'

    @classmethod
    def ugent(cls) -> 'ColourScheme':
        """
        UGent (Ghent University) brand colours.

        Uses UGent's official colour palette:
        - White/black for Game of Life patterns
        - Yellow/blue for ECA background

        Returns:
            ColourScheme with UGent brand colours

        Example:
            >>> colours = ColourScheme.ugent()
            >>> colours.eca_pixel  # UGent blue
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
               eca_pixel: str = '#1E64C8',
               rim_colour: str = '#FFFFFF') -> 'ColourScheme':
        """
        Custom colour scheme.

        Args:
            gol_background: Background colour for Game of Life mosaic (default: white)
            gol_pixel: Foreground/alive cell colour for Game of Life mosaic (default: black)
            eca_background: Background colour for Elementary Cellular Automaton overlay (default: yellow)
            eca_pixel: Foreground colour for Elementary Cellular Automaton overlay (default: blue)
            rim_colour: Colour for the outer rim of the mosaic (default: white)

        Returns:
            ColourScheme with custom colours

        Example:
            >>> colours = ColourScheme.custom(
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
            eca_pixel=eca_pixel,
            rim_colour=rim_colour
        )

    @classmethod
    def warhol(cls) -> 'ColourScheme':
        """
        Warhol-inspired colour scheme.

        Randomly selects colours from a 50-colour pop art palette. The GoL
        background is always one of the 25 lighter colours, and the GoL pixel
        is always one of the 25 darker colours. ECA colours are drawn randomly
        from the full palette.

        Returns:
            ColourScheme with Warhol-inspired colours

        Example:
            >>> colours = ColourScheme.warhol()
        """
        # 25 vivid light colours (perceived luminance 0.58–0.84)
        warhol_light_colours = {
            "pastel_pink":   "#ff9ecb",
            "bubblegum":     "#ff77cc",
            "rose":          "#ff6699",
            "coral_rose":    "#ff99aa",
            "light_rose":    "#ffaacc",
            "coral":         "#ff7755",
            "light_orange":  "#ff9955",
            "apricot":       "#ffbb55",
            "gold":          "#ffcc66",
            "warm_yellow":   "#ffdd44",
            "lime":          "#88ee22",
            "medium_green":  "#55dd55",
            "seafoam":       "#44dd99",
            "mint":          "#66ffaa",
            "turquoise":     "#00ddcc",
            "sky_blue":      "#44ccff",
            "light_blue":    "#77bbff",
            "periwinkle":    "#aabbff",
            "light_violet":  "#cc88ff",
            "orchid":        "#dd99ff",
            "light_magenta": "#ff88ff",
            "pink_lavender": "#ff99ee",
            "warm_pink":     "#ee66aa",
            "salmon":        "#ffaa77",
            "peach":         "#ffbb88",
        }

        # 25 vivid dark colours (perceived luminance < 0.50)
        warhol_dark_colours = {
            "hot_pink":       "#ff2d8d",
            "magenta":        "#ff0099",
            "fuchsia":        "#d1007a",
            "dark_pink":      "#cc0066",
            "deep_crimson":   "#b00020",
            "pop_red":        "#ff0033",
            "scarlet":        "#e60026",
            "deep_orange":    "#cc4400",
            "burnt_orange":   "#d94a00",
            "warm_gold":      "#997700",
            "dark_olive":     "#887700",
            "forest_green":   "#336600",
            "kelly_green":    "#00a550",
            "emerald":        "#009b5e",
            "jade":           "#00a86b",
            "deep_teal":      "#008080",
            "turquoise_blue": "#0096a7",
            "electric_blue":  "#0066ff",
            "cobalt":         "#0047ab",
            "royal_blue":     "#0033cc",
            "indigo":         "#3f00ff",
            "deep_violet":    "#6a00cc",
            "purple":         "#8000ff",
            "plum":           "#7b1fa2",
            "deep_magenta":   "#9b0056",
        }

        warhol_all_colours = list(warhol_light_colours.values()) + list(warhol_dark_colours.values())
        rng = np.random.default_rng()

        gol_background = rng.choice(list(warhol_light_colours.values()))
        gol_pixel = rng.choice(list(warhol_dark_colours.values()))
        eca_background, eca_pixel = rng.choice(warhol_all_colours, size=2, replace=False)

        return cls(
            gol_background=str(gol_background),
            gol_pixel=str(gol_pixel),
            eca_background=str(eca_background),
            eca_pixel=str(eca_pixel)
        )

    @classmethod
    def inverted(cls) -> 'ColourScheme':
        """
        Inverted UGent colours (light on dark).

        Uses black background with white patterns and
        blue background with yellow ECA overlay.

        Returns:
            ColourScheme with inverted UGent colours

        Example:
            >>> colours = ColourScheme.inverted()
            >>> colours.gol_background  # Black instead of white
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

        Returns:
            Dictionary with colour configuration

        Example:
            >>> colours = ColourScheme.ugent()
            >>> colours.to_dict()
            {'gol_background': '#FFFFFF', 'gol_pixel': '#000000', ...}
        """
        return {
            'gol_background': self.gol_background,
            'gol_pixel': self.gol_pixel,
            'eca_background': self.eca_background,
            'eca_pixel': self.eca_pixel,
            'rim_colour': self.rim_colour
        }

"""
Colour scheme management for Game of Life mosaics.

This module provides the ColorScheme dataclass for managing colour configurations
used in mosaic rendering.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

# Rejection-sampling cap for the Warhol luminance guard. High enough that a
# reachable gap is always found, low enough that an unreachable one still
# returns (the best pair seen) instead of spinning.
_MAX_DRAWS = 200


def _luma(hex_color: str) -> float:
    """
    Perceived brightness of a hex colour, in 0..1 (ITU-R BT.601 luma).

    Args:
        hex_color: Hex colour string, with or without '#'

    Returns:
        Luma in 0..1, where 0 is black and 1 is white

    Example:
        >>> round(_luma('#FFFFFF'), 3)
        1.0
    """
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def mix(first: str, second: str, weight: float) -> str:
    """
    Linear RGB blend of two hex colours.

    Args:
        first: Hex colour at weight 0
        second: Hex colour at weight 1
        weight: Blend position in 0..1

    Returns:
        Hex colour string, uppercase, with a leading '#'

    Example:
        >>> mix('#000000', '#FFFFFF', 0.5)
        '#808080'
    """
    a, b = (np.array([int(c.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)],
                     dtype=float) for c in (first, second))
    return '#%02X%02X%02X' % tuple(np.round(a + weight * (b - a)).astype(int))


#: How far a scheme with no `fill_pixel` pulls its filler colour from
#: `eca_pixel` towards `eca_background`. Far enough that the smallest still
#: lifes read as haze, short of vanishing into the field.
HAZE = 0.75


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
        fill_pixel: Far end of the filler ramp — the colour of the loose
            still lifes, which the smaller tile levels step towards from
            `eca_pixel` (see :func:`gol_mosaics.compose.filled_background`).
            None derives a haze three quarters of the way from `eca_pixel`
            to `eca_background`.

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
    fill_pixel: Optional[str] = None

    @property
    def fill(self) -> str:
        """Far end of the filler ramp, derived when `fill_pixel` is unset.

        The derived colour is :data:`HAZE` of the way from `eca_pixel` to
        `eca_background`, so a cascade of ever smaller tiles fades into the
        field instead of stopping on one flat colour.

        Example:
            >>> ColorScheme.ugent().fill
            '#C7B632'
        """
        if self.fill_pixel:
            return self.fill_pixel
        return mix(self.eca_pixel, self.eca_background, HAZE)

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
    def warhol(cls, force_white: bool = False,
               dark_on_light: bool = True,
               seed: int = None,
               min_luma_gap: float = 0.35) -> 'ColorScheme':
        """
        Warhol-inspired colour scheme.

        Uses bright, contrasting colours reminiscent of Andy Warhol's pop art style.

        Args:
            force_white: If True, forces the Game of Life background to be white (default: False)
            dark_on_light: If True, uses dark colours on a light background; if False, just picks randomly from the palette (default: True)
            seed: Seed for the colour draw. None (default) gives a different
                scheme on every call; an integer makes the scheme reproducible.
            min_luma_gap: Minimum brightness difference (0..1) between the two
                colours of each pair. Pairs that are too close are redrawn.
                Free-form still lifes carry tone as local live-cell density, so
                a low-contrast pair erases the picture rather than merely
                recolouring it. 0 disables the guard.

        Returns:
            ColorScheme with Warhol-inspired colours

        Example:
            >>> colors = ColorScheme.warhol()
            >>> colors.gol_pixel  # Bright magenta
            '#FF00FF'
            >>> # Reproducible: the same seed always gives the same scheme
            >>> ColorScheme.warhol(seed=7) == ColorScheme.warhol(seed=7)
            True
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
        rng = np.random.default_rng(seed)

        all_values = list(warhol_colors.values())
        dark_values = list(warhol_dark_colors.values())
        light_values = list(warhol_light_colors.values())

        def draw_gol():
            """One (background, pixel) candidate for the GoL layer."""
            if dark_on_light:
                # Force dark pixels on light background
                return str(rng.choice(light_values)), str(rng.choice(dark_values))
            pair = rng.choice(all_values, size=2, replace=False)
            return str(pair[0]), str(pair[1])

        def draw_eca():
            """One (background, pixel) candidate for the ECA layer."""
            pair = rng.choice(all_values, size=2, replace=False)
            return str(pair[0]), str(pair[1])

        def pick(draw):
            """Draw until the pair is far enough apart in brightness.

            Falls back to the widest pair seen rather than looping forever, so
            an unreachable min_luma_gap degrades instead of hanging.
            """
            best, best_gap = None, -1.0
            for _ in range(_MAX_DRAWS):
                background, pixel = draw()
                gap = abs(_luma(background) - _luma(pixel))
                if gap >= min_luma_gap:
                    return background, pixel
                if gap > best_gap:
                    best, best_gap = (background, pixel), gap
            return best

        gol_background, gol_pixel = pick(draw_gol)
        eca_background, eca_pixel = pick(draw_eca)

        # Force white background if requested
        if force_white:
            gol_background = '#FFFFFF'

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

        Useful for compatibility or serialisation. `fill_pixel` comes back
        resolved rather than as None, so every value is a hex string.

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
            'eca_pixel': self.eca_pixel,
            'fill_pixel': self.fill
        }

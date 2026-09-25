"""
Mosaic rendering and colour mapping.

This module provides the MosaicRenderer class for converting
numpy arrays to coloured PIL Images.
"""

import numpy as np
from PIL import Image
from typing import Dict, Optional

from .colours import ColourScheme, hex_to_rgb, mix


class MosaicRenderer:
    """
    Renders mosaic arrays as coloured PIL Images.

    Takes binary or multi-valued numpy arrays and applies colour
    mapping to create the final RGBA images. Handles both GoL
    mosaics and ECA overlays.

    Attributes:
        colours: ColourScheme instance defining colours to use

    Example:
        >>> from gol_mosaics import ColourScheme, MosaicRenderer
        >>> colors = ColourScheme.ugent()
        >>> renderer = MosaicRenderer(colors)
        >>> mosaic = np.random.randint(0, 2, (100, 100))
        >>> img = renderer.render_gol_mosaic(mosaic)
        >>> img.save('output.png')
    """

    def __init__(self, colours: ColourScheme):
        """
        Initialise renderer with colour scheme.

        Args:
            colours: ColourScheme instance defining colours

        Example:
            >>> colors = ColourScheme.ugent()
            >>> renderer = MosaicRenderer(colors)
        """
        self.colours = colours

    def render_gol_mosaic(self, mosaic: np.ndarray) -> Image.Image:
        """
        Render Game of Life mosaic with GoL colours.

        Args:
            mosaic: Binary array (0=background, 1=pixel)

        Returns:
            RGBA PIL Image

        Raises:
            ValueError: If mosaic is not 2D

        Example:
            >>> mosaic = np.array([[0, 1], [1, 0]])
            >>> img = renderer.render_gol_mosaic(mosaic)
            >>> img.mode
            'RGBA'
        """
        if mosaic.ndim != 2:
            raise ValueError(
                f"Mosaic must be 2D array, got shape {mosaic.shape}"
            )

        color_map = {
            0: self.colours.gol_background,
            1: self.colours.gol_pixel
        }

        rgb_array = self._array_to_rgb(mosaic, color_map)

        # Convert to RGBA
        rgba_array = np.zeros((*mosaic.shape, 4), dtype=np.uint8)
        rgba_array[:, :, :3] = rgb_array
        rgba_array[:, :, 3] = 255  # Fully opaque

        return Image.fromarray(rgba_array, mode='RGBA')

    def render_eca_overlay(self,
                           eca_mask: np.ndarray,
                           layers: Optional[int] = None) -> Image.Image:
        """
        Render ECA pattern as RGBA overlay.

        The eca_mask should have values:
        - 0: Transparent (no overlay)
        - 1: ECA background colour
        - 2: ECA pixel colour — the main mosaic
        - 3 and up: the filler layers of :func:`filled_background`, painted
          along a linear ramp from the ECA pixel colour to the scheme's
          `fill` colour. A single filler layer lands on `fill` exactly, so a
          hand-built 0-3 mask renders as it always did.

        Args:
            eca_mask: Array with values 0, 1, 2, 3, ...
            layers: How many layers the field has, main mosaic included, so
                the ramp spans the same range even when the last layer placed
                nothing. None reads it back from the mask.

        Returns:
            RGBA PIL Image with transparency

        Raises:
            ValueError: If eca_mask is not 2D

        Example:
            >>> eca_mask = np.array([[0, 1, 2], [2, 1, 0]])
            >>> overlay = renderer.render_eca_overlay(eca_mask)
            >>> overlay.mode
            'RGBA'
        """
        if eca_mask.ndim != 2:
            raise ValueError(
                f"ECA mask must be 2D array, got shape {eca_mask.shape}"
            )

        # One RGBA row per layer value: 0 transparent, 1 eca_background,
        # 2 eca_pixel, 3.. the filler ramp. Any value outside the table
        # (negative, above the ramp, or not an integer) stays transparent.
        top = int(eca_mask.max()) if layers is None else layers + 1
        palette = np.zeros((max(top, 2) + 1, 4), dtype=np.uint8)
        palette[1] = (*hex_to_rgb(self.colours.eca_background), 255)
        palette[2] = (*hex_to_rgb(self.colours.eca_pixel), 255)
        for value in range(3, top + 1):
            fraction = (value - 2) / max(top - 2, 1)
            palette[value] = (*hex_to_rgb(mix(self.colours.eca_pixel,
                                              self.colours.fill,
                                              fraction)), 255)

        index = eca_mask.astype(np.int64)
        known = (index == eca_mask) & (index >= 0) & (index < len(palette))
        overlay = palette[np.where(known, index, 0)]

        return Image.fromarray(overlay, mode='RGBA')

    def composite(self,
                 base: Image.Image,
                 overlay: Image.Image) -> Image.Image:
        """
        Alpha-composite overlay onto base image.

        Args:
            base: Base RGBA image
            overlay: Overlay RGBA image (same size as base)

        Returns:
            Composited RGBA image

        Raises:
            ValueError: If images have different sizes or wrong mode

        Example:
            >>> base = renderer.render_gol_mosaic(mosaic)
            >>> overlay = renderer.render_eca_overlay(eca_mask)
            >>> final = renderer.composite(base, overlay)
        """
        if base.size != overlay.size:
            raise ValueError(
                f"Images must have same size. "
                f"Base: {base.size}, Overlay: {overlay.size}"
            )

        if base.mode != 'RGBA' or overlay.mode != 'RGBA':
            raise ValueError(
                "Both images must be in RGBA mode"
            )

        return Image.alpha_composite(base, overlay)

    @staticmethod
    def _array_to_rgb(arr: np.ndarray, color_map: Dict[int, str]) -> np.ndarray:
        """
        Convert array to RGB using colour mapping.

        Args:
            arr: 2D array with integer values
            color_map: Dictionary mapping values to hex colours

        Returns:
            RGB array of shape (*arr.shape, 3)
        """
        rgb_array = np.zeros((*arr.shape, 3), dtype=np.uint8)

        for value, hex_color in color_map.items():
            mask = (arr == value)
            rgb_tuple = hex_to_rgb(hex_color)
            rgb_array[mask] = rgb_tuple

        return rgb_array

    def render_full_mosaic(self,
                          gol_mosaic: np.ndarray,
                          eca_mask: np.ndarray,
                          layers: Optional[int] = None) -> Image.Image:
        """
        Render complete mosaic with GoL pattern and ECA overlay.

        Convenience method that combines render_gol_mosaic,
        render_eca_overlay, and composite.

        Args:
            gol_mosaic: Binary GoL pattern array
            eca_mask: ECA overlay mask (values 0, 1, 2, ...)
            layers: Field layer count for the filler ramp, see
                :meth:`render_eca_overlay`

        Returns:
            Final composited RGBA image

        Example:
            >>> img = renderer.render_full_mosaic(gol_mosaic, eca_mask)
            >>> img.save('final.png')
        """
        base = self.render_gol_mosaic(gol_mosaic)
        overlay = self.render_eca_overlay(eca_mask, layers=layers)
        return self.composite(base, overlay)

    def __repr__(self) -> str:
        """String representation of renderer."""
        return (
            f"MosaicRenderer("
            f"gol_colors={self.colours.gol_background}/{self.colours.gol_pixel}, "
            f"eca_colors={self.colours.eca_background}/{self.colours.eca_pixel})"
        )

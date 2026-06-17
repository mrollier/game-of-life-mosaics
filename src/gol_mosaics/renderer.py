"""
Mosaic rendering and colour mapping.

This module provides the MosaicRenderer class for converting
numpy arrays to coloured PIL Images.
"""

import numpy as np
from PIL import Image
from typing import Dict

from .colors import ColorScheme


class MosaicRenderer:
    """
    Renders mosaic arrays as coloured PIL Images.

    Takes binary or multi-valued numpy arrays and applies colour
    mapping to create the final RGBA images. Handles both GoL
    mosaics and ECA overlays.

    Attributes:
        color_scheme: ColorScheme instance defining colours to use

    Example:
        >>> from gol_mosaics import ColorScheme, MosaicRenderer
        >>> colors = ColorScheme.ugent()
        >>> renderer = MosaicRenderer(colors)
        >>> mosaic = np.random.randint(0, 2, (100, 100))
        >>> img = renderer.render_gol_mosaic(mosaic)
        >>> img.save('output.png')
    """

    def __init__(self, color_scheme: ColorScheme):
        """
        Initialise renderer with colour scheme.

        Args:
            color_scheme: ColorScheme instance defining colours

        Example:
            >>> colors = ColorScheme.ugent()
            >>> renderer = MosaicRenderer(colors)
        """
        self.color_scheme = color_scheme

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
            0: self.color_scheme.gol_background,
            1: self.color_scheme.gol_pixel
        }

        rgb_array = self._array_to_rgb(mosaic, color_map)

        # Convert to RGBA
        rgba_array = np.zeros((*mosaic.shape, 4), dtype=np.uint8)
        rgba_array[:, :, :3] = rgb_array
        rgba_array[:, :, 3] = 255  # Fully opaque

        return Image.fromarray(rgba_array, mode='RGBA')

    def render_eca_overlay(self, eca_mask: np.ndarray) -> Image.Image:
        """
        Render ECA pattern as RGBA overlay.

        The eca_mask should have values:
        - 0: Transparent (no overlay)
        - 1: ECA background colour
        - 2: ECA pixel colour

        Args:
            eca_mask: Array with values 0, 1, 2

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

        h, w = eca_mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        # Convert hex colours to RGB
        rgb1 = self._hex_to_rgb(self.color_scheme.eca_background)
        rgb2 = self._hex_to_rgb(self.color_scheme.eca_pixel)

        # Value 1 -> eca_background, opaque
        mask1 = (eca_mask == 1)
        overlay[mask1, :3] = rgb1
        overlay[mask1, 3] = 255

        # Value 2 -> eca_pixel, opaque
        mask2 = (eca_mask == 2)
        overlay[mask2, :3] = rgb2
        overlay[mask2, 3] = 255

        # Value 0 stays (0,0,0,0) fully transparent

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
            rgb_tuple = MosaicRenderer._hex_to_rgb(hex_color)
            rgb_array[mask] = rgb_tuple

        return rgb_array

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        """
        Convert hex colour string to RGB tuple.

        Args:
            hex_color: Hex colour string (e.g., '#FFFFFF')

        Returns:
            RGB tuple (e.g., (255, 255, 255))

        Example:
            >>> MosaicRenderer._hex_to_rgb('#FFFFFF')
            (255, 255, 255)
            >>> MosaicRenderer._hex_to_rgb('#1E64C8')
            (30, 100, 200)
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def render_full_mosaic(self,
                          gol_mosaic: np.ndarray,
                          eca_mask: np.ndarray) -> Image.Image:
        """
        Render complete mosaic with GoL pattern and ECA overlay.

        Convenience method that combines render_gol_mosaic,
        render_eca_overlay, and composite.

        Args:
            gol_mosaic: Binary GoL pattern array
            eca_mask: ECA overlay mask (values 0, 1, 2)

        Returns:
            Final composited RGBA image

        Example:
            >>> img = renderer.render_full_mosaic(gol_mosaic, eca_mask)
            >>> img.save('final.png')
        """
        base = self.render_gol_mosaic(gol_mosaic)
        overlay = self.render_eca_overlay(eca_mask)
        return self.composite(base, overlay)

    def change_colors(self, new_color_scheme: ColorScheme) -> 'MosaicRenderer':
        """
        Create new renderer with different colours.

        Args:
            new_color_scheme: New ColorScheme to use

        Returns:
            New MosaicRenderer instance

        Example:
            >>> renderer1 = MosaicRenderer(ColorScheme.ugent())
            >>> renderer2 = renderer1.change_colors(ColorScheme.monochrome())
        """
        return MosaicRenderer(new_color_scheme)

    def __repr__(self) -> str:
        """String representation of renderer."""
        return (
            f"MosaicRenderer("
            f"gol_colors={self.color_scheme.gol_background}/{self.color_scheme.gol_pixel}, "
            f"eca_colors={self.color_scheme.eca_background}/{self.color_scheme.eca_pixel})"
        )

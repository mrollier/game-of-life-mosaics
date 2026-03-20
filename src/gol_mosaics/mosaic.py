"""
Main mosaic generation API.

This module provides the MosaicGenerator class, the primary interface
for converting images to Game of Life mosaics.
"""

import numpy as np
import math
from PIL import Image
from typing import Optional
from scipy.ndimage import binary_fill_holes

from .patterns import PatternLibrary
from .colours import ColourScheme
from .image_processing import ImageProcessor
from .eca import ECABackground
from .renderer import MosaicRenderer
from .export import GollyExporter


class MosaicGenerator:
    """
    Main API for generating Game of Life mosaics from images.

    This class orchestrates all components to produce the final artwork:
    pattern library, image processing, ECA backgrounds, and rendering.

    Attributes:
        level: Pattern complexity level (2-5 pre-computed, others need generation)
        grid_size: Number of tiles in the grid
        colour_scheme: ColourScheme for rendering
        eca_rule: Rule number for ECA background
        random_patterns: Whether to randomly select patterns
        invert: Whether to invert the density mapping

    Example:
        >>> from gol_mosaics import MosaicGenerator
        >>> generator = MosaicGenerator(level=5, grid_size=100)
        >>> mosaic = generator.generate_from_image('portrait.png')
        >>> mosaic.save('output.png')
    """

    def __init__(self,
                 level: Optional[int] = None,
                 grid_size: Optional[int] = None,
                 colour_scheme: Optional[ColourScheme] = None,
                 eca_rule: Optional[int] = None,
                 random_patterns: bool = True,
                 invert: bool = True):
        """
        Initialise mosaic generator.

        All parameters are optional. When omitted, sensible random values are
        chosen automatically so that every call to MosaicGenerator() produces
        a pleasingly varied result.

        Args:
            level: Pattern complexity level (2-5 for pre-computed patterns).
                Higher levels offer more density variety and finer greyscale
                reproduction, at the cost of computation time.
            grid_size: Number of Tiles along each axis of the grid (must be even).
                Higher values produce higher-resolution Mosaics.
            colour_scheme: ColourScheme instance. Defaults to a random
                Warhol-inspired palette with dark pixels on a light background.
            eca_rule: Rule number for Elementary Cellular Automaton background.
                If None, randomly selects from the interesting complex/chaotic rules.
            random_patterns: Use random pattern selection vs deterministic.
            invert: Invert the density mapping (dark image areas → dense Tiles).

        Raises:
            ValueError: If grid_size is odd

        Example:
            >>> from gol_mosaics import MosaicGenerator, ColourScheme
            >>> generator = MosaicGenerator(
            ...     level=5,
            ...     grid_size=100,
            ...     colour_scheme=ColourScheme.ugent(),
            ...     eca_rule=54
            ... )
        """
        # Pick random grid size and level if not provided
        self.grid_size = grid_size or self._auto_select_grid_size()
        if self.grid_size % 2 != 0:
            raise ValueError(
                f"grid_size must be even, got {self.grid_size}"
            )
        self.level = level or self._auto_select_level()

        # Pick random ECA rule from some interesting ones if not provided
        self.eca_rule = eca_rule or self._auto_select_eca_rule()

        # Default to a random Warhol-inspired colour scheme if not provided
        self.colour_scheme = colour_scheme or ColourScheme.warhol()

        # Pick random patterns and invert colours.
        # These can be touched but generally look better with default values.
        self.random_patterns = random_patterns
        self.invert = invert

        # Lazy-initialised components
        self._pattern_library: Optional[PatternLibrary] = None
        self._renderer: Optional[MosaicRenderer] = None
        self._eca_generator: Optional[ECABackground] = None

        # Stores the raw GoL mosaic array from the most recent generation,
        # allowing export_to_cells() to be called after generate_from_image().
        self._last_gol_mosaic: Optional[np.ndarray] = None

    @property
    def pattern_library(self) -> PatternLibrary:
        """Get pattern library (lazy-loaded)."""
        if self._pattern_library is None:
            self._pattern_library = PatternLibrary.load(self.level)
        return self._pattern_library

    @property
    def renderer(self) -> MosaicRenderer:
        """Get renderer (lazy-loaded)."""
        if self._renderer is None:
            self._renderer = MosaicRenderer(self.colour_scheme)
        return self._renderer

    @property
    def eca_generator(self) -> ECABackground:
        """Get ECA generator (lazy-loaded)."""
        if self._eca_generator is None:
            self._eca_generator = ECABackground(self.eca_rule)
        return self._eca_generator

    def generate_from_image(self,
                            image_path: str,
                            empty_tiles_cutoff: float = 0.75,
                            alpha_cutoff: float = 0.5,
                            supersample: Optional[int] = None,
                            no_eca: bool = False) -> Image.Image:
        """
        Generate mosaic from image file.

        This is the main method for converting images to mosaics. It handles
        the complete pipeline: loading, preprocessing, pattern mapping,
        ECA background generation, and final rendering.

        The raw GoL Tile array is stored on ``self._last_gol_mosaic`` after
        each call, enabling ``export_to_cells()`` to be called immediately
        afterwards.

        Args:
            image_path: Path to input image (PNG, JPG, etc.)
            empty_tiles_cutoff: Threshold for empty Tiles (0-1). Grayscale
                values above this become empty Tiles. Default: 0.75.
            alpha_cutoff: Threshold for transparency masking (0-1). Alpha
                values below this get filled with ECA. Default: 0.5.
            supersample: ECA upsampling factor (must divide mosaic width
                evenly). Higher values create finer ECA patterns. If None
                (default), automatically selects a valid value close to 15.
            no_eca: If True, skip ECA background generation entirely and
                leave the background regions transparent. Default: False.

        Returns:
            PIL Image in RGBA mode with mosaic and ECA background

        Raises:
            ValueError: If supersample doesn't divide mosaic width evenly
            FileNotFoundError: If image_path doesn't exist

        Example:
            >>> generator = MosaicGenerator(level=5, grid_size=100)
            >>> mosaic = generator.generate_from_image('portrait.png')
            >>> mosaic.save('output.png')
            >>> generator.export_to_cells('output.cells')
        """
        # Preprocess image
        results = ImageProcessor.preprocess_for_mosaic(
            image_path,
            self.grid_size
        )
        lowres_first, lowres_second, mask_first, mask_second, aspect_ratio = results

        # Build GoL mosaic
        gol_mosaic = self._build_mosaic(
            lowres_first,
            lowres_second,
            empty_tiles_cutoff
        )


        # Build transparency mask
        transparency_mask = self._build_mask(
            mask_first,
            mask_second,
            alpha_cutoff
        )

        # Build subject mask: pixels belonging to opaque (portrait subject) tiles.
        # A tile is "subject" if its alpha value >= alpha_cutoff. Subject tiles
        # keep gol_background even when they produce empty GoL patterns (e.g.
        # bright highlights). Only non-subject tiles are candidates for rim_colour.
        subject_mask = self._build_subject_mask(mask_first, mask_second, alpha_cutoff)

        # Adjust for original aspect ratio (same crop applied to all arrays)
        gol_mosaic, transparency_mask, subject_mask = self._adjust_aspect_ratio(
            gol_mosaic,
            transparency_mask,
            aspect_ratio,
            extra=subject_mask
        )
        # Store raw mosaic array for optional Golly export
        self._last_gol_mosaic = gol_mosaic

        # Auto-select supersample if not provided
        if supersample is None:
            supersample = self._auto_select_supersample(
                gol_mosaic.shape[1],
                target=15
            )

        # Apply ECA background and render
        final_image = self._apply_eca_background(
            gol_mosaic,
            transparency_mask,
            supersample,
            subject_mask=subject_mask,
            no_eca=no_eca
        )

        return final_image

    def export_to_cells(self, filename: str, add_glider: bool = False) -> None:
        """
        Export the most recently generated Mosaic as a Golly ``.cells`` file.

        Must be called after ``generate_from_image()``.

        Args:
            filename: Output file path (e.g. 'output/my_mosaic.cells')
            add_glider: If True, add a glider in the top-left corner for
                animation testing in Golly. Default: False.

        Raises:
            RuntimeError: If called before generate_from_image()

        Example:
            >>> generator = MosaicGenerator(level=5, grid_size=100)
            >>> generator.generate_from_image('portrait.png')
            >>> generator.export_to_cells('portrait.cells')
        """
        if self._last_gol_mosaic is None:
            raise RuntimeError(
                "No mosaic has been generated yet. "
                "Call generate_from_image() first."
            )
        GollyExporter.export_to_cells(
            self._last_gol_mosaic,
            filename=filename,
            add_glider=add_glider
        )

    def generate_from_gif(self,
                          gif_path: str,
                          empty_tiles_cutoff: float = 0.75,
                          alpha_cutoff: float = 0.5,
                          supersample: int = 15) -> Image.Image:
        """
        Convert animated GIF to mosaic GIF.

        Processes each frame independently and returns an animated GIF.

        Args:
            gif_path: Path to input GIF
            empty_tiles_cutoff: Threshold for empty Tiles (0-1)
            alpha_cutoff: Threshold for transparency masking (0-1)
            supersample: ECA upsampling factor

        Returns:
            Animated PIL Image

        Example:
            >>> generator = MosaicGenerator(level=4, grid_size=50)
            >>> mosaic_gif = generator.generate_from_gif('animation.gif')
            >>> mosaic_gif.save('output.gif', save_all=True)
        """
        import os
        import tempfile

        # Open GIF
        gif = Image.open(gif_path)
        frames = []
        durations = []

        # Process each frame
        frame_num = 0
        try:
            while True:
                # Convert frame to RGBA
                frame = gif.convert('RGBA')

                # Save to temporary file
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix='.png',
                    delete=False
                ) as tmp:
                    temp_path = tmp.name
                    frame.save(temp_path)

                # Process frame
                try:
                    mosaic = self.generate_from_image(
                        temp_path,
                        empty_tiles_cutoff=empty_tiles_cutoff,
                        alpha_cutoff=alpha_cutoff,
                        supersample=supersample
                    )
                    frames.append(mosaic)
                    durations.append(gif.info.get('duration', 100))
                finally:
                    # Clean up temp file
                    os.remove(temp_path)

                # Move to next frame
                frame_num += 1
                gif.seek(frame_num)

        except EOFError:
            # End of GIF
            pass

        if not frames:
            raise ValueError(f"No frames found in GIF: {gif_path}")

        # Return first frame (caller can save with save_all=True)
        frames[0].info['duration'] = durations[0]
        frames[0].info['loop'] = gif.info.get('loop', 0)

        return frames[0]

    def _build_mosaic(self,
                      lowres_first: np.ndarray,
                      lowres_second: np.ndarray,
                      empty_tiles_cutoff: float) -> np.ndarray:
        """
        Build GoL mosaic from diagonal patterns.

        Args:
            lowres_first: First diagonal grayscale pattern
            lowres_second: Second diagonal grayscale pattern
            empty_tiles_cutoff: Threshold for empty Tiles

        Returns:
            Complete GoL mosaic as binary array
        """
        # Map to patterns
        patterns_first = self.pattern_library.get_patterns_for_values(
            lowres_first / 255,
            random=self.random_patterns,
            invert=self.invert,
            empty_tiles_cutoff=empty_tiles_cutoff
        )

        patterns_second = self.pattern_library.get_patterns_for_values(
            lowres_second / 255,
            random=self.random_patterns,
            invert=self.invert,
            empty_tiles_cutoff=empty_tiles_cutoff
        )

        # Combine into big arrays
        big_array_first = np.block([
            [patterns_first[i, j] for j in range(patterns_first.shape[1])]
            for i in range(patterns_first.shape[0])
        ])

        big_array_second = np.block([
            [patterns_second[i, j] for j in range(patterns_second.shape[1])]
            for i in range(patterns_second.shape[0])
        ])

        # Add padding to align the two diagonal grids
        pond_width = 6
        pad_size = ((pond_width - 3) * (2 * self.level - 1) + 1 + 2) // 2

        pad_tuple = (pad_size, pad_size)
        mosaic_first = np.pad(
            big_array_first, pad_width=((0, 0), pad_tuple), constant_values=0
        )
        mosaic_second = np.pad(
            big_array_second, pad_width=(pad_tuple, (0, 0)), constant_values=0
        )

        return mosaic_first + mosaic_second

    def _build_mask(self,
                    mask_first: np.ndarray,
                    mask_second: np.ndarray,
                    alpha_cutoff: float) -> np.ndarray:
        """
        Build transparency mask from diagonal patterns.

        Args:
            mask_first: First diagonal alpha mask
            mask_second: Second diagonal alpha mask
            alpha_cutoff: Threshold for transparency

        Returns:
            Complete transparency mask as binary array
        """
        # Map to patterns
        patterns_first = self.pattern_library.get_patterns_for_mask(
            mask_first / 255,
            alpha_cutoff=alpha_cutoff
        )

        patterns_second = self.pattern_library.get_patterns_for_mask(
            mask_second / 255,
            alpha_cutoff=alpha_cutoff
        )

        # Combine into big arrays
        big_array_first = np.block([
            [patterns_first[i, j] for j in range(patterns_first.shape[1])]
            for i in range(patterns_first.shape[0])
        ])

        big_array_second = np.block([
            [patterns_second[i, j] for j in range(patterns_second.shape[1])]
            for i in range(patterns_second.shape[0])
        ])

        # Add padding (same geometry as _build_mosaic)
        pond_width = 6
        pad_size = ((pond_width - 3) * (2 * self.level - 1) + 1 + 2) // 2
        pad_tuple = (pad_size, pad_size)

        mask_padded_first = np.pad(
            big_array_first, pad_width=((0, 0), pad_tuple), constant_values=0
        )
        mask_padded_second = np.pad(
            big_array_second, pad_width=(pad_tuple, (0, 0)), constant_values=0
        )

        # Combine and fill holes so the subject interior is fully masked
        mask = mask_padded_first + mask_padded_second
        mask = binary_fill_holes(mask).astype(np.uint8)

        return mask

    def _build_subject_mask(self,
                            mask_first: np.ndarray,
                            mask_second: np.ndarray,
                            alpha_cutoff: float) -> np.ndarray:
        """
        Build a pixel-level boolean mask identifying "subject" tiles.

        A tile is a subject tile when its alpha value in the original image
        is >= alpha_cutoff (i.e. it belongs to the opaque portrait area).
        Subject tiles keep gol_background colour even when they produce empty
        GoL patterns (bright highlights), so they must never be painted rim_colour.

        The result has the same shape as the gol_mosaic array (before aspect
        ratio cropping).

        Args:
            mask_first: First diagonal alpha mask (from ImageProcessor)
            mask_second: Second diagonal alpha mask (from ImageProcessor)
            alpha_cutoff: Opacity threshold (0-1); tiles >= cutoff are subject tiles

        Returns:
            Boolean array matching gol_mosaic shape; True = subject tile pixel
        """
        tile_h = self.pattern_library.pond_pattern_edge().shape[0]
        tile_w = self.pattern_library.pond_pattern_edge().shape[1]

        # Boolean per lowres tile: True = opaque (subject)
        subject_first = (mask_first / 255 >= alpha_cutoff)
        subject_second = (mask_second / 255 >= alpha_cutoff)

        # Expand each tile cell to tile_h × tile_w pixels
        big_first = np.repeat(np.repeat(subject_first, tile_h, axis=0), tile_w, axis=1)
        big_second = np.repeat(np.repeat(subject_second, tile_h, axis=0), tile_w, axis=1)

        # Apply same padding as _build_mosaic
        pond_width = 6
        pad_size = ((pond_width - 3) * (2 * self.level - 1) + 1 + 2) // 2
        pad_tuple = (pad_size, pad_size)

        padded_first = np.pad(big_first, pad_width=((0, 0), pad_tuple), constant_values=False)
        padded_second = np.pad(big_second, pad_width=(pad_tuple, (0, 0)), constant_values=False)

        return padded_first | padded_second

    def _adjust_aspect_ratio(self,
                              mosaic: np.ndarray,
                              mask: np.ndarray,
                              aspect_ratio: float,
                              offset: int = 0,
                              extra: Optional[np.ndarray] = None) -> tuple:
        """
        Crop mosaic and mask to original aspect ratio.

        Args:
            mosaic: Square GoL mosaic
            mask: Square transparency mask
            aspect_ratio: Original width/height ratio
            offset: Optional offset for cropping
            extra: Optional extra array to crop identically (e.g. subject_mask)

        Returns:
            Tuple of (cropped_mosaic, cropped_mask, cropped_extra).
            cropped_extra is None when extra is None.
        """
        if aspect_ratio == 1.0:
            return mosaic, mask, extra

        # Get tile dimensions
        tile_height = self.pattern_library.pond_pattern_edge().shape[0]
        tile_width = self.pattern_library.pond_pattern_edge().shape[1]

        if aspect_ratio > 1:
            # Originally wider than tall: crop height
            new_height = int(mosaic.shape[1] / aspect_ratio)
            new_height = int(math.ceil(new_height / tile_height) * tile_height)

            start_idx = (mosaic.shape[1] - new_height) // 2
            s = slice(start_idx - offset, start_idx + new_height + offset)
            mosaic = mosaic[s, :]
            mask = mask[s, :]
            if extra is not None:
                extra = extra[s, :]

        else:
            # Originally taller than wide: crop width
            new_width = int(mosaic.shape[0] * aspect_ratio)
            new_width = int(math.ceil(new_width / tile_width) * tile_width)

            start_idx = (mosaic.shape[0] - new_width) // 2
            s = slice(start_idx - offset, start_idx + new_width + offset)
            mosaic = mosaic[:, s]
            mask = mask[:, s]
            if extra is not None:
                extra = extra[:, s]

        return mosaic, mask, extra

    def _apply_eca_background(self,
                               gol_mosaic: np.ndarray,
                               transparency_mask: np.ndarray,
                               supersample: int,
                               subject_mask: Optional[np.ndarray] = None,
                               no_eca: bool = False) -> Image.Image:
        """
        Generate ECA background and composite with GoL mosaic.

        After compositing, pixels in the rim are painted ``rim_colour``.
        The rim is identified using ``subject_mask``: pixels that do NOT belong
        to an opaque (subject) tile and are not live GoL cells and are not in
        the ECA area are rim pixels.  Subject tiles — even if they produce
        empty GoL patterns (bright highlights) — keep ``gol_background`` and
        are never painted ``rim_colour``.

        Args:
            gol_mosaic: Binary GoL mosaic
            transparency_mask: Binary transparency mask (1=ECA area, 0=GoL area)
            supersample: ECA upsampling factor
            subject_mask: Boolean array (same shape as gol_mosaic); True = pixel
                belongs to an opaque subject tile. When None, rim painting falls
                back to the tile-content heuristic.
            no_eca: If True, skip ECA pattern generation

        Returns:
            Final composited RGBA image

        Raises:
            ValueError: If supersample doesn't divide dimensions evenly
        """
        height, width = gol_mosaic.shape

        if no_eca:
            eca_pattern = np.zeros((height, width), dtype=np.uint8)
        else:
            # Validate supersample
            if not self.eca_generator.validate_supersample(width, supersample):
                valid = self.eca_generator.list_valid_supersamples(width)
                raise ValueError(
                    f"supersample={supersample} is incompatible with "
                    f"mosaic width={width}. Valid values: {valid}"
                )

            eca_pattern = self.eca_generator.generate(
                width=width,
                height=height,
                supersample=supersample
            )

        # Create ECA mask: 0=rim/transparent, 1=eca_background, 2=eca_pixel
        eca_mask = transparency_mask * (eca_pattern + transparency_mask)

        # Render and composite
        base_image = self.renderer.render_gol_mosaic(gol_mosaic)
        overlay_image = self.renderer.render_eca_overlay(eca_mask)
        final_image = self.renderer.composite(base_image, overlay_image)

        # Paint rim: pixels outside the portrait silhouette (non-subject tiles)
        # that have no live GoL cell and are not in the ECA area get rim_colour.
        rim_rgb = MosaicRenderer._hex_to_rgb(self.colour_scheme.rim_colour)
        final_array = np.array(final_image)

        if subject_mask is not None:
            rim_area = ~subject_mask & (gol_mosaic == 0) & (transparency_mask == 0)
        else:
            # Fallback: tile-content heuristic (has tile-alignment bug after crop)
            tile_h = self.pattern_library.pond_pattern_edge().shape[0]
            tile_w = self.pattern_library.pond_pattern_edge().shape[1]
            h, w = gol_mosaic.shape
            nh, nw = h // tile_h, w // tile_w
            tile_has_content = (
                gol_mosaic[:nh * tile_h, :nw * tile_w]
                .reshape(nh, tile_h, nw, tile_w)
                .any(axis=(1, 3))
            )
            content_mask = np.repeat(np.repeat(tile_has_content, tile_h, axis=0), tile_w, axis=1)
            rim_area = ~content_mask & (gol_mosaic == 0) & (transparency_mask == 0)

        final_array[rim_area, :3] = rim_rgb
        final_array[rim_area, 3] = 255  # fully opaque

        return Image.fromarray(final_array, mode='RGBA')

    def _auto_select_supersample(self, mosaic_width: int, target: int = 15) -> int:
        """
        Automatically select a valid supersample value close to target.

        Finds the divisor of mosaic_width that's closest to the target value.
        If there are two equally close values, randomly chooses between them
        for variety.

        Args:
            mosaic_width: Width of the mosaic in pixels
            target: Target supersample value (default: 15)

        Returns:
            Valid supersample value closest to target

        Example:
            >>> # For width=144, valid values include [1,2,3,4,6,8,9,12,16,18,...]
            >>> # Target 15 → returns 16 or 12 (randomly chosen, both distance 3)
        """
        valid_values = self.eca_generator.list_valid_supersamples(mosaic_width)

        distances = [abs(v - target) for v in valid_values]
        min_distance = min(distances)
        closest_values = [v for v, d in zip(valid_values, distances) if d == min_distance]

        if len(closest_values) > 1:
            return int(np.random.choice(closest_values))
        return closest_values[0]

    def _auto_select_grid_size(self) -> int:
        """Randomly select a grid size from predefined options."""
        GRID_SIZES = [40, 60, 80, 100, 120]
        return int(np.random.choice(GRID_SIZES))

    def _auto_select_level(self) -> int:
        """Randomly select a pattern complexity level from predefined options."""
        LEVELS = [3, 4, 5]
        return int(np.random.choice(LEVELS))

    def _auto_select_eca_rule(self) -> int:
        """Randomly select an ECA rule from interesting complex and chaotic rules."""
        interesting_rules = ECABackground.COMPLEX_RULES + ECABackground.CHAOTIC_RULES
        return int(np.random.choice(interesting_rules))

    def __repr__(self) -> str:
        """String representation of generator."""
        return (
            f"MosaicGenerator("
            f"level={self.level}, "
            f"grid_size={self.grid_size}, "
            f"eca_rule={self.eca_rule})"
        )

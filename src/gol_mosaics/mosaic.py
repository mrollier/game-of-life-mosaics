"""
Main mosaic generation API.

This module provides the MosaicGenerator class, the primary interface
for converting images to Game of Life mosaics.
"""

import numpy as np
import math
from PIL import Image
from typing import Optional, Union
from scipy.ndimage import binary_fill_holes

from .patterns import PatternLibrary
from .colors import ColorScheme
from .image_processing import ImageProcessor
from .eca import ECABackground
from .renderer import MosaicRenderer


class MosaicGenerator:
    """
    Main API for generating Game of Life mosaics from images.

    This class orchestrates all components to produce the final artwork:
    pattern library, image processing, ECA backgrounds, and rendering.

    Attributes:
        level: Pattern complexity level (1-5 pre-computed, others need generation)
        grid_size: Number of tiles in the grid
        color_scheme: ColorScheme for rendering
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
                 color_scheme: Optional[ColorScheme] = None,
                 eca_rule: Optional[int] = None,
                 random_patterns: bool = True,
                 invert: bool = True):
        """
        Initialise mosaic generator.

        Args:
            level: Pattern complexity level (2-5 for pre-computed patterns)
            grid_size: Number of tiles in the grid (must be even)
            color_scheme: ColorScheme instance (defaults to UGent colours)
            eca_rule: Rule number for Elementary Cellular Automaton background.
                     If None, randomly selects from interesting rules for variety.
            random_patterns: Use random pattern selection vs deterministic
            invert: Invert the density mapping (dark = dense patterns)

        Raises:
            ValueError: If grid_size is odd

        Example:
            >>> from gol_mosaics import MosaicGenerator, ColorScheme
            >>> colors = ColorScheme.monochrome()
            >>> generator = MosaicGenerator(
            ...     level=5,
            ...     grid_size=100,
            ...     color_scheme=colors,
            ...     eca_rule=54
            ... )
        """
        # Pick random grid size and level if not provided
        self.grid_size = grid_size or self._auto_select_grid_size()
        if self.grid_size % 2 != 0:
            raise ValueError(
                f"grid_size must be even, got {self.grid_size}. "
                "The diamond layout interlocks two diagonal grids and "
                "requires an even number of tiles."
            )
        self.level = level or self._auto_select_level()

        # Pick random ECA rule from some interesting ones if not provided
        self.eca_rule = eca_rule or self._auto_select_eca_rule()

        # Select default UGent colour scheme if not provided
        self.color_scheme = color_scheme or ColorScheme.ugent()

        # Pick random patterns and invert colours.
        # These can be touched but generally look better with default values.
        self.random_patterns = random_patterns
        self.invert = invert

        # Lazy-initialized components
        self._pattern_library: Optional[PatternLibrary] = None
        self._renderer: Optional[MosaicRenderer] = None
        self._eca_generator: Optional[ECABackground] = None

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
            self._renderer = MosaicRenderer(self.color_scheme)
        return self._renderer

    @property
    def eca_generator(self) -> ECABackground:
        """Get ECA generator (lazy-loaded)."""
        if self._eca_generator is None:
            self._eca_generator = ECABackground(self.eca_rule)
        return self._eca_generator

    def generate_from_image(self,
                           image_path: str,
                           empty_tiles_cutoff: float = 0.65,
                           alpha_cutoff: float = 0.5,
                           supersample: Optional[int] = None,
                           no_eca = False,
                           remove_background: Union[bool, str] = 'auto',
                           contrast: float = 5.0,
                           seed: Optional[int] = None) -> Image.Image:
        """
        Generate mosaic from an image file.

        Thin path-based wrapper around generate_from_pil: it opens the file and
        delegates the full pipeline. See generate_from_pil for the parameters.

        Args:
            image_path: Path to input image (PNG, JPG, etc.). All other
                arguments are forwarded to generate_from_pil.

        Returns:
            PIL Image in RGBA mode with mosaic and ECA background

        Raises:
            FileNotFoundError: If image_path doesn't exist

        Example:
            >>> generator = MosaicGenerator(level=5, grid_size=100)
            >>> mosaic = generator.generate_from_image('portrait.png')
            >>> mosaic.save('output.png')
        """
        return self.generate_from_pil(
            Image.open(image_path),
            empty_tiles_cutoff=empty_tiles_cutoff,
            alpha_cutoff=alpha_cutoff,
            supersample=supersample,
            no_eca=no_eca,
            remove_background=remove_background,
            contrast=contrast,
            seed=seed,
        )

    def generate_from_pil(self,
                          img: Image.Image,
                          empty_tiles_cutoff: float = 0.65,
                          alpha_cutoff: float = 0.5,
                          supersample: Optional[int] = None,
                          no_eca = False,
                          remove_background: Union[bool, str] = 'auto',
                          contrast: float = 5.0,
                          seed: Optional[int] = None) -> Image.Image:
        """
        Generate mosaic from an in-memory PIL image.

        This is the main pipeline: preprocessing, pattern mapping, ECA background
        generation, and final rendering. It accepts an already-loaded image so a
        web backend can process an upload without writing a temp file.

        Args:
            img: Input PIL Image (any mode; converted internally).
            empty_tiles_cutoff: Threshold for empty tiles (0-1).
                Greyscale values above this become empty tiles. Default: 0.65.
            alpha_cutoff: Threshold for transparency masking (0-1).
                Alpha values below this get filled with ECA. Default: 0.5.
            supersample: ECA upsampling factor (must divide mosaic width evenly).
                Higher values create finer ECA patterns.
                If None (default), automatically selects a valid value close to 15.
            no_eca: If True, skip the ECA background.
            remove_background: Background removal mode (default: 'auto').
                'auto' removes the background only when it is still present;
                True always removes it; False never does. Removal needs the
                optional 'rembg' package.
            contrast: Sigmoid contrast strength applied to the greyscale before
                tiling (default 5.0; 0 disables). See ImageProcessor.enhance_contrast.
            seed: Optional integer to seed numpy's global RNG before generation,
                so the same image and settings reproduce the same mosaic.
                Note: ColorScheme.warhol() uses its own np.random.default_rng()
                and is therefore NOT made reproducible by this seed.

        Returns:
            PIL Image in RGBA mode with mosaic and ECA background

        Raises:
            ValueError: If supersample doesn't divide mosaic width evenly

        Example:
            >>> from PIL import Image
            >>> generator = MosaicGenerator(level=5, grid_size=100)
            >>> mosaic = generator.generate_from_pil(Image.open('portrait.png'))
            >>> mosaic.save('output.png')
        """
        # Seed numpy's global RNG for reproducibility when requested. Pattern,
        # supersample and ECA selection all draw from np.random.
        if seed is not None:
            np.random.seed(seed)

        # Preprocess image
        results = ImageProcessor.preprocess_for_mosaic(
            img,
            self.grid_size,
            remove_background=remove_background,
            contrast=contrast
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

        # Adjust for original aspect ratio
        gol_mosaic, transparency_mask = self._adjust_aspect_ratio(
            gol_mosaic,
            transparency_mask,
            aspect_ratio
        )

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
            no_eca = no_eca
        )

        return final_image

    def generate_from_gif(self,
                         gif_path: str,
                         empty_tiles_cutoff: float = 0.75,
                         alpha_cutoff: float = 0.5,
                         supersample: int = 15,
                         remove_background: Union[bool, str] = 'auto',
                         contrast: float = 5.0) -> Image.Image:
        """
        Convert animated GIF to mosaic GIF.

        Processes each frame independently and returns an animated GIF.

        Args:
            gif_path: Path to input GIF
            empty_tiles_cutoff: Threshold for empty tiles (0-1)
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
                        supersample=supersample,
                        remove_background=remove_background,
                        contrast=contrast
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
            lowres_first: First diagonal greyscale pattern
            lowres_second: Second diagonal greyscale pattern
            empty_tiles_cutoff: Threshold for empty tiles

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

        # Add padding
        pond_width = 6
        pad_size = ((pond_width - 3) * (2 * self.level - 1) + 1 + 2) // 2

        # Pad first diagonal (top rows)
        pad_tuple = (pad_size, pad_size)
        pad_width_first = ((0, 0), pad_tuple)
        mosaic_first = np.pad(big_array_first, pad_width=pad_width_first, constant_values=0)

        # Pad second diagonal (side columns)
        pad_width_second = (pad_tuple, (0, 0))
        mosaic_second = np.pad(big_array_second, pad_width=pad_width_second, constant_values=0)

        # Combine
        mosaic = mosaic_first + mosaic_second

        return mosaic

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

        # Add padding
        pond_width = 6
        pad_size = ((pond_width - 3) * (2 * self.level - 1) + 1 + 2) // 2
        pad_tuple = (pad_size, pad_size)

        pad_width_first = ((0, 0), pad_tuple)
        mask_padded_first = np.pad(big_array_first, pad_width=pad_width_first, constant_values=0)

        pad_width_second = (pad_tuple, (0, 0))
        mask_padded_second = np.pad(big_array_second, pad_width=pad_width_second, constant_values=0)

        # Combine and fill holes
        mask = mask_padded_first + mask_padded_second
        mask = binary_fill_holes(mask).astype(np.uint8)

        return mask

    def _adjust_aspect_ratio(self,
                            mosaic: np.ndarray,
                            mask: np.ndarray,
                            aspect_ratio: float,
                            offset: int = 0) -> tuple:
        """
        Crop mosaic and mask to original aspect ratio.

        Args:
            mosaic: Square GoL mosaic
            mask: Square transparency mask
            aspect_ratio: Original width/height ratio
            offset: Optional offset for cropping

        Returns:
            Tuple of (cropped_mosaic, cropped_mask)
        """
        if aspect_ratio == 1.0:
            # Already square
            return mosaic, mask

        # Get tile dimensions
        tile_height = self.pattern_library.pond_pattern_edge().shape[0]
        tile_width = self.pattern_library.pond_pattern_edge().shape[1]

        if aspect_ratio > 1:
            # Originally wider than tall: crop height
            new_height = int(mosaic.shape[1] / aspect_ratio)
            new_height = int(math.ceil(new_height / tile_height) * tile_height)

            start_idx = (mosaic.shape[1] - new_height) // 2
            crop = slice(start_idx - offset, start_idx + new_height + offset)
            mosaic, mask = mosaic[crop, :], mask[crop, :]

        else:
            # Originally taller than wide: crop width
            new_width = int(mosaic.shape[0] * aspect_ratio)
            new_width = int(math.ceil(new_width / tile_width) * tile_width)

            start_idx = (mosaic.shape[0] - new_width) // 2
            crop = slice(start_idx - offset, start_idx + new_width + offset)
            mosaic, mask = mosaic[:, crop], mask[:, crop]

        return mosaic, mask

    def _apply_eca_background(self,
                             gol_mosaic: np.ndarray,
                             transparency_mask: np.ndarray,
                             supersample: int,
                             no_eca = False) -> Image.Image:
        """
        Generate ECA background and composite with GoL mosaic.

        Args:
            gol_mosaic: Binary GoL mosaic
            transparency_mask: Binary transparency mask
            supersample: ECA upsampling factor
            no_eca: Whether to skip ECA background generation
        Returns:
            Final composited RGBA image

        Raises:
            ValueError: If supersample doesn't divide dimensions evenly
        """
        height, width = gol_mosaic.shape

        if no_eca:
            eca_pattern = np.zeros((height, width), dtype=np.uint8)

        else:
            # Generate ECA pattern. Any positive supersample works: generate()
            # crops the upsampled pattern to the exact mosaic size, so it need
            # not divide the width or height.
            eca_pattern = self.eca_generator.generate(
                width=width,
                height=height,
                supersample=supersample
            )

        # Create ECA mask: 0=transparent, 1=eca_background, 2=eca_pixel
        eca_mask = transparency_mask * (eca_pattern + transparency_mask)

        # Render and composite
        base_image = self.renderer.render_gol_mosaic(gol_mosaic)
        overlay_image = self.renderer.render_eca_overlay(eca_mask)
        final_image = self.renderer.composite(base_image, overlay_image)

        return final_image

    def _auto_select_supersample(self, mosaic_width: int, target: int = 15) -> int:
        """
        Automatically select a supersample value.

        ECABackground.generate accepts any positive supersample (it crops the
        upsampled pattern to size), so this simply returns the target cell size.
        It is only clamped so at least one full ECA cell spans the mosaic on
        very small mosaics.

        Args:
            mosaic_width: Width of the mosaic in pixels
            target: Desired ECA cell size in pixels (default: 15)

        Returns:
            Supersample value to use

        Example:
            >>> # width=588 (no divisor equals 15) still uses 15 directly
            >>> # width=8 clamps the target down to 8
        """
        return max(1, min(target, mosaic_width))

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

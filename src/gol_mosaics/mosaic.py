"""
Main mosaic generation API.

This module provides the MosaicGenerator class, the primary interface
for converting images to Game of Life mosaics.
"""

import numpy as np
import math
from PIL import Image
from typing import Optional, Tuple, Union
from scipy.ndimage import binary_fill_holes, label

from .patterns import PatternLibrary
from .colors import ColorScheme
from .image_processing import ImageProcessor
from .eca import ECABackground
from .renderer import MosaicRenderer
from .tile_scheme import assemble as _assemble_scheme


def _fill_small_holes(binary: np.ndarray, max_hole_size: int) -> np.ndarray:
    """
    Fill enclosed holes strictly smaller than max_hole_size pixels.

    Unlike binary_fill_holes alone, this preserves large enclosed
    zero-regions such as a foreground subject that touches no image edge.

    Args:
        binary: Boolean array to fill
        max_hole_size: Holes with at least this many pixels stay open

    Returns:
        Boolean array with only the small holes filled
    """
    filled = binary_fill_holes(binary)
    holes = filled & ~binary
    labels, num_holes = label(holes)
    if num_holes == 0:
        return filled
    sizes = np.bincount(labels.ravel())
    big = sizes >= max_hole_size
    big[0] = False  # label 0 is the non-hole region
    return filled & ~big[labels]


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
                 invert: bool = True,
                 tile_shape: str = "diamond"):
        """
        Initialise mosaic generator.

        Args:
            level: Pattern complexity level (diamond: 1-6, square: 3-5
                pre-computed)
            grid_size: Number of tiles in the grid (must be even for the
                diamond layout; any positive integer for squares)
            color_scheme: ColorScheme instance (defaults to UGent colours)
            eca_rule: Rule number for Elementary Cellular Automaton background.
                     If None, randomly selects from interesting rules for variety.
            random_patterns: Use random pattern selection vs deterministic
            invert: Invert the density mapping (dark = dense patterns)
            tile_shape: Tile geometry, "diamond" (the historical 45-degree
                pond-diamond layout) or "square" (axis-aligned pond-frame
                square tiles)

        Raises:
            ValueError: If grid_size is odd for the diamond layout, or
                tile_shape is unknown

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
        if tile_shape not in ("diamond", "square"):
            raise ValueError(
                f"Unknown tile shape {tile_shape!r}; expected 'diamond' or "
                f"'square'."
            )
        self.tile_shape = tile_shape

        # Pick random grid size, level, and ECA rule if not provided. Use
        # explicit None checks (not `or`): falsy values like eca_rule=0 (a
        # valid Wolfram rule) must not be silently replaced by a random pick.
        self.grid_size = (self._auto_select_grid_size() if grid_size is None
                          else grid_size)
        if tile_shape == "diamond" and self.grid_size % 2 != 0:
            raise ValueError(
                f"grid_size must be even, got {self.grid_size}. "
                "The diamond layout interlocks two diagonal grids and "
                "requires an even number of tiles."
            )
        self.level = self._auto_select_level() if level is None else level
        self.eca_rule = (self._auto_select_eca_rule() if eca_rule is None
                         else eca_rule)

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
            self._pattern_library = PatternLibrary.load(self.level,
                                                        shape=self.tile_shape)
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
                           no_eca: bool = False,
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
                          no_eca: bool = False,
                          remove_background: Union[bool, str] = 'auto',
                          contrast: float = 5.0,
                          seed: Optional[int] = None,
                          return_arrays: bool = False
                          ) -> Union[Image.Image,
                                     Tuple[Image.Image, np.ndarray, np.ndarray]]:
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
            supersample: ECA upsampling factor (any positive value; the ECA
                pattern is cropped to the mosaic size). Higher values create
                finer ECA patterns. If None (default), a ~15-pixel cell size
                is used, clamped to the mosaic width.
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
            return_arrays: If True, also return the binary GoL mosaic and the
                transparency mask (both aspect-adjusted) alongside the image, as
                ``(image, gol_mosaic, transparency_mask)``. The GoL mosaic is the
                still-life pattern (0/1-valued), suitable for exporting to Golly.

        Returns:
            PIL Image in RGBA mode with mosaic and ECA background. If
            return_arrays is True, a tuple (image, gol_mosaic, transparency_mask).

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

        if self.tile_shape == "square":
            # Axis-aligned lattice: one rectangular tile grid sized straight
            # from the aspect ratio, so no diagonal split and no later crop.
            lowres, lowres_mask, _ = ImageProcessor.preprocess_for_square_mosaic(
                img,
                self.grid_size,
                remove_background=remove_background,
                contrast=contrast
            )
            gol_mosaic = self._build_square_mosaic(lowres, empty_tiles_cutoff)
            transparency_mask = self._build_square_mask(
                lowres_mask, alpha_cutoff, gol_mosaic.shape)
        else:
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

        # Auto-select supersample if not provided: any positive value works
        # (the ECA is cropped to size), so use a 15-pixel cell target clamped
        # so at least one full cell spans very small mosaics.
        if supersample is None:
            supersample = max(1, min(15, gol_mosaic.shape[1]))

        # Apply ECA background and render
        final_image = self._apply_eca_background(
            gol_mosaic,
            transparency_mask,
            supersample,
            no_eca = no_eca
        )

        if return_arrays:
            return final_image, gol_mosaic, transparency_mask
        return final_image

    def generate_from_gif(self,
                         gif_path: str,
                         empty_tiles_cutoff: float = 0.65,
                         alpha_cutoff: float = 0.5,
                         supersample: Optional[int] = None,
                         remove_background: Union[bool, str] = 'auto',
                         contrast: float = 5.0) -> Image.Image:
        """
        Convert animated GIF to mosaic GIF.

        Processes each frame independently, in memory, via generate_from_pil.
        Defaults match the single-image path.

        Known limitation: only the first processed frame is returned (carrying
        the source's duration/loop metadata), so saving it with save_all=True
        does not yet write a multi-frame animation.

        Args:
            gif_path: Path to input GIF
            empty_tiles_cutoff: Threshold for empty tiles (0-1)
            alpha_cutoff: Threshold for transparency masking (0-1)
            supersample: ECA upsampling factor (None = auto, as for images)
            remove_background: Forwarded to generate_from_pil
            contrast: Forwarded to generate_from_pil

        Returns:
            PIL Image (first processed frame with animation metadata)

        Example:
            >>> generator = MosaicGenerator(level=4, grid_size=50)
            >>> mosaic_gif = generator.generate_from_gif('animation.gif')
            >>> mosaic_gif.save('output.gif', save_all=True)
        """
        gif = Image.open(gif_path)
        frames = []
        durations = []

        # Process each frame
        frame_num = 0
        try:
            while True:
                mosaic = self.generate_from_pil(
                    gif.convert('RGBA'),
                    empty_tiles_cutoff=empty_tiles_cutoff,
                    alpha_cutoff=alpha_cutoff,
                    supersample=supersample,
                    remove_background=remove_background,
                    contrast=contrast
                )
                frames.append(mosaic)
                durations.append(gif.info.get('duration', 100))

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

        mosaic_first, mosaic_second = self._pad_diagonals(
            self._assemble_tiles(patterns_first),
            self._assemble_tiles(patterns_second)
        )

        # The offset grids interlock without overlap (each grid's live cells
        # fall inside the other grid's dead padding), so the sum stays binary.
        return mosaic_first + mosaic_second

    def _build_square_mosaic(self,
                             lowres: np.ndarray,
                             empty_tiles_cutoff: float) -> np.ndarray:
        """
        Build GoL mosaic on the axis-aligned square lattice.

        Adjacent square tiles share their border pond band, so tiles are
        selected by index and pasted with the scheme assembler (which checks
        overlap consistency) rather than block-stacked. Values above the
        cutoff leave true holes (index -1: no tile at that site), which stay
        globally stable because any subset of the frame lattice is a still
        life.

        Args:
            lowres: (rows, cols) greyscale array, values 0-255
            empty_tiles_cutoff: Threshold for empty tiles

        Returns:
            Complete GoL mosaic as binary array
        """
        library = self.pattern_library
        indices = library.get_indices_for_values(
            lowres / 255,
            random=self.random_patterns,
            invert=self.invert,
            empty_tiles_cutoff=empty_tiles_cutoff
        )
        return _assemble_scheme(library.scheme, indices, library.solutions)

    def _build_square_mask(self,
                           lowres_mask: np.ndarray,
                           alpha_cutoff: float,
                           out_shape: Tuple[int, int]) -> np.ndarray:
        """
        Build the transparency mask for the square lattice.

        Same semantics as the diamond path's _build_mask: 1 where the ECA
        background is drawn (alpha below the cutoff), 0 on the opaque
        subject. Each mosaic cell takes the value of the tile whose centre
        is nearest, so the mask is exact on the lattice with no interlock
        gaps to fill.

        Args:
            lowres_mask: (rows, cols) alpha values 0-255 (255 = opaque)
            alpha_cutoff: Threshold for transparency
            out_shape: Shape of the assembled mosaic

        Returns:
            Complete transparency mask as binary array
        """
        background = (lowres_mask / 255 < alpha_cutoff).astype(np.uint8)
        scheme = self.pattern_library.scheme
        n, pitch, pad = scheme.n, scheme.u[0], 2
        owners = []
        for size, count in zip(out_shape, background.shape):
            owner = (np.arange(size) - pad - (n - pitch) // 2) // pitch
            owners.append(np.clip(owner, 0, count - 1))
        return background[np.ix_(owners[0], owners[1])]

    def _assemble_tiles(self, patterns: np.ndarray) -> np.ndarray:
        """Assemble a (rows, cols, H, W) array of tiles into one 2D grid."""
        return np.block([
            [patterns[i, j] for j in range(patterns.shape[1])]
            for i in range(patterns.shape[0])
        ])

    def _pad_diagonals(self,
                       first: np.ndarray,
                       second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Offset the two diagonal grids so their tiles interlock.

        The first grid is padded horizontally and the second vertically by
        the library's tile_pad_size, shifting them half a tile relative to
        each other.
        """
        pad_tuple = (self.pattern_library.tile_pad_size,) * 2
        first_padded = np.pad(first, pad_width=((0, 0), pad_tuple),
                              constant_values=0)
        second_padded = np.pad(second, pad_width=(pad_tuple, (0, 0)),
                               constant_values=0)
        return first_padded, second_padded

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

        mask_padded_first, mask_padded_second = self._pad_diagonals(
            self._assemble_tiles(patterns_first),
            self._assemble_tiles(patterns_second)
        )

        # Combine the two diagonal grids. Where their tiles' dead borders
        # cross, the background is left with tiny enclosed gaps (a few pixels
        # each) that must be filled so the ECA field renders solid. A subject
        # that touches no image edge is *also* an enclosed zero-region, but a
        # vastly larger one (at least about half a tile), so only fill holes
        # smaller than a quarter tile to keep the foreground intact.
        mask = (mask_padded_first + mask_padded_second) > 0
        tile_h, tile_w = self.pattern_library.tile_shape
        mask = _fill_small_holes(
            mask,
            max_hole_size=tile_h * tile_w // 4
        ).astype(np.uint8)

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
        tile_height, tile_width = self.pattern_library.tile_shape

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
                             no_eca: bool = False) -> Image.Image:
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

        return self.renderer.render_full_mosaic(gol_mosaic, eca_mask)

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

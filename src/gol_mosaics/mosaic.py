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

from .tile_library import TileLibrary
from .colours import ColourScheme
from .compose import compose
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
    tile library, image processing, ECA backgrounds, and rendering.

    Attributes:
        level: Tile level (diamond 1-6, square 3-5); tiles are 6*level cells wide
        grid_size: Number of tiles in the grid
        colours: ColourScheme for rendering
        eca_rule: Rule number for ECA background
        random_tiles: Whether to draw at random among equally dense tiles
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
                 colours: Optional[ColourScheme] = None,
                 eca_rule: Optional[int] = None,
                 random_tiles: bool = True,
                 invert: bool = True,
                 layout: str = "diamond"):
        """
        Initialise mosaic generator.

        Args:
            level: Tile level (diamond: 1-6, square: 3-5)
            grid_size: Number of tiles in the grid (must be even for the
                diamond layout; any positive integer for squares)
            colours: ColourScheme instance (defaults to UGent colours)
            eca_rule: Rule number for Elementary Cellular Automaton background.
                     If None, randomly selects from interesting rules for variety.
            random_tiles: Draw at random among equally dense tiles (else
                the first one)
            invert: Invert the density mapping (dark = dense tiles)
            layout: Tile geometry, "diamond" (the historical 45-degree
                pond-diamond layout) or "square" (axis-aligned pond-frame
                square tiles)

        Raises:
            ValueError: If grid_size is odd for the diamond layout, or
                layout is unknown

        Example:
            >>> from gol_mosaics import MosaicGenerator, ColourScheme
            >>> colors = ColourScheme.monochrome()
            >>> generator = MosaicGenerator(
            ...     level=5,
            ...     grid_size=100,
            ...     colours=colors,
            ...     eca_rule=54
            ... )
        """
        if layout not in ("diamond", "square"):
            raise ValueError(
                f"Unknown tile shape {layout!r}; expected 'diamond' or "
                f"'square'."
            )
        self.layout = layout

        # Pick random grid size, level, and ECA rule if not provided. Use
        # explicit None checks (not `or`): falsy values like eca_rule=0 (a
        # valid Wolfram rule) must not be silently replaced by a random pick.
        self.grid_size = (self._auto_select_grid_size() if grid_size is None
                          else grid_size)
        if layout == "diamond" and self.grid_size % 2 != 0:
            raise ValueError(
                f"grid_size must be even, got {self.grid_size}. "
                "The diamond layout interlocks two diagonal grids and "
                "requires an even number of tiles."
            )
        self.level = self._auto_select_level() if level is None else level
        self.eca_rule = (self._auto_select_eca_rule() if eca_rule is None
                         else eca_rule)

        # Select default UGent colour scheme if not provided
        self.colours = colours or ColourScheme.ugent()

        # Draw tiles at random and invert the tone.
        # These can be touched but generally look better with default values.
        self.random_tiles = random_tiles
        self.invert = invert

        # Lazy-initialized components
        self._library: Optional[TileLibrary] = None
        self._renderer: Optional[MosaicRenderer] = None
        self._eca: Optional[ECABackground] = None

    @property
    def library(self) -> TileLibrary:
        """The tile library (loaded on first use)."""
        if self._library is None:
            self._library = TileLibrary.load(self.level,
                                                        layout=self.layout)
        return self._library

    @property
    def renderer(self) -> MosaicRenderer:
        """Get renderer (lazy-loaded)."""
        if self._renderer is None:
            self._renderer = MosaicRenderer(self.colours)
        return self._renderer

    @property
    def eca(self) -> ECABackground:
        """Get ECA generator (lazy-loaded)."""
        if self._eca is None:
            self._eca = ECABackground(self.eca_rule)
        return self._eca

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

        This is the main pipeline: preprocessing, tile mapping, ECA background
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
                Note: ColourScheme.warhol() uses its own np.random.default_rng()
                and is therefore NOT made reproducible by this seed.
            return_arrays: If True, also return the binary GoL mosaic and the
                transparency mask (both aspect-adjusted) alongside the image, as
                ``(image, gol_mosaic, transparency_mask)``. The GoL mosaic is the
                still-life pattern (0/1-valued), suitable for exporting to Golly.

        Returns:
            PIL Image in RGBA mode with mosaic and ECA background. If
            return_arrays is True, a tuple (image, gol_mosaic, transparency_mask).

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

        if self.layout == "square":
            # Axis-aligned lattice: one rectangular tile grid sized straight
            # from the aspect ratio, so no diagonal split and no later crop.
            lowres, lowres_mask, _ = ImageProcessor.preprocess_square(
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
            results = ImageProcessor.preprocess_diamond(
                img,
                self.grid_size,
                remove_background=remove_background,
                contrast=contrast
            )
            lowres_first, lowres_second, mask_first, mask_second, aspect_ratio = results

            # The crop back to the image's aspect ratio is decided first, so
            # that tiles it would cut through are left out of the mosaic.
            n = self.library.tile_size[0]
            full_shape = (lowres_first.shape[0] * n,
                          lowres_first.shape[1] * n
                          + 2 * self.library.lattice_offset)
            window = self._crop_window(full_shape, aspect_ratio)

            # Build GoL mosaic
            gol_mosaic = self._build_mosaic(
                lowres_first,
                lowres_second,
                empty_tiles_cutoff,
                window=window
            )

            # Build transparency mask
            transparency_mask = self._build_mask(
                mask_first,
                mask_second,
                alpha_cutoff
            )

            # Crop to the original aspect ratio
            rows, cols = slice(window[0], window[1]), slice(window[2], window[3])
            gol_mosaic = gol_mosaic[rows, cols]
            transparency_mask = transparency_mask[rows, cols]

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
                     empty_tiles_cutoff: float,
                     window: Optional[Tuple[int, int, int, int]] = None
                     ) -> np.ndarray:
        """
        Build the mosaic from the two diagonal grids.

        Args:
            lowres_first: Grey values of the first diagonal grid
            lowres_second: Grey values of the second diagonal grid
            empty_tiles_cutoff: Threshold for empty tiles
            window: (top, bottom, left, right) of the region that will be
                kept; tiles not wholly inside it are left empty, because a
                cut tile is no longer a still life. None keeps every tile.

        Returns:
            Complete GoL mosaic as binary array
        """
        # Map grey values to tiles
        tiles_first = self.library.tiles_for_values(
            lowres_first / 255,
            random=self.random_tiles,
            invert=self.invert,
            empty_tiles_cutoff=empty_tiles_cutoff
        )

        tiles_second = self.library.tiles_for_values(
            lowres_second / 255,
            random=self.random_tiles,
            invert=self.invert,
            empty_tiles_cutoff=empty_tiles_cutoff
        )

        if window is not None:
            pad = self.library.lattice_offset
            tiles_first[~self._inside(tiles_first.shape[:2], 0, pad, window)] = 0
            tiles_second[~self._inside(tiles_second.shape[:2], pad, 0, window)] = 0

        mosaic_first, mosaic_second = self._pad_diagonals(
            self._assemble_tiles(tiles_first),
            self._assemble_tiles(tiles_second)
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
        library = self.library
        indices = library.indices_for_values(
            lowres / 255,
            random=self.random_tiles,
            invert=self.invert,
            empty_tiles_cutoff=empty_tiles_cutoff
        )
        return _assemble_scheme(library.scheme, indices, library.tiles)

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
        scheme = self.library.scheme
        n, pitch, pad = scheme.n, scheme.u[0], 2
        owners = []
        for size, count in zip(out_shape, background.shape):
            owner = (np.arange(size) - pad - (n - pitch) // 2) // pitch
            owners.append(np.clip(owner, 0, count - 1))
        return background[np.ix_(owners[0], owners[1])]

    def _assemble_tiles(self, tiles: np.ndarray) -> np.ndarray:
        """Assemble a (rows, cols, H, W) array of tiles into one 2D grid."""
        rows, cols, h, w = tiles.shape
        return tiles.transpose(0, 2, 1, 3).reshape(rows * h, cols * w)

    def _pad_diagonals(self,
                       first: np.ndarray,
                       second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Offset the two diagonal grids so their tiles interlock.

        The first grid is padded horizontally and the second vertically by
        the library's lattice_offset, shifting them half a tile relative to
        each other.
        """
        pad_tuple = (self.library.lattice_offset,) * 2
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
        Build the transparency mask from the two diagonal grids.

        Args:
            mask_first: First diagonal alpha mask
            mask_second: Second diagonal alpha mask
            alpha_cutoff: Threshold for transparency

        Returns:
            Complete transparency mask as binary array
        """
        # Map alpha values to solid and empty tiles
        tiles_first = self.library.tiles_for_mask(
            mask_first / 255,
            alpha_cutoff=alpha_cutoff
        )

        tiles_second = self.library.tiles_for_mask(
            mask_second / 255,
            alpha_cutoff=alpha_cutoff
        )

        mask_padded_first, mask_padded_second = self._pad_diagonals(
            self._assemble_tiles(tiles_first),
            self._assemble_tiles(tiles_second)
        )

        # Combine the two diagonal grids. Where their tiles' dead borders
        # cross, the background is left with tiny enclosed gaps (a few pixels
        # each) that must be filled so the ECA field renders solid. A subject
        # that touches no image edge is *also* an enclosed zero-region, but a
        # vastly larger one (at least about half a tile), so only fill holes
        # smaller than a quarter tile to keep the foreground intact.
        mask = (mask_padded_first + mask_padded_second) > 0
        tile_h, tile_w = self.library.tile_size
        mask = _fill_small_holes(
            mask,
            max_hole_size=tile_h * tile_w // 4
        ).astype(np.uint8)

        return mask

    def _inside(self,
                grid_shape: Tuple[int, int],
                row_offset: int,
                col_offset: int,
                window: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Which tiles of one diagonal grid lie wholly inside the window.

        Tile (r, c) occupies the box starting at (row_offset + r*n,
        col_offset + c*n). Its outermost ring is always dead and the diamond
        reaches every side of the ring's interior, so the tile survives an
        axis-aligned crop exactly when the box minus that ring fits.
        """
        n = self.library.tile_size[0]
        top, bottom, left, right = window
        i0 = row_offset + np.arange(grid_shape[0]) * n
        j0 = col_offset + np.arange(grid_shape[1]) * n
        rows = (i0 + 1 >= top) & (i0 + n - 1 <= bottom)
        cols = (j0 + 1 >= left) & (j0 + n - 1 <= right)
        return rows[:, None] & cols[None, :]

    def _crop_window(self,
                     shape: Tuple[int, int],
                     aspect_ratio: float) -> Tuple[int, int, int, int]:
        """
        (top, bottom, left, right) of the square diamond mosaic that matches
        the original aspect ratio, rounded up to whole tiles and centred.

        Args:
            shape: Shape of the square mosaic
            aspect_ratio: Original width / height
        """
        height, width = shape
        if aspect_ratio == 1.0:
            return 0, height, 0, width
        tile_height, tile_width = self.library.tile_size
        if aspect_ratio > 1:
            # Originally wider than tall: crop the height
            new_height = int(width / aspect_ratio)
            new_height = int(math.ceil(new_height / tile_height) * tile_height)
            start = (width - new_height) // 2
            return start, start + new_height, 0, width
        # Originally taller than wide: crop the width
        new_width = int(height * aspect_ratio)
        new_width = int(math.ceil(new_width / tile_width) * tile_width)
        start = (height - new_width) // 2
        return 0, height, start, start + new_width

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
        """
        # Any positive supersample works: compose() crops the upsampled ECA
        # pattern to the exact mosaic size, so it need not divide the width
        # or height. no_eca leaves the flat background colour behind.
        return compose(
            gol_mosaic,
            transparency_mask,
            self.colours,
            style='flat' if no_eca else 'eca',
            rule=self.eca_rule,
            supersample=supersample
        )

    def _auto_select_grid_size(self) -> int:
        """Randomly select a grid size from predefined options."""
        GRID_SIZES = [40, 60, 80, 100, 120]
        return int(np.random.choice(GRID_SIZES))
    
    def _auto_select_level(self) -> int:
        """Randomly select a tile level from predefined options."""
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

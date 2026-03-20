"""
Image loading and preprocessing for Game of Life mosaics.

This module provides the ImageProcessor class for loading images,
applying transformations, and preparing them for mosaic conversion.
"""

import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Union


class ImageProcessor:
    """
    Handles image loading, preprocessing, and transformation.

    All methods are static as image processing is stateless.
    The class provides utilities for:
    - Loading images with alpha channel handling
    - Making images square with padding
    - Rotating and pixelating images
    - Extracting diagonal patterns for diamond layout

    Example:
        >>> img = ImageProcessor.load_image('portrait.png')
        >>> square_img, aspect = ImageProcessor.square_image(img)
        >>> lowres = ImageProcessor.rotate_and_pixelate(square_img, grid_size=30)
    """

    @staticmethod
    def load_image(image_path: str,
                  alpha_colour: str = 'white',
                  return_alpha: bool = False) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """
        Load an image from file with alpha channel handling.

        Transparent pixels are composited onto a solid colour background
        before converting to grayscale.

        Args:
            image_path: Path to image file (PNG, JPG, etc.)
            alpha_colour: Colour for transparent background (default: 'white')
            return_alpha: If True, also return the alpha mask

        Returns:
            If return_alpha=False: Grayscale PIL Image
            If return_alpha=True: Tuple of (grayscale Image, alpha mask Image)

        Raises:
            FileNotFoundError: If image_path doesn't exist
            IOError: If image cannot be loaded

        Example:
            >>> img = ImageProcessor.load_image('portrait.png')
            >>> img, mask = ImageProcessor.load_image('portrait.png', return_alpha=True)
        """
        # Open image
        img = Image.open(image_path)

        # Convert to RGBA and extract alpha channel
        img = img.convert('RGBA')
        mask = img.split()[-1]

        # Composite onto background colour
        bg = Image.new('RGBA', img.size, alpha_colour)
        bg.paste(img, mask=mask)
        img = bg

        # Convert to grayscale
        img = img.convert('L')

        if return_alpha:
            return img, mask
        return img

    @staticmethod
    def square_image(img: Image.Image,
                    return_aspect: bool = True,
                    fill_colour: str = 'white') -> Union[Image.Image, Tuple[Image.Image, float]]:
        """
        Make image square by padding with specified colour.

        Preserves the original aspect ratio by adding padding to the shorter
        dimension rather than cropping.

        Args:
            img: PIL Image to make square
            return_aspect: If True, also return original aspect ratio
            fill_colour: Colour for padding (default: 'white')

        Returns:
            If return_aspect=False: Square PIL Image
            If return_aspect=True: Tuple of (square Image, width/height ratio)

        Example:
            >>> img = Image.open('portrait.jpg')
            >>> square_img, aspect = ImageProcessor.square_image(img)
            >>> print(f"Original aspect ratio: {aspect:.2f}")
        """
        width, height = img.size
        width_over_height = width / height

        if width != height:
            size = max(width, height)
            img = ImageOps.pad(img, (size, size), color=fill_colour)

        if return_aspect:
            return img, width_over_height
        return img

    @staticmethod
    def rotate_and_pixelate(img: Image.Image,
                           grid_size: int,
                           expand: bool = True) -> np.ndarray:
        """
        Rotate image 45 degrees and pixelate to low resolution.

        This creates the diagonal tile layout characteristic of this
        mosaic style.

        Process:
        1. Resize to grid_size x grid_size
        2. Rotate 45 degrees
        3. Return as numpy array (edges trimmed)

        Args:
            img: PIL Image to process
            grid_size: Target grid size (must be even)
            expand: If True, expand canvas during rotation to avoid cropping

        Returns:
            Low-resolution numpy array (slightly smaller than rotated size)

        Raises:
            ValueError: If grid_size is odd

        Example:
            >>> img = Image.open('portrait.jpg')
            >>> lowres = ImageProcessor.rotate_and_pixelate(img, grid_size=30)
            >>> lowres.shape
            (41, 41)  # Approximately sqrt(2) * grid_size
        """
        if grid_size % 2 != 0:
            raise ValueError(
                f"Grid size must be even, got {grid_size}"
            )

        # Resample to desired grid size
        img = img.resize((grid_size, grid_size), resample=Image.LANCZOS)

        # Rotate 45 degrees
        img = img.rotate(45, expand=expand, fillcolor=255)

        # Convert to numpy and trim edges
        arr = np.array(img)[1:-1, 1:-1]
        return arr

    @staticmethod
    def extract_diagonal_patterns(lowres: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract two diagonal pattern grids from rotated image.

        After rotating 45 degrees, the image has a diamond structure.
        This function extracts two interlocking diagonal grids that will
        become the even and odd tiles in the final mosaic.

        Args:
            lowres: Low-resolution array from rotate_and_pixelate()

        Returns:
            Tuple of (first_diagonal_pattern, second_diagonal_pattern)
            Both are 2D arrays representing the tile grids

        Example:
            >>> lowres = ImageProcessor.rotate_and_pixelate(img, 30)
            >>> diag1, diag2 = ImageProcessor.extract_diagonal_patterns(lowres)
            >>> diag1.shape
            (16, 16)  # Approximately grid_size/2 + 1
        """
        grid_size = lowres.shape[0]

        # First diagonal indices
        diag_indices_first = [
            [(grid_size // 2 - 1 - i + j, i + j) for i in range(grid_size // 2)]
            for j in range(grid_size // 2 + 1)
        ]
        diag_indices_first = np.array(diag_indices_first)
        diag_indices_first_shape = diag_indices_first.shape
        diag_indices_first = diag_indices_first.reshape(-1, diag_indices_first.shape[-1])

        # Second diagonal indices
        diag_indices_second = [
            [(grid_size // 2 - i + j, i + j) for i in range(grid_size // 2 + 1)]
            for j in range(grid_size // 2)
        ]
        diag_indices_second = np.array(diag_indices_second)
        diag_indices_second_shape = diag_indices_second.shape
        diag_indices_second = diag_indices_second.reshape(-1, diag_indices_second.shape[-1])

        # Extract patterns
        rows_first, cols_first = zip(*diag_indices_first)
        rows_second, cols_second = zip(*diag_indices_second)

        lowres_first = lowres[rows_first, cols_first].reshape(diag_indices_first_shape[:2])
        lowres_second = lowres[rows_second, cols_second].reshape(diag_indices_second_shape[:2])

        return lowres_first, lowres_second

    @classmethod
    def preprocess_for_mosaic(cls,
                             image_path: str,
                             grid_size: int,
                             alpha_colour: str = 'white',
                             fill_colour: str = 'white') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Complete preprocessing pipeline from image file to diagonal patterns.

        This is a convenience method that chains together the full preprocessing
        workflow.

        Args:
            image_path: Path to input image
            grid_size: Target grid size (must be even)
            alpha_colour: Background colour for transparent pixels
            fill_colour: Padding colour for squaring

        Returns:
            Tuple of:
            - lowres_first: First diagonal grayscale pattern
            - lowres_second: Second diagonal grayscale pattern
            - mask_first: First diagonal alpha mask
            - mask_second: Second diagonal alpha mask
            - aspect_ratio: Original width/height ratio

        Example:
            >>> results = ImageProcessor.preprocess_for_mosaic('portrait.png', grid_size=30)
            >>> lowres_first, lowres_second, mask_first, mask_second, aspect = results
            >>> print(f"Aspect ratio: {aspect:.2f}")
        """
        # Load image and mask
        img, mask = cls.load_image(image_path, alpha_colour=alpha_colour, return_alpha=True)

        # Process grayscale image
        square_img, aspect_ratio = cls.square_image(img, return_aspect=True, fill_colour=fill_colour)
        lowres = cls.rotate_and_pixelate(square_img, grid_size, expand=True)
        lowres_first, lowres_second = cls.extract_diagonal_patterns(lowres)

        # Process alpha mask
        square_mask = cls.square_image(mask, return_aspect=False, fill_colour='white')
        lowres_mask = cls.rotate_and_pixelate(square_mask, grid_size, expand=True)
        mask_first, mask_second = cls.extract_diagonal_patterns(lowres_mask)

        return lowres_first, lowres_second, mask_first, mask_second, aspect_ratio

    @staticmethod
    def list_denominators(n: int) -> list:
        """
        Return all positive divisors of a positive integer.

        Useful for finding valid supersample values that divide
        the mosaic width evenly (no pixel interpolation needed).

        Args:
            n: Positive integer

        Returns:
            Sorted list of divisors

        Raises:
            ValueError: If n is not a positive integer

        Example:
            >>> ImageProcessor.list_denominators(60)
            [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
            >>> # These are all valid supersample values for width=60
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        small, large = [], []
        i = 1
        while i * i <= n:
            if n % i == 0:
                small.append(i)
                j = n // i
                if j != i:
                    large.append(j)
            i += 1

        return small + large[::-1]

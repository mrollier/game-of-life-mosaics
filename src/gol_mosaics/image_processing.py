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

    # Cached rembg sessions, keyed by model name (created lazily on first use).
    _REMBG_SESSIONS = {}

    # ONNX Runtime execution providers for background removal. None auto-detects
    # (prefer GPU/accelerator, fall back to CPU). Set to a list to force, e.g.
    # ImageProcessor.background_removal_providers = ['CPUExecutionProvider'].
    # Must be set before the first removal, as the session is cached.
    background_removal_providers = None

    @staticmethod
    def load_image(image_path: str,
                  alpha_color: str = 'white',
                  return_alpha: bool = False,
                  remove_background: Union[bool, str] = 'auto',
                  contrast: float = 5.0) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """
        Load an image from file with alpha channel handling.

        Transparent pixels are composited onto a solid color background
        before converting to grayscale. The alpha channel becomes the mask
        that tells the mosaic subject from background, so an image whose
        background is still present is optionally removed first.

        Args:
            image_path: Path to image file (PNG, JPG, etc.)
            alpha_color: Color for transparent background (default: 'white')
            return_alpha: If True, also return the alpha mask
            remove_background: Background removal mode (default: 'auto').
                'auto' removes the background only when has_background() detects
                it is still present; True always removes it; False never does.
            contrast: Sigmoid contrast strength applied to the greyscale
                (default 5.0; 0 disables). See enhance_contrast().

        Returns:
            If return_alpha=False: Grayscale PIL Image
            If return_alpha=True: Tuple of (grayscale Image, alpha mask Image)

        Raises:
            FileNotFoundError: If image_path doesn't exist
            IOError: If image cannot be loaded
            ValueError: If remove_background is not True, False or 'auto'
            ImportError: If removal is needed but 'rembg' is not installed

        Example:
            >>> img = ImageProcessor.load_image('portrait.png')
            >>> img, mask = ImageProcessor.load_image('portrait.png', return_alpha=True)
        """
        # Open image
        img = Image.open(image_path)

        # Convert to RGBA
        img = img.convert('RGBA')

        # Optionally strip the background so the alpha channel marks the subject
        if remove_background not in (True, False, 'auto'):
            raise ValueError(
                f"remove_background must be True, False or 'auto', "
                f"got {remove_background!r}"
            )
        if remove_background is True or (
            remove_background == 'auto' and ImageProcessor.has_background(img)
        ):
            img = ImageProcessor.remove_background(img)

        # Extract alpha channel
        mask = img.split()[-1]

        # Composite onto background color
        bg = Image.new('RGBA', img.size, alpha_color)
        bg.paste(img, mask=mask)
        img = bg

        # Convert to grayscale
        img = img.convert('L')

        # Increase contrast (S-curve) so the mosaic reads more clearly
        img = ImageProcessor.enhance_contrast(img, contrast=contrast)

        if return_alpha:
            return img, mask
        return img

    @staticmethod
    def has_background(img: Image.Image, opaque_threshold: float = 0.99) -> bool:
        """
        Detect whether an image still has its (opaque) background.

        The mosaic pipeline uses the alpha channel to separate subject from
        background. A photo with no alpha channel — or whose alpha is
        essentially all-opaque — has not had its background removed yet.

        Args:
            img: PIL Image in any mode.
            opaque_threshold: Fraction of fully-opaque pixels at or above which
                the background is considered still present (default: 0.99, i.e.
                fewer than ~1% transparent pixels).

        Returns:
            True if the background appears to still be present.

        Example:
            >>> ImageProcessor.has_background(Image.open('photo.jpg'))
            True
        """
        alpha = np.asarray(img.convert('RGBA').split()[-1])
        return bool((alpha == 255).mean() >= opaque_threshold)

    @staticmethod
    def enhance_contrast(img: Image.Image,
                        contrast: float = 5.0,
                        midpoint: float = 0.5) -> Image.Image:
        """
        Increase contrast with a sigmoid (S-curve).

        Pushes light tones lighter and dark tones darker while keeping pure
        black and white fixed, so high-contrast portraits map to clearer
        mosaics. The curve is normalised so the endpoints stay at 0 and 255.

        Args:
            img: Greyscale ('L') PIL Image.
            contrast: Sigmoid steepness. 0 leaves the image unchanged; higher
                values increase contrast (default 5.0).
            midpoint: Pivot of the curve in 0-1 (default 0.5 = mid-grey). Tones
                below it are darkened, tones above it are lightened.

        Returns:
            Contrast-enhanced greyscale PIL Image.

        Example:
            >>> grey = ImageProcessor.load_image('portrait.png')
            >>> punchy = ImageProcessor.enhance_contrast(grey, contrast=6)
        """
        if not contrast:
            return img

        arr = np.asarray(img, dtype=np.float64) / 255.0
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-contrast * (x - midpoint)))
        lo, hi = sigmoid(0.0), sigmoid(1.0)
        out = np.clip((sigmoid(arr) - lo) / (hi - lo), 0.0, 1.0)
        return Image.fromarray((out * 255).round().astype(np.uint8), mode='L')

    @classmethod
    def remove_background(cls, img: Image.Image,
                         model: str = 'u2net',
                         alpha_matting: bool = False,
                         **kwargs) -> Image.Image:
        """
        Remove an image's background, returning RGBA with the subject opaque.

        Uses the optional ``rembg`` package. Sessions are created once per model
        and reused across calls.

        Args:
            img: PIL Image to process.
            model: rembg segmentation model. 'u2net' (default) is general
                purpose; 'u2net_human_seg' is tuned for people; others include
                'isnet-general-use' and 'silueta'. Each downloads on first use.
            alpha_matting: Refine the cutout edges with alpha matting (slower,
                softer edges). Default False.
            **kwargs: Forwarded to ``rembg.remove`` — e.g. the alpha-matting
                thresholds ``alpha_matting_foreground_threshold``,
                ``alpha_matting_background_threshold``, ``alpha_matting_erode_size``.

        Returns:
            RGBA PIL Image whose background pixels are transparent.

        Raises:
            ImportError: If the optional ``rembg`` package is not installed.

        Example:
            >>> clean = ImageProcessor.remove_background(Image.open('photo.jpg'))
            >>> clean.mode
            'RGBA'
        """
        try:
            from rembg import remove
        except (ImportError, SystemExit) as exc:
            # rembg raises ImportError when absent, and sys.exit()s on import
            # when installed without an onnxruntime backend.
            raise ImportError(
                "Background removal requires the optional 'rembg' package with "
                "an onnxruntime backend. Install it with `pip install \"rembg[cpu]\"` "
                "(or `pip install gol-mosaics[bg-removal]`), "
                "or pass remove_background=False to skip removal."
            ) from exc
        return remove(img, session=cls._rembg_session(model),
                      alpha_matting=alpha_matting, **kwargs).convert('RGBA')

    @classmethod
    def _rembg_session(cls, model: str = 'u2net'):
        """Create (once per model) and return a cached rembg session.

        rembg's own auto-selection never considers Apple's CoreML provider, so
        we pass an explicit provider list that prefers hardware acceleration
        (CUDA/ROCm on NVIDIA/AMD, CoreML on Apple Silicon) and falls back to CPU.
        """
        if model not in cls._REMBG_SESSIONS:
            import onnxruntime as ort
            from rembg import new_session
            providers = cls.background_removal_providers or cls._preferred_providers(
                ort.get_available_providers()
            )
            cls._REMBG_SESSIONS[model] = new_session(model, providers=providers)
        return cls._REMBG_SESSIONS[model]

    @staticmethod
    def _preferred_providers(available) -> list:
        """Order available ONNX Runtime providers best-first, CPU last.

        Args:
            available: Provider names from onnxruntime.get_available_providers().

        Returns:
            Accelerated providers (if present) followed by CPU as a fallback.
        """
        preferred = [
            'CUDAExecutionProvider',    # NVIDIA
            'ROCMExecutionProvider',    # AMD
            'CoreMLExecutionProvider',  # Apple Silicon
            'CPUExecutionProvider',
        ]
        ordered = [p for p in preferred if p in available]
        if 'CPUExecutionProvider' not in ordered:
            ordered.append('CPUExecutionProvider')
        return ordered

    @staticmethod
    def square_image(img: Image.Image,
                    return_aspect: bool = True,
                    fill_color: str = 'white') -> Union[Image.Image, Tuple[Image.Image, float]]:
        """
        Make image square by padding with specified color.

        Preserves the original aspect ratio by adding padding to the shorter
        dimension rather than cropping.

        Args:
            img: PIL Image to make square
            return_aspect: If True, also return original aspect ratio
            fill_color: Color for padding (default: 'white')

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
            img = ImageOps.pad(img, (size, size), color=fill_color)

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
                             alpha_color: str = 'white',
                             fill_color: str = 'white',
                             remove_background: Union[bool, str] = 'auto',
                             contrast: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Complete preprocessing pipeline from image file to diagonal patterns.

        This is a convenience method that chains together the full preprocessing
        workflow.

        Args:
            image_path: Path to input image
            grid_size: Target grid size (must be even)
            alpha_color: Background color for transparent pixels
            fill_color: Padding color for squaring
            remove_background: Background removal mode passed to load_image
                ('auto', True or False; default 'auto')
            contrast: Sigmoid contrast strength passed to load_image
                (default 5.0; 0 disables)

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
        img, mask = cls.load_image(image_path, alpha_color=alpha_color,
                                   return_alpha=True, remove_background=remove_background,
                                   contrast=contrast)

        # Process grayscale image
        square_img, aspect_ratio = cls.square_image(img, return_aspect=True, fill_color=fill_color)
        lowres = cls.rotate_and_pixelate(square_img, grid_size, expand=True)
        lowres_first, lowres_second = cls.extract_diagonal_patterns(lowres)

        # Process alpha mask
        square_mask = cls.square_image(mask, return_aspect=False, fill_color='white')
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

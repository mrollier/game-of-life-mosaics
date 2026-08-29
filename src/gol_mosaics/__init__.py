"""
Game of Life Mosaics - Digital art using GoL still-lifes.

This package provides tools for creating artistic mosaics from images using
Conway's Game of Life still-life patterns combined with Elementary Cellular
Automaton backgrounds.

Example:
    >>> from gol_mosaics import MosaicGenerator
    >>> generator = MosaicGenerator(level=5, grid_size=100)
    >>> mosaic = generator.generate_from_image('portrait.png')
    >>> mosaic.save('output.png')
"""

__version__ = "2.3.0"
__author__ = "Michiel Rollier"

from .mosaic import MosaicGenerator
from .patterns import PatternLibrary
from .colors import ColorScheme
from .export import GollyExporter
from .image_processing import ImageProcessor
from .eca import ECABackground
from .renderer import MosaicRenderer
from .compose import (compose, agar_background, density_band,
                      fill_layer_count, filled_background, life_safe_pattern,
                      mosaic_background, scatter_background)

__all__ = [
    'MosaicGenerator',
    'PatternLibrary',
    'ColorScheme',
    'GollyExporter',
    'ImageProcessor',
    'ECABackground',
    'MosaicRenderer',
    'compose',
    'agar_background',
    'density_band',
    'life_safe_pattern',
    'mosaic_background',
    'filled_background',
    'fill_layer_count',
    'scatter_background',
]

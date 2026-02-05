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

__version__ = "2.0.0"
__author__ = "Michiel Rollier"

from .mosaic import MosaicGenerator
from .patterns import PatternLibrary
from .colors import ColorScheme
from .export import GollyExporter
from .image_processing import ImageProcessor
from .eca import ECABackground
from .renderer import MosaicRenderer

__all__ = [
    'MosaicGenerator',
    'PatternLibrary',
    'ColorScheme',
    'GollyExporter',
    'ImageProcessor',
    'ECABackground',
    'MosaicRenderer',
]

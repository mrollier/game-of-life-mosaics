"""
Game of Life Mosaics: images as Conway's Game of Life still lifes.

Two ways to turn a picture into a still life:

- **Tile mosaics** (MosaicGenerator): the image is tiled with pre-computed
  still-life tiles whose density follows its grey values, on an elementary
  cellular automaton background.
- **Free-form still lifes** (gol_mosaics.freeform, needs the [beyond]
  extra): the whole pattern is solved cell by cell with CP-SAT.

Example:
    >>> from gol_mosaics import MosaicGenerator
    >>> generator = MosaicGenerator(level=5, grid_size=100)
    >>> mosaic = generator.generate_from_image('input/images/john.png')
    >>> mosaic.save('output.png')
"""

__version__ = "3.0.0"
__author__ = "Michiel Rollier"

from .mosaic import MosaicGenerator
from .tile_library import TileLibrary
from .colours import ColourScheme
from .export import GollyExporter
from .image_processing import ImageProcessor
from .eca import ECABackground
from .renderer import MosaicRenderer
from .compose import (compose, agar_background, density_band,
                      fill_layer_count, filled_background, merge_background,
                      mosaic_background, scatter_background)

__all__ = [
    'MosaicGenerator',
    'TileLibrary',
    'ColourScheme',
    'GollyExporter',
    'ImageProcessor',
    'ECABackground',
    'MosaicRenderer',
    'compose',
    'agar_background',
    'density_band',
    'merge_background',
    'mosaic_background',
    'filled_background',
    'fill_layer_count',
    'scatter_background',
]

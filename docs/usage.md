# Usage guide

How to make mosaics with the library. The [tutorial notebooks](../notebooks/README.md)
cover the same ground with pictures; the [API reference](api.md) lists every
argument.

## Installing

```bash
git clone https://github.com/mrollier/game-of-life-mosaics.git
cd game-of-life-mosaics
pip install -e .                  # the library: numpy, scipy, Pillow
```

Python 3.10 or newer. Extras, as needed:

| Extra | Adds | Needed for |
|---|---|---|
| `[bg-removal]` | rembg | removing a photo's background automatically |
| `[sat]` | python-sat | enumerating tiles yourself (`sat_search`) |
| `[beyond]` | OR-Tools | free-form still lifes (`gol_mosaics.freeform`) |
| `[app]` | gradio, rembg | running the web app locally |
| `[dev]` | pytest, python-sat, matplotlib, jupyter, nbstripout | tests and notebooks |
| `[gurobi]` | gurobipy | the historical ILP generator (needs a licence) |

For example `pip install -e ".[bg-removal,dev]"`.

## Your first mosaic

```python
from gol_mosaics import MosaicGenerator

generator = MosaicGenerator(level=4, grid_size=60)
mosaic = generator.generate_from_image('input/images/john.png')
mosaic.save('john_mosaic.png')
```

`john.png` already has a transparent background. A photo that still has its
background works too: it is removed automatically (`remove_background='auto'`),
which needs the `[bg-removal]` extra; the first removal downloads a 176 MB
model. Or remove it beforehand with any photo editor and pass
`remove_background=False`.

Arguments left out are chosen at random when the generator is made (level
3-5, grid size 40-120, an interesting ECA rule), so every generator without
them gives a new variation. For a reproducible result, set those three
explicitly and pass `seed=` to `generate_from_image`, which fixes the tile draw
and the background.

## The settings

| Argument | Meaning | Typical |
|---|---|---|
| `level` | tile size and richness: tiles are 6 x level cells; diamonds 1-6, squares 3-5 | 4-5 |
| `grid_size` | tiles across the image; must be even for diamonds | 40-150 |
| `layout` | `'diamond'` (45-degree interlocking grids) or `'square'` | `'diamond'` |
| `colours` | a `ColourScheme` | `ColourScheme.ugent()` |
| `eca_rule` | background automaton, Wolfram rule 0-255; complex: 54, 110, 124, 137, 147, 193; chaotic: 30, 45, 106, 150 | `None` (random) |
| `random_tiles` | draw at random among equally dense tiles | `True` |
| `empty_tiles_cutoff` | grey values above this get no tile at all (lower: more white) | 0.65 |
| `alpha_cutoff` | where the background starts, on the alpha channel | 0.5 |
| `supersample` | background cell size in pixels | `None` (about 15) |
| `contrast` | S-curve contrast boost before tiling; 0 disables | 5.0 |
| `no_eca` | a flat background instead of the automaton | `False` |

Level 6 holds 332,321 tiles and takes about half a second and 430 MB to load
the first time. Level 1 and 2 hold one and two tiles: they give a pattern, not a
picture.

## Colours

```python
from gol_mosaics import ColourScheme, MosaicGenerator

colours = ColourScheme.monochrome()               # black on white
colours = ColourScheme.warhol(seed=7)             # random pop colours, reproducible
colours = ColourScheme(gol_background='#FFFFFF',  # the cells' dead colour
                       gol_pixel='#000000',       # the cells' live colour
                       eca_background='#FFD200',  # the background's two colours
                       eca_pixel='#1E64C8')
generator = MosaicGenerator(level=5, grid_size=100, colours=colours)
```

## Square tiles

```python
generator = MosaicGenerator(level=4, grid_size=50, layout='square')
```

Square tiles sit on an axis-aligned lattice and share their border ponds; any
grid size works. Levels 3-5 ship (3, 65 and 10,398 tiles).

## The mosaic as a Game of Life pattern

A mosaic *is* a still life. `generate_from_pil(..., return_arrays=True)`
returns the pattern itself next to the image; export it for
[Golly](https://golly.sourceforge.io/), optionally with a glider that will
break it:

```python
from PIL import Image
from gol_mosaics import GollyExporter, MosaicGenerator
from gol_mosaics.life import is_still_life

image, cells, background = MosaicGenerator(level=4, grid_size=40).generate_from_pil(
    Image.open('input/images/john.png'), return_arrays=True)
assert is_still_life(cells)
GollyExporter.export_to_cells(cells, 'john.cells', add_glider='bottom right')
```

Only the still life is exported; the coloured background is decoration and
would not be stable. See [04_golly_export](../notebooks/tutorials/04_golly_export.ipynb).

## Working with the tiles

```python
import numpy as np
from gol_mosaics import TileLibrary

library = TileLibrary.load(level=5)                    # 2632 diamond tiles
squares = TileLibrary.load(level=5, layout='square')   # 10,398 square tiles
tile = library.tile_for_value(0.3)                     # one tile for a grey value
tiles = library.tiles_for_values(np.array([[0.2, 0.8]]))
library.tiles.shape, library.densities[:5]
```

To enumerate tiles yourself, for any level or Life-like rule, see
[tiles.md](tiles.md).

## Animated GIFs

`generate_from_gif('animation.gif')` processes every frame, but currently
returns only the first processed frame (with the animation metadata), so
saving it gives a still image. Writing the frames as an animation is on the
[roadmap](roadmap.md).

## Backgrounds for any still life

`gol_mosaics.compose` paints a finished pattern with a colour scheme and a
backdrop, and can grow a background that is itself part of the still life:

```python
from gol_mosaics import ColourScheme, compose, filled_background, merge_background

# cells: a 0/1 still life; subject: True where the subject is
compose(cells, ~subject, ColourScheme.warhol(seed=3), style='mosaic',
        level=4, fill='auto', scale=3).save('art.png')

# the subject and a tile background together, as one still life for Golly
whole = merge_background(cells, ~subject, field=filled_background(~subject, level=4))
```

Styles: `'none'` (transparent), `'flat'`, `'eca'`, `'agar'` (a block agar) and
`'mosaic'` (a field of this project's tiles, its gap to the subject packed
with smaller tiles and loose still lifes by `fill='auto'`). Only `'agar'` and
`'mosaic'` are real cells; every background keeps two dead cells between
itself and the subject, which is what keeps the union stable. The
[beyond_tiles_art](../notebooks/research/beyond_tiles/beyond_tiles_art.ipynb)
notebook shows them all.

## Free-form still lifes

Instead of tiles, a still life can be solved cell by cell to follow an image
more freely. That takes the `[beyond]` extra and minutes to hours of solving:
see [freeform.md](freeform.md).

## Sharp output

A saved mosaic is pixel-perfect: one cell per pixel, lossless PNG. If it looks
blurry when you zoom in, the viewer is smoothing it. How to view, scale and
share mosaics without blurring them: [pixel_perfect.md](pixel_perfect.md).

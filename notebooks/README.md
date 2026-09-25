# Notebooks

Every notebook finds the repository root on its own, so it runs from
wherever Jupyter or VS Code starts it. Install the package first
(`pip install -e ".[dev]"` from the root); the extras a notebook needs on top
are listed below.

## Tutorials: making mosaics

Start here. Each runs in under a minute.

| Notebook | What you learn | Extra needs |
|---|---|---|
| [01_quickstart](tutorials/01_quickstart.ipynb) | a mosaic from a portrait in three lines | – |
| [02_parameters](tutorials/02_parameters.ipynb) | what every setting does, then a playground for your own | – |
| [03_preprocessing](tutorials/03_preprocessing.ipynb) | from a photo with a background: removal, greyscale, contrast | `[bg-removal]` (downloads a 176 MB model once) |
| [04_golly_export](tutorials/04_golly_export.ipynb) | the mosaic as a Game of Life pattern, and a glider breaking it | [Golly](https://golly.sourceforge.io/) to watch it |
| [05_how_tiles_work](tutorials/05_how_tiles_work.ipynb) | ponds, the tile frame, symmetry, tile levels | – |

## Research

Write-ups of the methods behind the package, with live computations. Their
outputs are not versioned (see CONTRIBUTING.md); run them to see the
figures.

| Notebook | Topic | Extra needs, runtime |
|---|---|---|
| [tiles/sat_tile_generation](research/tiles/sat_tile_generation.ipynb) | the SAT enumeration of the tile census, validated against brute force and the ILP | `[sat]`, minutes |
| [tiles/sat_tile_search](research/tiles/sat_tile_search.ipynb) | run the search for any Life-like rule and level | `[sat]`, seconds to hours |
| [tiles/tile_scheme_generalisation](research/tiles/tile_scheme_generalisation.ipynb) | tile schemes beyond diamonds: the square family and its census | `[sat]`, minutes |
| [tiles/tile_nosym_enumeration](research/tiles/tile_nosym_enumeration.ipynb) | the census without the symmetry requirement | `[sat]`, seconds |
| [beyond_tiles/beyond_tiles](research/beyond_tiles/beyond_tiles.ipynb) | free-form still lifes with CP-SAT: a guided tour of `gol_mosaics.freeform` | `[beyond]`, minutes |
| [beyond_tiles/beyond_tiles_art](research/beyond_tiles/beyond_tiles_art.ipynb) | recolouring and backgrounds for a solved still life (`compose`) | `[beyond]`, minutes |
| [archive/tile_generation_gurobi](research/archive/tile_generation_gurobi.ipynb) | the original ILP tile generator, kept for the record | a Gurobi licence |

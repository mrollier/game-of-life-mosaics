# Repository map

What lives where, and who each part is for.

```
game-of-life-mosaics/
├── src/gol_mosaics/         the library (pip install -e .)
│   ├── freeform/            free-form still lifes with CP-SAT ([beyond] extra)
│   └── data/                the shipped tile databases (packed)
├── app/                     the web app (Gradio) and its Hugging Face deploy
├── notebooks/
│   ├── tutorials/           start here: five notebooks, a few minutes each
│   └── research/            the methods, with live computations
├── docs/                    this documentation
├── studies/                 finished pieces made with the library
├── experiments/             the research harness behind the free-form solver
├── search/                  the SAT searches that produced the tile censuses
├── tools/                   maintenance scripts (packing the tile data)
├── tests/                   the test suite (pytest)
├── input/, output/          example images and example results
└── pyproject.toml           package metadata, dependencies, extras
```

## The library: `src/gol_mosaics/`

| Module | What it does |
|---|---|
| `mosaic.py` | `MosaicGenerator`, the main entry point: image in, mosaic out |
| `tile_library.py` | `TileLibrary`: loads a level's tiles and matches grey values to tiles by density |
| `image_processing.py` | `ImageProcessor`: loading, background removal (rembg), contrast, the diamond and square preprocessing |
| `colours.py` | `ColourScheme` and its presets (UGent, monochrome, Warhol, inverted) |
| `renderer.py` | `MosaicRenderer`: arrays of cells and backgrounds to RGBA images |
| `eca.py` | `ECABackground`: elementary cellular automaton backgrounds |
| `compose.py` | backgrounds and recolouring for a finished pattern: agar, tile fields, scatter, `compose()` |
| `export.py` | `GollyExporter`: `.cells` and RLE files for Golly |
| `life.py` | a small Game of Life stepper and still-life check |
| `tile_domain.py` | the pond geometry, symmetry orbits, interlock cells and the packed tile format |
| `tile_scheme.py` | general tile schemes (diamond, square) and lattice assembly |
| `sat_search.py` | the SAT encoding and exhaustive tile enumeration, for any Life-like rule |
| `nosym_tiles.py` | the census without the symmetry requirement |
| `legacy_ilp.py` | the original Gurobi tile generator, kept for the record |
| `freeform/` | `targets`, `solver`, `decompose`, `lns`, `seeds`, `metrics`, `io`, `poster`: see [freeform.md](freeform.md) |

## Everything else

| Folder | For | Read first |
|---|---|---|
| `app/` | using or deploying the web app | [deploy.md](deploy.md) |
| `notebooks/` | learning by running | [notebooks/README.md](../notebooks/README.md) |
| `studies/` | seeing what the library can do on real commissions | [studies/README.md](../studies/README.md) |
| `experiments/beyond_tiles/` | the research record of the free-form solver (experiments E1-E10, benchmarks) | [REPORT.md](../experiments/beyond_tiles/REPORT.md) |
| `experiments/archive/` | a documented negative result (parallel tempering) | its module docstring |
| `search/` | reproducing the tile censuses, certificates and the level-8 count | [search/README.md](../search/README.md) |
| `tools/` | rebuilding or checking the shipped tile data | `python tools/pack_tiles.py --help` |
| `tests/` | contributing | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| `input/images/` | the example portraits (John Conway with and without background, Marilyn Monroe, the Ghent Altarpiece, two landscapes) | – |
| `output/` | example mosaics referenced by the README and notebooks | – |

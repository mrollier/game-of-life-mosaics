# API reference

The public API of `gol_mosaics` 3.0. Every function and class has a docstring
with the full argument list; this page is the map. Moving from 2.x? See
[the renames](#from-2x-to-30) at the end.

```python
from gol_mosaics import (MosaicGenerator, TileLibrary, ColourScheme,
                         ImageProcessor, ECABackground, MosaicRenderer,
                         GollyExporter, compose, agar_background,
                         mosaic_background, filled_background,
                         scatter_background, merge_background, density_band,
                         fill_layer_count)
```

## MosaicGenerator

```python
MosaicGenerator(level=None, grid_size=None, colours=None, eca_rule=None,
                random_tiles=True, invert=True, layout='diamond')
```

`None` picks a level (3-5), grid size (40-120) or ECA rule at random.

| Method | Returns |
|---|---|
| `generate_from_image(path, empty_tiles_cutoff=0.65, alpha_cutoff=0.5, supersample=None, no_eca=False, remove_background='auto', contrast=5.0, seed=None)` | RGBA image |
| `generate_from_pil(image, ..., return_arrays=False)` | the image, or `(image, cells, background)` with the 0/1 still life and the background mask |
| `generate_from_gif(path, ...)` | the first processed frame (a known limitation) |

Attributes: `level`, `grid_size`, `layout`, `colours`, `eca_rule`, and the
lazily built `library` (a `TileLibrary`), `renderer`, `eca`.

## TileLibrary

| Member | |
|---|---|
| `TileLibrary.load(level, layout='diamond')` | the shipped tiles: diamonds 1-6, squares 3-5; cached, read-only |
| `TileLibrary.from_tiles(tiles, level, layout='diamond')` | wrap tiles computed elsewhere |
| `.tiles` | `(N, 6L, 6L)` uint8 |
| `.densities` | live-cell density per tile, normalised to 0-1 per library |
| `.tile_for_value(value, random=True, invert=True)` | one tile |
| `.tiles_for_values(values, random=True, invert=True, empty_tiles_cutoff=1.0)` | `(*values.shape, 6L, 6L)` |
| `.indices_for_values(...)` | the same selection as indices; -1 for empty |
| `.tiles_for_mask(mask, alpha_cutoff=0.5)` | solid and empty tiles for a transparency mask |
| `.scheme`, `.pond`, `.tile_size`, `.lattice_offset` | the tile geometry |

`gol_mosaics.tile_library.nearest_density_indices(densities, wanted, rng, random, candidates)`
is the selection used by all of these and by `mosaic_background`.

## ColourScheme

A frozen dataclass of hex colours: `gol_background`, `gol_pixel` (the
cells), `eca_background`, `eca_pixel` (the backdrop), and `fill_pixel` (the
far end of the filler ramp of `compose`; `None` derives one). Presets:
`ugent()` (default), `monochrome(foreground, background)`,
`warhol(force_white=False, dark_on_light=True, seed=None, min_luma_gap=0.35)`,
`inverted()`. Module helpers: `hex_to_rgb`, `mix`.

## ImageProcessor

Static methods: `load_image`, `has_background`, `remove_background` (rembg),
`enhance_contrast`, `square_image`, `rotate_and_pixelate`, `split_diagonals`,
and the two complete preprocessing paths `preprocess_diamond` and
`preprocess_square`. Colour arguments are `alpha_colour` and `fill_colour`.

## ECABackground, MosaicRenderer, GollyExporter

| Call | |
|---|---|
| `ECABackground(rule).generate(width, height, supersample=15)` | 0/1 array; `COMPLEX_RULES`, `CHAOTIC_RULES`, `from_category()` |
| `MosaicRenderer(colours).render(cells, backdrop, layers=None)` | cells plus a layered backdrop (0 transparent, 1 field, 2 pattern, 3+ filler) |
| `MosaicRenderer.render_gol_mosaic(cells)`, `.render_backdrop(backdrop, layers)`, `.composite(base, overlay)` | the parts of `render` |
| `GollyExporter.export_to_cells(cells, path, add_glider=None)` | Golly `.cells`; `add_glider` is a corner such as `'bottom right'` |
| `GollyExporter.export_to_rle(cells, path, name=None, comments=None)` | run-length encoded |

## compose

Backgrounds and recolouring for a finished pattern (see the
[usage guide](usage.md#backgrounds-for-any-still-life)).

| Function | |
|---|---|
| `compose(cells, background_mask, colours=None, style='eca', ..., layout='diamond', centred=False, field=None, scale=1)` | RGBA image; styles `'none'`, `'flat'`, `'eca'`, `'agar'`, `'mosaic'` |
| `agar_background(mask, pitch=(3, 4), gap=2)` | a still-life block agar |
| `mosaic_background(mask, level=4, layout='diamond', density=(0, 1), tone=None, gap=2, seed=None, centred=False)` | a field of tiles |
| `filled_background(mask, level=4, layout='diamond', fill='auto', ...)` | the same, its gap packed with smaller tiles and loose still lifes; numbered by layer |
| `scatter_background(mask, occupied=None, gap=2, band=None, ...)` | loose elementary still lifes |
| `merge_background(cells, background_mask, field=None, ...)` | pattern and background as one checked still life |
| `density_band`, `fill_layer_count`, `centring_pad` | helpers |

## Tile mathematics

For research use; see [tiles.md](tiles.md).

| Module | Main entries |
|---|---|
| `tile_domain` | `POND`, `pond_lattice(L)`, `pond_frame(L)`, `free_octant(L)`, `derive_dead_edges(L)`, `build_domain(L)` → `Domain` with `pack`, `unpack`, `expand_many`; `canonical_order(tiles)` |
| `tile_scheme` | `TileScheme`, `diamond_scheme(L)`, `square_scheme(L)`, `derive_interlock`, `build_scheme_domain`, `enumerate_scheme_tiles`, `assemble(scheme, indices, tiles)`, `check_frame_mosaic` |
| `sat_search` | `build_cnf(L, birth, survival)`, `enumerate_all`, `enumerate_tiles(L, birth, survival)`, `bruteforce_tiles`, `rule_violations`, `domain_clauses`, `CONWAY`, `HIGHLIFE`; CLI `python -m gol_mosaics.sat_search bruteforce --level 4` |
| `nosym_tiles` | `build_nosym_domain`, `enumerate_nosym_tiles`, `load_nosym_tiles`, `d4_classes` |
| `life` | `neighbour_counts`, `life_step`, `is_still_life` |
| `legacy_ilp` | `generate_tiles(level, solution_limit)` (Gurobi) |

## Free-form (`gol_mosaics.freeform`, `[beyond]` extra)

| Module | Main entries |
|---|---|
| `poster` | `PosterConfig`, `solve_poster(grey, free, cfg, out=None, resume=False, log=None)` → `PosterResult` |
| `targets` | `load_rect_target`, `grey_and_mask_from_image`, `cell_targets`, `window_slices`, `window_targets`, `window_live_counts`, `normalise_grey`, `equalise_grey` |
| `solver` | `SolveConfig`, `solve_image`, `build_model`, `solve`, `verify_still_life` |
| `decompose` | `plan_strips`, `solve_strips(isolate=False)`, `plan_blocks`, `solve_blocks`, `lower_bound_strips` |
| `lns` | `LnsConfig`, `improve`, `repair_diagonal_runs`, `window_devs`, `seam_occupancy` |
| `seeds`, `metrics`, `io` | warm starts; deviation and texture statistics; `save_packed` / `load_packed` |

## From 2.x to 3.0

3.0 renamed the API to one vocabulary (pond, tile, mosaic, layout) and to
British spelling, without compatibility aliases.

| 2.x | 3.0 |
|---|---|
| `gol_mosaics.colors`, `ColorScheme` | `gol_mosaics.colours`, `ColourScheme` |
| `MosaicGenerator(color_scheme=, tile_shape=, random_patterns=)` | `MosaicGenerator(colours=, layout=, random_tiles=)` |
| `generator.pattern_library`, `.eca_generator`, `.color_scheme`, `.tile_shape` | `.library`, `.eca`, `.colours`, `.layout` |
| `gol_mosaics.patterns`, `PatternLibrary` | `gol_mosaics.tile_library`, `TileLibrary` |
| `PatternLibrary.load(level, shape=)` | `TileLibrary.load(level, layout=)` |
| `.solutions`, `.shape` | `.tiles`, `.layout` |
| `get_pattern_for_value`, `get_patterns_for_values` | `tile_for_value`, `tiles_for_values` |
| `get_indices_for_values`, `get_patterns_for_mask` | `indices_for_values`, `tiles_for_mask` |
| `.tile_shape` (the tuple), `.tile_pad_size` | `.tile_size`, `.lattice_offset` |
| `PatternLibrary.generate(level, n)` | `TileLibrary.from_tiles(legacy_ilp.generate_tiles(level, n), level)` |
| `PatternLibrary.pond_pattern()`, `.pond_pattern_multiple()`, `.pond_pattern_edge()`, `.pond_pattern_eighth()` | `tile_domain.POND`, `pond_lattice(L)`, `pond_frame(L)` (or `library.pond`), `free_octant(L)` |
| `MosaicRenderer(color_scheme).render_full_mosaic`, `.render_eca_overlay` | `MosaicRenderer(colours).render`, `.render_backdrop` |
| `renderer.hex_to_rgb` | `colours.hex_to_rgb` |
| `compose(pattern, mask, scheme, ..., shape=)`, `life_safe_pattern` | `compose(cells, mask, colours, ..., layout=)`, `merge_background` |
| `shape=` on `mosaic_background`, `filled_background`, `density_band`, `fill_layer_count` | `layout=` |
| `ImageProcessor.extract_diagonal_patterns`, `preprocess_for_mosaic`, `preprocess_for_square_mosaic` | `split_diagonals`, `preprocess_diamond`, `preprocess_square` |
| `alpha_color=`, `fill_color=` | `alpha_colour=`, `fill_colour=` |
| `GollyExporter.export_to_cells(mosaic, filename)` | `export_to_cells(cells, path)` |
| `tile_domain.neighbors`, `tile_scheme.neighbor_offsets`, `pond_square_scheme` | `neighbours`, `neighbour_offsets`, `square_scheme` |
| `pack_solutions` / `unpack_solutions` (and the `_scheme_`, `_nosym_` pairs) | `build_domain(L).pack(tiles)` / `.unpack(packed)` (and `build_scheme_domain`, `build_nosym_domain`) |
| `sat_search._domain_clauses` | `domain_clauses` |
| data `solutions_pattern_level_N.npy` etc. | `tiles_{diamond,square}_level_N_orbits.npy` (all packed) |
| `experiments/beyond_tiles/*` library modules | `gol_mosaics.freeform.*` (`still_image` → `solver`) |
| `SpikeConfig`, `SpikeResult` | `SolveConfig`, `SolveResult` |
| `save_pattern_asset`, `load_pattern_asset` | `freeform.io.save_packed`, `load_packed` |
| `normalize_grey`, `equalize_grey` | `normalise_grey`, `equalise_grey` |

Behaviour changes worth knowing: a seeded mosaic with `random_tiles=True`
differs from 2.x (the tile draw no longer builds a dense difference matrix);
tiles are uint8 at every level; `.cells` and RLE files use `\n` line endings
on every platform; the bottom-left glider now heads into the grid.

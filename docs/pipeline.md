# The tile-mosaic pipeline

This document follows an input image through `MosaicGenerator` to the finished
RGBA mosaic, one stage at a time, with links to the code that does each step.
It is written for anyone who wants to change the pipeline or find out why an
output looks wrong. Each stage gives the arrays it produces, the invariants it
relies on and the places where it is known to be fragile. Two tile layouts share
the same front and back end. The diamond layout, the original one, places
45-degree pond diamonds on two interlocking diagonal grids. The square layout
places axis-aligned tiles on a single lattice, where neighbours share their
border ponds. Line numbers refer to the 3.0 code and will drift, so treat the
function names as the stable reference.

1. **Set-up.** `__init__` picks any missing parameters; `seed` reseeds numpy.
2. **Load.** `load_image` splits the input into alpha and greyscale.
3. **Downsample.** One value per tile site: two diagonal grids or one rectangular grid.
4. **Select tiles.** `TileLibrary` matches each grey value to a tile by density.
5. **Assemble.** The tiles are placed on the lattice: the binary pattern.
6. **Mask.** The alpha grid becomes a binary mask of where the backdrop goes.
7. **Crop (diamond only).** Back to the input's aspect ratio.
8. **Backdrop and render.** `compose` paints the pattern and an elementary
   cellular automaton (ECA) backdrop.

## Notation

- `G` = `grid_size`: even for diamonds; for squares, the number of tile columns.
- `L` = `level`. Diamond tiles ship for levels 1-6, square tiles for 3-5
  ([tile_library.py:97](../src/gol_mosaics/tile_library.py#L97)).
- `n = 6L` = tile side in cells (`TileLibrary.tile_size`); one cell is one pixel.
- `R` = side of the rotated image in the diamond path, about √2·G (exactly
  1.4·G for G = 40, 60 and 100). Diamond shapes after stage 3 follow `R`, not `G`.
- `p = 6(L-1)` = square-lattice pitch; neighbouring square tiles overlap by one
  6-cell pond band.

## Entry points

| Call | Where | Notes |
|------|-------|-------|
| `generate_from_image(path, …)` | [mosaic.py:166](../src/gol_mosaics/mosaic.py#L166) | Opens the file and calls `generate_from_pil`. |
| `generate_from_pil(img, …)` | [mosaic.py:207](../src/gol_mosaics/mosaic.py#L207) | The pipeline. `return_arrays=True` also returns the binary pattern and the mask. |
| `generate_from_gif(path, …)` | [mosaic.py:335](../src/gol_mosaics/mosaic.py#L335) | Runs `generate_from_pil` per frame (fragile spot 4). |
| Web app `render_mosaic(…)` | [app.py:299](../app/app.py#L299) | Seeds, calls `generate_from_pil`, then pads to the upload's ratio (stage 9). |

## 1. Set-up and seeding

`__init__` ([mosaic.py:70](../src/gol_mosaics/mosaic.py#L70)) fills in
whatever was left as None, at random: grid size 40-120, level 3-5, and an ECA
rule from `COMPLEX_RULES + CHAOTIC_RULES`
([mosaic.py:656-669](../src/gol_mosaics/mosaic.py#L656-L669)). The diamond
layout rejects an odd grid size ([mosaic.py:121](../src/gol_mosaics/mosaic.py#L121)).
The `library` property ([mosaic.py:145](../src/gol_mosaics/mosaic.py#L145))
calls `TileLibrary.load(level, layout=)`, which caches one shared, read-only
instance per (level, layout).

If `seed` is given, `generate_from_pil` calls `np.random.seed(seed)`
([mosaic.py:264-265](../src/gol_mosaics/mosaic.py#L264-L265)), which covers
the tile draw and the ECA's initial row ([eca.py:125](../src/gol_mosaics/eca.py#L125)).
It does not cover the random choices in `__init__`, which come earlier (the web
app seeds before building the generator, [app.py:261](../app/app.py#L261)), nor
`ColourScheme.warhol()`, which has its own generator.

## 2. Load the image

`ImageProcessor.load_image` ([image_processing.py:40](../src/gol_mosaics/image_processing.py#L40)),
shared by both layouts, converts to RGBA and removes the background with
`rembg` if `remove_background` is True, or if it is `'auto'` and
`has_background` ([image_processing.py:116](../src/gol_mosaics/image_processing.py#L116))
finds at least 99% of the pixels fully opaque (the web app passes False). The
alpha channel becomes the mask, the only source of the subject/background
split. The image is composited onto white, converted to greyscale and passed
through the sigmoid `enhance_contrast` ([image_processing.py:141](../src/gol_mosaics/image_processing.py#L141);
`contrast=5.0` by default, 0 switches it off). It returns `(grey, alpha)`.

## 3. Downsample to tile grids

**Diamond: `preprocess_diamond`**
([image_processing.py:393](../src/gol_mosaics/image_processing.py#L393)). The
grey and alpha images go through the same three steps
([image_processing.py:436-443](../src/gol_mosaics/image_processing.py#L436-L443)),
so they stay aligned:

1. `square_image` ([image_processing.py:260](../src/gol_mosaics/image_processing.py#L260))
   pads the shorter side with white to a square and records
   `aspect_ratio = width / height`. It pads and never crops.
2. `rotate_and_pixelate` ([image_processing.py:295](../src/gol_mosaics/image_processing.py#L295))
   resizes to G×G (Lanczos), rotates 45° with `expand=True` and white fill, and
   trims one pixel off every edge (`arr[1:-1, 1:-1]`,
   [image_processing.py:338](../src/gol_mosaics/image_processing.py#L338)),
   giving R×R.
3. `split_diagonals` ([image_processing.py:342](../src/gol_mosaics/image_processing.py#L342))
   samples the rotated image along two interlocking diagonal lattices. It takes
   its size from `lowres.shape[0]`, which is R: the first grid is
   `(R/2 + 1, R/2)`, the second `(R/2, R/2 + 1)` (29×28 and 28×29 for G = 40).

Each grid entry becomes one tile, so a diamond mosaic holds about R²/2 ≈ G²
tiles, one per pixel of the resized image. A diagonal of the rotated image runs
along an axis of the original, so the mosaic comes out the same way up as the
input. The rotation corners are white in both images, so their sites get empty
tiles and count as subject: they render in the opaque `gol_background` colour,
never as backdrop.

**Square: `preprocess_square`**
([image_processing.py:448](../src/gol_mosaics/image_processing.py#L448)). No
squaring, rotation or split: both images are resized straight to `(rows, G)`
with `rows = max(1, round(G / aspect_ratio))`
([image_processing.py:493-497](../src/gol_mosaics/image_processing.py#L493-L497)).
The grid shape carries the aspect ratio, so there is no crop later.

## 4. Select tiles by density

`TileLibrary` ([tile_library.py:107](../src/gol_mosaics/tile_library.py#L107))
holds every tile of one level and layout as an `(N, n, n)` uint8 array
(`.tiles`). Its `densities`
([tile_library.py:211](../src/gol_mosaics/tile_library.py#L211)) are live-cell
fractions min-max normalised to [0, 1] per library, so the same value means a
different absolute fill at different levels.

`indices_for_values`
([tile_library.py:321](../src/gol_mosaics/tile_library.py#L321)) does the
matching. Values above `empty_tiles_cutoff` (0.65 by default) become index -1,
an empty site. The rest are divided by the cutoff, stretching [0, cutoff] onto
[0, 1], and inverted when `invert=True` so that dark means dense
([tile_library.py:357-361](../src/gol_mosaics/tile_library.py#L357-L361)).
`nearest_density_indices`
([tile_library.py:29](../src/gol_mosaics/tile_library.py#L29)) then picks a
tile of the nearest density for each value, by `searchsorted` on the sorted
unique densities; the dense (values × tiles) difference matrix it replaced ran
to about 14 GB at level 6. With `random=True` it draws uniformly among the
tiles sharing that density, otherwise it takes the lowest index.

`tiles_for_values` ([tile_library.py:287](../src/gol_mosaics/tile_library.py#L287))
wraps this and returns the tiles, `(*shape, n, n)`, with all-zero tiles at the
-1 sites. The diamond path uses it; the square path uses the indices directly.

## 5. Assemble the pattern

**Diamond: `_build_mosaic`** ([mosaic.py:404](../src/gol_mosaics/mosaic.py#L404)).
Each diagonal grid is mapped to tiles (two independent draws) and stacked into
one array by `_assemble_tiles`
([mosaic.py:512](../src/gol_mosaics/mosaic.py#L512)). `_pad_diagonals`
([mosaic.py:517](../src/gol_mosaics/mosaic.py#L517)) then pads the first grid
left and right and the second top and bottom, both by
`TileLibrary.lattice_offset` = n/2 = 3L
([tile_library.py:247](../src/gol_mosaics/tile_library.py#L247)). Both become
`(R/2 + 1)·n` square and sit half a tile apart, so each grid's diamonds fill
the other's gaps. The two are added
([mosaic.py:451](../src/gol_mosaics/mosaic.py#L451)); the diamond supports
never overlap, so the sum stays 0/1. For G = 40 the side is 522 at L = 3 and
696 at L = 4.

**Square: `_build_square_mosaic`** ([mosaic.py:453](../src/gol_mosaics/mosaic.py#L453)).
Square tiles overlap their neighbours by one pond band, so they cannot be
stacked as blocks. The indices go to `tile_scheme.assemble`
([tile_scheme.py:236](../src/gol_mosaics/tile_scheme.py#L236)) with
`library.scheme` (from `square_scheme`,
[tile_scheme.py:320](../src/gol_mosaics/tile_scheme.py#L320)). It pastes
`tiles[indices[a, b]]` at `a·u + b·v` with u = (p, 0) and v = (0, p), leaves -1
sites empty, adds a 2-cell dead margin and asserts that overlapping tiles agree
on every shared cell. The result is
`((rows - 1)·p + n + 4) × ((G - 1)·p + n + 4)`. An empty site drops its whole
tile, frame included; the neighbours still draw their half of the shared band.

## 6. Build the mask

The mask convention is **1 = backdrop region (ECA drawn), 0 = subject (pattern
shown)**.

**Diamond: `_build_mask`** ([mosaic.py:534](../src/gol_mosaics/mosaic.py#L534)).
`tiles_for_mask` ([tile_library.py:368](../src/gol_mosaics/tile_library.py#L368))
turns alpha below `alpha_cutoff` (0.5 by default, i.e. transparent) into a
filled tile and alpha at or above it into an empty tile. The filled tile is
`binary_fill_holes(tiles[-1])`
([tile_library.py:399](../src/gol_mosaics/tile_library.py#L399)): the filled
diamond, about 44% of the box at level 4, not a solid n×n square. The two mask
grids are padded and summed like the pattern, so the diamonds cover the
background apart from tiny gaps where their dead borders meet.
`_fill_small_holes` ([mosaic.py:23](../src/gol_mosaics/mosaic.py#L23)) closes
those, filling only holes smaller than a quarter tile, n²/4
([mosaic.py:573-576](../src/gol_mosaics/mosaic.py#L573-L576)). Plain
`binary_fill_holes` would also fill a subject that touches no image edge, since
that is an enclosed zero-region too; the size threshold tells the two apart.

**Square: `_build_square_mask`** ([mosaic.py:482](../src/gol_mosaics/mosaic.py#L482)).
The alpha grid is thresholded directly, and each mosaic cell takes the value of
the tile site whose centre is nearest, clipped at the edges
([mosaic.py:503-510](../src/gol_mosaics/mosaic.py#L503-L510)). This is exact on
the lattice, so no hole filling is needed.

## 7. Crop to the aspect ratio (diamond only)

The diamond mosaic is built on a square canvas and cropped back towards
`aspect_ratio`: the height for landscape inputs, the width for portrait ones.
`_crop_window` ([mosaic.py:601](../src/gol_mosaics/mosaic.py#L601)) rounds the
kept extent up to whole tiles (`ceil(new / n)·n`) and centres it with integer
division, so the crop never matches the input ratio exactly.

On the diamond lattice every straight line crosses tiles, so the window is
worked out *before* assembly ([mosaic.py:295](../src/gol_mosaics/mosaic.py#L295))
and `_build_mosaic` leaves empty every tile that is not wholly inside it
(`_inside`, [mosaic.py:580](../src/gol_mosaics/mosaic.py#L580)). A cut tile is
no longer a still life; an empty one is allowed anywhere. The crop itself then
only removes empty cells and dead padding. (Before 3.0 the crop came last and
sliced tiles, and very wide or tall images gave patterns that were not still
lifes.)

## 8. Backdrop and render

If `supersample` is None, `generate_from_pil` uses `max(1, min(15, width))`
([mosaic.py:320-321](../src/gol_mosaics/mosaic.py#L320-L321)): 15-pixel ECA
cells, clamped for very small mosaics. Any positive integer works, because the
ECA is cropped to size. `_apply_eca_background`
([mosaic.py:628](../src/gol_mosaics/mosaic.py#L628)) passes everything to
`compose(cells, mask, colours, style=…, rule=, supersample=)`
([compose.py:907](../src/gol_mosaics/compose.py#L907)), with `style='eca'`, or
`'flat'` when `no_eca=True`:

1. For `'eca'`, `ECABackground(rule).generate` ([eca.py:57](../src/gol_mosaics/eca.py#L57))
   steps an elementary CA at `ceil(width/s) × ceil(height/s)` from a random
   row with a numpy lookup table ([eca.py:113-129](../src/gol_mosaics/eca.py#L113-L129)),
   repeats each cell into an s×s block and crops to `(height, width)`. For
   `'flat'` the field is all zeros.
2. `backdrop = mask · (field + mask)` ([compose.py:1050-1053](../src/gol_mosaics/compose.py#L1050-L1053))
   gives 0 on the subject, 1 for ECA background and 2 for ECA pixel.
3. `MosaicRenderer.render` ([renderer.py:198](../src/gol_mosaics/renderer.py#L198))
   paints the pattern opaque (`render_gol_mosaic`), the backdrop as 0
   transparent, 1 `eca_background`, 2 `eca_pixel` (`render_backdrop`), and
   alpha-composites the second over the first.

The output is an opaque RGBA image, one pixel per cell. The ECA is only paint:
it is not in the pattern that `return_arrays` returns and never reaches a Golly
export (for a backdrop that is itself a still life, see `merge_background`,
[compose.py:815](../src/gol_mosaics/compose.py#L815)). `compose` also serves the
free-form pipeline, hence its other styles and filler layers 3 and up.

## 9. Web app: pad to the upload's ratio

`_fit_to_aspect` ([app.py:193](../app/app.py#L193)) pads the shorter axis to
exactly the upload's ratio with `eca_background` and centres the mosaic on it.
The bars are in the ECA background colour; the diamond mosaic's own rotation
corners are in the GoL background colour.

## Where the tiles come from

Each level ships as packed symmetry-orbit bits in
`src/gol_mosaics/data/tiles_{layout}_level_{L}_orbits.npy`, one bit per free D4
orbit per tile (59 bits at diamond level 6). `_load_tile_library`
([tile_library.py:413](../src/gol_mosaics/tile_library.py#L413)) builds the
level's `Domain` (`tile_domain.build_domain` for diamonds,
`tile_scheme.build_scheme_domain` for squares) and expands the bits with
`Domain.unpack` ([tile_domain.py:339](../src/gol_mosaics/tile_domain.py#L339));
`Domain.pack` ([tile_domain.py:322](../src/gol_mosaics/tile_domain.py#L322)) is
the inverse, for writing new data. The frames are built from the pond
(`tile_domain.POND`, `pond_lattice`, `pond_frame`,
[tile_domain.py:57-115](../src/gol_mosaics/tile_domain.py#L57-L115)); the tiles
were enumerated with SAT (`gol_mosaics.sat_search`), and higher levels can be
wrapped with `TileLibrary.from_tiles`. Why any assignment of tiles, empty sites
included, is a global still life is set out in the `tile_scheme` module
docstring ([tile_scheme.py:1-49](../src/gol_mosaics/tile_scheme.py#L1-L49)).

## Data shapes

| Stage | Diamond | Square |
|-------|---------|--------|
| 3 | G × G, then R × R; grids (R/2+1, R/2) and (R/2, R/2+1) | rows × G, rows = round(G·H₀/W₀) |
| 5, 6 | (R/2+1)·n square | ((rows−1)·p + n + 4) × ((G−1)·p + n + 4) |
| 7 | (R/2+1)·n on one axis, tile-rounded on the other | no crop |
| 8 | as stage 7 | as stage 5 |

For a 3:2 landscape input with G = 40 and L = 4, the diamond path gives 696×696
before the crop and 480×696 after it; the square path gives a 27×40 grid and a
496×730 mosaic.

## Invariants

- `lattice_offset` equals n/2. `_pad_diagonals`, `derive_dead_edges_full`
  ([tile_domain.py:151](../src/gol_mosaics/tile_domain.py#L151)) and
  `diamond_scheme` ([tile_scheme.py:301](../src/gol_mosaics/tile_scheme.py#L301))
  all assume it; if it changes, the diagonal grids overlap or leave gaps.
- The pattern is 0/1 after assembly, and pattern and mask share one shape. The
  grey and alpha images always take the same geometric path.
- Mask polarity is 1 = backdrop, 0 = subject in both layouts; `compose` takes
  the mask as its `background_mask`.
- `_build_square_mask` hard-codes the assembly margin as `pad = 2`
  ([mosaic.py:505](../src/gol_mosaics/mosaic.py#L505)); it must match the
  default `pad` of `assemble`, or the mask shifts against the pattern.

## Fragile spots

1. **Diamond edge geometry.** The `[1:-1, 1:-1]` trim in `rotate_and_pixelate`
   and the half-open `range(R//2 …)` bounds in `split_diagonals` set how R
   relates to G and how many tiles land on the rim. Missing half-tiles or a
   short row at the edge start here.
2. **The diamond crop window.** Any change to `_crop_window` or to the tile
   offsets must keep `_inside` in step: a tile the crop cuts has to be left out
   before assembly, or the edge is no longer a still life
   (`test_cropped_diamond_mosaics_stay_still_lifes`). The square layout has no
   crop.
3. **Mask hole filling.** Going back to plain `binary_fill_holes`, or raising
   the n²/4 threshold, can make a subject that touches no edge vanish under the
   backdrop (`test_enclosed_foreground_survives_mask_building`).
4. **GIFs.** `generate_from_gif` processes every frame but returns only the
   first ([mosaic.py:395-402](../src/gol_mosaics/mosaic.py#L395-L402)), so
   saving with `save_all=True` writes a still image. It also passes no seed, so
   with `random_tiles=True` reassembled frames would flicker.
5. **Library order.** Diamond libraries are not sorted by population; square
   ones are in canonical (population, bytes) order. `tiles_for_mask` works
   anyway, because every diamond tile fills to the same diamond, but nothing
   should rely on `tiles[-1]` being the densest tile.

## Symptom table

| Symptom | Stage | Where to look |
|---------|-------|---------------|
| Half-tiles, short rows or a ragged edge (diamond) | 3, 7 | `rotate_and_pixelate` trim, `split_diagonals`, `_crop_window` |
| Diagonal grids doubled or gapped | 5 | `lattice_offset`, `_pad_diagonals` |
| Pattern unstable at the edge (diamond, very wide or tall input) | 7 | `_inside` out of step with `_crop_window` |
| `inconsistent overlap while assembling` (square) | 5 | tiles from another scheme or level passed to `assemble` |
| Subject vanishes under the backdrop | 6 | `_fill_small_holes` threshold |
| Backdrop on the subject rather than the background | 6 | `alpha_cutoff`, polarity in `tiles_for_mask` or `_build_square_mask` |
| Specks of GoL background inside the backdrop (diamond) | 6 | hole threshold too low |
| Too dense, too sparse or inverted tones | 2, 4 | `contrast`, `empty_tiles_cutoff`, `invert` |
| Corner colour differs from the app's padding bars | 3, 9 | rotation corners count as subject; bars are `eca_background` |
| Different output with the same seed | 1 | random choices in `__init__`; `ColourScheme.warhol` |

The free-form pipeline, which solves the whole pattern cell by cell with CP-SAT
instead of placing tiles, lives in `gol_mosaics.freeform` and is described in
[freeform.md](freeform.md) and [experiments/beyond_tiles/REPORT.md](../experiments/beyond_tiles/REPORT.md).

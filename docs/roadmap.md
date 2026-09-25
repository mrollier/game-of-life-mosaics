# Roadmap

Where the project could go next: the priorities first, then known problems,
performance work that is identified but not done, and a longer list of ideas.
Effort tags: **(S)** an afternoon, **(M)** a few days, **(L)** a project.

## Next steps

### 1. Free-form still lifes in the web app (M)

`gol_mosaics.freeform.solve_poster` now runs as a library call, so the app can
offer a "beyond tiles" mode next to the two tile layouts.

- Small canvases only: 200x200 cells solve to proven optimality in about a
  minute; cap the canvas and the solver time, and show progress.
- The Space needs OR-Tools (`[beyond]`), and each strip solve commits about
  1.5 GB, so a free CPU Space must solve one strip at a time
  (`strip_procs=1`) or run the monolithic solve for small canvases.
- Render with `compose` so the colour schemes and backgrounds already in the
  app apply; offer the `.cells` download of the verified pattern.
- Stretch: queue long solves and e-mail or poll for the result.

### 2. Level-7 diamonds on demand (M)

The level-7 database (108,492,376 tiles, 1.19 GB packed, sha256 recorded in
[search/tiles/README.md](../search/tiles/README.md)) exists only as a local
file. Host it as a Hugging Face dataset, and let `TileLibrary.load(7)` fetch
it on first use (behind `huggingface_hub`, an optional extra). This also
gives the artefact the off-site copy it still lacks.

- Expanding 10^8 tiles is impossible in memory, so level 7 needs selection
  on the packed bits: densities from `bits @ orbit_sizes + forced_alive_count`
  (`Domain.orbit_sizes`, `Domain.forced_alive_count` exist), then expand only
  the chosen tiles. The same change makes level 6 cheap (see *Performance*).
- Publish the DRAT completeness certificates (161 MB) with it, or on Zenodo,
  and fill in the DOI placeholder in `search/tiles/README.md`.

### 3. Other Life-like rules, end to end (M)

`sat_search.enumerate_tiles(level, birth=..., survival=...)` already
enumerates tiles for any Life-like rule (HighLife B36/S23 is tested). What is
missing is the rest of the pipeline:

- a rule argument on `TileLibrary` and `MosaicGenerator`, with shipped packed
  databases for, say, HighLife levels 3-5 (`tools/pack_tiles.py` writes them);
- the interlock derivation and hypothesis (H) checked per rule (the frame of
  ponds must itself be a still life under the rule);
- `life.is_still_life(..., birth, survival)` in the tests and a rule selector
  in the app; `GollyExporter` writing the rule into RLE headers.

## Open problems

- **Ragged outer rim (diamonds).** The edge of a diamond mosaic can show too
  few tiles or a short row, from the 45-degree rotation and the crop back to
  the image's aspect ratio. Since 3.0 the tiles that crop would cut are left
  out, which keeps the pattern a still life but makes wide and tall images'
  edges raggeder; a crop aligned to the lattice, or a canvas sized from the
  aspect ratio as the square layout does, would give a clean edge. (M)
- **`grid_size` crops.** Some values give a surprisingly cropped image; the
  relation between `grid_size`, the aspect ratio and the final canvas should be
  made explicit (and documented). (S)
- **Animated GIFs.** `generate_from_gif` processes every frame but returns
  only the first; it should return all frames so `save(..., save_all=True)`
  writes an animation. (S)
- **One cell per pixel on a canvas.** Composing several mosaics (the Marilyn
  diptych) breaks the one-cell-one-pixel correspondence because each mosaic
  has its own size. (M)
- **Dead-edge minimality.** The derived interlock cells are provably
  sufficient; which could be freed for *some* tile pairings? One SAT query per
  candidate cell. (M)
- **No-symmetry census, level 4.** More than two million tiles; the
  cube-and-conquer runner is ready (`search/nosym/search_nosym.py run --level 4
  --cube-bits 16`). (S to run, hours of compute)
- **A lower bound for free-form solves.** The best proven bound at 400x400 is
  2 (the pipeline reaches 3); a real bound needs wider relaxed strips
  (`freeform.decompose.lower_bound_strips`). (M)

## Performance

Identified during the 3.0 clean-up and not done yet, largest gain first.

- **Level-6 memory.** `TileLibrary.load(6)` expands 332,321 tiles to 430 MB.
  Selection only needs densities (computable from the packed bits, see
  *Level-7 diamonds*) and the chosen tiles, which would cut the Space's
  largest allocation to a few MB and start level 6 instantly. (M)
- **Free-form model building.** `solver.build_model` and the LNS patch
  model loop over cells in Python; building the neighbourhood sums from index
  arrays would halve the time of small patch solves. (M)
- **Background site loops.** `compose.agar_background` and
  `mosaic_background` test lattice sites one by one; a binary erosion of the
  mask by the tile footprint answers all sites at once. (S)
- **Diamond mask assembly.** `_build_mask` assembles full tiles to build a
  two-valued mask; a Kronecker product with the filled tile does the same.
  (S)
- **Search merges.** `search.py merge` verifies tiles serially (about 3,000
  tiles/s at level 7); `search_adaptive.py` already verifies inside workers,
  and `search.py` could do the same. (S)

## Ideas

### Tile mathematics

- **Oscillating mosaics** — period-2 tiles (`step(A) = B`, `step(B) = A`) for
  portraits that blink in Golly. (M)
- **Rectangular tiles** — relax D4 to D2 in `TileScheme` for n x m tiles and a
  third census. (M)
- **Brick lattice** — offset-row square tiles; needs a lattice symmetry
  parameter. (L)
- **Mixed-level mosaics** — large tiles in flat regions, small ones where the
  image has detail; the interlock across levels is the research question. (L)
- **OEIS** — submit the diamond census (1, 2, 7, 85, 2632, 332321, 108492376,
  172693540438) and the square census (1, 3, 65, 10398, 19287185). (S)

### Rendering

- **QR codes as still lifes** — one square tile per module, holes for light
  modules; an adapter from a QR matrix to `tile_scheme.assemble`. (S)
- **Error diffusion for tiles** — Floyd-Steinberg over tile densities for
  smoother tone at small grid sizes (the free-form targets already have it,
  `dither='fs'`). (S)
- **Stained glass** — colour each tile from the image's local colour. (M)
- **Glider-crash GIF** — the portrait stable until a glider arrives, then
  coming apart; the tutorial [04_golly_export](../notebooks/tutorials/04_golly_export.ipynb)
  already simulates it. (S)
- **Vector export** — SVG or PDF for large prints, plotters, stencils. (M)
- **Photomosaic of mosaics.** (L)

### Web app

- **Show and accept the seed** — the per-session seed exists; showing it and
  accepting one makes a look shareable. (S)
- **Warhol canvas** — one mosaic in a 2x2 grid of palettes. (S)
- **"Verified still life" badge** — run `is_still_life` on the result. (S)
- **Before/after slider.** (S)
- **Square level-6 by sampling** — 19,287,185 tiles cannot ship; sample them
  by density with the cube-and-conquer runner. (L)

### Outreach

- **Second paper** on tile schemes, hypothesis (H) and the square census;
  material in [tile_scheme_generalisation](../notebooks/research/tiles/tile_scheme_generalisation.ipynb). (L)
- **An interactive explainer** from a single pond to a full mosaic. (M)

## Recently done

In 3.0: one vocabulary (pond, tile, mosaic, layout) and British spelling in the
API; the free-form solver in the package; every tile database packed (21 MB to
2.7 MB); level-6 selection without a 14 GB matrix; the level-7 enumeration and
the level-8 count; the `.cells` download in the app; a guarded deploy. See
[CHANGELOG.md](../CHANGELOG.md).

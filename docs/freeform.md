# Free-form still lifes

A tile mosaic is fast, but its texture is the tile vocabulary: the same few
tiles recur everywhere. `gol_mosaics.freeform` drops the tiles. It solves the
whole image as *one* still life, cell by cell, with the constraint solver
CP-SAT: stability is a hard constraint, and the live-cell count of every 8x8
window is pulled towards the image's tone. The result follows the image far
more closely and its texture is its own (92 % of its 6x6 blocks are unique,
where a tile mosaic reuses a handful).

It needs the `[beyond]` extra (OR-Tools) and real computing time: seconds for
a 200x200 portrait, minutes for 400x400, hours for a poster.

## A poster in a few lines

```python
from gol_mosaics.freeform.poster import PosterConfig, solve_poster
from gol_mosaics.freeform.targets import load_rect_target

grey, free = load_rect_target('input/images/john.png', width=400, height=560)
result = solve_poster(grey, free, PosterConfig(), out='runs/john', log=print)

result.pattern                 # the still life, 0/1, verified
result.report['deviation']     # how closely it follows the image
```

`solve_poster` is the pipeline every large piece here was made with:

1. **Strips or blocks.** The canvas is cut into horizontal strips (or, when
   wide, into blocks), separated by two dead lines, and each piece is solved
   on its own, in parallel. Two dead lines are exactly enough to keep the
   pieces from interacting, so the stitched result is a still life.
2. **Polish.** Large-neighbourhood search re-solves the worst 40x40 patches
   against their frozen surroundings, in rounds, while the objective keeps
   falling (`freeform.lns.improve`).
3. **Seams.** A polished decomposition still shows its separators: a window
   total cannot see *where* inside the window the cells are. Seam rounds give
   every separator a proportional sub-target.
4. **Diagonal repair.** Long diagonal chains of live cells read as drawn
   lines; the solver forbids chains longer than five, and a last pass repairs
   any that patching created.

With `out=`, every stage is saved, and `resume=True` continues a killed run.
The same pipeline as a command line, with the render and the Golly export:

```bash
python experiments/beyond_tiles/poster.py input/images/john.png \
    --width 1416 --height 2000 --out results/john
```

## Settings that matter

| Setting | Default | Why |
|---|---|---|
| `d_max` | 0.40 | the density of pure black; 0.45 sits on the still-life packing ceiling, where dark images stall |
| `dither` | `'round'` | `'fs'` (error diffusion) for smooth synthetic gradients, which otherwise turn into dotted lines |
| `strip_rows`, `block_cols` | 48, 0 | set `block_cols` (e.g. 416) for wide canvases: a 48x2480 strip does not close, a 48x416 block closes in minutes |
| `strip_procs`, `polish_procs` | 5, 4 | each strip solve commits about 1.5 GB; on a 16 GB machine use 2 and 3 |
| `isolate` | `False` | one process per strip, so a crashed solver worker costs one retry, not the batch |
| `max_diag_run` | 5 | the longest allowed diagonal chain; `None` lifts the cap |

Sides must be multiples of 8 for the window lattice (or use
`edge_windows='partial'` with strips).

## Backgrounds and colours

A solve costs minutes to hours, so colours and backgrounds are chosen
afterwards from the stored pattern: `gol_mosaics.compose` paints it in any
`ColourScheme` and can grow a tile or agar background that is itself part of
the still life (see the [usage guide](usage.md#backgrounds-for-any-still-life)).
Store solves with `freeform.io.save_packed` (about 20 kB for a million
cells).

## Modules

| Module | |
|---|---|
| `targets` | image to per-window live-cell targets; window geometry; tone equalisation |
| `solver` | the CP-SAT model (`SolveConfig`, `solve_image`) and `verify_still_life` |
| `decompose` | strips and blocks, and a valid lower bound from relaxed strips |
| `lns` | patch polishing, seam sub-targets, diagonal repair |
| `seeds` | a block-agar warm start |
| `metrics` | deviation, texture and tile-overlap statistics |
| `io` | bit-packed storage of patterns and solver snapshots |
| `poster` | the pipeline above |

## Where to read more

- The research record, with every experiment, benchmark and design decision:
  [experiments/beyond_tiles/REPORT.md](../experiments/beyond_tiles/REPORT.md).
- A guided tour with live solves:
  [beyond_tiles notebook](../notebooks/research/beyond_tiles/beyond_tiles.ipynb);
  backgrounds and recolouring:
  [beyond_tiles_art](../notebooks/research/beyond_tiles/beyond_tiles_art.ipynb).
- Finished pieces: the [studies](../studies/README.md) (a PhD-defence flyer,
  banners, the whole Ghent Altarpiece at one cell per pixel).

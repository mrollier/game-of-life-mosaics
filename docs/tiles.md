# The tiles

Every mosaic is assembled from *tiles*: still lifes inside a shared frame of
ponds, glued to their neighbours through that frame. This page explains what
a tile is, how the complete set of tiles of each level is computed, and where
the results live. The words are defined in the [glossary](glossary.md).

## What makes a tile

A level-*L* diamond tile is a 6*L* x 6*L* grid whose border is a chain of
ponds, forced alive. Inside, any pattern is allowed that

1. keeps the whole tile a still life (every live cell has two or three live
   neighbours, no dead cell has exactly three),
2. is D4-symmetric (unchanged by the four rotations and four reflections),
3. keeps the *interlock cells* dead: the interior cells near where two tiles
   meet.

The interlock cells are what make tiles freely combinable. They are derived
from the geometry (`tile_domain.derive_dead_edges_full`): a cell must be dead
if it can influence a cell whose neighbourhood reaches into two tiles. With
them dead, every cell near a seam sees only frame cells, whose configuration
is the same for every choice of tiles and is a still life, so *any*
assignment of tiles to a lattice is one global still life. The same argument
works for any tile scheme (a support, a frame and a lattice) that satisfies
the hypothesis that the frame-only mosaic is a still life; the square family
is the second instance (`gol_mosaics.tile_scheme`).

The walk-through from pond to frame to tile is the tutorial
[05_how_tiles_work](../notebooks/tutorials/05_how_tiles_work.ipynb).

## The census

| Level | Grid | Live cells in the basic (frame-only) tile | Interior cells enclosed by the frame | Free cells after the interlock | Free D4 orbits | Symmetric tiles |
|---|---|---|---|---|---|---|
| 1 | 6×6 | 8 | 4 | 4 | 1 | 1 |
| 2 | 12×12 | 24 | 32 | 16 | 3 | 2 |
| 3 | 18×18 | 40 | 96 | 68 | 10 | 7 |
| 4 | 24×24 | 56 | 196 | 156 | 22 | 85 |
| 5 | 30×30 | 72 | 332 | 280 | 38 | 2,632 |
| 6 | 36×36 | 88 | 504 | 440 | 59 | 332,321 |
| 7 | 42×42 | 104 | 712 | 636 | 84 | 108,492,376 |
| 8 | 48×48 | 120 | 956 | 868 | 114 | 172,693,540,438 |

- The **basic tile** is the pond frame with an empty interior: the sparsest
  tile at every level. Its live count is 16*L* - 8, the interior it encloses
  18*L*² - 26*L* + 12 cells, and the interlock ring holds 12*L* - 8 of those
  for *L* ≥ 2.
- **Levels 1-7 are enumerated exhaustively.** Levels 1-6 ship with the
  package; the level-7 database (1.2 GB packed) was computed twice by
  independent runs and is kept outside git (see [search/tiles](../search/tiles/README.md)).
- **Level 8 is a count, not a database.** The tile formula has one variable
  per free orbit and no auxiliary variables, so its number of solutions *is*
  the census. An exact model counter (sharpSAT-td) computes it by splitting
  the formula into independent parts, without listing the tiles: it
  reproduces levels 3-7 and gives 172,693,540,438 for level 8 in 6.5 hours
  on one core. Listing them would take weeks and about 2.5 TB.

**Square tiles** (`layout='square'`): 1, 3, 65, 10,398 and 19,287,185 tiles
for levels 2-6; levels 3-5 ship. **Without the symmetry requirement**
(`gol_mosaics.nosym_tiles`): 1, 2 and 1,061 tiles for levels 1-3 (181
classes up to symmetry); level 3 ships, level 4 exceeds two million and has
not been run.

## How the census is computed

`gol_mosaics.sat_search` turns a level into a propositional formula with one
boolean per free symmetry orbit; the still-life rule becomes clauses of the
form "this cell, in this state, does not have exactly *c* live neighbours".
A SAT solver (CaDiCaL, through python-sat) then lists every solution,
blocking each one as it is found. Because there are no auxiliary variables,
each blocking clause removes exactly one tile: the enumeration is exact and
free of duplicates.

```python
from gol_mosaics.sat_search import enumerate_tiles, HIGHLIFE

tiles = enumerate_tiles(level=5)                      # all 2632 tiles, seconds
highlife = enumerate_tiles(level=4, birth=HIGHLIFE[0],
                           survival=HIGHLIFE[1])      # any Life-like rule
```

The result is validated five ways: byte-exact agreement with the databases
the original Gurobi ILP produced (levels 3-5), a SAT-free brute force over
all 2²² assignments at level 4 (`python -m gol_mosaics.sat_search
bruteforce --level 4`), agreement between two unrelated solvers, an
independent per-tile checker, and whole-mosaic stability tests. DRAT proofs
certify that nothing is missing (`search/tiles/certify.py`).

Level 7 needs cube-and-conquer: the formula is split on its first variables
into independent sub-problems, solved in parallel and checkpointed. The
production runs, the adaptive splitter, the model counter and the
certificates are in [search/](../search/README.md); the method and its
validation are written up in the notebook
[sat_tile_generation](../notebooks/research/tiles/sat_tile_generation.ipynb),
and [sat_tile_search](../notebooks/research/tiles/sat_tile_search.ipynb)
runs it for any rule and level.

## Where the tiles live

| | |
|---|---|
| `src/gol_mosaics/data/tiles_{diamond,square}_level_N_orbits.npy` | the shipped databases: one bit per free orbit per tile, packed (`Domain.pack`); level 6 is 2.7 MB |
| `TileLibrary.load(level, layout)` | expands a database to uint8 grids, cached |
| `tools/pack_tiles.py` | writes a database from grids or a fresh enumeration; `check` verifies all of them |
| [reproduce.md](reproduce.md) | file hashes, CNF fingerprints and the commands behind every number of the paper |

# Glossary

The words this project uses, from the smallest object to the largest. The
code, the docs and the notebooks all follow this vocabulary.

**Still life**
: A Game of Life pattern that does not change from one generation to the
  next: every live cell has two or three live neighbours, and no dead cell
  has exactly three.

**Pond**
: The 4x4 still life with eight live cells arranged in a ring
  (`gol_mosaics.tile_domain.POND`). Every tile frame is built from ponds.

**Level**
: The size parameter of the tiles. A level-*L* tile is 6*L* cells wide, and
  its frame is a chain of ponds. Higher levels give more tiles to choose from,
  and so finer tone.

**Pond frame** (or frame)
: The chain of ponds around the edge of a tile, forced alive in every tile
  of a level. Neighbouring tiles share frame ponds, which is what glues them
  together (`tile_domain.pond_frame`, `TileLibrary.pond`).

**Tile**
: One still life inside a level's pond frame. The interior can hold any
  arrangement that keeps the whole tile a still life *and* keeps the
  interlock cells dead.

**Interlock cells** (dead edges)
: The interior cells next to where two tiles meet, forced dead so that any
  two tiles can sit side by side without disturbing each other. They are
  derived from the geometry (`tile_domain.derive_dead_edges`).

**Census**
: The complete set of tiles of one level: 1, 2, 7, 85, 2632, 332,321 and
  108,492,376 symmetric diamond tiles for levels 1-7. See [tiles.md](tiles.md).

**Symmetry, orbit**
: Diamond and square tiles are D4-symmetric (unchanged by rotations and
  reflections). The cells fall into *orbits* of symmetric positions; the SAT
  search has one variable per free orbit, and the databases store one bit per
  free orbit per tile.

**Layout**
: How tiles are arranged: `'diamond'` (tiles on two interlocking 45-degree
  grids) or `'square'` (axis-aligned tiles that share their border ponds).
  Each layout has its own tile family and census.

**Tile scheme**
: The general description of a layout: a tile's support, its frame and the
  lattice it is placed on (`gol_mosaics.tile_scheme.TileScheme`). Diamonds and
  squares are two schemes; the composition theorem holds for any scheme that
  satisfies its hypotheses.

**Mosaic**
: Tiles assembled on a lattice, one tile per image region, chosen so that the
  tile's density follows the image's grey value. A mosaic is itself one still
  life.

**Density**
: The fraction of live cells in a tile. Dark regions of the image get dense
  tiles, light regions sparse or empty ones.

**Background, ECA**
: The coloured texture behind the subject: an elementary cellular automaton
  (a one-dimensional Wolfram rule run downwards). It is decoration, not part
  of the still life.

**Free-form still life**
: A still life solved cell by cell (`gol_mosaics.freeform`) instead of being
  assembled from tiles. The live-cell count of every small window follows the
  image; the whole pattern is still one still life.

**Window, target**
: In the free-form solver, the image is covered by small windows (8x8 cells);
  each gets a target number of live cells from its grey value, and the solver
  minimises the total deviation from the targets.

**Strips, blocks, seams**
: Large free-form canvases are solved in horizontal strips (or 2-D blocks)
  separated by two dead lines, the *seams*, which keep the pieces from
  interacting. Seam rounds afterwards fill those lines to the density around
  them.

**LNS (large-neighbourhood search)**
: Polishing a solved pattern by re-solving small rectangular patches against
  their frozen surroundings (`freeform.lns`).

**Cells** (in code)
: A 0/1 array of Game of Life cells, the pattern itself, as opposed to a
  rendered image.

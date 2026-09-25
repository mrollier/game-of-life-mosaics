"""
Free-form still lifes: a greyscale image as one Game of Life still life.

Where the tile mosaics assemble pre-computed tiles, this subpackage solves
the whole pattern cell by cell with CP-SAT: stability is a hard constraint
and the live-cell count of every small window is pulled towards the
image's tone. Large canvases are solved in strips or blocks and polished
by re-solving rectangular patches (large-neighbourhood search).

Modules:
    targets    image -> per-window live-cell targets, window geometry
    solver     the CP-SAT model and single-shot solve (needs OR-Tools)
    decompose  strip and block decomposition with dead separators
    lns        patch re-solving, seam passes, diagonal-run repair
    seeds      a still-life block agar used as a warm start
    metrics    deviation, texture and tile-overlap statistics
    io         bit-packed storage for solved patterns and snapshots

The solver needs the optional OR-Tools dependency:
``pip install gol-mosaics[beyond]``. targets, seeds, metrics and io import
without it. The research record behind every design choice is
experiments/beyond_tiles/REPORT.md.
"""

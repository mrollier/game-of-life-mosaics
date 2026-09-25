# Search campaigns

The tile databases shipped in `src/gol_mosaics/data/` come from exhaustive
SAT searches. The encoding and the solver loop live in the package
(`gol_mosaics.tile_domain`, `gol_mosaics.sat_search`); this folder holds the
production runs around them, which need hours of CPU and write files too
large for git.

| Folder | What it computes | Status |
|---|---|---|
| [`tiles/`](tiles/README.md) | The symmetric (D4) diamond census: enumeration with cube-and-conquer, #SAT counting, DRAT completeness certificates | Levels 1-7 enumerated (level 7: 108,492,376 tiles); level 8 counted (172,693,540,438) |
| [`nosym/`](nosym/search_nosym.py) | The census without the symmetry requirement | Level 3 enumerated and shipped (1,061 tiles); level 4 prepared, not yet run |

Everything here runs from its own folder in a checkout with the package
installed (`pip install -e ".[sat]"`). Where results are reported and how to
reproduce the paper's tables is described in [`docs/reproduce.md`](../docs/reproduce.md).

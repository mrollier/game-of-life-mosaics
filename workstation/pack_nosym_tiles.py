"""
One-off: enumerate the level-3 no-symmetry diamond tile census (1061
tiles) and write it as packed free-cell bits into the package data
directory (solutions_pattern_nosym_level_3_cells.npy, ~9.5 KB).

Usage:  python pack_nosym_tiles.py

Requires python-sat (enumeration); the shipped file is expanded at load
time without it. Aborts unless the packed file expands back
byte-identically. Levels 1-2 are not stored (their nosym censuses equal
the symmetric ones and re-derive in milliseconds); the level-4 census
is uncomputed (> 2 x 10^6 tiles — see nosym_search/search_nosym.py and
REPRODUCE.md).
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from gol_mosaics.nosym_tiles import (  # noqa: E402
    build_nosym_cnf,
    enumerate_nosym_tiles,
    pack_nosym_solutions,
    unpack_nosym_solutions,
)

LEVEL = 3


def main() -> int:
    data_dir = REPO / "src" / "gol_mosaics" / "data"
    t0 = time.time()
    grids = enumerate_nosym_tiles(LEVEL)
    packed = pack_nosym_solutions(grids, LEVEL)
    if unpack_nosym_solutions(packed, LEVEL).tobytes() != grids.tobytes():
        print(f"FATAL: level {LEVEL} does not expand back byte-identically")
        return 1
    dst = data_dir / f"solutions_pattern_nosym_level_{LEVEL}_cells.npy"
    with open(dst, "wb") as f:
        np.save(f, packed)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"level {LEVEL}: {len(packed)} tiles x {packed.shape[1]} packed "
          f"bytes -> {dst.name} ({dst.stat().st_size / 1e3:.1f} KB, "
          f"{time.time() - t0:.1f}s)")
    print(f"file sha256:     {sha}")
    print(f"encoding sha256: {build_nosym_cnf(LEVEL).sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

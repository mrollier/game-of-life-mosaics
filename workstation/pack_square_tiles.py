"""
One-off: enumerate the pond-frame square tile sets (levels 3-5) and write
them as packed free-orbit bits into the package data directory
(solutions_square_level_{3,4,5}_orbits.npy, ~60 KB total).

Usage:  python pack_square_tiles.py

Requires python-sat (enumeration); the shipped files are expanded at load
time without it. Aborts unless every packed file expands back
byte-identically.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from gol_mosaics.tile_scheme import (  # noqa: E402
    enumerate_scheme_tiles,
    pack_scheme_solutions,
    pond_square_scheme,
    unpack_scheme_solutions,
)

LEVELS = (3, 4, 5)


def main() -> int:
    data_dir = REPO / "src" / "gol_mosaics" / "data"
    for level in LEVELS:
        t0 = time.time()
        scheme = pond_square_scheme(level)
        grids = enumerate_scheme_tiles(scheme)
        packed = pack_scheme_solutions(scheme, grids)
        if unpack_scheme_solutions(scheme, packed).tobytes() != grids.tobytes():
            print(f"FATAL: level {level} does not expand back byte-identically")
            return 1
        dst = data_dir / f"solutions_square_level_{level}_orbits.npy"
        with open(dst, "wb") as f:
            np.save(f, packed)
        print(f"level {level}: {len(packed)} tiles x {packed.shape[1]} packed "
              f"bytes -> {dst.name} ({dst.stat().st_size / 1e3:.1f} KB, "
              f"{time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

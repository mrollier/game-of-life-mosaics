"""
One-off: compress the level-6 tile database into the packed free-orbit-bit
format shipped inside the package (~2.7 MB instead of ~430 MB).

Usage:  python pack_level6.py [solutions_pattern_level_6.npy]

Aborts unless the packed file expands back byte-identically.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from gol_mosaics.tile_domain import pack_solutions, unpack_solutions  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "solutions_pattern_level_6.npy"
    dst = REPO / "src" / "gol_mosaics" / "data" / "solutions_pattern_level_6_orbits.npy"

    grids = np.load(src)
    packed = pack_solutions(grids, level=6)
    if unpack_solutions(packed, level=6).tobytes() != grids.astype(np.uint8).tobytes():
        print("FATAL: packed file does not expand back byte-identically")
        return 1

    with open(dst, "wb") as f:
        np.save(f, packed)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.2f} MB, "
          f"{len(packed)} solutions x {packed.shape[1]} packed bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

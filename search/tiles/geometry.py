"""
Shim: the tile geometry now lives in the gol_mosaics package
(src/gol_mosaics/tile_domain.py), where the dead-edge lists are *derived*
from the interlocking mosaic geometry for any level (levels 2-6 reproduce
the historical hard-coded lists exactly; level 7+ are new).

This module re-exports it so the search scripts keep working. Run from a
checkout of the repository (sys.path fallback below), or `pip install -e .`
first. The independent checker verify.py deliberately does NOT go through
the encoding — only through forced_masks — preserving its independence.
"""

import sys
from pathlib import Path

try:
    from gol_mosaics.tile_domain import *  # noqa: F401,F403
    from gol_mosaics.tile_domain import _base_masks  # noqa: F401
except ImportError:
    _src = Path(__file__).resolve().parents[2] / "src"
    if not _src.is_dir():
        raise ImportError(
            "gol_mosaics is not importable and the repository src/ directory "
            "was not found next to this bundle. Copy the whole repository to "
            "the target machine (or pip install gol-mosaics)."
        )
    sys.path.insert(0, str(_src))
    from gol_mosaics.tile_domain import *  # noqa: F401,F403
    from gol_mosaics.tile_domain import _base_masks  # noqa: F401

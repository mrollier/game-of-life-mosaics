"""
Shim: the SAT encoding now lives in the gol_mosaics package
(src/gol_mosaics/sat_search.py), generalized to arbitrary Life-like rules
B{birth}/S{survival} (Conway is the default). This module re-exports it so
the search scripts keep working; see geometry.py for the import fallback.
"""

import geometry  # noqa: F401  (installs the sys.path fallback first)

from gol_mosaics.sat_search import (  # noqa: F401
    CONWAY,
    HIGHLIFE,
    Clause,
    Encoding,
    Lit,
    build_cnf,
    bruteforce_tiles,
    cube_units,
    enumerate_all,
    enumerate_tiles,
    rule_violations,
)

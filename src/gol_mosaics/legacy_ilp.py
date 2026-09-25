"""
The original integer-linear-programming tile generator (Gurobi).

Historical: the shipped tile databases are now enumerated with SAT
(:mod:`gol_mosaics.sat_search`), which is exact, needs no licence and scales
to level 7. This module keeps the ILP that produced the first databases,
following Rob Bosch's "Opt Art" (2019), for comparison and for the
`notebooks/research/archive` walkthrough. gurobipy is imported only when a
function here runs, so ``import gol_mosaics`` never touches it.

Constraint semantics (mirrored by the SAT encoding and pinned by its tests):
an alive cell has 2-3 live neighbours, a dead cell does not have exactly 3,
the tile is D4-symmetric, the pond frame is forced alive, cells outside the
diamond and the derived interlock cells are forced dead. The objective
maximises the live-cell count, and each solution is excluded in turn until
the model is infeasible.
"""

import logging

import numpy as np
from scipy.ndimage import binary_fill_holes

from .tile_domain import (derive_dead_edges, neighbours, pond_frame,
                          symmetric_coords)

logger = logging.getLogger(__name__)


def _gurobi():
    try:
        import gurobipy
    except ImportError as exc:
        raise ImportError(
            "Gurobi is required for the legacy ILP generator but is not "
            "installed. Install with: pip install gol-mosaics[gurobi] (a "
            "licence is needed; free academic licences at gurobi.com). The "
            "shipped tiles come from the SAT search instead: see "
            "gol_mosaics.sat_search.enumerate_tiles."
        ) from exc
    return gurobipy


def generate_tiles(level: int, solution_limit: int = 1000) -> np.ndarray:
    """
    Find symmetric still-life tiles of one level with the Gurobi ILP.

    Args:
        level: Tile level (dead-edge data is derived for any level >= 1)
        solution_limit: Stop after this many tiles (default 1000)

    Returns:
        (N, 6*level, 6*level) array of tiles, in the order found (densest
        first, since the objective maximises live cells)

    Raises:
        ImportError: If gurobipy is not available
    """
    gp = _gurobi()
    GRB, quicksum = gp.GRB, gp.quicksum

    pp_edge = pond_frame(level)
    n = pp_edge.shape[0]
    logger.info("Looking for pattern level %d with grid size %dx%d",
                level, n, n)

    # Mask for cells outside the tile pattern
    pp_edge_binary = (pp_edge > 0).astype(np.uint8)
    pp_outside = 1 - binary_fill_holes(pp_edge_binary).astype(np.uint8)

    model = gp.Model("still_life")
    model.setParam('OutputFlag', 0)

    alive = {}  # Alive cells
    ldead = {}  # Low dead (< 2 neighbours)
    hdead = {}  # High dead (> 3 neighbours)
    for i in range(n):
        for j in range(n):
            alive[i, j] = model.addVar(vtype=GRB.BINARY, name=f"A_{i}_{j}")
            ldead[i, j] = model.addVar(vtype=GRB.BINARY, name=f"L_{i}_{j}")
            hdead[i, j] = model.addVar(vtype=GRB.BINARY, name=f"H_{i}_{j}")
    model.update()

    dead_edges = derive_dead_edges(level)

    for i in range(n):
        for j in range(n):
            neighbour_sum = quicksum(alive[ii, jj]
                                     for (ii, jj) in neighbours(i, j, n))

            # Low-dead: cells with < 2 neighbours
            model.addConstr(4 * ldead[i, j] + neighbour_sum <= 6,
                            name=f"low_dead_{i}_{j}")
            # High-dead: cells with > 3 neighbours
            model.addConstr(4 * hdead[i, j] <= neighbour_sum,
                            name=f"high_dead_{i}_{j}")
            # Stayin' alive: alive cells need 2-3 neighbours
            model.addConstr(2 * alive[i, j] <= neighbour_sum,
                            name=f"stay1_{i}_{j}")
            model.addConstr(3 * alive[i, j] + neighbour_sum <= 6,
                            name=f"stay2_{i}_{j}")
            # Exactly one of L, H, or A is true
            model.addConstr(ldead[i, j] + hdead[i, j] + alive[i, j] == 1,
                            name=f"oneof_{i}_{j}")

            # D4 symmetry (the seven non-identity images)
            for (ii, jj) in symmetric_coords(i, j, n)[1:]:
                model.addConstr(alive[i, j] == alive[ii, jj])
                model.addConstr(ldead[i, j] == ldead[ii, jj])
                model.addConstr(hdead[i, j] == hdead[ii, jj])

            if pp_edge_binary[i, j]:
                model.addConstr(alive[i, j] == int(pp_edge_binary[i, j]),
                                name=f"force_alive_{i}_{j}")
            if pp_outside[i, j]:
                model.addConstr(alive[i, j] == 0, name=f"force_dead_{i}_{j}")
            if (i, j) in dead_edges:
                model.addConstr(alive[i, j] == 0,
                                name=f"force_dead_edge_{i}_{j}")

    # Objective: maximise number of living cells
    model.setObjective(
        quicksum(alive[i, j] for i in range(n) for j in range(n)),
        GRB.MAXIMIZE
    )

    # Iterative exclusion to find all solutions
    solutions = []
    while True:
        model.optimize()
        if model.status != GRB.OPTIMAL:
            logger.info("Found %d optimal solutions.", len(solutions))
            break

        sol = np.array([[round(alive[i, j].X) for j in range(n)]
                        for i in range(n)])
        solutions.append(sol)

        alive_cells = [(i, j) for i in range(n) for j in range(n)
                       if round(alive[i, j].X) == 1]
        model.addConstr(
            quicksum(1 - alive[i, j] for (i, j) in alive_cells) +
            quicksum(alive[i, j] for i in range(n) for j in range(n)
                     if (i, j) not in alive_cells) >= 1,
            name=f"exclude_solution_{len(solutions)}"
        )

        if len(solutions) >= solution_limit:
            logger.info("Reached solution limit (%d).", solution_limit)
            break

    return np.array(solutions)

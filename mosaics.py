import numpy as np
from gurobipy import Model, GRB, quicksum
from scipy.ndimage import binary_fill_holes
import os

def pond_pattern():
    pp = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]])
    return pp

def pond_pattern_multiple(level=4):
    pond_width = 6
    width = pond_width * level

    pp = pond_pattern()

    # create a double pattern by stacking the original pattern
    if level > 1:
        pp_multiple = np.vstack((np.vstack([pp[:-1]]*(level-1)),
                                        pp,
                                        np.vstack([pp[1:]]*(level-1))))
        pp_multiple = np.hstack((np.hstack([pp_multiple[:,:-1]]*(level-1)),
                                        pp_multiple,
                                        np.hstack([pp_multiple[:,1:]]*(level-1))))
    else:
        pp_multiple = pp.copy()

    # add an edge to the pattern
    pp_multiple = np.pad(pp_multiple, pad_width=1, constant_values=0)

    ### mask all the corners
    # diagonal corners
    mask_even = np.array([[(i+j)<pond_width*level/2 for j in range(width)] for i in range(width)])
    mask_even = mask_even + mask_even[::-1, ::-1]

    # off-diagonal corners
    mask_odd = np.array([[(i-j)>=pond_width*level/2 for j in range(width)] for i in range(width)])
    mask_odd = mask_odd + mask_odd.T

    mask = mask_even + mask_odd
    pp_multiple = np.where(1-mask, pp_multiple, 0)
    return pp_multiple

def pond_pattern_edge(level=4):
    pond_width = 6
    width = pond_width * level
    pp_multiple = pond_pattern_multiple(level=level)

    ### mask the interior
    mask_corner = np.array([[(i+j)<pond_width*level/2+3 for j in range(width)] for i in range(width)])
    mask = mask_corner + mask_corner[::-1, ::-1] + mask_corner[::-1] + mask_corner[:, ::-1]

    pp_edge = np.where(mask, pp_multiple, 0)
    return pp_edge

def empty_mosaic(level=4, n_odd_tiles=1):
    pond_width = 6
    pp_edge = pond_pattern_edge(level=level)

    # add odd tiles
    em_odd = np.hstack([pp_edge]*n_odd_tiles)
    em_odd = np.vstack([em_odd]*n_odd_tiles)

    # add white edge around odd tiles
    pad_width = ( (pond_width-3)*(2*level-1) + 1 + 2 ) // 2
    em_odd = np.pad(em_odd, pad_width=pad_width, constant_values=0)

    # add even tiles
    em_even = np.hstack([pp_edge]*(n_odd_tiles+1))
    em_even = np.vstack([em_even]*(n_odd_tiles+1))

    # add even tiles to odd tiles
    em = em_odd + em_even

    return em

def find_all_symmetric_gol_mosaics(level=4, solution_limit=1e3):
    # create empty mosaic and print initial information
    pp_edge = pond_pattern_edge(level=level)
    n = pp_edge.shape[0]
    print(f"Looking for pattern level {level} with grid size {n}x{n}")

    # make mask for cells that are outside the tile pattern
    pp_edge = (pp_edge > 0).astype(np.uint8)
    # Fill the interior
    pp_outside = 1-binary_fill_holes(pp_edge).astype(np.uint8)

    # Create Gurobi model
    model = Model("still_life")

    # Decision variables
    alive = {}  # Alive
    ldead = {}  # Low dead
    hdead = {}  # High dead

    for i in range(n):
        for j in range(n):
            alive[i, j] = model.addVar(vtype=GRB.BINARY, name=f"A_{i}_{j}")
            ldead[i, j] = model.addVar(vtype=GRB.BINARY, name=f"L_{i}_{j}")
            hdead[i, j] = model.addVar(vtype=GRB.BINARY, name=f"H_{i}_{j}")

    model.update()

    # Define neighbors function
    def neighbors(i, j, n):
        return [
            ((i + di) % n, (j + dj) % n)
            for di in [-1, 0, 1]
            for dj in [-1, 0, 1]
            if not (di == 0 and dj == 0)
        ]

    def symmetric_coords(i, j, n):
        return {
            "ver": (n - 1 - i, j),
            "hor": (i, n - 1 - j),
            "ver_and_hor": (n - 1 - i, n - 1 - j),
            "diag": (j, i),
            "diag_and_ver": (j, n - 1 - i),
            "diag_and_hor": (n - 1 - j, i),
            "all_sym": (n - 1 - j, n - 1 - i)
        }

    def _return_dead_edges(level:int):
        dead_edges = {
            "level_2" : [
                (2,6),
                (3,6)],
            "level_3" : [
                (2,9),
                (3,9),
                (5,11),
                (5,12)],
            "level_4" : [
                (2,11),
                (3,11),
                (5,14),
                (5,15),
                (6,15)],
            "level_5" : [
                (2,15),
                (3,15),
                (5,17),
                (5,18),
                (6,18),
                (8,20),
                (8,21)],
            "level_6" : [
                (2,18),
                (3,18),
                (5,20),
                (5,21),
                (6,21),
                (8,23),
                (8,24),
                (9,24)
            ]
        }
        return dead_edges[f"level_{level}"]

    # Constraints
    for i in range(n):
        for j in range(n):
            N = neighbors(i, j, n)
            neighbor_sum = quicksum(alive[ii, jj] for (ii, jj) in N)

            # Low-dead
            model.addConstr(4 * ldead[i, j] + neighbor_sum <= 6, name=f"low_dead_{i}_{j}")

            # High-dead
            model.addConstr(4 * hdead[i, j] <= neighbor_sum, name=f"high_dead_{i}_{j}")

            # Stayin' alive constraints
            model.addConstr(2 * alive[i, j] <= neighbor_sum, name=f"stay1_{i}_{j}")
            model.addConstr(3 * alive[i, j] + neighbor_sum <= 6, name=f"stay2_{i}_{j}")

            # Exactly one of L, H, or A is true
            model.addConstr(ldead[i, j] + hdead[i, j] + alive[i, j] == 1, name=f"oneof_{i}_{j}")

            # Symmetry constraints
            sym_coords = symmetric_coords(i, j, n)
            for (ii, jj) in sym_coords.values():
                model.addConstr(alive[i, j] == alive[ii, jj])
                model.addConstr(ldead[i, j] == ldead[ii, jj])
                model.addConstr(hdead[i, j] == hdead[ii, jj])

            # Constraint to ensure that some values are alive along the tile pattern
            if pp_edge[i,j]:
                model.addConstr(alive[i, j] == int(pp_edge[i,j]), name=f"force_alive_{i}_{j}")
            # Constraint to ensure that cells outside the tile pattern are dead
            if pp_outside[i, j]:
                model.addConstr(alive[i, j] == 0, name=f"force_dead_{i}_{j}")

            dead_edges = _return_dead_edges(level)
            # Add constraints for dead edges based on the level
            # Constraint to ensure that cells on the edges are dead
            if (i, j) in dead_edges:
                model.addConstr(alive[i, j] == 0, name=f"force_dead_edge_{i}_{j}")

    # Objective: maximize number of living cells
    model.setObjective(quicksum(alive[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE)

    # === Iterative exclusion ===
    solutions = []
    iterator = 0

    # I find 352 solutions for pattern_level=4
    while True:
        # Solve
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print("================================")
            print("No more optimal solutions found.")
            print("================================")
            break

        # Extract current solution
        sol = np.array([[round(alive[i, j].X) for j in range(n)] for i in range(n)])
        solutions.append(sol)

        # Identify which cells are alive
        alive_cells = [(i, j) for i in range(n) for j in range(n) if round(alive[i, j].X) == 1]

        # Add exclusion constraint: prevent rediscovery of this exact solution
        model.addConstr(
            quicksum(1 - alive[i, j] for (i, j) in alive_cells) +
            quicksum(alive[i, j] for i in range(n) for j in range(n) if (i, j) not in alive_cells)
            >= 1,
            name=f"exclude_solution_{len(solutions)}"
        )

        if len(solutions) >= solution_limit:
            print(f"Reached solution limit ({solution_limit}).")
            break

    solutions = np.array(solutions)
    return solutions

def load_all_symmetric_gol_mosaics(level:int=4):
    """
    Load all symmetric Game of Life mosaic solutions for a given level,
    using a path relative to this script's location.
    The solutions are loaded ranging from high density to low density.
    """
    if level < 3 or level > 5:
        raise ValueError("Level must be 3, 4 or 5.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", f"solutions_pattern_level_{level}.npy")
    solutions = np.load(data_path)
    return solutions

def map_greyscale_to_mosaic(greyscale_value, solutions, random=True):
    """
    Map one or more greyscale values (scalar or numpy array) to mosaics from the solutions array.
    Returns a numpy array of mosaics with shape (N, H, W) if greyscale_value has N elements.
    """
    greyscale_value = np.asarray(greyscale_value)
    # Calculate the mean density of each mosaic
    densities = np.mean(solutions, axis=(1, 2))
    # Normalize the densities to the range [0, 1]
    densities = (densities - densities.min()) / (densities.max() - densities.min())

    # Prepare output array
    output_shape = greyscale_value.shape + solutions.shape[1:]
    mosaics = np.empty(output_shape, dtype=solutions.dtype)

    # Flatten greyscale_value for iteration
    flat_grey = greyscale_value.ravel()
    selected_indices = []

    for idx, val in enumerate(flat_grey):
        if val < 0 or val > 1:
            raise ValueError("Greyscale value must be between 0 and 1")
        diffs = np.abs(densities - val)
        min_diff = diffs.min()
        indices = np.where(diffs == min_diff)[0]
        if random:
            index = np.random.choice(indices)
        else:
            index = indices[0]
        selected_indices.append(index)
        mosaics.reshape(-1, *solutions.shape[1:])[idx, ...] = solutions[index]

    return mosaics
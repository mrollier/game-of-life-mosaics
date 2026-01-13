# %% MOSAICS

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
    data_path = os.path.join(base_dir, "../data", f"solutions_pattern_level_{level}.npy")
    solutions = np.load(data_path)
    return solutions

def map_greyscale_to_mosaic(greyscale_values, solutions, random=True, invert=True, empty_tiles_cutoff=1.):
    """
    Map one or more greyscale values (scalar or numpy array) to mosaics from the solutions array.
    Returns a numpy array of mosaics with shape (N, H, W) if greyscale_value has N elements.
    """
    greyscale_values = np.asarray(greyscale_values)
    # Calculate the mean density of each mosaic
    densities = np.mean(solutions, axis=(1, 2))
    # Normalise the densities to the range [0, 1]
    densities = (densities - densities.min()) / (densities.max() - densities.min())

    # if invert, the meaning (colour) of 0 and 1 are inverted
    if invert:
        densities = densities[::-1]

    # Prepare output array
    output_shape = greyscale_values.shape + solutions.shape[1:]
    mosaics = np.empty(output_shape, dtype=solutions.dtype)

    # Flatten greyscale_value for iteration
    flat_grey = greyscale_values.ravel()
    # selected_indices = [] # not sure what this is for

    # TODO: this for loop is probably not very efficient
    for idx, val in enumerate(flat_grey):
        if val < 0 or val > 1:
            raise ValueError("Greyscale value must be between 0 and 1")
        if val > empty_tiles_cutoff:
            solution = np.zeros_like(solutions[0])
        else:
            diffs = np.abs(densities - val)
            min_diff = diffs.min()
            indices = np.where(diffs == min_diff)[0]
            if random:
                index = np.random.choice(indices)
            else:
                # TODO: it would be nice to play around with this in an animation, cycling through the options in a non-random way
                index = indices[0]
            solution = solutions[index]
        # selected_indices.append(index)
        mosaics.reshape(-1, *solutions.shape[1:])[idx, ...] = solution

    return mosaics

def numpy_to_cells(array, filename="output.cells", glider=False):
    """
    Function that saves a NumPy array to a .cells file format that can be used in Golly
    """
    if glider:
        # If glider is True, we add a glider pattern to the top-left corner
        glider_pattern = np.array([[0, 1, 0],
                                   [0, 0, 1],
                                   [1, 1, 1]])
        array[:glider_pattern.shape[0], :glider_pattern.shape[1]] = glider_pattern

    with open(filename, 'w') as f:
        f.write('!Generated from NumPy array\n')
        for row in array:
            line = ''.join('O' if cell else '.' for cell in row)
            f.write(line + '\n')

# %% IMAGES

from PIL import Image

def load_image(image_path, grayscale=True):
    """
    Load an image from the given path.
    If grayscale is True, convert the image to grayscale.
    Returns a PIL Image object.
    """
    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    return img

def square_image(img, grayscale=True, return_aspect=True):
    # Make the image square by cropping or padding
    width, height = img.size
    width_over_height = width / height
    if width != height:
        new_size = max(width, height)
        new_img = Image.new('L' if grayscale else 'RGB', (new_size, new_size), color=255)
        new_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
        img = new_img
    if return_aspect:
        return img, width_over_height
    return img

def rotate_and_pixelate(img, grid_size, expand=True):
    """
    1. Resample
    2. Rotate 45 degrees
    3. Pixelate by averaging over blocks
    4. Return low-res numpy array
    """
    if grid_size % 2 != 0:
        raise ValueError("Grid size must be even for this implementation.")

    # Resample to desired grid size
    img = img.resize((grid_size, grid_size), resample=Image.LANCZOS)

    # Rotate 45 degrees with expand so nothing is cut off
    img = img.rotate(45, expand=expand, fillcolor=255)

    # cut off edges and return as numpy array
    return np.array(img)[1:-1,1:-1]

def extract_diagonal_patterns(lowres):
    """
    Extract diagonal patterns from a low-res array.
    Returns two arrays: one for the first diagonal pattern and one for the second.
    """
    grid_size = lowres.shape[0]
    # first diagonals
    diag_indices_first = [[(grid_size//2-1-i+j, i+j) for i in range(grid_size//2)] for j in range(grid_size//2+1)]
    diag_indices_first = np.array(diag_indices_first)
    diag_indices_first_shape = diag_indices_first.shape
    diag_indices_first = diag_indices_first.reshape(-1,diag_indices_first.shape[-1])
    # second diagonals
    diag_indices_second = [[(grid_size//2-i+j, i+j) for i in range(grid_size//2+1)] for j in range(grid_size//2)]
    diag_indices_second = np.array(diag_indices_second)
    diag_indices_second_shape = diag_indices_second.shape
    diag_indices_second = diag_indices_second.reshape(-1,diag_indices_second.shape[-1])
    # zip indices
    rows_first, cols_first = zip(*diag_indices_first)
    rows_second, cols_second = zip(*diag_indices_second)
    # extract patterns
    lowres_first = lowres[rows_first, cols_first].reshape(diag_indices_first_shape[:2])
    lowres_second = lowres[rows_second, cols_second].reshape(diag_indices_second_shape[:2])
    # return
    return lowres_first, lowres_second

def diagonal_patterns_to_mosaic(lowres_first, lowres_second, level=4, invert=True, random=True, empty_tiles_cutoff=1.):
    solutions = load_all_symmetric_gol_mosaics(level=level)

    mosaics_first = map_greyscale_to_mosaic(lowres_first/255, solutions, invert=invert, random=random,
                                            empty_tiles_cutoff=empty_tiles_cutoff)
    mosaics_second = map_greyscale_to_mosaic(lowres_second/255, solutions, invert=invert, random=random,
                                             empty_tiles_cutoff=empty_tiles_cutoff)

    big_array_first = np.block([[mosaics_first[i, j] for j in range(mosaics_first.shape[1])]
                      for i in range(mosaics_first.shape[0])])
    big_array_second = np.block([[mosaics_second[i, j] for j in range(mosaics_second.shape[1])]
                      for i in range(mosaics_second.shape[0])])

    # add white edge around top row tiles
    pond_width = 6
    pad_tuple = ( ((pond_width-3)*(2*level-1) + 1 + 2) // 2, ((pond_width-3)*(2*level-1) + 1 + 2) // 2 )
    pad_width = ((0,0), pad_tuple)
    solution_mosaic_first = np.pad(big_array_first, pad_width=pad_width, constant_values=0)

    # add white edge around second-row tiles
    pad_width = (pad_tuple, (0,0))
    solution_mosaic_second = np.pad(big_array_second, pad_width=pad_width, constant_values=0)

    solution_mosaic = solution_mosaic_first + solution_mosaic_second
    return solution_mosaic

def image_to_still_life(image_path, grid_size=30, level=4, random=True, invert=True, empty_tiles_cutoff=1.):
    """
    Convert an image to a still life mosaic pattern.
    """
    # load image
    img = load_image(image_path, grayscale=True)
    # make the image square but save the original aspect ratio
    square_img, original_width_over_height = square_image(img, grayscale=True, return_aspect=True)
    # Make a low-resolution version of the image that can be converted to a mosaic
    lowres = rotate_and_pixelate(square_img, grid_size, expand=True)
    lowres_first, lowres_second = extract_diagonal_patterns(lowres)
    # create the actual mosaic
    solution_mosaic = diagonal_patterns_to_mosaic(lowres_first, lowres_second,
                                                  level=level, invert=invert, random=random,
                                                  empty_tiles_cutoff=empty_tiles_cutoff)
    # adjust for original aspect ratio by cropping (with a particular offset, if required)
    offset = 0
    if original_width_over_height > 1:
        # originally wider than tall: readjust square by cropping height
        new_height = int(solution_mosaic.shape[1] / original_width_over_height)
        start_height_idx = (solution_mosaic.shape[1] - new_height) // 2
        solution_mosaic = solution_mosaic[start_height_idx-offset:start_height_idx+new_height+offset, :]
    elif original_width_over_height < 1:
        # originally taller than wide: readjust square by cropping width
        new_width = int(solution_mosaic.shape[0] * original_width_over_height)
        start_width_idx = (solution_mosaic.shape[0] - new_width) // 2
        solution_mosaic = solution_mosaic[:, start_width_idx-offset:start_width_idx+new_width+offset]
    return solution_mosaic


def gif_to_still_life(gif_path, grid_size=30, level=4, random=True, invert=True, empty_tiles_cutoff=1.):
    """
    Convert a GIF to a still life mosaic pattern.
    """
    # Load GIF frames
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = img.convert('L')  # Convert to grayscale
            frames.append(frame.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # End of sequence

    # Process each frame
    mosaics = []
    for frame in frames:
        # Save frame to a temporary path
        temp_path = "temp_frame.png"
        frame.save(temp_path)
        # Convert frame to still life mosaic
        mosaic = image_to_still_life(temp_path, grid_size=grid_size, level=level, random=random, invert=invert, empty_tiles_cutoff=empty_tiles_cutoff)
        mosaics.append(mosaic)
        # Remove temporary file
        os.remove(temp_path)

    return np.array(mosaics)

def save_mosaic_as_image(mosaic, filename="output.png"):
    """
    Save a mosaic as an image file.
    """
    # invert
    mosaic = 1 - mosaic
    # Convert mosaic to 0-255 range
    mosaic_256 = (mosaic * 255).astype(np.uint8)
    img = Image.fromarray(mosaic_256, mode='L')  # 'L' for grayscale
    img.save(filename)
    print(f"Mosaic saved as {filename}")

def list_denominators(n):
    """
    Return a sorted list of all positive divisors of the positive integer n.
    Raises ValueError for non-positive or non-integer input.
    Used to find suitable grid sizes for saving mosaics *without pixel interpolation*.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    import math
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            j = n // i
            if j != i:
                large.append(j)
        i += 1
    return small + large[::-1]
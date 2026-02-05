# Game of Life Mosaics

> Create stunning artistic mosaics from images using Conway's Game of Life still-life patterns

Transform your photos into unique digital art pieces composed entirely of Game of Life patterns, overlaid with Elementary Cellular Automaton backgrounds. Each mosaic is a carefully crafted arrangement of living, stable patterns from Conway's Game of Life.

## Gallery

*Example mosaics will be added here - the code generates beautiful portraits and landscapes using symmetric GoL patterns in a diamond layout with colorful ECA backgrounds.*

## What is This?

This project combines two fascinating computational concepts to create digital art:

1. **Game of Life Still-Lifes**: Stable patterns in Conway's Game of Life that never change. These symmetric patterns are computed using integer linear programming to find all valid configurations.

2. **Elementary Cellular Automata (ECA)**: Simple one-dimensional cellular automata that create complex background patterns according to Wolfram rules.

The result is a unique mosaic where:
- Darker/lighter regions of your image are mapped to denser/sparser GoL patterns
- The patterns are arranged in a diamond (45° rotation) layout for visual interest
- An ECA pattern fills transparent areas with intricate backgrounds
- Everything is mathematically generated and fully customizable

## Features

- **Automatic mosaic generation** from any image (PNG, JPG, GIF)
- **Three complexity levels** (3, 4, 5) with pre-computed patterns
- **Customizable color schemes** (UGent colors, monochrome, or custom)
- **ECA background overlays** with multiple rule options
- **Transparency support** for creative compositing
- **Export to Golly format** for Game of Life simulation
- **Clean object-oriented API** designed for extensibility
- **Comprehensive documentation** and examples

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/mrollier/game-of-life-mosaics.git
cd game-of-life-mosaics

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **numpy** (>=1.20.0) - Numerical computing
- **scipy** (>=1.7.0) - Scientific computing (binary_fill_holes)
- **cellpylib** (>=2.0.0) - Cellular automaton simulation
- **Pillow** (>=9.0.0) - Image processing
- **gurobipy** (>=11.0.0) - Optimization solver (only for generating new patterns)

> **Note on Gurobi**: Gurobi is only required for *generating* new patterns. Using pre-computed patterns (levels 3-5) works without a Gurobi license. For pattern generation, obtain a free academic license or trial from [gurobi.com](https://www.gurobi.com/).

## Quick Start

```python
from gol_mosaics import MosaicGenerator

# Create generator
generator = MosaicGenerator(level=5, grid_size=100)

# Generate mosaic from image
mosaic = generator.generate_from_image('portrait.png')

# Save result
mosaic.save('output.png')
```

That's it! You now have a Game of Life mosaic.

## Usage Guide

### Basic Usage

```python
from gol_mosaics import MosaicGenerator, ColorScheme

# Simple usage with defaults (UGent colors)
generator = MosaicGenerator(level=5, grid_size=100)
mosaic = generator.generate_from_image('input.png')
mosaic.save('output.png')
```

### Custom Colors

```python
from gol_mosaics import MosaicGenerator, ColorScheme

# Use a preset color scheme
colors = ColorScheme.monochrome()

# Or create custom colors
colors = ColorScheme(
    gol_background='#FFFFFF',  # white
    gol_pixel='#000000',       # black
    eca_background='#FFD200',  # yellow
    eca_pixel='#1E64C8'        # blue
)

generator = MosaicGenerator(
    level=5,
    grid_size=100,
    color_scheme=colors
)

mosaic = generator.generate_from_image('input.png')
mosaic.save('output.png')
```

### Advanced Parameters

```python
from gol_mosaics import MosaicGenerator, ColorScheme

generator = MosaicGenerator(
    level=5,                    # Pattern complexity (3, 4, or 5)
    grid_size=100,              # Number of tiles (must be even)
    color_scheme=ColorScheme.ugent(),
    eca_rule=106,               # ECA rule (30, 45, 54, 106, 110, etc.)
    random_patterns=True,       # Random vs deterministic pattern selection
    invert=True                 # Invert density mapping
)

mosaic = generator.generate_from_image(
    'portrait.png',
    empty_tiles_cutoff=0.6,     # Threshold for empty tiles (0-1)
    alpha_cutoff=0.8,           # Transparency threshold (0-1)
    supersample=15              # ECA detail level
)

mosaic.save('output.png')
```

### Parameter Guide

- **level** (3-5): Pattern complexity. Higher = more detailed but larger files. Pre-computed levels: 3, 4, 5.
- **grid_size** (must be even): Number of tiles. Higher = more detail but slower. Typical: 30-100.
- **eca_rule**: Wolfram rule for background pattern.
  - Complex: 54, 147, 110, 124, 137, 193
  - Chaotic: 30, 45, 106, 150
- **empty_tiles_cutoff** (0-1): Brightness threshold above which tiles are empty. Lower = more empty tiles.
- **alpha_cutoff** (0-1): Transparency threshold. Transparent areas get filled with ECA pattern.
- **supersample**: ECA upsampling factor. Must divide mosaic width evenly. Higher = finer ECA detail.

### Working with Pattern Library

```python
from gol_mosaics import PatternLibrary
import numpy as np

# Load pre-computed patterns
library = PatternLibrary.load(level=5)

# Get a single pattern for a grayscale value
pattern = library.get_pattern_for_value(0.5, random=True)

# Map array of values to patterns
values = np.array([[0.2, 0.5], [0.7, 0.9]])
patterns = library.get_patterns_for_values(values, random=True, invert=True)

# Generate new patterns (requires Gurobi license)
library = PatternLibrary.generate(level=6, solution_limit=500)
np.save('solutions_level_6.npy', library.solutions)
```

### Export to Golly

```python
from gol_mosaics import GollyExporter
import numpy as np

# After generating your mosaic, convert to binary and export
mosaic_array = np.array(mosaic.convert('L')) > 128
GollyExporter.export_to_cells(mosaic_array, 'output/golly/my_mosaic.cells', add_glider=True)

# Now open output/golly/my_mosaic.cells in Golly to see your mosaic as a Game of Life pattern!
```

### Processing GIFs

```python
from gol_mosaics import MosaicGenerator

generator = MosaicGenerator(level=4, grid_size=50)
mosaic_gif = generator.generate_from_gif('animation.gif')
mosaic_gif.save('output.gif', save_all=True)
```

## How It Works

### Game of Life Still-Lifes

Conway's Game of Life is a cellular automaton where cells live or die based on their neighbors:
- A living cell with 2-3 neighbors survives
- A dead cell with exactly 3 neighbors becomes alive
- All other cells die

**Still-lifes** are stable patterns that never change. This project uses **8-fold symmetric still-lifes** computed via integer linear programming (Gurobi) to find all valid patterns at each complexity level.

For level 4, there are **352 unique symmetric patterns** ranging from sparse to dense.

### Pattern Mapping

1. **Image preprocessing**: Load image, make square with padding, convert to grayscale
2. **Rotation**: Rotate 45° to create diamond layout
3. **Pixelation**: Downsample to grid of tiles
4. **Diagonal extraction**: Extract two interlocking diagonal grids
5. **Density matching**: Map each tile's brightness to the closest GoL pattern by density
6. **Reconstruction**: Assemble patterns into final mosaic
7. **Aspect ratio adjustment**: Crop to original proportions

### Elementary Cellular Automata

ECAs evolve from a random initial row according to simple rules. Each new row depends only on the three cells above it. Different rules create different patterns:

- **Rule 30**: Random, chaotic
- **Rule 54**: Complex, intricate
- **Rule 106**: Chaotic with structure
- **Rule 110**: Turing-complete, complex

The ECA pattern is generated at low resolution and upsampled to overlay on the mosaic, filling transparent areas with texture.

### Diamond Layout

The 45° rotation creates a diamond-like tiling effect. Two diagonal grids interlock, with padding between tiles. This gives the characteristic diagonal aesthetic of these mosaics.

## API Reference

### MosaicGenerator

Main API for generating mosaics.

**Constructor:**
```python
MosaicGenerator(level=4, grid_size=30, color_scheme=None,
                eca_rule=106, random_patterns=True, invert=True)
```

**Methods:**
- `generate_from_image(image_path, empty_tiles_cutoff=1.0, alpha_cutoff=0.5, supersample=15)` - Generate from image file
- `generate_from_gif(gif_path, ...)` - Process animated GIF

### PatternLibrary

Manages Game of Life patterns.

**Class Methods:**
- `PatternLibrary.load(level)` - Load pre-computed patterns (levels 3-5)
- `PatternLibrary.generate(level, solution_limit)` - Generate new patterns

**Methods:**
- `get_pattern_for_value(value, random, invert)` - Get single pattern
- `get_patterns_for_values(values, random, invert, empty_tiles_cutoff)` - Map multiple values
- `get_patterns_for_mask(mask, alpha_cutoff)` - Map transparency mask

### ColorScheme

Immutable color configuration (dataclass).

**Class Methods:**
- `ColorScheme.ugent()` - UGent brand colors (default)
- `ColorScheme.monochrome(foreground, background)` - Two-color scheme
- `ColorScheme.inverted()` - Inverted UGent colors

**Attributes:**
- `gol_background` - Background color for GoL patterns
- `gol_pixel` - Foreground color for GoL patterns
- `eca_background` - Background color for ECA overlay
- `eca_pixel` - Foreground color for ECA overlay

### GollyExporter

Export to Golly simulator format.

**Static Methods:**
- `export_to_cells(mosaic, filename, add_glider)` - Export to .cells format
- `export_to_rle(mosaic, filename, name, comments)` - Export to RLE format

### ECABackground

Generate Elementary Cellular Automaton patterns.

**Constructor:**
```python
ECABackground(rule=106)
```

**Methods:**
- `generate(width, height, supersample)` - Generate ECA pattern
- `list_valid_supersamples(width)` - Get valid supersample values

**Class Methods:**
- `from_category(category)` - Create with 'complex' or 'chaotic' rule

### ImageProcessor

Image loading and preprocessing (all static methods).

**Static Methods:**
- `load_image(image_path, alpha_color, return_alpha)` - Load with alpha handling
- `square_image(img, return_aspect, fill_color)` - Pad to square
- `rotate_and_pixelate(img, grid_size, expand)` - Rotate 45° and pixelate
- `extract_diagonal_patterns(lowres)` - Extract two diagonal grids
- `preprocess_for_mosaic(image_path, grid_size)` - Complete preprocessing pipeline

### MosaicRenderer

Render arrays as colored images.

**Constructor:**
```python
MosaicRenderer(color_scheme)
```

**Methods:**
- `render_gol_mosaic(mosaic)` - Render GoL pattern with colors
- `render_eca_overlay(eca_mask)` - Render ECA overlay with transparency
- `composite(base, overlay)` - Alpha-composite images
- `render_full_mosaic(gol_mosaic, eca_mask)` - Complete rendering pipeline

## Examples

See the [notebooks/](notebooks/) directory for Jupyter notebooks with detailed examples:

- **quickstart.ipynb** - Simple 10-line example to get started
- **bw26-portraits.ipynb** - Portrait processing examples
- **explore_tiles.ipynb** - Pattern library exploration
- **animate_mosaics.ipynb** - Animated GIF processing

## Contributing

Contributions are welcome! Areas for improvement:

- Additional color schemes and presets
- More ECA rules and categories
- Pattern generation for levels 2 and 6+
- Performance optimizations
- Additional export formats
- Web-based interface
- More example notebooks

Please open an issue to discuss major changes before submitting a PR.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src/gol_mosaics --cov-report=html
```

## Project Structure

```
game-of-life-mosaics/
├── src/
│   └── gol_mosaics/          # Main package
│       ├── mosaic.py          # MosaicGenerator (main API)
│       ├── patterns.py        # PatternLibrary (GoL patterns)
│       ├── colors.py          # ColorScheme (color management)
│       ├── image_processing.py # ImageProcessor (preprocessing)
│       ├── eca.py             # ECABackground (background generation)
│       ├── renderer.py        # MosaicRenderer (color rendering)
│       └── export.py          # GollyExporter (format export)
├── data/                      # Pre-computed pattern solutions
│   ├── solutions_pattern_level_3.npy
│   ├── solutions_pattern_level_4.npy
│   └── solutions_pattern_level_5.npy
├── tests/                     # Unit and integration tests
├── notebooks/                 # Example Jupyter notebooks
├── input/                     # Example input images
│   └── images/                # Example images for testing
├── output/                    # Generated mosaics
│   ├── images/                # PNG output examples
│   └── golly/                 # .cells files for Golly simulator
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Acknowledgments

This project builds on several excellent libraries and concepts:

- **gurobipy**: Powerful optimization solver for finding GoL patterns
- **cellpylib**: Elementary Cellular Automaton simulation
- **NumPy/SciPy**: Numerical computing foundation
- **Pillow**: Image processing
- **Conway's Game of Life**: Classic cellular automaton by John Conway
- **Stephen Wolfram**: Elementary Cellular Automata classification

Special thanks to:
- The Ghent University (UGent) for color inspiration
- The Game of Life community for pattern catalogs and inspiration
- Everyone who contributed to the underlying mathematical and computational concepts

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Michiel Rollier

## Citation

If you use this project in academic work, please cite:

```bibtex
@software{rollier2026golmosaics,
  author = {Rollier, Michiel},
  title = {Game of Life Mosaics: Digital Art using GoL Still-Lifes},
  year = {2026},
  url = {https://github.com/mrollier/game-of-life-mosaics}
}
```

---

Made with ❤️ and Game of Life patterns

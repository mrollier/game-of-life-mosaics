#!/bin/bash
# Cleanup script for public release
# Review this script before running!

set -e  # Exit on error

echo "🧹 Game of Life Mosaics - Cleanup for Public Release"
echo "===================================================="
echo ""

# Function to safely remove directory
safe_remove() {
    if [ -d "$1" ]; then
        echo "  ✓ Removing: $1 ($(du -sh "$1" 2>/dev/null | cut -f1))"
        rm -rf "$1"
    else
        echo "  ⊘ Not found: $1 (already clean)"
    fi
}

# Function to safely remove file
safe_remove_file() {
    if [ -f "$1" ]; then
        echo "  ✓ Removing: $1 ($(du -sh "$1" 2>/dev/null | cut -f1))"
        rm -f "$1"
    else
        echo "  ⊘ Not found: $1 (already clean)"
    fi
}

echo "1️⃣  Removing Python cache directories..."
safe_remove "__pycache__"
safe_remove "src/__pycache__"
safe_remove "src/gol_mosaics/__pycache__"
echo ""

echo "2️⃣  Removing IDE/tool-specific directories..."
safe_remove ".vscode"
safe_remove ".claude"
echo ""

echo "3️⃣  Removing personal BW26 project files..."
safe_remove "output/bw26"
safe_remove "input/bw26"
echo ""

echo "4️⃣  Removing personal iteration outputs..."
safe_remove "output/gravensteen_iterations"
safe_remove "output/michielshelling_iterations"
echo ""

echo "5️⃣  Removing video directories..."
safe_remove "output/videos"
safe_remove "input/videos"
echo ""

echo "6️⃣  Removing old development notebooks..."
echo "   (Keeping quickstart.ipynb only)"
safe_remove_file "notebooks/bw26-portraits.ipynb"
safe_remove_file "notebooks/bw26-wall_art.ipynb"
safe_remove_file "notebooks/explore_tiles.ipynb"
safe_remove_file "notebooks/animate_mosaics.ipynb"
safe_remove_file "notebooks/apply_texture.ipynb"
safe_remove_file "notebooks/manipulate_images.ipynb"
safe_remove_file "notebooks/make_website_background.ipynb"
echo ""

echo "7️⃣  Cleaning up Jupyter notebook checkpoints..."
safe_remove "notebooks/.ipynb_checkpoints"
echo ""

echo "8️⃣  Removing loose output PNG files in root output/ directory..."
rm -f output/*.png 2>/dev/null || true
echo "  ✓ Removed loose PNG files"
echo ""

echo "✅ Cleanup complete!"
echo ""
echo "📊 Final directory sizes:"
echo "--------------------------------"
du -sh data/ input/ output/ notebooks/ src/ tests/ 2>/dev/null | sort -h
echo ""
echo "📁 Final project structure:"
echo "--------------------------------"
echo "✓ src/gol_mosaics/      - Clean OOP code"
echo "✓ data/                 - Pre-computed patterns"
echo "✓ tests/                - Test suite"
echo "✓ notebooks/            - Quickstart example"
echo "✓ input/images/         - Example input images"
echo "✓ output/images/        - Example PNG outputs"
echo "✓ output/golly/         - Example .cells files"
echo "✓ README.md             - Documentation"
echo "✓ requirements.txt      - Dependencies"
echo "✓ LICENSE               - MIT license"
echo ""
echo "💡 Next steps:"
echo "  1. Review remaining files in input/images/ (keep only examples)"
echo "  2. Review remaining files in output/images/ (keep only examples)"
echo "  3. Run: git status"
echo "  4. Run: git add -A"
echo "  5. Run: git commit -m 'Refactor to OOP and clean up for public release'"
echo ""

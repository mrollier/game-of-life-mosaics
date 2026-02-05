"""
Export Game of Life mosaics to various formats.

This module provides utilities for exporting mosaics to formats
compatible with external tools like the Golly simulator.
"""

import numpy as np
from typing import Optional


class GollyExporter:
    """
    Export mosaics to .cells format for Golly simulator.

    Golly is a popular Game of Life simulator that can read .cells files.
    This exporter converts numpy arrays (with 0=dead, 1=alive) to the
    .cells text format.

    Example:
        >>> mosaic = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> GollyExporter.export_to_cells(mosaic, 'glider.cells')
    """

    @staticmethod
    def export_to_cells(mosaic: np.ndarray,
                       filename: str = "output.cells",
                       add_glider: bool = False) -> None:
        """
        Export mosaic to .cells format for Golly simulator.

        The .cells format is a simple text format where:
        - 'O' represents alive cells
        - '.' represents dead cells
        - Each line represents a row

        Args:
            mosaic: Binary numpy array (0=dead, 1=alive)
            filename: Output filename (should end with .cells)
            add_glider: If True, adds a glider pattern to top-left corner
                       for animation testing

        Raises:
            ValueError: If mosaic is not 2D or contains values other than 0/1

        Example:
            >>> mosaic = np.zeros((10, 10))
            >>> mosaic[4:7, 4:7] = 1  # Add a block
            >>> GollyExporter.export_to_cells(mosaic, 'block.cells')

            >>> # With glider for animation
            >>> GollyExporter.export_to_cells(
            ...     mosaic,
            ...     'animated.cells',
            ...     add_glider=True
            ... )
        """
        # Validate input
        if mosaic.ndim != 2:
            raise ValueError(
                f"Mosaic must be 2D array, got shape {mosaic.shape}"
            )

        if not np.all(np.isin(mosaic, [0, 1])):
            raise ValueError(
                "Mosaic must be binary (only 0 and 1 values)"
            )

        # Make a copy if we're adding a glider
        if add_glider:
            mosaic = mosaic.copy()
            mosaic = GollyExporter._add_glider_pattern(mosaic)

        # Write to file
        with open(filename, 'w') as f:
            f.write('!Generated from Game of Life Mosaic\n')
            for row in mosaic:
                line = ''.join('O' if cell else '.' for cell in row)
                f.write(line + '\n')

    @staticmethod
    def _add_glider_pattern(mosaic: np.ndarray) -> np.ndarray:
        """
        Add a glider pattern to the top-left corner of the mosaic.

        The glider is a simple moving pattern in Conway's Game of Life
        that travels diagonally across the grid.

        Args:
            mosaic: Binary numpy array to add glider to

        Returns:
            Modified mosaic with glider in top-left corner
        """
        glider_pattern = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ])

        # Only add if mosaic is large enough
        if mosaic.shape[0] >= 3 and mosaic.shape[1] >= 3:
            mosaic[:glider_pattern.shape[0], :glider_pattern.shape[1]] = glider_pattern

        return mosaic

    @staticmethod
    def export_to_rle(mosaic: np.ndarray,
                     filename: str = "output.rle",
                     name: Optional[str] = None,
                     comments: Optional[str] = None) -> None:
        """
        Export mosaic to RLE (Run Length Encoded) format.

        RLE is a more compact format for Game of Life patterns, using
        run-length encoding to reduce file size.

        Note: This is a basic implementation. For complex patterns,
        consider using dedicated libraries.

        Args:
            mosaic: Binary numpy array (0=dead, 1=alive)
            filename: Output filename (should end with .rle)
            name: Optional pattern name for the header
            comments: Optional comments for the header

        Example:
            >>> mosaic = np.zeros((10, 10))
            >>> mosaic[4:7, 4:7] = 1
            >>> GollyExporter.export_to_rle(
            ...     mosaic,
            ...     'block.rle',
            ...     name='Block Pattern'
            ... )
        """
        # Validate input
        if mosaic.ndim != 2:
            raise ValueError(
                f"Mosaic must be 2D array, got shape {mosaic.shape}"
            )

        if not np.all(np.isin(mosaic, [0, 1])):
            raise ValueError(
                "Mosaic must be binary (only 0 and 1 values)"
            )

        height, width = mosaic.shape

        with open(filename, 'w') as f:
            # Write header
            if name:
                f.write(f'#N {name}\n')
            if comments:
                for line in comments.split('\n'):
                    f.write(f'#C {line}\n')

            # Write size line
            f.write(f'x = {width}, y = {height}, rule = B3/S23\n')

            # Encode pattern
            rle_lines = []
            for row in mosaic:
                rle_line = GollyExporter._encode_rle_row(row)
                rle_lines.append(rle_line)

            # Join lines with $ separator, end with !
            rle_pattern = '$'.join(rle_lines) + '!'

            # Write pattern in chunks of 70 characters
            for i in range(0, len(rle_pattern), 70):
                f.write(rle_pattern[i:i+70] + '\n')

    @staticmethod
    def _encode_rle_row(row: np.ndarray) -> str:
        """
        Encode a single row as RLE format.

        Args:
            row: 1D binary array

        Returns:
            RLE-encoded string (b=dead, o=alive)
        """
        if len(row) == 0:
            return ''

        encoded = []
        current_val = row[0]
        count = 1

        for val in row[1:]:
            if val == current_val:
                count += 1
            else:
                # Output the run
                symbol = 'o' if current_val == 1 else 'b'
                if count > 1:
                    encoded.append(f'{count}{symbol}')
                else:
                    encoded.append(symbol)

                current_val = val
                count = 1

        # Output final run
        symbol = 'o' if current_val == 1 else 'b'
        if count > 1:
            encoded.append(f'{count}{symbol}')
        else:
            encoded.append(symbol)

        return ''.join(encoded)

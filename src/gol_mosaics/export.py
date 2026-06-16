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
                       add_glider: Optional[str] = None) -> None:
        """
        Export mosaic to .cells format for Golly simulator.

        The .cells format is a simple text format where:
        - 'O' represents alive cells
        - '.' represents dead cells
        - Each line represents a row

        Args:
            mosaic: Binary numpy array (0=dead, 1=alive)
            filename: Output filename (should end with .cells)
            add_glider: If None (default), no glider is added. Otherwise a
                       corner name ('top left', 'top right', 'bottom left',
                       'bottom right') that places a glider in that corner
                       for animation testing

        Raises:
            ValueError: If mosaic is not 2D, contains values other than 0/1,
                       or add_glider is not a recognised corner name

        Example:
            >>> mosaic = np.zeros((10, 10))
            >>> mosaic[4:7, 4:7] = 1  # Add a block
            >>> GollyExporter.export_to_cells(mosaic, 'block.cells')

            >>> # With glider for animation
            >>> GollyExporter.export_to_cells(
            ...     mosaic,
            ...     'animated.cells',
            ...     add_glider='bottom right'
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
            mosaic = GollyExporter._add_glider_pattern(mosaic, add_glider)

        # Write to file
        with open(filename, 'w') as f:
            f.write('!Generated from Game of Life Mosaic\n')
            for row in mosaic:
                line = ''.join('O' if cell else '.' for cell in row)
                f.write(line + '\n')

    @staticmethod
    def _add_glider_pattern(mosaic: np.ndarray, add_glider: str = 'bottom right') -> np.ndarray:
        """
        Add a glider pattern to the chosen corner of the mosaic.

        The glider is a simple moving pattern in Conway's Game of Life
        that travels diagonally across the grid. It is oriented to travel
        inward from the chosen corner.

        Args:
            mosaic: Binary numpy array to add glider to
            add_glider: Corner to place the glider in ('top left', 'top right',
                       'bottom left', 'bottom right')

        Returns:
            Modified mosaic with a glider in the chosen corner
        """

        if add_glider not in ['top left', 'bottom right', 'top right', 'bottom left']:
            raise ValueError(
                f"Invalid add_glider option: {add_glider}. "
                "Choose from 'top left', 'bottom right', 'top right', 'bottom left'."
            )

        glider_pattern = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ])

        # Only add if mosaic is large enough
        if mosaic.shape[0] >= 3 and mosaic.shape[1] >= 3:
            # add to top-left corner
            if add_glider == 'top left':
                mosaic[:glider_pattern.shape[0], :glider_pattern.shape[1]] = glider_pattern
            # add to bottom-right corner
            elif add_glider == 'bottom right':
                # rotate glider_pattern 180 degrees for correct orientation
                glider_pattern = np.rot90(glider_pattern, 2)
                mosaic[-glider_pattern.shape[0]:, -glider_pattern.shape[1]:] = glider_pattern
            # add to top-right corner
            elif add_glider == 'top right':
                # rotate glider_pattern 90 degrees for correct orientation
                glider_pattern = np.rot90(glider_pattern, 3)
                mosaic[:glider_pattern.shape[0], -glider_pattern.shape[1]:] = glider_pattern
            # add to bottom-left corner
            elif add_glider == 'bottom left':
                # rotate glider_pattern 270 degrees for correct orientation
                glider_pattern = np.rot90(glider_pattern, 3)
                mosaic[-glider_pattern.shape[0]:, :glider_pattern.shape[1]] = glider_pattern

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

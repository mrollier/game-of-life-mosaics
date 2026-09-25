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
        >>> cells = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> GollyExporter.export_to_cells(cells, 'glider.cells')
    """

    @staticmethod
    def export_to_cells(cells: np.ndarray,
                       path: str = "output.cells",
                       add_glider: Optional[str] = None) -> None:
        """
        Export cells to .cells format for Golly simulator.

        The .cells format is a simple text format where:
        - 'O' represents alive cells
        - '.' represents dead cells
        - Each line represents a row

        Args:
            cells: Binary numpy array (0=dead, 1=alive)
            path: Output path (should end with .cells)
            add_glider: If None (default), no glider is added. Otherwise a
                       corner name ('top left', 'top right', 'bottom left',
                       'bottom right') that places a glider in that corner
                       for animation testing

        Raises:
            ValueError: If cells is not 2D, contains values other than 0/1,
                       or add_glider is not a recognised corner name

        Example:
            >>> cells = np.zeros((10, 10))
            >>> cells[4:7, 4:7] = 1  # Add a block
            >>> GollyExporter.export_to_cells(cells, 'block.cells')

            >>> # With glider for animation
            >>> GollyExporter.export_to_cells(
            ...     cells,
            ...     'animated.cells',
            ...     add_glider='bottom right'
            ... )
        """
        # Validate input
        if cells.ndim != 2:
            raise ValueError(
                f"Mosaic must be 2D array, got shape {cells.shape}"
            )

        if not np.all(np.isin(cells, [0, 1])):
            raise ValueError(
                "Mosaic must be binary (only 0 and 1 values)"
            )

        # Make a copy if we're adding a glider
        if add_glider:
            cells = cells.copy()
            cells = GollyExporter._add_glider_pattern(cells, add_glider)

        # One byte per cell, then one line per row. newline='\n' keeps the
        # file identical on every platform.
        chars = np.where(cells == 1, ord('O'), ord('.')).astype(np.uint8)
        with open(path, 'w', newline='\n') as f:
            f.write('!Generated from Game of Life Mosaic\n')
            for row in chars:
                f.write(row.tobytes().decode('ascii') + '\n')

    @staticmethod
    def _add_glider_pattern(cells: np.ndarray, add_glider: str = 'bottom right') -> np.ndarray:
        """
        Add a glider pattern to the chosen corner of the cells.

        The glider is a simple moving pattern in Conway's Game of Life
        that travels diagonally across the grid. It is oriented to travel
        inward from the chosen corner.

        Args:
            cells: Binary numpy array to add glider to
            add_glider: Corner to place the glider in ('top left', 'top right',
                       'bottom left', 'bottom right')

        Returns:
            Modified cells with a glider in the chosen corner
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

        # Only add if cells is large enough
        if cells.shape[0] >= 3 and cells.shape[1] >= 3:
            # add to top-left corner
            if add_glider == 'top left':
                cells[:glider_pattern.shape[0], :glider_pattern.shape[1]] = glider_pattern
            # add to bottom-right corner
            elif add_glider == 'bottom right':
                # rotate glider_pattern 180 degrees for correct orientation
                glider_pattern = np.rot90(glider_pattern, 2)
                cells[-glider_pattern.shape[0]:, -glider_pattern.shape[1]:] = glider_pattern
            # add to top-right corner
            elif add_glider == 'top right':
                # rotate glider_pattern 90 degrees for correct orientation
                glider_pattern = np.rot90(glider_pattern, 3)
                cells[:glider_pattern.shape[0], -glider_pattern.shape[1]:] = glider_pattern
            # add to bottom-left corner
            elif add_glider == 'bottom left':
                # rotate glider_pattern 90 degrees anticlockwise so it
                # travels up and to the right, into the grid
                glider_pattern = np.rot90(glider_pattern, 1)
                cells[-glider_pattern.shape[0]:, :glider_pattern.shape[1]] = glider_pattern

        return cells

    @staticmethod
    def export_to_rle(cells: np.ndarray,
                     path: str = "output.rle",
                     name: Optional[str] = None,
                     comments: Optional[str] = None) -> None:
        """
        Export cells to RLE (Run Length Encoded) format.

        RLE is a more compact format for Game of Life patterns, using
        run-length encoding to reduce file size.

        Note: This is a basic implementation. For complex patterns,
        consider using dedicated libraries.

        Args:
            cells: Binary numpy array (0=dead, 1=alive)
            path: Output path (should end with .rle)
            name: Optional pattern name for the header
            comments: Optional comments for the header

        Example:
            >>> cells = np.zeros((10, 10))
            >>> cells[4:7, 4:7] = 1
            >>> GollyExporter.export_to_rle(
            ...     cells,
            ...     'block.rle',
            ...     name='Block Pattern'
            ... )
        """
        # Validate input
        if cells.ndim != 2:
            raise ValueError(
                f"Mosaic must be 2D array, got shape {cells.shape}"
            )

        if not np.all(np.isin(cells, [0, 1])):
            raise ValueError(
                "Mosaic must be binary (only 0 and 1 values)"
            )

        height, width = cells.shape

        with open(path, 'w', newline='\n') as f:
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
            for row in cells:
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
        row = np.asarray(row)
        if len(row) == 0:
            return ''

        # Runs start at 0 and wherever the value changes
        starts = np.concatenate(([0], np.flatnonzero(row[1:] != row[:-1]) + 1))
        lengths = np.diff(np.append(starts, len(row)))
        return ''.join(
            (str(count) if count > 1 else '') + ('o' if value == 1 else 'b')
            for count, value in zip(lengths.tolist(), row[starts].tolist()))

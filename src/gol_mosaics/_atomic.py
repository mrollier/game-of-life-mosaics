"""
Crash-safe .npy writes for long-running jobs.

The search campaigns and the free-form block solver checkpoint one file per
finished unit of work and resume by listing what exists, so a file must
appear complete or not at all. Writing to a temporary name in the same
folder and renaming it into place gives that on every platform. On Windows
the rename fails while another process (a reader, a virus scanner, a sync
client such as OneDrive) holds the target open, so it is retried briefly.
"""

import os
import time
from pathlib import Path

import numpy as np


def atomic_save(path, array: np.ndarray, retries: int = 6) -> Path:
    """
    np.save `array` to `path` so that `path` is never seen half-written.

    The data goes to ``<path>.<pid>.tmp`` first (same folder, so the rename
    stays on one filesystem; the pid keeps concurrent writers apart) and is
    then moved over `path` with os.replace, which overwrites an existing
    file on every platform (unlike Path.rename on Windows).

    Args:
        path: Destination file, conventionally ending in .npy
        array: Array to save
        retries: Attempts at the rename before giving up on a locked target

    Returns:
        The destination path
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * 2 ** attempt)
    return path

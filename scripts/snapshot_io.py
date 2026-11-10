"""Read native float32 coordinates and pressure-only SEM snapshots."""

from pathlib import Path

import numpy as np


def load_mesh(path):
    """Read one (x, y, z) triplet per node, in snapshot pressure order."""
    path = Path(path)
    size = path.stat().st_size
    if size == 0 or size % 12 != 0:
        raise ValueError(f"Expected nonempty mesh.bin, 12 bytes/node: {path}")
    coordinates = np.fromfile(path, dtype="=f4")
    if coordinates.size != size // 4:
        raise ValueError(f"Incomplete mesh: {path}")
    coordinates = coordinates.reshape(-1, 3)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"Nonfinite mesh coordinates: {path}")
    return coordinates


def load_snapshot(path, node_count):
    """Read pressures and require exactly one value per mesh node."""
    path = Path(path)
    if node_count <= 0 or path.stat().st_size != 4 * node_count:
        raise ValueError(f"Expected one float32 pressure per mesh node: {path}")
    pressures = np.fromfile(path, dtype="=f4")
    if pressures.size != node_count:
        raise ValueError(f"Incomplete snapshot: {path}")
    if not np.all(np.isfinite(pressures)):
        raise ValueError(f"Nonfinite snapshot pressures: {path}")
    return pressures

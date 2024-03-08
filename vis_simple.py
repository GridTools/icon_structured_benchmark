import numpy as np

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    IndexTransformation,
)

from plotting import plot_torus
from utilities import import_grid

# Define vertices
vertices = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
        [0, 2, 0],
        [1, 2, 0],
        [2, 2, 0],
    ]
)

grid = import_grid(
    "tests/data/simple_grid/simple_grid_gridfile.nc", IndexTransformation()
)

plot_torus(
    vertices,
    grid.get_offset_provider("E2V").table,
    grid.get_offset_provider("C2V").table,
)

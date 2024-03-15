import numpy as np
from os import path

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
    path.dirname(__file__) + "/tests/data/simple_grid/simple_grid_gridfile.nc",
    IndexTransformation(),
)

print(grid.get_offset_provider("E2V").table)

plot_torus(
    vertices,
    grid.get_offset_provider("E2V").table,
)

print(grid.get_offset_provider("E2C2V").table[:])

for i, vec in enumerate(grid.get_offset_provider("E2C2V").table):
    print("{}: {}".format(i, vec))

for i, vec in enumerate(grid.get_offset_provider("E2V").table):
    print("{}: {}".format(i, vec))

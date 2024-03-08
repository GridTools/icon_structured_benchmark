from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    ToGt4PyTransformation,
)

from plotting import plot_torus  # type: ignore [import-not-found]
from utilities import import_grid, get_vertices_coordinates  # type: ignore [import-not-found]

grid_file = "tests/data/torus_grid/torus_100000_100000_24576.nc"

grid = import_grid(grid_file, ToGt4PyTransformation())

vertice_coords = get_vertices_coordinates(grid_file)

plot_torus(
    vertice_coords,
    grid.get_offset_provider("E2V").table,
    grid.get_offset_provider("C2V").table,
)

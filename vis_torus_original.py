import argparse
from os import path
import numpy as np

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    ToGt4PyTransformation,
)

from plotting import plot_torus  # type: ignore [import-not-found]
from utilities import (
    import_grid,
    get_vertices_coordinates,
    print_torus_file_information,
    print_lat_lon_dimensions,
    get_torus_cartesian_dimensions,
)  # type: ignore [import-not-found]

# grid_file = path.dirname(__file__) + "/tests/data/torus_grid/torus_100000_100000_24576.nc"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("grid")
    return parser.parse_args()


args = parse_arguments()
grid_file = args.grid

grid = import_grid(grid_file, ToGt4PyTransformation())

vertice_coords = get_vertices_coordinates(grid_file)

print_lat_lon_dimensions(grid_file)

print_torus_file_information(grid_file)

# print(grid.get_offset_provider("E2C2V").table)
for i in range(len(grid.get_offset_provider("E2C2V").table)):
    print("{} {}".format(i, grid.get_offset_provider("E2C2V").table[i]))

np.set_printoptions(threshold=np.inf)  # type: ignore [arg-type]

for i in range(len(grid.get_offset_provider("E2V").table)):
    print("E2V[{}]: {}".format(i, grid.get_offset_provider("E2V").table[i]))
print(get_torus_cartesian_dimensions(grid_file))
plot_torus(
    vertice_coords,
    grid.get_offset_provider("E2V").table,
)

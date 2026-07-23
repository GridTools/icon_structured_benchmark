import argparse
from icon4py.model.common.grid.gridfile import (  # type: ignore [import-not-found]
    ToZeroBasedIndexTransformation,
)

from plotting import plot_torus  # type: ignore [import-not-found]
from utilities import (
    import_grid,
    get_vertices_coordinates,
    print_torus_file_information,
    print_lat_lon_dimensions,
)  # type: ignore [import-not-found]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid")
    return parser.parse_args()


args = parse_arguments()
grid_file = args.grid

grid = import_grid(grid_file, ToZeroBasedIndexTransformation())

vertice_coords = get_vertices_coordinates(grid_file)

print_lat_lon_dimensions(grid_file)

print_torus_file_information(grid_file)

plot_torus(
    vertice_coords,
    grid.get_connectivity("E2V").ndarray,
)

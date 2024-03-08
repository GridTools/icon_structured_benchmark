import netCDF4  # type: ignore [import-not-found]
import numpy as np

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToGt4PyTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize  # type: ignore [import-not-found]


def import_grid(filename, indexing):
    def init_grid_manager(fname, num_levels=10, transformation=ToGt4PyTransformation()):
        grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
        grid_manager()
        return grid_manager

    grid_manager = init_grid_manager(filename, 10, indexing)
    grid_manager()
    grid = grid_manager.get_grid()
    return grid


def get_vertices_coordinates(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    x_coords = nc["cartesian_x_vertices"][:]
    y_coords = nc["cartesian_y_vertices"][:]
    z_coords = nc["cartesian_z_vertices"][:]
    return np.dstack((x_coords, y_coords, z_coords))[0]

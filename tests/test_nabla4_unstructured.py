from icon4py.model.common.grid.grid_manager import (
    GridManager,
    IndexTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize

from icon4py.model.common.dimension import E2C2VDim

import icon_benchmark

import pytest

from os import path


def init_grid_manager(fname, num_levels=65, transformation=IndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
    grid_manager()
    return grid_manager


@pytest.fixture
def simple_grid():
    grid_manager = init_grid_manager(
        path.dirname(__file__) + "/data/simple_grid_gridfile.nc"
    )
    grid_manager()
    simple_grid = grid_manager.get_grid()
    yield simple_grid


def test_gridtools(simple_grid):
    repetitions = 3
    runtimes = icon_benchmark.nabla4_benchmark(
        simple_grid.get_offset_provider("E2C2V").table,
        simple_grid.num_cells,
        simple_grid.num_vertices,
        simple_grid.num_edges,
        simple_grid.num_levels,
        simple_grid.size[E2C2VDim],
        repetitions,
    )

    assert len(runtimes) == repetitions

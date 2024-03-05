import numpy as np
from os import path
import pytest

from icon4py.model.common.grid.grid_manager import (
    GridManager,
    IndexTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize

from icon4py.model.common.dimension import E2C2VDim

import icon_benchmark

SIMPLE_GRID_FILE = path.dirname(__file__) + "/data/simple_grid_gridfile.nc"


def init_grid_manager(fname, num_levels=65, transformation=IndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
    grid_manager()
    return grid_manager


@pytest.fixture
def simple_grid():
    grid_manager = init_grid_manager(SIMPLE_GRID_FILE)
    grid_manager()
    simple_grid = grid_manager.get_grid()
    yield simple_grid


def get_simple_grid():
    grid_manager = init_grid_manager(SIMPLE_GRID_FILE)
    grid_manager()
    simple_grid = grid_manager.get_grid()
    return simple_grid


def test_gridtools(simple_grid):
    repetitions = 3
    dry_runs = 0
    runtimes = icon_benchmark.nabla4_benchmark_naive(
        simple_grid.get_offset_provider("E2C2V").table,
        simple_grid.get_offset_provider("E2ECV").table,
        simple_grid.num_cells,
        simple_grid.num_vertices,
        simple_grid.num_edges,
        simple_grid.num_levels,
        simple_grid.size[E2C2VDim],
        repetitions,
        dry_runs,
    )

    assert simple_grid.num_cells == 18
    assert simple_grid.num_vertices == 9
    assert simple_grid.num_edges == 27
    assert simple_grid.num_levels == 65
    assert len(runtimes) == repetitions


if __name__ == "__main__":
    simple_grid_inst = get_simple_grid()
    print("E2C2V: {}".format(simple_grid_inst.get_offset_provider("E2C2V").table))
    print("E2ECV: {}".format(simple_grid_inst.get_offset_provider("E2ECV").table))
    print("CellDim: {}".format(simple_grid_inst.num_cells))
    print("VertexDim: {}".format(simple_grid_inst.num_vertices))
    print("EdgeDim: {}".format(simple_grid_inst.num_edges))
    print("KDim: {}".format(simple_grid_inst.num_levels))
    print("E2C2VDim: {}".format(simple_grid_inst.size[E2C2VDim]))

    repetitions = 101
    dry_runs = 1
    runtimes = icon_benchmark.nabla4_benchmark_cpu_ifirst(
        simple_grid_inst.get_offset_provider("E2C2V").table,
        simple_grid_inst.get_offset_provider("E2ECV").table,
        simple_grid_inst.num_cells,
        simple_grid_inst.num_vertices,
        simple_grid_inst.num_edges,
        simple_grid_inst.num_levels,
        simple_grid_inst.size[E2C2VDim],
        repetitions,
        dry_runs,
    )

    print("cpu_ifirst mean runtime: {}".format(np.mean(runtimes)))

    runtimes = icon_benchmark.nabla4_benchmark_cpu_kfirst(
        simple_grid_inst.get_offset_provider("E2C2V").table,
        simple_grid_inst.get_offset_provider("E2ECV").table,
        simple_grid_inst.num_cells,
        simple_grid_inst.num_vertices,
        simple_grid_inst.num_edges,
        simple_grid_inst.num_levels,
        simple_grid_inst.size[E2C2VDim],
        repetitions,
        dry_runs,
    )

    print("cpu_kfirst mean runtime: {}".format(np.mean(runtimes)))

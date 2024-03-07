import numpy as np
from os import path
import pytest

import serialbox as ser  # type: ignore [import-not-found]

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize  # type: ignore [import-not-found]

from icon4py.model.common.dimension import E2C2VDim  # type: ignore [import-not-found]

import icon_benchmark  # type: ignore [import-not-found]

SIMPLE_GRID_FILE = path.dirname(__file__) + "/data/simple_grid/simple_grid_gridfile.nc"


def init_grid_manager(fname, num_levels=10, transformation=IndexTransformation()):
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


def test_grid(simple_grid):
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


def test_validate_nabla4_unstructured_naive(simple_grid):
    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/simple_grid",
        "nabla4_fields",
    )
    savepoint = serializer.savepoint["ValidationTest"].time[1]
    u_vert = serializer.read("u_vert", savepoint)
    v_vert = serializer.read("v_vert", savepoint)
    primal_normal_vert_v1 = serializer.read("primal_normal_vert_v1_new", savepoint)
    primal_normal_vert_v2 = serializer.read("primal_normal_vert_v2_new", savepoint)
    z_nabla2_e = serializer.read("z_nabla2_e", savepoint)
    inv_vert_vert_length = serializer.read("inv_vert_vert_length", savepoint)
    inv_primal_edge_length = serializer.read("inv_primal_edge_length", savepoint)

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/simple_grid",
        "nabla4_output",
    )
    out_savepoint = serializer.savepoint["OutputValidationTest"].time[1]
    z_nabla4_e2 = serializer.read("z_nabla4_e2", out_savepoint)

    z_nabla4_e2_comp = icon_benchmark.nabla4_validate_naive(
        simple_grid.get_offset_provider("E2C2V").table,
        simple_grid.get_offset_provider("E2ECV").table,
        simple_grid.num_cells,
        simple_grid.num_vertices,
        simple_grid.num_edges,
        simple_grid.num_levels,
        simple_grid.size[E2C2VDim],
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )

    assert np.allclose(
        z_nabla4_e2_comp, z_nabla4_e2, equal_nan=True, atol=1e-8, rtol=1e-4
    )


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

    test_validate_nabla4_unstructured_naive(simple_grid_inst)

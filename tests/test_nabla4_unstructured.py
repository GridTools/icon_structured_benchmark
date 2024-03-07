from dataclasses import dataclass
import numpy as np
from os import path
import pytest

import serialbox as ser  # type: ignore [import-not-found]

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToGt4PyTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize  # type: ignore [import-not-found]

from icon4py.model.common.dimension import E2C2VDim  # type: ignore [import-not-found]

import icon_benchmark  # type: ignore [import-not-found]

SIMPLE_GRID_FILE = path.dirname(__file__) + "/data/simple_grid/simple_grid_gridfile.nc"
SMALL_TORUS_GRID_FILE = (
    path.dirname(__file__) + "/data/torus_grid/torus_100000_100000_24576.nc"
)


def init_grid_manager(fname, num_levels=10, transformation=IndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
    grid_manager()
    return grid_manager


@pytest.fixture
def simple_grid():
    grid_manager = init_grid_manager(SIMPLE_GRID_FILE, 10, IndexTransformation())
    grid_manager()
    simple_grid = grid_manager.get_grid()
    yield simple_grid


@pytest.fixture
def small_torus_grid():
    grid_manager = init_grid_manager(SMALL_TORUS_GRID_FILE, 65, ToGt4PyTransformation())
    grid_manager()
    small_torus_grid = grid_manager.get_grid()
    yield small_torus_grid


def test_simple_grid(simple_grid):
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
    assert simple_grid.num_levels == 10
    assert len(runtimes) == repetitions


@dataclass
class Nabla4KernelValidationData:
    e2c2v: np.ndarray
    e2ecv: np.ndarray
    num_cells: int
    num_vertices: int
    num_edges: int
    num_levels: int
    E2C2VDim: int
    u_vert: np.ndarray
    v_vert: np.ndarray
    primal_normal_vert_v1: np.array
    primal_normal_vert_v2: np.array
    z_nabla2_e: np.ndarray
    inv_vert_vert_length: np.array
    inv_primal_edge_length: np.array
    ref_z_nabla4_e2: np.ndarray


@pytest.fixture
def simple_grid_kernel_input(simple_grid):
    nabla4_kernel_validation_data = Nabla4KernelValidationData

    nabla4_kernel_validation_data.e2c2v = simple_grid.get_offset_provider("E2C2V").table
    nabla4_kernel_validation_data.e2ecv = simple_grid.get_offset_provider("E2ECV").table
    nabla4_kernel_validation_data.num_cells = simple_grid.num_cells
    nabla4_kernel_validation_data.num_vertices = simple_grid.num_vertices
    nabla4_kernel_validation_data.num_edges = simple_grid.num_edges
    nabla4_kernel_validation_data.num_levels = simple_grid.num_levels
    nabla4_kernel_validation_data.E2C2VDim = simple_grid.size[E2C2VDim]

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/simple_grid",
        "nabla4_fields",
    )
    savepoint = serializer.savepoint["ValidationTest"].time[1]
    nabla4_kernel_validation_data.u_vert = serializer.read("u_vert", savepoint)
    nabla4_kernel_validation_data.v_vert = serializer.read("v_vert", savepoint)
    nabla4_kernel_validation_data.primal_normal_vert_v1 = serializer.read(
        "primal_normal_vert_v1_new", savepoint
    )
    nabla4_kernel_validation_data.primal_normal_vert_v2 = serializer.read(
        "primal_normal_vert_v2_new", savepoint
    )
    nabla4_kernel_validation_data.z_nabla2_e = serializer.read("z_nabla2_e", savepoint)
    nabla4_kernel_validation_data.inv_vert_vert_length = serializer.read(
        "inv_vert_vert_length", savepoint
    )
    nabla4_kernel_validation_data.inv_primal_edge_length = serializer.read(
        "inv_primal_edge_length", savepoint
    )

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/simple_grid",
        "nabla4_output",
    )
    out_savepoint = serializer.savepoint["OutputValidationTest"].time[1]
    nabla4_kernel_validation_data.ref_z_nabla4_e2 = serializer.read(
        "z_nabla4_e2", out_savepoint
    )

    yield nabla4_kernel_validation_data


@pytest.fixture
def small_torus_grid_kernel_input(small_torus_grid):
    nabla4_kernel_validation_data = Nabla4KernelValidationData

    nabla4_kernel_validation_data.e2c2v = small_torus_grid.get_offset_provider(
        "E2C2V"
    ).table
    nabla4_kernel_validation_data.e2ecv = small_torus_grid.get_offset_provider(
        "E2ECV"
    ).table
    nabla4_kernel_validation_data.num_cells = small_torus_grid.num_cells
    nabla4_kernel_validation_data.num_vertices = small_torus_grid.num_vertices
    nabla4_kernel_validation_data.num_edges = small_torus_grid.num_edges
    nabla4_kernel_validation_data.num_levels = small_torus_grid.num_levels
    nabla4_kernel_validation_data.E2C2VDim = small_torus_grid.size[E2C2VDim]

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/torus_grid",
        "nabla4_fields",
    )
    savepoint = serializer.savepoint["ValidationTest"].time[1]
    nabla4_kernel_validation_data.u_vert = serializer.read("u_vert", savepoint)
    nabla4_kernel_validation_data.v_vert = serializer.read("v_vert", savepoint)
    nabla4_kernel_validation_data.primal_normal_vert_v1 = serializer.read(
        "primal_normal_vert_v1_new", savepoint
    )
    nabla4_kernel_validation_data.primal_normal_vert_v2 = serializer.read(
        "primal_normal_vert_v2_new", savepoint
    )
    nabla4_kernel_validation_data.z_nabla2_e = serializer.read("z_nabla2_e", savepoint)
    nabla4_kernel_validation_data.inv_vert_vert_length = serializer.read(
        "inv_vert_vert_length", savepoint
    )
    nabla4_kernel_validation_data.inv_primal_edge_length = serializer.read(
        "inv_primal_edge_length", savepoint
    )

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/torus_grid",
        "nabla4_output",
    )
    out_savepoint = serializer.savepoint["OutputValidationTest"].time[1]
    nabla4_kernel_validation_data.ref_z_nabla4_e2 = serializer.read(
        "z_nabla4_e2", out_savepoint
    )

    yield nabla4_kernel_validation_data


@pytest.mark.parametrize(
    "grid", ("simple_grid_kernel_input", "small_torus_grid_kernel_input")
)
def test_validate_nabla4_unstructured_naive(request, grid):
    grid = request.getfixturevalue(grid)
    z_nabla4_e2_comp = icon_benchmark.nabla4_validate_naive(
        grid.e2c2v,
        grid.e2ecv,
        grid.num_cells,
        grid.num_vertices,
        grid.num_edges,
        grid.num_levels,
        grid.E2C2VDim,
        grid.u_vert,
        grid.v_vert,
        grid.primal_normal_vert_v1,
        grid.primal_normal_vert_v2,
        grid.z_nabla2_e,
        grid.inv_vert_vert_length,
        grid.inv_primal_edge_length,
    )

    assert np.allclose(
        z_nabla4_e2_comp,
        grid.ref_z_nabla4_e2,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "grid", ("simple_grid_kernel_input", "small_torus_grid_kernel_input")
)
def test_validate_nabla4_unstructured_cpu_ifirst(request, grid):
    grid = request.getfixturevalue(grid)
    z_nabla4_e2_comp = icon_benchmark.nabla4_validate_cpu_ifirst(
        grid.e2c2v,
        grid.e2ecv,
        grid.num_cells,
        grid.num_vertices,
        grid.num_edges,
        grid.num_levels,
        grid.E2C2VDim,
        grid.u_vert,
        grid.v_vert,
        grid.primal_normal_vert_v1,
        grid.primal_normal_vert_v2,
        grid.z_nabla2_e,
        grid.inv_vert_vert_length,
        grid.inv_primal_edge_length,
    )

    assert np.allclose(
        z_nabla4_e2_comp,
        grid.ref_z_nabla4_e2,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "grid", ("simple_grid_kernel_input", "small_torus_grid_kernel_input")
)
def test_validate_nabla4_unstructured_cpu_kfirst(request, grid):
    grid = request.getfixturevalue(grid)
    z_nabla4_e2_comp = icon_benchmark.nabla4_validate_cpu_kfirst(
        grid.e2c2v,
        grid.e2ecv,
        grid.num_cells,
        grid.num_vertices,
        grid.num_edges,
        grid.num_levels,
        grid.E2C2VDim,
        grid.u_vert,
        grid.v_vert,
        grid.primal_normal_vert_v1,
        grid.primal_normal_vert_v2,
        grid.z_nabla2_e,
        grid.inv_vert_vert_length,
        grid.inv_primal_edge_length,
    )

    assert np.allclose(
        z_nabla4_e2_comp,
        grid.ref_z_nabla4_e2,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )

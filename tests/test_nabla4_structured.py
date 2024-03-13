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


def test_simple_grid(simple_grid):
    repetitions = 3
    dry_runs = 0
    runtimes = icon_benchmark.nabla4_benchmark_structured_naive(
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


def apply_permutation(perm, vec):
    ret = []
    for i in range(len(vec)):
        ret.append(vec[perm[i]])
    return ret


@pytest.fixture
def simple_grid_kernel_input(simple_grid):
    nabla4_kernel_validation_data = Nabla4KernelValidationData

    nabla4_kernel_validation_data.e2c2v = simple_grid.get_offset_provider("E2C2V").table
    # Needed to alter the E2ECV elements in some cases because the order of the vertices in the SimpleGrid differ
    # with the order of vertices being calculated for each edge
    # With the following E2ECV matrix we get same data as the tests in icon4py
    nabla4_kernel_validation_data.e2ecv = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 11, 10],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],
        [32, 33, 34, 35],
        [36, 37, 39, 38],
        [40, 41, 42, 43],
        [44, 45, 47, 46],
        [48, 49, 51, 50],
        [52, 53, 54, 55],
        [56, 57, 58, 59],
        [60, 61, 63, 62],
        [64, 65, 66, 67],
        [68, 69, 70, 71],
        [72, 73, 75, 74],
        [76, 77, 78, 79],
        [80, 81, 83, 82],
        [84, 85, 87, 86],
        [88, 89, 90, 91],
        [92, 93, 94, 95],
        [96, 97, 99, 98],
        [100, 101, 102, 103],
        [104, 105, 106, 107],
    ]
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

    primal_normal_permutation = sum(nabla4_kernel_validation_data.e2ecv, [])
    nabla4_kernel_validation_data.primal_normal_vert_v1 = serializer.read(
        "primal_normal_vert_v1_new", savepoint
    )
    nabla4_kernel_validation_data.primal_normal_vert_v1 = apply_permutation(
        primal_normal_permutation, nabla4_kernel_validation_data.primal_normal_vert_v1
    )
    nabla4_kernel_validation_data.primal_normal_vert_v2 = serializer.read(
        "primal_normal_vert_v2_new", savepoint
    )
    nabla4_kernel_validation_data.primal_normal_vert_v2 = apply_permutation(
        primal_normal_permutation, nabla4_kernel_validation_data.primal_normal_vert_v2
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


def test_validate_nabla4_structured_naive(simple_grid_kernel_input):
    z_nabla4_e2_comp = icon_benchmark.nabla4_validate_structured_naive(
        simple_grid_kernel_input.num_cells,
        simple_grid_kernel_input.num_vertices,
        simple_grid_kernel_input.num_edges,
        simple_grid_kernel_input.num_levels,
        simple_grid_kernel_input.E2C2VDim,
        simple_grid_kernel_input.u_vert,
        simple_grid_kernel_input.v_vert,
        simple_grid_kernel_input.primal_normal_vert_v1,
        simple_grid_kernel_input.primal_normal_vert_v2,
        simple_grid_kernel_input.z_nabla2_e,
        simple_grid_kernel_input.inv_vert_vert_length,
        simple_grid_kernel_input.inv_primal_edge_length,
    )

    assert np.allclose(
        z_nabla4_e2_comp,
        simple_grid_kernel_input.ref_z_nabla4_e2,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )

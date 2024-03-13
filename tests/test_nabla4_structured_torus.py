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

SMALL_TORUS_GRID_FILE = (
    path.dirname(__file__) + "/data/torus_grid/torus_100000_100000_24576.nc"
)


def init_grid_manager(fname, num_levels=10, transformation=IndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
    grid_manager()
    return grid_manager


@pytest.fixture
def small_torus_grid():
    grid_manager = init_grid_manager(SMALL_TORUS_GRID_FILE, 65, ToGt4PyTransformation())
    grid_manager()
    small_torus_grid = grid_manager.get_grid()
    yield small_torus_grid


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


def test_validate_nabla4_structured_torus_naive(small_torus_grid_kernel_input):
    z_nabla4_e2_comp = icon_benchmark.nabla4_validate_structured_torus_naive(
        small_torus_grid_kernel_input.num_cells,
        small_torus_grid_kernel_input.num_vertices,
        small_torus_grid_kernel_input.num_edges,
        small_torus_grid_kernel_input.num_levels,
        small_torus_grid_kernel_input.E2C2VDim,
        3,
        4,
        small_torus_grid_kernel_input.u_vert,
        small_torus_grid_kernel_input.v_vert,
        small_torus_grid_kernel_input.primal_normal_vert_v1,
        small_torus_grid_kernel_input.primal_normal_vert_v2,
        small_torus_grid_kernel_input.z_nabla2_e,
        small_torus_grid_kernel_input.inv_vert_vert_length,
        small_torus_grid_kernel_input.inv_primal_edge_length,
    )

    assert np.allclose(
        z_nabla4_e2_comp,
        small_torus_grid_kernel_input.ref_z_nabla4_e2,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )

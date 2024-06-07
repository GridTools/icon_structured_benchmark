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

from icon4py.model.common.dimension import V2EDim  # type: ignore [import-not-found]

from test_nabla4_unstructured import small_torus_grid

import icon_benchmark  # type: ignore [import-not-found]


@dataclass
class MoIntpRbfRbfVecInterpolVertexValidationData:
    v2e: np.ndarray
    num_vertices: int
    num_edges: int
    num_levels: int
    p_e_in: np.array
    ptr_coeff_1: np.array
    ptr_coeff_2: np.array
    p_u_out: np.array
    p_v_out: np.array


@pytest.fixture
def small_torus_grid_kernel_input(small_torus_grid):
    interpolate_kernel_validation_data = MoIntpRbfRbfVecInterpolVertexValidationData

    interpolate_kernel_validation_data.v2e = small_torus_grid.get_offset_provider(
        "V2E"
    ).table
    interpolate_kernel_validation_data.num_vertices = small_torus_grid.num_vertices
    interpolate_kernel_validation_data.num_edges = small_torus_grid.num_edges
    interpolate_kernel_validation_data.num_levels = small_torus_grid.num_levels

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/torus_grid",
        "MoIntpRbfRbfVecInterpolVertex_fields",
    )
    savepoint = serializer.savepoint["ValidationTest"].time[1]
    interpolate_kernel_validation_data.p_e_in = serializer.read("p_e_in", savepoint)
    interpolate_kernel_validation_data.ptr_coeff_1 = serializer.read(
        "ptr_coeff_1", savepoint
    )
    interpolate_kernel_validation_data.ptr_coeff_2 = serializer.read(
        "ptr_coeff_2", savepoint
    )

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        path.dirname(__file__) + "/data/torus_grid",
        "MoIntpRbfRbfVecInterpolVertex_output",
    )
    out_savepoint = serializer.savepoint["OutputValidationTest"].time[1]
    interpolate_kernel_validation_data.p_u_out = serializer.read(
        "p_u_out", out_savepoint
    )
    interpolate_kernel_validation_data.p_v_out = serializer.read(
        "p_v_out", out_savepoint
    )

    yield interpolate_kernel_validation_data


def test_validate_MoIntpRbfRbfVecInterpolVertex_unstructured_cpu_ifirst(
    small_torus_grid_kernel_input,
):
    p_u_out, p_v_out = icon_benchmark.interpolate_validate_unstructured_cpu_ifirst(
        small_torus_grid_kernel_input.num_vertices,
        small_torus_grid_kernel_input.num_edges,
        small_torus_grid_kernel_input.num_levels,
        small_torus_grid_kernel_input.v2e,
        small_torus_grid_kernel_input.p_e_in,
        small_torus_grid_kernel_input.ptr_coeff_1,
        small_torus_grid_kernel_input.ptr_coeff_2,
    )

    assert np.allclose(
        p_u_out,
        small_torus_grid_kernel_input.p_u_out,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )
    assert np.allclose(
        p_v_out,
        small_torus_grid_kernel_input.p_v_out,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )


def test_validate_MoIntpRbfRbfVecInterpolVertex_unstructured_cpu_kfirst(
    small_torus_grid_kernel_input,
):
    p_u_out, p_v_out = icon_benchmark.interpolate_validate_unstructured_cpu_kfirst(
        small_torus_grid_kernel_input.num_vertices,
        small_torus_grid_kernel_input.num_edges,
        small_torus_grid_kernel_input.num_levels,
        small_torus_grid_kernel_input.v2e,
        small_torus_grid_kernel_input.p_e_in,
        small_torus_grid_kernel_input.ptr_coeff_1,
        small_torus_grid_kernel_input.ptr_coeff_2,
    )

    assert np.allclose(
        p_u_out,
        small_torus_grid_kernel_input.p_u_out,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )
    assert np.allclose(
        p_v_out,
        small_torus_grid_kernel_input.p_v_out,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )

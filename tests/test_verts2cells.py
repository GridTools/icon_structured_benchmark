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
class verts2cellsValidationData:
    c2v: np.ndarray
    num_vertices: int
    num_cells: int
    num_levels: int
    p_vert_in: np.array
    ptr_coeff: np.array
    p_cell_out: np.array

def reference_interpolate_unstructured_cpu_ifirst(
    num_cells: int,
    num_levels: int,
    c2v: np.ndarray,
    p_vert_in: np.array,
    ptr_coeff: np.array,
):
    p_cell_out = np.zeros((len(c2v), num_levels))
    for i in range(len(c2v)):
        for k in range(num_levels):
            for j in range(3):
                p_cell_out[i, k] += p_vert_in[c2v[i, j], k] * ptr_coeff[i, j]
    return p_cell_out

@pytest.fixture
def small_torus_grid_kernel_input(small_torus_grid):
    interpolate_kernel_validation_data = verts2cellsValidationData

    interpolate_kernel_validation_data.c2v = small_torus_grid.get_offset_provider(
        "C2V"
    ).table
    interpolate_kernel_validation_data.num_vertices = small_torus_grid.num_vertices
    interpolate_kernel_validation_data.num_cells = small_torus_grid.num_cells
    interpolate_kernel_validation_data.num_levels = small_torus_grid.num_levels

    interpolate_kernel_validation_data.p_vert_in = np.random.rand(
        small_torus_grid.num_vertices, small_torus_grid.num_levels
    )
    interpolate_kernel_validation_data.ptr_coeff = np.random.rand(
        small_torus_grid.num_cells, 3
    )

    interpolate_kernel_validation_data.p_cell_out = reference_interpolate_unstructured_cpu_ifirst(
        interpolate_kernel_validation_data.num_cells,
        interpolate_kernel_validation_data.num_levels,
        interpolate_kernel_validation_data.c2v,
        interpolate_kernel_validation_data.p_vert_in,
        interpolate_kernel_validation_data.ptr_coeff,
    )

    yield interpolate_kernel_validation_data


def test_validate_verts2cells_unstructured_cpu_kfirst(
    small_torus_grid_kernel_input,
):
    p_cell_out = icon_benchmark.verts2cells_validate_unstructured_cpu_kfirst(
        small_torus_grid_kernel_input.num_vertices,
        small_torus_grid_kernel_input.num_cells,
        small_torus_grid_kernel_input.num_levels,
        small_torus_grid_kernel_input.c2v,
        small_torus_grid_kernel_input.p_vert_in,
        small_torus_grid_kernel_input.ptr_coeff,
    )

    assert np.allclose(
        p_cell_out,
        small_torus_grid_kernel_input.p_cell_out,
        equal_nan=True,
        atol=1e-8,
        rtol=1e-4,
    )

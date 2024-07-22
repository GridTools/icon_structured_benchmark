import argparse
import numpy as np
from os import path
from pathlib import Path

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToGt4PyTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize  # type: ignore [import-not-found]

from icon4py.model.common.dimension import E2C2VDim  # type: ignore [import-not-found]

import icon_benchmark  # type: ignore [import-not-found]

import nabla4_gtfn  # type: ignore [import-not-found]

import netCDF4  # type: ignore [import-not-found]

from json import dump

from run_filtered_torus_grid_int_interpolate import (
    process_v2e_per_orientation,
    process_v2e_per_vertex,
)


def print_median_runtimes(runtimes):
    for key in runtimes.keys():
        values = runtimes[key]
        print(
            "{} median runtime: {}".format(
                key,
                np.median(values),
            )
        )


def get_torus_cartesian_dimensions(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    sorted_y_coordinates = np.sort(nc["cartesian_y_vertices"][:])
    longitude_dimension = np.count_nonzero(sorted_y_coordinates == 0.0)
    latitude_dimension = int(len(sorted_y_coordinates) / longitude_dimension)
    return (longitude_dimension, latitude_dimension)


def init_grid_manager(
    fname,
    num_levels=65,
    transformation=ToGt4PyTransformation(),
    e2c2v_ordering="per-vertex",
):
    grid_manager = GridManager(
        transformation,
        fname,
        VerticalGridSize(num_levels),
        True,
        e2c2v_ordering == "per-orientation",
    )
    grid_manager()
    return grid_manager


def get_torus_grid(filename, num_levels, transformation, e2c2v_ordering="per-vertex"):
    grid_manager = init_grid_manager(
        filename, num_levels, transformation, e2c2v_ordering
    )
    simple_grid = grid_manager.get_grid()
    return simple_grid


def filter_edge_vector(
    vector, grid_cartesian_dimensions, e2c2v_ordering="per-vertex", halo=2
):
    filtered_vector = []
    if e2c2v_ordering == "per-vertex":
        for i in range(grid_cartesian_dimensions[0]):
            for j in range(grid_cartesian_dimensions[1]):
                if (
                    i > halo - 1
                    and j > halo - 1
                    and i < grid_cartesian_dimensions[0] - halo
                    and j < grid_cartesian_dimensions[1] - halo
                ):
                    filtered_vector.append(
                        vector[(i * grid_cartesian_dimensions[1] + j) * 3]
                    )
                    filtered_vector.append(
                        vector[(i * grid_cartesian_dimensions[1] + j) * 3 + 1]
                    )
                    filtered_vector.append(
                        vector[(i * grid_cartesian_dimensions[1] + j) * 3 + 2]
                    )
    else:
        for i in range(grid_cartesian_dimensions[0]):
            for j in range(grid_cartesian_dimensions[1]):
                if (
                    i > halo - 1
                    and j > halo - 1
                    and i < grid_cartesian_dimensions[0] - halo
                    and j < grid_cartesian_dimensions[1] - halo
                ):
                    filtered_vector.append(
                        vector[(i * grid_cartesian_dimensions[1] + j)]
                    )
        for i in range(grid_cartesian_dimensions[0]):
            for j in range(grid_cartesian_dimensions[1]):
                if (
                    i > halo - 1
                    and j > halo - 1
                    and i < grid_cartesian_dimensions[0] - halo
                    and j < grid_cartesian_dimensions[1] - halo
                ):
                    filtered_vector.append(
                        vector[
                            (i * grid_cartesian_dimensions[1] + j)
                            + grid_cartesian_dimensions[0]
                            * grid_cartesian_dimensions[1]
                        ]
                    )
        for i in range(grid_cartesian_dimensions[0]):
            for j in range(grid_cartesian_dimensions[1]):
                if (
                    i > halo - 1
                    and j > halo - 1
                    and i < grid_cartesian_dimensions[0] - halo
                    and j < grid_cartesian_dimensions[1] - halo
                ):
                    filtered_vector.append(
                        vector[
                            (i * grid_cartesian_dimensions[1] + j)
                            + (
                                grid_cartesian_dimensions[0]
                                * grid_cartesian_dimensions[1]
                            )
                            * 2
                        ]
                    )
    return np.array(filtered_vector)


def run_sanity_checks(
    filtered_e2c2v_separate,
    filtered_e2ecv_separate,
    filtered_e2v_separate,
    filtered_e2c2v_inlined,
    filtered_e2ecv_inlined,
    filtered_e2v_inlined,
    grid,
    lon_dim,
    lat_dim,
    backend="all_cpu",
    halo=2,
    e2c2v_ordering="per-vertex",
    combination="separate",
):
    print("Generating separate validation data")
    random_validation_data_separate = (
        icon_benchmark.get_nabla4_benchmark_validation_data(
            filtered_e2c2v_separate,
            filtered_e2ecv_separate,
            grid.num_cells,
            grid.num_vertices,
            grid.num_edges,
            grid.num_levels,
            grid.size[E2C2VDim],
        )
    )
    p_e_in = np.array(random_validation_data_separate.z_nabla4_e2_wp).T
    ptr_coeff_1 = np.random.rand(grid.num_edges, 6)
    ptr_coeff_2 = np.random.rand(grid.num_edges, 6)
    (
        p_u_out_ref_separate,
        p_v_out_ref_separate,
    ) = icon_benchmark.interpolate_validate_unstructured_cpu_ifirst(
        grid.num_vertices,
        (lon_dim - 2 * halo) * (lat_dim - 2 * halo) * 3,
        grid.num_levels,
        filtered_e2v_separate,
        p_e_in,
        ptr_coeff_1,
        ptr_coeff_2,
    )
    print("Generated separated validation data")

    print("Generating inlined validation data")
    inv_vert_vert_length_inlined = np.zeros(
        random_validation_data_separate.EdgeDim, dtype=np.float64
    )
    inv_primal_edge_length_inlined = np.zeros(
        random_validation_data_separate.EdgeDim, dtype=np.float64
    )
    z_nabla2_e_inlined = np.zeros(
        (random_validation_data_separate.EdgeDim, grid.num_levels),
        dtype=np.float64,
    )
    z_nabla2_e_ref = np.array(random_validation_data_separate.z_nabla2_e).T
    if e2c2v_ordering == "per-orientation":
        for edge_index in range(random_validation_data_separate.EdgeDim):
            j = edge_index % (lon_dim * lat_dim) // lat_dim
            i = edge_index % (lon_dim * lat_dim) % lat_dim
            orientation = edge_index // (lon_dim * lat_dim)
            internal_index = (
                i
                - halo
                + (j - halo) * (lat_dim - 2 * halo)
                + orientation * (lon_dim - 2 * halo) * (lat_dim - 2 * halo)
            )
            if i >= halo and j >= halo and i < lat_dim - halo and j < lon_dim - halo:
                inv_vert_vert_length_inlined[
                    edge_index
                ] = random_validation_data_separate.inv_vert_vert_length[internal_index]
                inv_primal_edge_length_inlined[
                    edge_index
                ] = random_validation_data_separate.inv_primal_edge_length[
                    internal_index
                ]
                z_nabla2_e_inlined[edge_index] = z_nabla2_e_ref[internal_index]
    else:
        for edge_index in range(random_validation_data_separate.EdgeDim):
            j = edge_index // 3 // lat_dim
            i = edge_index // 3 % lat_dim
            orientation = edge_index % 3
            internal_index = (
                i - halo + (j - halo) * (lat_dim - 2 * halo)
            ) * 3 + orientation
            if i >= halo and j >= halo and i < lat_dim - halo and j < lon_dim - halo:
                inv_vert_vert_length_inlined[
                    edge_index
                ] = random_validation_data_separate.inv_vert_vert_length[internal_index]
                inv_primal_edge_length_inlined[
                    edge_index
                ] = random_validation_data_separate.inv_primal_edge_length[
                    internal_index
                ]
                z_nabla2_e_inlined[edge_index] = z_nabla2_e_ref[internal_index]
    print("Generated inlined validation data")

    if backend in ["all_cpu", "cpu_ifirst"]:
        if combination in ["all", "separate"]:
            print("Running unstructured cpu_ifirst separate sanity check")
            (
                p_u_out_cpu_ifirst_unstructured_separate,
                p_v_out_cpu_ifirst_unstructured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_cpu_ifirst_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_e2v_separate,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_ifirst_unstructured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_ifirst_unstructured_separate, p_v_out_ref_separate
            )
            print("unstructured cpu_ifirst separate sanity check passed")

            print("Running structured cpu_ifirst separate sanity check")
            (
                p_u_out_cpu_ifirst_structured_separate,
                p_v_out_cpu_ifirst_structured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_cpu_ifirst_separate(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_ifirst_structured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_ifirst_structured_separate, p_v_out_ref_separate
            )
            print("structured cpu_ifirst separate sanity check passed")

        if combination in ["all", "inlined"]:
            print("Running unstructured cpu_ifirst inlined sanity check")
            (
                p_u_out_cpu_ifirst_unstructured_inlined,
                p_v_out_cpu_ifirst_unstructured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_cpu_ifirst_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_e2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_ifirst_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_ifirst_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured cpu_ifirst inlined sanity check passed")

            print("Running unstructured cpu_ifirst inlined v2e2c2v sanity check")
            (
                p_u_out_cpu_ifirst_unstructured_inlined_v2v,
                p_v_out_cpu_ifirst_unstructured_inlined_v2v,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_cpu_ifirst_inlined_v2v(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_e2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_ifirst_unstructured_inlined_v2v, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_ifirst_unstructured_inlined_v2v, p_v_out_ref_separate
            )
            print("unstructured cpu_ifirst inlined v2e2c2v sanity check passed")

            print("Running structured cpu_ifirst inlined sanity check")
            (
                p_u_out_cpu_ifirst_structured_inlined,
                p_v_out_cpu_ifirst_structured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_cpu_ifirst_inlined(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_ifirst_structured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_ifirst_structured_inlined, p_v_out_ref_separate
            )
            print("structured cpu_ifirst inlined sanity check passed")

    if backend in ["all_cpu", "cpu_kfirst"]:
        if combination in ["all", "separate"]:
            print("Running unstructured cpu_kfirst separate sanity check")
            (
                p_u_out_cpu_kfirst_unstructured_separate,
                p_v_out_cpu_kfirst_unstructured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_cpu_kfirst_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_e2v_separate,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_kfirst_unstructured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_kfirst_unstructured_separate, p_v_out_ref_separate
            )
            print("unstructured cpu_kfirst separate sanity check passed")

            print("Running structured cpu_kfirst separate sanity check")
            (
                p_u_out_cpu_kfirst_structured_separate,
                p_v_out_cpu_kfirst_structured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_cpu_kfirst_separate(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_kfirst_structured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_kfirst_structured_separate, p_v_out_ref_separate
            )
            print("structured cpu_kfirst separate sanity check passed")

        if combination in ["all", "inlined"]:
            print("Running unstructured cpu_kfirst inlined sanity check")
            (
                p_u_out_cpu_kfirst_unstructured_inlined,
                p_v_out_cpu_kfirst_unstructured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_cpu_kfirst_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_e2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_kfirst_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_kfirst_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured cpu_kfirst inlined sanity check passed")

            print("Running unstructured cpu_kfirst v2e2c2v inlined sanity check")
            (
                p_u_out_cpu_kfirst_unstructured_inlined_v2v,
                p_v_out_cpu_kfirst_unstructured_inlined_v2v,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_cpu_kfirst_inlined_v2v(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_e2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_kfirst_unstructured_inlined_v2v, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_kfirst_unstructured_inlined_v2v, p_v_out_ref_separate
            )
            print("unstructured cpu_kfirst inlined v2e2c2v sanity check passed")

            print("Running structured cpu_kfirst inlined sanity check")
            (
                p_u_out_cpu_kfirst_structured_inlined,
                p_v_out_cpu_kfirst_structured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_cpu_kfirst_inlined(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_cpu_kfirst_structured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_cpu_kfirst_structured_inlined, p_v_out_ref_separate
            )
            print("structured cpu_kfirst inlined sanity check passed")

    if backend in ["all_gpu", "gpu_naive"]:
        if combination in ["all", "separate"]:
            print("Running unstructured gpu_naive separate sanity check")
            (
                p_u_out_gpu_naive_unstructured_separate,
                p_v_out_gpu_naive_unstructured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_naive_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_e2v_separate,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_gpu_naive_unstructured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_unstructured_separate, p_v_out_ref_separate
            )
            print("unstructured gpu_naive separate sanity check passed")

            print("Running structured gpu_naive separate sanity check")
            (
                p_u_out_gpu_naive_structured_separate,
                p_v_out_gpu_naive_structured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_gpu_naive_separate(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_gpu_naive_structured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_structured_separate, p_v_out_ref_separate
            )
            print("structured gpu_naive separate sanity check passed")

        if combination in ["all", "inlined"]:
            print("Running unstructured gpu_naive inlined sanity check")
            (
                p_u_out_gpu_naive_unstructured_inlined,
                p_v_out_gpu_naive_unstructured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_naive_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_e2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_gpu_naive_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_naive inlined sanity check passed")

            print("Running unstructured gpu_naive inlined v2v sanity check")
            (
                p_u_out_gpu_naive_unstructured_inlined,
                p_v_out_gpu_naive_unstructured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_naive_inlined_v2v(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_e2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_gpu_naive_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_naive inlined v2v sanity check passed")

            print("Running structured gpu_naive inlined sanity check")
            (
                p_u_out_gpu_naive_structured_inlined,
                p_v_out_gpu_naive_structured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_gpu_naive_inlined(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            print("p_u_out_gpu_naive_structured_inlined", p_u_out_gpu_naive_structured_inlined)
            print("p_u_out_ref_separate", p_u_out_ref_separate)
            assert np.allclose(
                p_u_out_gpu_naive_structured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_structured_inlined, p_v_out_ref_separate
            )
            print("structured gpu_naive inlined sanity check passed")

    if backend in ["all_gpu", "gpu_kloop"]:
        if combination in ["all", "separate"]:
            print("Running unstructured gpu_kloop separate sanity check")
            (
                p_u_out_gpu_kloop_unstructured_separate,
                p_v_out_gpu_kloop_unstructured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_kloop_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_e2v_separate,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_gpu_kloop_unstructured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_unstructured_separate, p_v_out_ref_separate
            )
            print("unstructured gpu_kloop separate sanity check passed")

            print("Running structured gpu_kloop separate sanity check")
            (
                p_u_out_gpu_kloop_structured_separate,
                p_v_out_gpu_kloop_structured_separate,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_gpu_kloop_separate(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
            )
            assert np.allclose(
                p_u_out_gpu_kloop_structured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_structured_separate, p_v_out_ref_separate
            )
            print("structured gpu_kloop separate sanity check passed")

    print("Sanity checks pass")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("grid", help="Torus grid netCDF4 file")
    parser.add_argument(
        "--transformation",
        choices=["gt4py", "index"],
        default="gt4py",
        help="Use either ToGt4PyTransformation or IndexTransformation",
    )
    parser.add_argument("--klevels", type=int, default=65, help="Number of k levels")
    parser.add_argument(
        "--repetitions", type=int, default=101, help="Number of repetitions"
    )
    parser.add_argument(
        "--dry-run", default=False, help="Do a dry run or not", action="store_true"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="JSON output file name"
    )
    parser.add_argument(
        "--sanity-checks",
        default=False,
        help="Do a validation with random data between structured and unstructured for the given grid",
        action="store_true",
    )
    parser.add_argument(
        "--backend",
        choices=[
            "all_cpu",
            "all_gpu",
            "gtfn_cpu",
            "gtfn_gpu",
            "naive",
            "cpu_ifirst",
            "cpu_kfirst",
            "gpu_kloop",
            "gpu_naive",
        ],
        default="all_cpu",
        help="Which backend to benchmark",
    )
    parser.add_argument(
        "--combination",
        choices=["separate", "inlined", "all"],
        default="all",
        help="Which combination of kernels to benchmark",
    )
    parser.add_argument(
        "--e2c2v-ordering",
        choices=["per-vertex", "per-orientation"],
        default="per-vertex",
        help="E2C2V ordering",
    )
    parser.add_argument(
        "--halo", type=int, default=2, help="Halo size for structured grids"
    )

    args = parser.parse_args()
    if "gpu" in args.backend:
        args.e2c2v_ordering = "per-orientation"
    else:
        args.e2c2v_ordering = "per-vertex"

    return args


def run_benchmarks():
    args = parse_arguments()

    transformation = (
        ToGt4PyTransformation()
        if args.transformation == "gt4py"
        else IndexTransformation()
    )

    torus_grid = get_torus_grid(
        args.grid, args.klevels, transformation, args.e2c2v_ordering
    )

    repetitions = args.repetitions
    dry_runs = 10 if args.dry_run else 0

    grid_cartesian_dimensions = get_torus_cartesian_dimensions(args.grid)

    print(
        "CellsDim: {} VertexDim: {} EdgeDim: {} KDim: {} E2C2VDim: {} Longitude dimension: {} Latitude dimension: {} Halo: {}".format(
            torus_grid.num_cells,
            torus_grid.num_vertices,
            torus_grid.num_edges,
            torus_grid.num_levels,
            torus_grid.size[E2C2VDim],
            grid_cartesian_dimensions[0],
            grid_cartesian_dimensions[1],
            args.halo,
        )
    )

    def filter_neighbors(
        grid_e2c2v,
        grid_e2ecv,
        grid_v2e,
        e2c2v_ordering="per-vertex",
        combination="separate",
    ):
        filtered_e2c2v = filter_edge_vector(
            grid_e2c2v,
            grid_cartesian_dimensions,
            args.e2c2v_ordering,
            args.halo if combination == "separate" else 0,
        )

        def _get_gpu_coalesced_permuted_e2ecv():
            orientation_permuted_e2ecv = np.zeros_like(grid_e2ecv)
            edges_size = len(grid_e2ecv)
            for i in range(edges_size):
                for j in range(4):
                    orientation_permuted_e2ecv[i][j] = j * edges_size + i
            return orientation_permuted_e2ecv

        permuted_e2ecv = (
            grid_e2ecv
            if args.e2c2v_ordering == "per-vertex"
            else _get_gpu_coalesced_permuted_e2ecv()
        )
        filtered_e2ecv = filter_edge_vector(
            permuted_e2ecv,
            grid_cartesian_dimensions,
            args.e2c2v_ordering,
            args.halo if combination == "separate" else 0,
        )

        filtered_v2e = (
            process_v2e_per_orientation(
                grid_v2e,
                grid_cartesian_dimensions[1],
                grid_cartesian_dimensions[0],
                args.halo + 1,
            )
            if e2c2v_ordering == "per-orientation"
            else process_v2e_per_vertex(
                grid_v2e,
                grid_cartesian_dimensions[1],
                grid_cartesian_dimensions[0],
                args.halo + 1,
            )
        )

        def transform_v2e_to_nabla4_region_per_orientation(v2e, x_dim, y_dim, halo):
            x_dim_inner = x_dim - 2 * (halo + 1)
            y_dim_inner = y_dim - 2 * (halo + 1)
            x_dim_nabla4 = x_dim - 2 * halo
            y_dim_nabla4 = y_dim - 2 * halo
            nabla4_dim = x_dim_nabla4 * y_dim_nabla4
            transformed_v2e = np.zeros_like(v2e)
            assert (x_dim_inner * y_dim_inner, 6) == v2e.shape
            for i in range(x_dim_inner):
                for j in range(y_dim_inner):
                    for k in range(6):
                        global_vertex_i = (
                            v2e[i + j * x_dim_inner][k] % (x_dim * y_dim)
                        ) % x_dim
                        global_vertex_j = (
                            v2e[i + j * x_dim_inner][k] % (x_dim * y_dim)
                        ) // x_dim
                        orientation = v2e[i + j * x_dim_inner][k] // (x_dim * y_dim)
                        nabla4_local_vertex_i = global_vertex_i - halo
                        nabla4_local_vertex_j = global_vertex_j - halo
                        transformed_v2e[i + j * x_dim_inner][k] = (
                            nabla4_local_vertex_i
                            + nabla4_local_vertex_j * x_dim_nabla4
                            + orientation * nabla4_dim
                        )
            return transformed_v2e

        def transform_v2e_to_nabla4_region_per_vertex(v2e, x_dim, y_dim, halo):
            x_dim_inner = x_dim - 2 * (halo + 1)
            y_dim_inner = y_dim - 2 * (halo + 1)
            x_dim_nabla4 = x_dim - 2 * halo
            y_dim_nabla4 = y_dim - 2 * halo
            transformed_v2e = np.zeros_like(v2e)
            assert (x_dim_inner * y_dim_inner, 6) == v2e.shape
            for i in range(x_dim_inner):
                for j in range(y_dim_inner):
                    for k in range(6):
                        global_vertex_i = (v2e[i + j * x_dim_inner][k] // 3) % x_dim
                        global_vertex_j = (v2e[i + j * x_dim_inner][k] // 3) // x_dim
                        orientation = v2e[i + j * x_dim_inner][k] % 3
                        nabla4_local_vertex_i = global_vertex_i - halo
                        nabla4_local_vertex_j = global_vertex_j - halo
                        transformed_v2e[i + j * x_dim_inner][k] = (
                            nabla4_local_vertex_i + nabla4_local_vertex_j * x_dim_nabla4
                        ) * 3 + orientation
            return transformed_v2e

        if combination == "separate":
            filtered_v2e = (
                transform_v2e_to_nabla4_region_per_orientation(
                    filtered_v2e,
                    grid_cartesian_dimensions[1],
                    grid_cartesian_dimensions[0],
                    args.halo,
                )
                if e2c2v_ordering == "per-orientation"
                else transform_v2e_to_nabla4_region_per_vertex(
                    filtered_v2e,
                    grid_cartesian_dimensions[1],
                    grid_cartesian_dimensions[0],
                    args.halo,
                )
            )
        return filtered_e2c2v, filtered_e2ecv, filtered_v2e

    (
        filtered_e2c2v_separate,
        filtered_e2ecv_separate,
        filtered_v2e_separate,
    ) = filter_neighbors(
        torus_grid.get_offset_provider("E2C2V").table,
        torus_grid.get_offset_provider("E2ECV").table,
        torus_grid.get_offset_provider("V2E").table,
        args.e2c2v_ordering,
        "separate",
    )
    (
        filtered_e2c2v_inlined,
        filtered_e2ecv_inlined,
        filtered_v2e_inlined,
    ) = filter_neighbors(
        torus_grid.get_offset_provider("E2C2V").table,
        torus_grid.get_offset_provider("E2ECV").table,
        torus_grid.get_offset_provider("V2E").table,
        args.e2c2v_ordering,
        "inlined",
    )

    def create_v2e2c2v_neighbor(e2c2v, v2e):
        v2e2c2v = np.zeros((len(v2e), 24), dtype=np.int32)
        for i, v2e_neighbors in enumerate(v2e):
            for j, v2e_neighbor in enumerate(v2e_neighbors):
                for k, e2c2v_neighbor in enumerate(e2c2v[v2e_neighbor]):
                    v2e2c2v[i][j * 4 + k] = e2c2v_neighbor
        return v2e2c2v

    def optimize_v2e2c2v(v2e2c2v):
        v2e2c2v_opt = np.zeros((len(v2e2c2v), 7), dtype=np.int32)
        for i, v2e2c2v_neighbors in enumerate(v2e2c2v):
            v2e2c2v_opt[i][0] = v2e2c2v_neighbors[0]
            v2e2c2v_opt[i][1] = v2e2c2v_neighbors[1]
            v2e2c2v_opt[i][2] = v2e2c2v_neighbors[2]
            v2e2c2v_opt[i][3] = v2e2c2v_neighbors[3]
            v2e2c2v_opt[i][4] = v2e2c2v_neighbors[5]
            v2e2c2v_opt[i][5] = v2e2c2v_neighbors[6]
            v2e2c2v_opt[i][6] = v2e2c2v_neighbors[7]
            assert np.isin(v2e2c2v_opt[i], v2e2c2v[i]).all()
            # print("i: ", i)
            # for j in v2e2c2v_neighbors:
            #     print(list(v2e2c2v_opt[i]).index(j))
        return v2e2c2v_opt

    v2e2c2v = create_v2e2c2v_neighbor(filtered_e2c2v_inlined, filtered_v2e_inlined)
    v2e2c2v_opt = optimize_v2e2c2v(v2e2c2v)

    # v2e2ecv = create_v2e2c2v_neighbor(filtered_e2ecv_inlined, filtered_v2e_inlined)
    # v2e2ecv_opt = optimize_v2e2c2v(v2e2ecv)

    runtimes = {}

    halo = args.halo

    if args.sanity_checks:
        run_sanity_checks(
            filtered_e2c2v_separate,
            filtered_e2ecv_separate,
            filtered_v2e_separate,
            filtered_e2c2v_inlined,
            filtered_e2ecv_inlined,
            filtered_v2e_inlined,
            torus_grid,
            grid_cartesian_dimensions[0],
            grid_cartesian_dimensions[1],
            args.backend,
            halo,
            args.e2c2v_ordering,
            args.combination,
        )

    if args.backend in ["all_cpu", "cpu_ifirst"]:
        if args.combination in ["all", "separate"]:
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_ifirst_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_structured_cpu_ifirst_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_ifirst_separate(
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )
        if args.combination in ["all", "inlined"]:
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined_v2v"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined_v2v(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined"
            ] = icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined(
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )

    if args.backend in ["all_cpu", "cpu_kfirst"]:
        if args.combination in ["all", "separate"]:
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_kfirst_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_structured_cpu_kfirst_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_kfirst_separate(
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )

        if args.combination in ["all", "inlined"]:
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined_v2v"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined_v2v(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined"
            ] = icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined(
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )

    if args.backend in ["all_gpu", "gpu_naive"]:
        if args.combination in ["all", "separate"]:
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_gpu_naive_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_naive_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_structured_gpu_naive_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_structured_gpu_naive_separate(
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )

        if args.combination in ["all", "inlined"]:
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_structured_gpu_naive_inlined"
            ] = icon_benchmark.nabla4_interpolate_benchmark_structured_gpu_naive_inlined(
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )

    if args.backend in ["all_gpu", "gpu_kloop"]:
        if args.combination in ["all", "separate"]:
            runtimes[
                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            runtimes[
                "nabla4_interpolate_benchmark_structured_gpu_kloop_separate"
            ] = icon_benchmark.nabla4_interpolate_benchmark_structured_gpu_kloop_separate(
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )

    print_median_runtimes(runtimes)

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

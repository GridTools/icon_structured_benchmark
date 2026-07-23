import argparse
import numpy as np
from os import path
from pathlib import Path

from icon4py.model.common.grid.gridfile import (  # type: ignore [import-not-found]
    NoTransformation,
    ToZeroBasedIndexTransformation,
)

from icon4py.model.common.dimension import E2C2VDim  # type: ignore [import-not-found]

import icon_benchmark  # type: ignore [import-not-found]

import nabla4_gtfn  # type: ignore [import-not-found]

from json import dump

from run_filtered_torus_grid_int_common import (
    filter_edge_vector,
    generate_filtered_e2c2v,
    generate_original_e2ecv,
    get_torus_cartesian_dimensions,
    get_torus_grid,
    print_median_runtimes,
    process_v2e_per_orientation,
    process_v2e_per_vertex,
)


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
    np.random.seed(10)
    ptr_coeff_1 = np.random.uniform(-10, 10, size=(grid.num_edges, 6))
    ptr_coeff_2 = np.random.uniform(-10, 10, size=(grid.num_edges, 6))
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
        # Generate all edge indices
        edge_indices = np.arange(random_validation_data_separate.EdgeDim)

        # Compute j, i, orientation, and internal_index
        j = (edge_indices % (lon_dim * lat_dim)) // lat_dim
        i = (edge_indices % (lon_dim * lat_dim)) % lat_dim
        orientation = edge_indices // (lon_dim * lat_dim)

        internal_index = (
            i
            - halo
            + (j - halo) * (lat_dim - 2 * halo)
            + orientation * (lon_dim - 2 * halo) * (lat_dim - 2 * halo)
        )

        # Find valid indices
        valid_mask = (
            (i >= halo) & (j >= halo) & (i < lat_dim - halo) & (j < lon_dim - halo)
        )
        valid_edge_indices = edge_indices[valid_mask]
        valid_internal_indices = internal_index[valid_mask]

        # Assign the corresponding values using valid indices
        inv_vert_vert_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_vert_vert_length
        )[valid_internal_indices]
        inv_primal_edge_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_primal_edge_length
        )[valid_internal_indices]
        z_nabla2_e_inlined[valid_edge_indices] = z_nabla2_e_ref[valid_internal_indices]
    else:
        # Generate all edge indices
        edge_indices = np.arange(random_validation_data_separate.EdgeDim)

        # Compute j, i, and orientation
        j = (edge_indices // 3) // lat_dim
        i = (edge_indices // 3) % lat_dim
        orientation = edge_indices % 3

        # Compute internal_index
        internal_index = (
            (i - halo) + (j - halo) * (lat_dim - 2 * halo)
        ) * 3 + orientation

        # Find valid indices
        valid_mask = (
            (i >= halo) & (j >= halo) & (i < lat_dim - halo) & (j < lon_dim - halo)
        )
        valid_edge_indices = np.array(edge_indices[valid_mask], dtype=np.int32)
        # import pdb; pdb.set_trace()
        valid_internal_indices = np.array(internal_index[valid_mask], dtype=np.int32)

        # Assign the corresponding values using valid indices
        inv_vert_vert_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_vert_vert_length
        )[valid_internal_indices]
        inv_primal_edge_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_primal_edge_length
        )[valid_internal_indices]
        z_nabla2_e_inlined[valid_edge_indices] = z_nabla2_e_ref[valid_internal_indices]
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

            print("Running unstructured gpu_naive vertical separate sanity check")
            (
                p_u_out_gpu_naive_vertical_unstructured_separate,
                p_v_out_gpu_naive_vertical_unstructured_separate,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_unstructured_gpu_naive_separate(
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
                p_u_out_gpu_naive_vertical_unstructured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_vertical_unstructured_separate, p_v_out_ref_separate
            )
            print("unstructured gpu_naive vertical separate sanity check passed")

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

            print("Running structured gpu_naive vertical separate sanity check")
            (
                p_u_out_gpu_naive_vertical_structured_separate,
                p_v_out_gpu_naive_vertical_structured_separate,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_structured_gpu_naive_separate(
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
                p_u_out_gpu_naive_vertical_structured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_vertical_structured_separate, p_v_out_ref_separate
            )
            print("structured gpu_naive vertical separate sanity check passed")

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

            print("Running unstructured gpu_naive vertical inlined sanity check")
            (
                p_u_out_gpu_naive_vertical_unstructured_inlined,
                p_v_out_gpu_naive_vertical_unstructured_inlined,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_unstructured_gpu_naive_inlined(
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
                p_u_out_gpu_naive_vertical_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_vertical_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_naive vertical inlined sanity check passed")

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

            print("Running unstructured gpu_naive inlined v2v_general sanity check")
            (
                p_u_out_gpu_naive_unstructured_inlined_v2v_general,
                p_v_out_gpu_naive_unstructured_inlined_v2v_general,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_naive_inlined_v2v_general(
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
                p_u_out_gpu_naive_unstructured_inlined_v2v_general, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_unstructured_inlined_v2v_general, p_v_out_ref_separate
            )
            print("unstructured gpu_naive inlined v2v_general sanity check passed")

            print("Running unstructured gpu_naive vertical inlined v2v sanity check")
            (
                p_u_out_gpu_naive_vertical_unstructured_inlined,
                p_v_out_gpu_naive_vertical_unstructured_inlined,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_unstructured_gpu_naive_inlined_v2v(
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
                p_u_out_gpu_naive_vertical_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_vertical_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_naive vertical inlined v2v sanity check passed")

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
            assert np.allclose(
                p_u_out_gpu_naive_structured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_structured_inlined, p_v_out_ref_separate
            )
            print("structured gpu_naive inlined sanity check passed")

            print("Running structured gpu_naive vertical inlined sanity check")
            (
                p_u_out_gpu_naive_vertical_structured_inlined,
                p_v_out_gpu_naive_vertical_structured_inlined,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_structured_gpu_naive_inlined(
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
                p_u_out_gpu_naive_vertical_structured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_vertical_structured_inlined, p_v_out_ref_separate
            )
            print("structured gpu_naive vertical inlined sanity check passed")

            print("Running structured gpu_naive inlined cached sanity check")
            (
                p_u_out_gpu_naive_structured_inlined_cached,
                p_v_out_gpu_naive_structured_inlined_cached,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_gpu_naive_inlined_cached(
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
                p_u_out_gpu_naive_structured_inlined_cached, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_naive_structured_inlined_cached, p_v_out_ref_separate
            )
            print("structured gpu_naive inlined cached sanity check passed")

            print("Running structured gpu_naive vertical inlined cached sanity check")
            (
                p_u_out_gpu_naive_vertical_structured_inlined_cached,
                p_v_out_gpu_naive_vertical_structured_inlined_cached,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_structured_gpu_naive_inlined_cached(
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
                p_u_out_gpu_naive_vertical_structured_inlined_cached,
                p_u_out_ref_separate,
            )
            assert np.allclose(
                p_v_out_gpu_naive_vertical_structured_inlined_cached,
                p_v_out_ref_separate,
            )
            print("structured gpu_naive inlined cached sanity check passed")

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

            print("Running unstructured gpu_kloop vertical separate sanity check")
            (
                p_u_out_gpu_kloop_vertical_unstructured_separate,
                p_v_out_gpu_kloop_vertical_unstructured_separate,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_separate(
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
                p_u_out_gpu_kloop_vertical_unstructured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_vertical_unstructured_separate, p_v_out_ref_separate
            )
            print("unstructured gpu_kloop vertical separate sanity check passed")

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

            print("Running structured gpu_kloop vertical separate sanity check")
            (
                p_u_out_gpu_kloop_vertical_structured_separate,
                p_v_out_gpu_kloop_vertical_structured_separate,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_structured_gpu_kloop_separate(
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
                p_u_out_gpu_kloop_vertical_structured_separate, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_vertical_structured_separate, p_v_out_ref_separate
            )
            print("structured gpu_kloop vertical separate sanity check passed")

        if combination in ["all", "inlined"]:
            print("Running unstructured gpu_kloop inlined sanity check")
            (
                p_u_out_gpu_kloop_unstructured_inlined,
                p_v_out_gpu_kloop_unstructured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_kloop_inlined(
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
                p_u_out_gpu_kloop_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_kloop inlined sanity check passed")

            print("Running unstructured gpu_kloop inlined cached sanity check")
            (
                p_u_out_gpu_kloop_unstructured_inlined_cached,
                p_v_out_gpu_kloop_unstructured_inlined_cached,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_cached(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_e2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
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
                p_u_out_gpu_kloop_unstructured_inlined_cached, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_unstructured_inlined_cached, p_v_out_ref_separate
            )
            print("unstructured gpu_kloop inlined sanity check passed")

            print("Running unstructured gpu_kloop vertical inlined sanity check")
            (
                p_u_out_gpu_kloop_vertical_unstructured_inlined,
                p_v_out_gpu_kloop_vertical_unstructured_inlined,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_inlined(
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
                p_u_out_gpu_kloop_vertical_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_vertical_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_kloop vertical inlined sanity check passed")

            print("Running unstructured gpu_kloop inlined v2v sanity check")
            (
                p_u_out_gpu_kloop_unstructured_inlined,
                p_v_out_gpu_kloop_unstructured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_v2v(
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
                p_u_out_gpu_kloop_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_kloop inlined v2v sanity check passed")

            # print("Running unstructured gpu_kloop inlined v2v_general sanity check")
            # (
            #     p_u_out_gpu_kloop_unstructured_inlined_v2v_general,
            #     p_v_out_gpu_kloop_unstructured_inlined_v2v_general,
            # ) = icon_benchmark.nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_v2v_general(
            #     filtered_e2c2v_inlined,
            #     filtered_e2ecv_inlined,
            #     filtered_e2v_inlined,
            #     random_validation_data_separate.CellDim,
            #     random_validation_data_separate.VertexDim,
            #     random_validation_data_separate.EdgeDim,
            #     random_validation_data_separate.KDim,
            #     random_validation_data_separate.ECVDim,
            #     np.array(random_validation_data_separate.u_vert).T,
            #     np.array(random_validation_data_separate.v_vert).T,
            #     random_validation_data_separate.primal_normal_vert_v1,
            #     random_validation_data_separate.primal_normal_vert_v2,
            #     z_nabla2_e_inlined,
            #     inv_vert_vert_length_inlined,
            #     inv_primal_edge_length_inlined,
            #     ptr_coeff_1,
            #     ptr_coeff_2,
            # )
            # assert np.allclose(
            #     p_u_out_gpu_kloop_unstructured_inlined_v2v_general, p_u_out_ref_separate
            # )
            # assert np.allclose(
            #     p_v_out_gpu_kloop_unstructured_inlined_v2v_general, p_v_out_ref_separate
            # )
            # print("unstructured gpu_kloop inlined v2v_general sanity check passed")

            print("Running unstructured gpu_kloop vertical inlined v2v sanity check")
            (
                p_u_out_gpu_kloop_vertical_unstructured_inlined,
                p_v_out_gpu_kloop_vertical_unstructured_inlined,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_inlined_v2v(
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
                p_u_out_gpu_kloop_vertical_unstructured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_vertical_unstructured_inlined, p_v_out_ref_separate
            )
            print("unstructured gpu_kloop inlined v2v sanity check passed")

            print("Running structured gpu_kloop inlined sanity check")
            (
                p_u_out_gpu_kloop_structured_inlined,
                p_v_out_gpu_kloop_structured_inlined,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_gpu_kloop_inlined(
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
                p_u_out_gpu_kloop_structured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_structured_inlined, p_v_out_ref_separate
            )
            print("structured gpu_kloop inlined sanity check passed")

            print("Running structured gpu_kloop vertical inlined sanity check")
            (
                p_u_out_gpu_kloop_vertical_structured_inlined,
                p_v_out_gpu_kloop_vertical_structured_inlined,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_structured_gpu_kloop_inlined(
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
                p_u_out_gpu_kloop_vertical_structured_inlined, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_vertical_structured_inlined, p_v_out_ref_separate
            )
            print("structured gpu_kloop vertical inlined sanity check passed")

            print("Running structured gpu_kloop inlined cached sanity check")
            (
                p_u_out_gpu_kloop_structured_inlined_cached,
                p_v_out_gpu_kloop_structured_inlined_cached,
            ) = icon_benchmark.nabla4_interpolate_validate_structured_gpu_kloop_inlined_cached(
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
                p_u_out_gpu_kloop_structured_inlined_cached, p_u_out_ref_separate
            )
            assert np.allclose(
                p_v_out_gpu_kloop_structured_inlined_cached, p_v_out_ref_separate
            )
            print("structured gpu_kloop inlined cached sanity check passed")

            print("Running structured gpu_kloop vertical inlined cached sanity check")
            (
                p_u_out_gpu_kloop_vertical_structured_inlined_cached,
                p_v_out_gpu_kloop_vertical_structured_inlined_cached,
            ) = icon_benchmark.nabla4_vertical_interpolate_validate_structured_gpu_kloop_inlined_cached(
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
                p_u_out_gpu_kloop_vertical_structured_inlined_cached,
                p_u_out_ref_separate,
            )
            assert np.allclose(
                p_v_out_gpu_kloop_vertical_structured_inlined_cached,
                p_v_out_ref_separate,
            )
            print("structured gpu_kloop vertical inlined cached sanity check passed")

    print("Sanity checks pass")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("grid", help="Torus grid netCDF4 file")
    parser.add_argument(
        "--transformation",
        choices=["gt4py", "index"],
        default="gt4py",
        help="Use either ToZeroBasedIndexTransformation or NoTransformation (gt4py by default)",
    )
    parser.add_argument(
        "--klevels", type=int, default=80, help="Number of k levels (80 default)"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=101,
        help="Number of repetitions (101 default)",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        help="Enable dry runs (not taken into runtime results) (disabled by default)",
        action="store_true",
    )
    parser.add_argument(
        "--output", type=str, default="output", help="JSON output file name"
    )
    parser.add_argument(
        "--sanity-checks",
        default=False,
        help="Do a validation with random data between structured and unstructured for the given grid (disabled by default)",
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
        help="Which backend to benchmark (default all_cpu)",
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
        help="E2C2V ordering (per-vertex in CPU and per-orienteation in GPU by default)",
    )
    parser.add_argument(
        "--halo",
        type=int,
        default=2,
        help="Halo size for structured grids (default 2) [Shouldn't be changed]",
    )
    parser.add_argument(
        "--vertical",
        default=False,
        help="Use nabla4_vertical kernel instead of nabla4 (disabled by default)",
        action="store_true",
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
        ToZeroBasedIndexTransformation()
        if args.transformation == "gt4py"
        else NoTransformation()
    )

    torus_grid = get_torus_grid(
        args.grid, args.klevels, transformation, args.e2c2v_ordering, True
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
        generated_e2c2v = generate_filtered_e2c2v(
            grid_cartesian_dimensions,
            args.e2c2v_ordering,
            args.halo if combination == "separate" else 0,
        )
        filtered_e2c2v = filter_edge_vector(
            grid_e2c2v,
            grid_cartesian_dimensions,
            args.e2c2v_ordering,
            args.halo if combination == "separate" else 0,
        )
        if not np.array_equal(generated_e2c2v, filtered_e2c2v):
            raise ValueError(
                f"Generated e2c2v and filtered e2c2v are not equal for combination: {combination}. Please check the filtering logic."
            )

        # generate_original_e2ecv already produces the per-orientation layout
        # that the structured GPU kernels expect (slot * n_edges + color * n_vertices + vertex),
        # so no extra permutation is needed.
        filtered_e2ecv = filter_edge_vector(
            grid_e2ecv,
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

    original_e2c2v = torus_grid.get_connectivity("E2C2V").ndarray
    e2ecv = generate_original_e2ecv(original_e2c2v, args.e2c2v_ordering)

    (
        filtered_e2c2v_separate,
        filtered_e2ecv_separate,
        filtered_v2e_separate,
    ) = filter_neighbors(
        original_e2c2v,
        e2ecv,
        torus_grid.get_connectivity("V2E").ndarray,
        args.e2c2v_ordering,
        "separate",
    )
    (
        filtered_e2c2v_inlined,
        filtered_e2ecv_inlined,
        filtered_v2e_inlined,
    ) = filter_neighbors(
        original_e2c2v,
        e2ecv,
        torus_grid.get_connectivity("V2E").ndarray,
        args.e2c2v_ordering,
        "inlined",
    )

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

    if repetitions == 0:
        print("Repetitions is 0, skipping benchmarks")
        return

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
            print(
                "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_separate"
                        ]
                    )
                )
            )
            runtimes["nabla4_interpolate_benchmark_structured_cpu_ifirst_separate"] = (
                icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_ifirst_separate(
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
            )
            print(
                "nabla4_interpolate_benchmark_structured_cpu_ifirst_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_structured_cpu_ifirst_separate"
                        ]
                    )
                )
            )
        if args.combination in ["all", "inlined"]:
            runtimes["nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined"] = (
                icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined(
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
            )
            print(
                "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined"
                        ]
                    )
                )
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
            print(
                "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined_v2v median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined_v2v"
                        ]
                    )
                )
            )
            runtimes["nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined"] = (
                icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined(
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
            )
            print(
                "nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined"
                        ]
                    )
                )
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
            print(
                "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_separate"
                        ]
                    )
                )
            )
            runtimes["nabla4_interpolate_benchmark_structured_cpu_kfirst_separate"] = (
                icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_kfirst_separate(
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
            )
            print(
                "nabla4_interpolate_benchmark_structured_cpu_kfirst_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_structured_cpu_kfirst_separate"
                        ]
                    )
                )
            )

        if args.combination in ["all", "inlined"]:
            runtimes["nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined"] = (
                icon_benchmark.nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined(
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
            )
            print(
                "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined"
                        ]
                    )
                )
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
            print(
                "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined_v2v median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined_v2v"
                        ]
                    )
                )
            )
            runtimes["nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined"] = (
                icon_benchmark.nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined(
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
            )
            print(
                "nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined"
                        ]
                    )
                )
            )

    if args.backend in ["all_gpu", "gpu_naive"]:
        if args.combination in ["all", "separate"]:
            if not args.vertical:
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_naive_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_naive_separate"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_separate"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_separate(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_separate"
                            ]
                        )
                    )
                )
            if not args.vertical:
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
                print(
                    "nabla4_interpolate_benchmark_structured_gpu_naive_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_structured_gpu_naive_separate"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_separate"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_structured_gpu_naive_separate(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_separate"
                            ]
                        )
                    )
                )

        if args.combination in ["all", "inlined"]:
            if not args.vertical:
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined"
                            ]
                        )
                    )
                )
            if not args.vertical:
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v"
                            ]
                        )
                    )
                )
                runtimes[
                    "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v_general"
                ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v_general(
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v_general median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v_general"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v"
                            ]
                        )
                    )
                )
            if not args.vertical:
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
                print(
                    "nabla4_interpolate_benchmark_structured_gpu_naive_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_structured_gpu_naive_inlined"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined"
                            ]
                        )
                    )
                )
            if not args.vertical:
                runtimes[
                    "nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached"
                ] = icon_benchmark.nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached(
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
                print(
                    "nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined_cached"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined_cached(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined_cached median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined_cached"
                            ]
                        )
                    )
                )

    if args.backend in ["all_gpu", "gpu_kloop"]:
        if args.combination in ["all", "separate"]:
            if not args.vertical:
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_separate"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_separate(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_separate"
                            ]
                        )
                    )
                )
            if not args.vertical:
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
                print(
                    "nabla4_interpolate_benchmark_structured_gpu_kloop_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_structured_gpu_kloop_separate"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_separate"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_separate(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_separate median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_separate"
                            ]
                        )
                    )
                )

        if args.combination in ["all", "inlined"]:
            if not args.vertical:
                runtimes[
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined"
                ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined(
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined"
                            ]
                        )
                    )
                )
                runtimes[
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_cached"
                ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_cached(
                    filtered_e2c2v_inlined,
                    filtered_e2ecv_inlined,
                    filtered_v2e_inlined,
                    torus_grid.num_cells,
                    torus_grid.num_vertices,
                    torus_grid.num_edges,
                    torus_grid.num_levels,
                    torus_grid.size[E2C2VDim],
                    grid_cartesian_dimensions[0],
                    repetitions,
                    dry_runs,
                )
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_cached median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_cached"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined"
                            ]
                        )
                    )
                )
            if not args.vertical:
                runtimes[
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v"
                ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v(
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v"
                            ]
                        )
                    )
                )
                runtimes[
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v_general"
                ] = icon_benchmark.nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v_general(
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
                print(
                    "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v_general median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v_general"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v"
                            ]
                        )
                    )
                )
            if not args.vertical:
                runtimes["nabla4_interpolate_benchmark_gpu_kloop_roofline"] = (
                    icon_benchmark.nabla4_interpolate_benchmark_gpu_kloop_roofline(
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
                )
                print(
                    "nabla4_interpolate_benchmark_gpu_kloop_roofline median: {}".format(
                        np.median(
                            runtimes["nabla4_interpolate_benchmark_gpu_kloop_roofline"]
                        )
                    )
                )
                runtimes[
                    "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined"
                ] = icon_benchmark.nabla4_interpolate_benchmark_structured_gpu_kloop_inlined(
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
                print(
                    "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined"
                            ]
                        )
                    )
                )
            else:
                runtimes["nabla4_vertical_interpolate_benchmark_gpu_kloop_roofline"] = (
                    icon_benchmark.nabla4_vertical_interpolate_benchmark_gpu_kloop_roofline(
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
                )
                print(
                    "nabla4_vertical_interpolate_benchmark_gpu_kloop_roofline median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_gpu_kloop_roofline"
                            ]
                        )
                    )
                )
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined"
                            ]
                        )
                    )
                )
            if not args.vertical:
                runtimes[
                    "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached"
                ] = icon_benchmark.nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached(
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
                print(
                    "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached"
                            ]
                        )
                    )
                )
            else:
                runtimes[
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined_cached"
                ] = icon_benchmark.nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined_cached(
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
                print(
                    "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined_cached median: {}".format(
                        np.median(
                            runtimes[
                                "nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined_cached"
                            ]
                        )
                    )
                )

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

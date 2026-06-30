import argparse
import numpy as np
from os import path
from pathlib import Path

from icon4py.model.common.grid.grid_manager import GridManager  # type: ignore [import-not-found]
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
from icon4py.model.common.grid.gridfile import (  # type: ignore [import-not-found]
    NoTransformation,
    ToZeroBasedIndexTransformation,
)

from icon4py.model.common.dimension import E2C2VDim  # type: ignore [import-not-found]

import icon_benchmark  # type: ignore [import-not-found]

import nabla4_gtfn  # type: ignore [import-not-found]

import netCDF4  # type: ignore [import-not-found]

from json import dump


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
    transformation=ToZeroBasedIndexTransformation(),
):
    grid_manager = GridManager(
        grid_file=fname,
        config=VerticalGridConfig(num_levels=num_levels),
        offset_transformation=transformation,
    )
    grid_manager(allocator=None, keep_skip_values=True)
    return grid_manager


def get_torus_grid(filename, num_levels, transformation, e2c2v_ordering="per-vertex"):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    return grid_manager.grid


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


def compare_ndarrays(a, b):
    same = True
    if a.shape != b.shape:
        same = False

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if not np.isclose(a[i][j], b[i][j]):
                print(
                    "Difference at index ({}, {}) is {}".format(i, j, a[i][j] - b[i][j])
                )
                same = False
    if not same:
        print("Arrays are not the same")
        import sys

        sys.exit(1)


def run_gtfn(repetitions, dry_runs, e2c2v, e2ecv, nabla4_data, grid, backend):
    if backend == "gt:gpu":
        import cupy as cp  # type: ignore [import-not-found]

        float_dtype = cp.float64
        int_dtype = cp.int32
    else:
        float_dtype = np.float64
        int_dtype = np.int32

    from gt4py.storage import zeros, from_array  # type: ignore [import-not-found]

    z_nabla4_e2_wp_gtfn = zeros(
        shape=(len(e2c2v), grid.num_levels), dtype=float_dtype, backend=backend
    )

    if backend == "gt:gpu":
        runtime = nabla4_gtfn.calculate_nabla4_gpu(
            repetitions,
            dry_runs,
            (
                from_array(
                    np.array(nabla4_data.u_vert).T, dtype=float_dtype, backend=backend
                ),
                (0, 0),
            ),
            (
                from_array(
                    np.array(nabla4_data.v_vert).T, dtype=float_dtype, backend=backend
                ),
                (0, 0),
            ),
            (
                from_array(
                    np.array(nabla4_data.primal_normal_vert_v1),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (
                from_array(
                    np.array(nabla4_data.primal_normal_vert_v2),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (
                from_array(
                    np.array(nabla4_data.z_nabla2_e).T,
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0, 0),
            ),
            (
                from_array(
                    np.array(nabla4_data.inv_vert_vert_length),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (
                from_array(
                    np.array(nabla4_data.inv_primal_edge_length),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (z_nabla4_e2_wp_gtfn, (0, 0)),
            0,
            len(e2c2v),
            0,
            grid.num_levels,
            (
                from_array(np.array(e2c2v), dtype=int_dtype, backend=backend),
                (0, 0),
            ),
            (
                from_array(np.array(e2ecv), dtype=int_dtype, backend=backend),
                (0, 0),
            ),
        )
    else:
        runtime = nabla4_gtfn.calculate_nabla4_cpu(
            repetitions,
            dry_runs,
            (
                from_array(
                    np.array(nabla4_data.u_vert).T, dtype=float_dtype, backend=backend
                ),
                (0, 0),
            ),
            (
                from_array(
                    np.array(nabla4_data.v_vert).T, dtype=float_dtype, backend=backend
                ),
                (0, 0),
            ),
            (
                from_array(
                    np.array(nabla4_data.primal_normal_vert_v1),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (
                from_array(
                    np.array(nabla4_data.primal_normal_vert_v2),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (
                from_array(
                    np.array(nabla4_data.z_nabla2_e).T,
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0, 0),
            ),
            (
                from_array(
                    np.array(nabla4_data.inv_vert_vert_length),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (
                from_array(
                    np.array(nabla4_data.inv_primal_edge_length),
                    dtype=float_dtype,
                    backend=backend,
                ),
                (0,),
            ),
            (z_nabla4_e2_wp_gtfn, (0, 0)),
            0,
            len(e2c2v),
            0,
            grid.num_levels,
            (
                from_array(np.array(e2c2v), dtype=int_dtype, backend=backend),
                (0, 0),
            ),
            (
                from_array(np.array(e2ecv), dtype=int_dtype, backend=backend),
                (0, 0),
            ),
        )
    return z_nabla4_e2_wp_gtfn, runtime


def run_sanity_checks(
    filtered_e2c2v, filtered_e2ecv, grid, lon_dim, lat_dim, backend="all_cpu", halo=2
):
    print("Generating validation data")
    random_validation_data = icon_benchmark.get_nabla4_benchmark_validation_data(
        filtered_e2c2v,
        filtered_e2ecv,
        grid.num_cells,
        grid.num_vertices,
        grid.num_edges,
        grid.num_levels,
        grid.size[E2C2VDim],
    )
    print("Generated validation data")

    if backend in ["all_cpu", "gtfn_cpu"]:
        print("Running gtfn_cpu sanity check")
        z_nabla4_e2_wp_gtfn, _ = run_gtfn(
            1,
            0,
            filtered_e2c2v,
            filtered_e2ecv,
            random_validation_data,
            grid,
            "gt:cpu_ifirst",
        )
        compare_ndarrays(
            z_nabla4_e2_wp_gtfn, np.array(random_validation_data.z_nabla4_e2_wp).T
        )
        print("gtfn_cpu sanity check passed")

    if backend in ["all_gpu", "gtfn_gpu"]:
        print("Running gtfn_gpu sanity check")
        z_nabla4_e2_wp_gtfn, _ = run_gtfn(
            1, 0, filtered_e2c2v, filtered_e2ecv, random_validation_data, grid, "gt:gpu"
        )
        compare_ndarrays(
            z_nabla4_e2_wp_gtfn, np.array(random_validation_data.z_nabla4_e2_wp).T
        )
        print("gtfn_gpu sanity check passed")

    if backend in ["all_cpu", "cpu_ifirst"]:
        print("Running unstructured cpu_ifirst sanity check")
        z_nabla4_e2_comp_unstructured_cpu_ifirst_gridtools = (
            icon_benchmark.nabla4_validate_unstructured_cpu_ifirst_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_unstructured_cpu_ifirst_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("unstructured cpu_ifirst sanity check passed")

    if backend in ["all_cpu", "cpu_kfirst"]:
        print("Running unstructured cpu_kfirst sanity check")
        z_nabla4_e2_comp_unstructured_cpu_kfirst_gridtools = (
            icon_benchmark.nabla4_validate_unstructured_cpu_kfirst_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_unstructured_cpu_kfirst_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("unstructured cpu_kfirst sanity check passed")

    if backend in ["all_gpu", "gpu_kloop"]:
        print("Running unstructured gpu kloop sanity check")
        z_nabla4_e2_comp_unstructured_gpu_kloop_gridtools = (
            icon_benchmark.nabla4_validate_unstructured_gpu_kloop_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_unstructured_gpu_kloop_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("unstructured gpu kloop sanity check passed")

        print("Running unstructured gpu kloop vertical sanity check")
        z_nabla4_e2_comp_unstructured_gpu_kloop_vertical_gridtools = (
            icon_benchmark.nabla4_vertical_validate_unstructured_gpu_kloop_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_unstructured_gpu_kloop_vertical_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("unstructured gpu kloop sanity check passed")

    if backend in ["all_gpu", "gpu_naive"]:
        print("Running unstructured gpu_naive sanity check")
        z_nabla4_e2_comp_unstructured_gpu_naive_gridtools = (
            icon_benchmark.nabla4_validate_unstructured_gpu_naive_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_unstructured_gpu_naive_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("unstructured gpu_naive sanity check passed")

        print("Running unstructured gpu_naive vertical sanity check")
        z_nabla4_e2_comp_unstructured_gpu_naive_vertical_gridtools = (
            icon_benchmark.nabla4_vertical_validate_unstructured_gpu_naive_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_unstructured_gpu_naive_vertical_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("unstructured gpu_naive vertical sanity check passed")

    if backend in ["all_cpu", "cpu_ifirst"]:
        print("Running structured cpu_ifirst sanity check")
        z_nabla4_e2_comp_structured_cpu_ifirst_gridtools = (
            icon_benchmark.nabla4_validate_structured_torus_cpu_ifirst_gridtools_halo(
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_cpu_ifirst_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured cpu_ifirst sanity check passed")

    if backend in ["all_cpu", "cpu_kfirst"]:
        print("Running structured cpu_kfirst sanity check")
        z_nabla4_e2_comp_structured_cpu_kfirst_gridtools = (
            icon_benchmark.nabla4_validate_structured_torus_cpu_kfirst_gridtools_halo(
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_cpu_kfirst_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured cpu_kfirst sanity check passed")

    if backend in ["all_gpu", "gpu_kloop"]:
        print("Running structured gpu kloop sanity check")
        z_nabla4_e2_comp_structured_torus_gpu_kloop_gridtools_halo = (
            icon_benchmark.nabla4_validate_structured_torus_gpu_kloop_gridtools_halo(
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_torus_gpu_kloop_gridtools_halo,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured gpu kloop sanity check passed")

        print("Running structured gpu kloop vertical sanity check")
        z_nabla4_e2_comp_structured_torus_gpu_kloop_vertical_gridtools_halo = icon_benchmark.nabla4_vertical_validate_structured_torus_gpu_kloop_gridtools_halo(
            random_validation_data.CellDim,
            random_validation_data.VertexDim,
            random_validation_data.EdgeDim,
            random_validation_data.KDim,
            random_validation_data.ECVDim,
            lon_dim,
            lat_dim,
            halo,
            np.array(random_validation_data.u_vert).T,
            np.array(random_validation_data.v_vert).T,
            random_validation_data.primal_normal_vert_v1,
            random_validation_data.primal_normal_vert_v2,
            np.array(random_validation_data.z_nabla2_e).T,
            random_validation_data.inv_vert_vert_length,
            random_validation_data.inv_primal_edge_length,
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_torus_gpu_kloop_vertical_gridtools_halo,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured gpu kloop vertical sanity check passed")

        print("Running structured gpu kloop cutile sanity check")
        z_nabla4_e2_comp_structured_torus_gpu_kloop_cutile_halo = (
            icon_benchmark.nabla4_validate_structured_torus_gpu_kloop_cutile_halo(
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_torus_gpu_kloop_cutile_halo,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured gpu kloop cutile sanity check passed")

    if backend in ["all_gpu", "gpu_naive"]:
        print("Running structured gpu_naive sanity check")
        z_nabla4_e2_comp_structured_torus_gpu_naive_gridtools_halo = (
            icon_benchmark.nabla4_validate_structured_torus_gpu_naive_gridtools_halo(
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_torus_gpu_naive_gridtools_halo,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured gpu_naive sanity check passed")

        print("Running structured gpu_naive vertical sanity check")
        z_nabla4_e2_comp_structured_torus_gpu_naive_vertical_gridtools_halo = icon_benchmark.nabla4_vertical_validate_structured_torus_gpu_naive_gridtools_halo(
            random_validation_data.CellDim,
            random_validation_data.VertexDim,
            random_validation_data.EdgeDim,
            random_validation_data.KDim,
            random_validation_data.ECVDim,
            lon_dim,
            lat_dim,
            halo,
            np.array(random_validation_data.u_vert).T,
            np.array(random_validation_data.v_vert).T,
            random_validation_data.primal_normal_vert_v1,
            random_validation_data.primal_normal_vert_v2,
            np.array(random_validation_data.z_nabla2_e).T,
            random_validation_data.inv_vert_vert_length,
            random_validation_data.inv_primal_edge_length,
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_torus_gpu_naive_vertical_gridtools_halo,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured gpu_naive vertical sanity check passed")
        print("Running structured gpu naive cutile sanity check")
        z_nabla4_e2_comp_structured_torus_gpu_naive_cutile_halo = (
            icon_benchmark.nabla4_validate_structured_torus_gpu_naive_cutile_halo(
                random_validation_data.CellDim,
                random_validation_data.VertexDim,
                random_validation_data.EdgeDim,
                random_validation_data.KDim,
                random_validation_data.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data.u_vert).T,
                np.array(random_validation_data.v_vert).T,
                random_validation_data.primal_normal_vert_v1,
                random_validation_data.primal_normal_vert_v2,
                np.array(random_validation_data.z_nabla2_e).T,
                random_validation_data.inv_vert_vert_length,
                random_validation_data.inv_primal_edge_length,
            )
        )
        assert np.allclose(
            z_nabla4_e2_comp_structured_torus_gpu_naive_cutile_halo,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured gpu naive cutile sanity check passed")

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
        args.grid, args.klevels, transformation, args.e2c2v_ordering
    )

    repetitions = args.repetitions
    dry_runs = 10 if args.dry_run else 0

    grid_cartesian_dimensions = get_torus_cartesian_dimensions(args.grid)

    filtered_e2c2v = filter_edge_vector(
        torus_grid.get_connectivity("E2C2V").ndarray,
        grid_cartesian_dimensions,
        args.e2c2v_ordering,
        args.halo,
    )

    original_e2c2v = torus_grid.get_connectivity("E2C2V").ndarray
    e2ecv = []
    for i in range(len(original_e2c2v)):
        e2ecv.append(original_e2c2v[i])

    def _get_gpu_coalesced_permuted_e2ecv():
        orientation_permuted_e2ecv = np.zeros_like(
            e2ecv
        )
        edges_size = len(torus_grid.get_connectivity("E2C2V").ndarray)
        for i in range(edges_size):
            for j in range(4):
                orientation_permuted_e2ecv[i][j] = j * edges_size + i
        return orientation_permuted_e2ecv

    permuted_e2ecv = (
        e2ecv
        if args.e2c2v_ordering == "per-vertex"
        else _get_gpu_coalesced_permuted_e2ecv()
    )
    filtered_e2ecv = filter_edge_vector(
        permuted_e2ecv,
        grid_cartesian_dimensions,
        args.e2c2v_ordering,
        args.halo,
    )

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

    if args.sanity_checks:
        run_sanity_checks(
            filtered_e2c2v,
            filtered_e2ecv,
            torus_grid,
            grid_cartesian_dimensions[0],
            grid_cartesian_dimensions[1],
            args.backend,
            args.halo,
        )

    runtimes = {}

    halo = args.halo

    if args.backend in ["all_cpu", "gtfn_cpu"]:
        random_validation_data_gtfn = (
            icon_benchmark.get_nabla4_benchmark_validation_data(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
            )
        )

        _, runtimes_gtfn = run_gtfn(
            repetitions,
            dry_runs,
            filtered_e2c2v,
            filtered_e2ecv,
            random_validation_data_gtfn,
            torus_grid,
            "gt:cpu_ifirst",
        )

        runtimes["nabla4_benchmark_unstructured_gtfn_cpu"] = runtimes_gtfn

    if args.backend in ["all_gpu", "gtfn_gpu"]:
        random_validation_data_gtfn = (
            icon_benchmark.get_nabla4_benchmark_validation_data(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
            )
        )

        _, runtimes_gtfn = run_gtfn(
            repetitions,
            dry_runs,
            filtered_e2c2v,
            filtered_e2ecv,
            random_validation_data_gtfn,
            torus_grid,
            "gt:gpu",
        )
        runtimes["nabla4_benchmark_unstructured_gtfn_gpu"] = runtimes_gtfn

    if args.backend in ["all_cpu", "cpu_ifirst"]:
        runtimes["nabla4_benchmark_unstructured_cpu_ifirst_gridtools"] = (
            icon_benchmark.nabla4_benchmark_unstructured_cpu_ifirst_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
        )

        runtimes["nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo"] = (
            icon_benchmark.nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo(
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

    if args.backend in ["all_cpu", "cpu_kfirst"]:
        runtimes["nabla4_benchmark_unstructured_cpu_kfirst_gridtools"] = (
            icon_benchmark.nabla4_benchmark_unstructured_cpu_kfirst_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
        )

        runtimes["nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo"] = (
            icon_benchmark.nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo(
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

    if args.backend in ["all_gpu", "gpu_kloop"]:
        runtimes["nabla4_benchmark_unstructured_gpu_kloop_gridtools"] = (
            icon_benchmark.nabla4_benchmark_unstructured_gpu_kloop_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
        )
        runtimes["nabla4_vertical_benchmark_unstructured_gpu_kloop_gridtools"] = (
            icon_benchmark.nabla4_vertical_benchmark_unstructured_gpu_kloop_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
        )
        runtimes["nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo"] = (
            icon_benchmark.nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo(
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
        runtimes[
            "nabla4_vertical_benchmark_structured_torus_gpu_kloop_gridtools_halo"
        ] = icon_benchmark.nabla4_vertical_benchmark_structured_torus_gpu_kloop_gridtools_halo(
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
        runtimes["nabla4_benchmark_structured_torus_gpu_kloop_cutile_halo"] = (
            icon_benchmark.nabla4_benchmark_structured_torus_gpu_kloop_cutile_halo(
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

    if args.backend in ["all_gpu", "gpu_naive"]:
        runtimes["nabla4_benchmark_unstructured_gpu_naive_gridtools"] = (
            icon_benchmark.nabla4_benchmark_unstructured_gpu_naive_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
        )
        runtimes["nabla4_vertical_benchmark_unstructured_gpu_naive_gridtools"] = (
            icon_benchmark.nabla4_vertical_benchmark_unstructured_gpu_naive_gridtools(
                filtered_e2c2v,
                filtered_e2ecv,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
        )
        runtimes["nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo"] = (
            icon_benchmark.nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo(
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
        runtimes[
            "nabla4_vertical_benchmark_structured_torus_gpu_naive_gridtools_halo"
        ] = icon_benchmark.nabla4_vertical_benchmark_structured_torus_gpu_naive_gridtools_halo(
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
        runtimes["nabla4_benchmark_structured_torus_gpu_naive_cutile_halo"] = (
            icon_benchmark.nabla4_benchmark_structured_torus_gpu_naive_cutile_halo(
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

    print_median_runtimes(runtimes)

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

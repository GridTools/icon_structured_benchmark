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


def generate_v2e(x_dim, y_dim, internal_halo=1):
    y_dim_internal = y_dim - 2 * internal_halo
    x_dim_internal = x_dim - 2 * internal_halo
    v2e = np.zeros((y_dim_internal * x_dim_internal, 6), dtype=np.int32)
    for i in range(internal_halo, x_dim - internal_halo):
        for j in range(internal_halo, y_dim - internal_halo):
            i_internal = i - internal_halo
            j_internal = j - internal_halo
            i_j = i + j * x_dim
            i_jm1 = i + (j - 1) * x_dim
            v2e[j_internal * x_dim_internal + i_internal][0] = (x_dim * y_dim) + i_j - 1
            v2e[j_internal * x_dim_internal + i_internal][1] = (x_dim * y_dim) + i_j
            v2e[j_internal * x_dim_internal + i_internal][2] = i_jm1
            v2e[j_internal * x_dim_internal + i_internal][3] = i_j
            v2e[j_internal * x_dim_internal + i_internal][4] = 2 * (x_dim * y_dim) + i_j
            v2e[j_internal * x_dim_internal + i_internal][5] = (
                2 * (x_dim * y_dim) + i_j + x_dim - 1
            )
    return v2e


def filter_v2e_per_orientation(v2e_table, x_dim, y_dim):
    e2e_table = []
    for i in range(x_dim * y_dim):
        e2e_table.append(i * 3)
        e2e_table.append(i * 3 + 1)
        e2e_table.append(
            i * 3 + 3 * x_dim + 2
            if i * 3 + 2 + 3 * x_dim < x_dim * y_dim * 3
            else i * 3 + 3 * x_dim + 2 - x_dim * y_dim * 3
        )

    for edges in v2e_table:
        for edge_id in range(6):
            edges[edge_id] = e2e_table[edges[edge_id]]

    e2e_table_orientation_sorting = []
    for i in range(0, x_dim * y_dim * 3):
        e2e_table_orientation_sorting.append(i // 3 + (i % 3) * x_dim * y_dim)

    for edges in v2e_table:
        for edge_id in range(6):
            edges[edge_id] = e2e_table_orientation_sorting[edges[edge_id]]
    return v2e_table


def halo_filter(v2e_table, halo, x_dim, y_dim):
    filtered_v2e_table = []
    for i in range(x_dim):
        for j in range(y_dim):
            if i >= halo and i < x_dim - halo and j >= halo and j < y_dim - halo:
                vertex_id = i + j * x_dim
                filtered_v2e_table.append(v2e_table[vertex_id])
    filtered_v2e_table = np.array(filtered_v2e_table)
    assert filtered_v2e_table.shape == ((x_dim - 2 * halo) * (y_dim - 2 * halo), 6)
    return filtered_v2e_table


def transpose_ij(v2e_table, x_dim, y_dim):
    transposed_v2e_table = []
    for j in range(y_dim):
        for i in range(x_dim):
            vertex_id = i * y_dim + j
            transposed_v2e_table.append(v2e_table[vertex_id])
    transposed_v2e_table = np.array(transposed_v2e_table)
    assert transposed_v2e_table.shape == (x_dim * y_dim, 6)
    return transposed_v2e_table


def process_v2e_per_orientation(v2e_table, x_dim, y_dim, halo=1):
    return transpose_ij(
        halo_filter(
            filter_v2e_per_orientation(
                v2e_table,
                x_dim,
                y_dim,
            ),
            halo,
            x_dim,
            y_dim,
        ),
        x_dim - 2 * halo,
        y_dim - 2 * halo,
    )


def filter_v2e_per_vertex(v2e_table, x_dim, y_dim):
    e2e_table = []
    for i in range(x_dim * y_dim):
        e2e_table.append(i * 3)
        e2e_table.append(i * 3 + 1)
        e2e_table.append(
            i * 3 + 3 * x_dim + 2
            if i * 3 + 2 + 3 * x_dim < x_dim * y_dim * 3
            else i * 3 + 3 * x_dim + 2 - x_dim * y_dim * 3
        )

    for edges in v2e_table:
        for edge_id in range(6):
            edges[edge_id] = e2e_table[edges[edge_id]]

    return v2e_table


def process_v2e_per_vertex(v2e_table, x_dim, y_dim, halo=1):
    return transpose_ij(
        halo_filter(
            filter_v2e_per_vertex(
                v2e_table,
                x_dim,
                y_dim,
            ),
            halo,
            x_dim,
            y_dim,
        ),
        x_dim - 2 * halo,
        y_dim - 2 * halo,
    )


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


def run_sanity_checks(
    v2e, nvertices, nedges, nlevels, lon_dim, lat_dim, backend="all_cpu", halo=2
):
    print("Generating validation data")
    p_e_in = np.random.rand(nedges, nlevels)
    ptr_coeff_1 = np.random.rand(nedges, 6)
    ptr_coeff_2 = np.random.rand(nedges, 6)
    (
        p_u_out_ref,
        p_v_out_ref,
    ) = icon_benchmark.interpolate_validate_unstructured_cpu_ifirst(
        nvertices,
        nedges,
        nlevels,
        v2e,
        p_e_in,
        ptr_coeff_1,
        ptr_coeff_2,
    )
    print("Generated validation data")

    if backend in ["all_cpu", "cpu_kfirst"]:
        print("Running unstructured cpu_kfirst sanity check")
        (
            p_u_out_cpu_kfirst,
            p_v_out_cpu_kfirst,
        ) = icon_benchmark.interpolate_validate_unstructured_cpu_kfirst(
            nvertices,
            nedges,
            nlevels,
            v2e,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_cpu_kfirst, p_u_out_ref)
        assert np.allclose(p_v_out_cpu_kfirst, p_v_out_ref)
        print("unstructured cpu_kfirst sanity check passed")

    if backend in ["all_gpu", "gpu_kloop"]:
        print("Running unstructured gpu naive sanity check")
        (
            p_u_out_gpu_naive,
            p_v_out_gpu_naive,
        ) = icon_benchmark.interpolate_validate_unstructured_gpu_naive(
            nvertices,
            nedges,
            nlevels,
            v2e,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_gpu_naive, p_u_out_ref)
        assert np.allclose(p_v_out_gpu_naive, p_v_out_ref)
        print("unstructured gpu naive sanity check passed")

        print("Running unstructured gpu kloop sanity check")
        (
            p_u_out_gpu_kloop,
            p_v_out_gpu_kloop,
        ) = icon_benchmark.interpolate_validate_unstructured_gpu_kloop(
            nvertices,
            nedges,
            nlevels,
            v2e,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_gpu_kloop, p_u_out_ref)
        assert np.allclose(p_v_out_gpu_kloop, p_v_out_ref)
        print("unstructured gpu kloop sanity check passed")

    if backend in ["all_cpu", "cpu_ifirst"]:
        print("Running structured cpu_ifirst sanity check")
        (
            p_u_out_cpu_ifirst,
            p_v_out_cpu_ifirst,
        ) = icon_benchmark.interpolate_validate_structured_cpu_ifirst(
            nvertices,
            nedges,
            nlevels,
            lon_dim,
            lat_dim,
            halo,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_cpu_ifirst, p_u_out_ref)
        assert np.allclose(p_v_out_cpu_ifirst, p_v_out_ref)
        print("structured cpu_ifirst sanity check passed")

    if backend in ["all_cpu", "cpu_kfirst"]:
        print("Running structured cpu_kfirst sanity check")
        (
            p_u_out_cpu_kfirst,
            p_v_out_cpu_kfirst,
        ) = icon_benchmark.interpolate_validate_structured_cpu_kfirst(
            nvertices,
            nedges,
            nlevels,
            lon_dim,
            lat_dim,
            halo,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_cpu_kfirst, p_u_out_ref)
        assert np.allclose(p_v_out_cpu_kfirst, p_v_out_ref)
        print("structured cpu_ifirst sanity check passed")

    if backend in ["all_gpu", "gpu_kloop"]:
        print("Running structured gpu naive sanity check")
        (
            p_u_out_gpu_naive,
            p_v_out_gpu_naive,
        ) = icon_benchmark.interpolate_validate_structured_gpu_naive(
            nvertices,
            nedges,
            nlevels,
            lon_dim,
            lat_dim,
            halo,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_gpu_naive, p_u_out_ref)
        assert np.allclose(p_v_out_gpu_naive, p_v_out_ref)
        print("structured gpu naive sanity check passed")

        print("Running structured gpu kloop sanity check")
        (
            p_u_out_gpu_kloop,
            p_v_out_gpu_kloop,
        ) = icon_benchmark.interpolate_validate_structured_gpu_kloop(
            nvertices,
            nedges,
            nlevels,
            lon_dim,
            lat_dim,
            halo,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_gpu_kloop, p_u_out_ref)
        assert np.allclose(p_v_out_gpu_kloop, p_v_out_ref)
        print("structured gpu kloop sanity check passed")

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
        default=1,
        help="Halo size for structured grids (default 1) [Shouldn't be changed]",
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

    v2e_filtered = (
        process_v2e_per_orientation(
            torus_grid.get_connectivity("V2E").ndarray,
            grid_cartesian_dimensions[1],
            grid_cartesian_dimensions[0],
            args.halo,
        )
        if args.e2c2v_ordering == "per-orientation"
        else process_v2e_per_vertex(
            torus_grid.get_connectivity("V2E").ndarray,
            grid_cartesian_dimensions[1],
            grid_cartesian_dimensions[0],
            args.halo,
        )
    )
    v2e_generated = generate_v2e(
        grid_cartesian_dimensions[1], grid_cartesian_dimensions[0], args.halo
    )

    if args.e2c2v_ordering == "per-orientation":
        assert np.allclose(
            v2e_filtered, v2e_generated
        ), "Filtered V2E table is incorrect"

    if args.sanity_checks:
        run_sanity_checks(
            v2e_filtered,
            torus_grid.num_vertices,
            torus_grid.num_edges,
            torus_grid.num_levels,
            grid_cartesian_dimensions[0],
            grid_cartesian_dimensions[1],
            args.backend,
            args.halo,
        )

    runtimes = {}

    halo = args.halo

    if args.backend in ["all_cpu", "cpu_ifirst"]:
        runtimes["interpolate_benchmark_unstructured_cpu_ifirst"] = (
            icon_benchmark.interpolate_benchmark_unstructured_cpu_ifirst(
                v2e_filtered,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                repetitions,
                dry_runs,
            )
        )

        runtimes["interpolate_benchmark_structured_cpu_ifirst"] = (
            icon_benchmark.interpolate_benchmark_structured_cpu_ifirst(
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )
        )

    if args.backend in ["all_cpu", "cpu_kfirst"]:
        runtimes["interpolate_benchmark_unstructured_cpu_kfirst"] = (
            icon_benchmark.interpolate_benchmark_unstructured_cpu_kfirst(
                v2e_filtered,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                repetitions,
                dry_runs,
            )
        )

        runtimes["interpolate_benchmark_structured_cpu_kfirst"] = (
            icon_benchmark.interpolate_benchmark_structured_cpu_kfirst(
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )
        )

    if args.backend in ["all_gpu", "gpu_kloop"]:
        runtimes["interpolate_benchmark_unstructured_gpu_naive"] = (
            icon_benchmark.interpolate_benchmark_unstructured_gpu_naive(
                v2e_filtered,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                repetitions,
                dry_runs,
            )
        )
        runtimes["interpolate_benchmark_unstructured_gpu_kloop"] = (
            icon_benchmark.interpolate_benchmark_unstructured_gpu_kloop(
                v2e_filtered,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                repetitions,
                dry_runs,
            )
        )
        runtimes["interpolate_benchmark_structured_gpu_naive"] = (
            icon_benchmark.interpolate_benchmark_structured_gpu_naive(
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                halo,
                repetitions,
                dry_runs,
            )
        )
        runtimes["interpolate_benchmark_structured_gpu_kloop"] = (
            icon_benchmark.interpolate_benchmark_structured_gpu_kloop(
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
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

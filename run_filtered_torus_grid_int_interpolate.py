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
    v2e, nvertices, nedges, nlevels, lon_dim, lat_dim, backend="all_cpu", halo=2
):
    print("Generating validation data")
    p_e_in = np.random.rand(nedges, nlevels)
    ptr_coeff_1 = np.random.rand(nvertices, 6)
    ptr_coeff_2 = np.random.rand(nvertices, 6)
    p_u_out_ref, p_v_out_ref = icon_benchmark.interpolate_validate_unstructured_cpu_ifirst(
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
        p_u_out_cpu_kfirst, p_v_out_cpu_kfirst = icon_benchmark.interpolate_validate_unstructured_cpu_kfirst(
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

    if backend in ["all_gpu", "gpu"]:
        print("Running unstructured gpu sanity check")
        p_u_out_gpu, p_v_out_gpu= icon_benchmark.interpolate_validate_unstructured_gpu(
            nvertices,
            nedges,
            nlevels,
            v2e,
            p_e_in,
            ptr_coeff_1,
            ptr_coeff_2,
        )
        assert np.allclose(p_u_out_gpu, p_u_out_ref)
        assert np.allclose(p_v_out_gpu, p_v_out_ref)
        print("unstructured gpu sanity check passed")

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
            "gpu",
            "gpu_naive",
        ],
        default="all_cpu",
        help="Which backend to benchmark",
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

    if args.sanity_checks:
        run_sanity_checks(
            torus_grid.get_offset_provider("V2E").table,
            torus_grid.num_vertices,
            torus_grid.num_edges,
            torus_grid.num_levels,
            grid_cartesian_dimensions[0],
            grid_cartesian_dimensions[1],
            args.backend,
            args.halo,
        )
    quit()
    runtimes = {}

    halo = args.halo

    if args.backend in ["all_cpu", "cpu_ifirst"]:
        runtimes[
            "nabla4_benchmark_unstructured_cpu_ifirst_gridtools"
        ] = icon_benchmark.nabla4_benchmark_unstructured_cpu_ifirst_gridtools(
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

        runtimes[
            "nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo"
        ] = icon_benchmark.nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo(
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
        runtimes[
            "nabla4_benchmark_unstructured_cpu_kfirst_gridtools"
        ] = icon_benchmark.nabla4_benchmark_unstructured_cpu_kfirst_gridtools(
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

        runtimes[
            "nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo"
        ] = icon_benchmark.nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo(
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

    if args.backend in ["all_gpu", "gpu"]:
        runtimes[
            "nabla4_benchmark_unstructured_gpu_gridtools"
        ] = icon_benchmark.nabla4_benchmark_unstructured_gpu_gridtools(
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
        runtimes[
            "nabla4_benchmark_structured_torus_gpu_gridtools_halo"
        ] = icon_benchmark.nabla4_benchmark_structured_torus_gpu_gridtools_halo(
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
        runtimes[
            "nabla4_benchmark_unstructured_gpu_gridtools_naive"
        ] = icon_benchmark.nabla4_benchmark_unstructured_gpu_gridtools_naive(
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
        runtimes[
            "nabla4_benchmark_structured_torus_gpu_gridtools_halo_naive"
        ] = icon_benchmark.nabla4_benchmark_structured_torus_gpu_gridtools_halo_naive(
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

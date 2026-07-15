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
    filter_c2v_vector,
    get_torus_cartesian_dimensions,
    get_torus_grid,
    print_median_runtimes,
)


def run_sanity_checks(
    c2v, nvertices, ncells, nlevels, lon_dim, lat_dim, backend="all_cpu", halo=4
):
    print("Generating validation data")
    p_vert_u_in = np.random.rand(nvertices, nlevels)
    p_vert_v_in = np.random.rand(nvertices, nlevels)
    ptr_coeff_1 = np.random.rand(ncells, 3)
    ptr_coeff_2 = np.random.rand(ncells, 3)
    p_cell_out_ref = icon_benchmark.verts2cells_validate_unstructured_cpu_kfirst(
        nvertices,
        ncells,
        nlevels,
        c2v,
        p_vert_u_in,
        p_vert_v_in,
        ptr_coeff_1,
        ptr_coeff_2,
    )
    print("Generated validation data")

    if backend in ["all_cpu", "cpu_kfirst"]:
        print("Running structured cpu_kfirst sanity check")
        p_cell_out_structured_cpu_kfirst = (
            icon_benchmark.verts2cells_validate_structured_cpu_kfirst(
                nvertices,
                ncells,
                nlevels,
                lon_dim,
                lat_dim,
                halo,
                p_vert_u_in,
                p_vert_v_in,
                ptr_coeff_1,
                ptr_coeff_2,
            )
        )
        assert np.allclose(p_cell_out_structured_cpu_kfirst, p_cell_out_ref)
        print("structured cpu_kfirst sanity check passed")

    if backend in ["all_gpu", "gpu_naive"]:
        print("Running unstructured gpu naive sanity check")
        p_cell_out_gpu_naive = (
            icon_benchmark.verts2cells_validate_unstructured_gpu_naive(
                nvertices,
                ncells,
                nlevels,
                c2v,
                p_vert_u_in,
                p_vert_v_in,
                ptr_coeff_1,
                ptr_coeff_2,
            )
        )
        assert np.allclose(p_cell_out_gpu_naive, p_cell_out_ref)
        print("unstructured gpu naive sanity check passed")

        print("Running structured gpu_naive sanity check")
        p_cell_out_structured_gpu_naive = (
            icon_benchmark.verts2cells_validate_structured_gpu_naive(
                nvertices,
                ncells,
                nlevels,
                lon_dim,
                lat_dim,
                halo,
                p_vert_u_in,
                p_vert_v_in,
                ptr_coeff_1,
                ptr_coeff_2,
            )
        )
        assert np.allclose(p_cell_out_structured_gpu_naive, p_cell_out_ref)
        print("structured gpu_naive sanity check passed")

    if backend in ["all_gpu", "gpu_kloop"]:
        print("Running unstructured gpu kloop sanity check")
        p_cell_out_gpu_kloop = (
            icon_benchmark.verts2cells_validate_unstructured_gpu_kloop(
                nvertices,
                ncells,
                nlevels,
                c2v,
                p_vert_u_in,
                p_vert_v_in,
                ptr_coeff_1,
                ptr_coeff_2,
            )
        )
        assert np.allclose(p_cell_out_gpu_kloop, p_cell_out_ref)
        print("unstructured gpu kloop sanity check passed")

        print("Running structured gpu_kloop sanity check")
        p_cell_out_structured_gpu_kloop = (
            icon_benchmark.verts2cells_validate_structured_gpu_kloop(
                nvertices,
                ncells,
                nlevels,
                lon_dim,
                lat_dim,
                halo,
                p_vert_u_in,
                p_vert_v_in,
                ptr_coeff_1,
                ptr_coeff_2,
            )
        )
        assert np.allclose(p_cell_out_structured_gpu_kloop, p_cell_out_ref)
        print("structured gpu_kloop sanity check passed")

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
        default=4,
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

    original_c2v = torus_grid.get_connectivity("C2V").ndarray
    filtered_c2v = filter_c2v_vector(original_c2v, grid_cartesian_dimensions, args.halo)

    if args.sanity_checks:
        run_sanity_checks(
            filtered_c2v,
            torus_grid.num_vertices,
            torus_grid.num_cells,
            torus_grid.num_levels,
            grid_cartesian_dimensions[0],
            grid_cartesian_dimensions[1],
            args.backend,
            args.halo,
        )

    if repetitions == 0:
        print("Repetitions is 0, skipping benchmarks")
        return

    runtimes = {}

    if args.backend in ["all_cpu", "cpu_kfirst"]:
        runtimes["verts2cells_benchmark_unstructured_cpu_kfirst"] = (
            icon_benchmark.verts2cells_benchmark_unstructured_cpu_kfirst(
                filtered_c2v,
                torus_grid.num_vertices,
                torus_grid.num_cells,
                torus_grid.num_levels,
                repetitions,
                dry_runs,
            )
        )
        runtimes["verts2cells_benchmark_structured_cpu_kfirst"] = (
            icon_benchmark.verts2cells_benchmark_structured_cpu_kfirst(
                torus_grid.num_vertices,
                torus_grid.num_cells,
                torus_grid.num_levels,
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                args.halo,
                repetitions,
                dry_runs,
            )
        )

    if args.backend in ["all_gpu", "gpu_naive"]:
        runtimes["verts2cells_benchmark_unstructured_gpu_naive"] = (
            icon_benchmark.verts2cells_benchmark_unstructured_gpu_naive(
                filtered_c2v,
                torus_grid.num_vertices,
                torus_grid.num_cells,
                torus_grid.num_levels,
                repetitions,
                dry_runs,
            )
        )
        runtimes["verts2cells_benchmark_structured_gpu_naive"] = (
            icon_benchmark.verts2cells_benchmark_structured_gpu_naive(
                torus_grid.num_vertices,
                torus_grid.num_cells,
                torus_grid.num_levels,
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                args.halo,
                repetitions,
                dry_runs,
            )
        )

    if args.backend in ["all_gpu", "gpu_kloop"]:
        runtimes["verts2cells_benchmark_unstructured_gpu_kloop"] = (
            icon_benchmark.verts2cells_benchmark_unstructured_gpu_kloop(
                filtered_c2v,
                torus_grid.num_vertices,
                torus_grid.num_cells,
                torus_grid.num_levels,
                repetitions,
                dry_runs,
            )
        )
        runtimes["verts2cells_benchmark_structured_gpu_kloop"] = (
            icon_benchmark.verts2cells_benchmark_structured_gpu_kloop(
                torus_grid.num_vertices,
                torus_grid.num_cells,
                torus_grid.num_levels,
                grid_cartesian_dimensions[0],
                grid_cartesian_dimensions[1],
                args.halo,
                repetitions,
                dry_runs,
            )
        )

    print_median_runtimes(runtimes)

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

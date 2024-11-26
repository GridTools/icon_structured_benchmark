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

def run_sanity_checks(
    c2v, nvertices, ncells, nlevels, lon_dim, lat_dim, backend="all_cpu", halo=4
):
    print("Generating validation data")
    p_vert_in = np.random.rand(nvertices, nlevels)
    ptr_coeff = np.random.rand(ncells, 3)
    p_cell_out = icon_benchmark.verts2cells_validate_unstructured_cpu_kfirst(
        nvertices,
        ncells,
        nlevels,
        c2v,
        p_vert_in,
        ptr_coeff,
    )
    print("Generated validation data")

    if backend in ["all_gpu", "gpu_naive"]:
        print("Running unstructured gpu naive sanity check")
        p_cell_out_gpu_naive = icon_benchmark.verts2cells_validate_unstructured_gpu_naive(
            nvertices,
            ncells,
            nlevels,
            c2v,
            p_vert_in,
            ptr_coeff,
        )
        assert np.allclose(p_cell_out_gpu_naive, p_cell_out)
        print("unstructured gpu naive sanity check passed")
    
    if backend in ["all_gpu", "gpu_kloop"]:
        print("Running unstructured gpu kloop sanity check")
        p_cell_out_gpu_kloop = icon_benchmark.verts2cells_validate_unstructured_gpu_kloop(
            nvertices,
            ncells,
            nlevels,
            c2v,
            p_vert_in,
            ptr_coeff,
        )
        assert np.allclose(p_cell_out_gpu_kloop, p_cell_out)
        print("unstructured gpu kloop sanity check passed")

    print("Sanity checks pass")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("grid", help="Torus grid netCDF4 file")
    parser.add_argument(
        "--transformation",
        choices=["gt4py", "index"],
        default="gt4py",
        help="Use either ToGt4PyTransformation or IndexTransformation (gt4py by default)",
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

def filter_c2v_vector(
    c2v, grid_cartesian_dimensions, halo=3
):
    filtered_c2v = []
    print("grid_cartesian_dimensions: ", grid_cartesian_dimensions)
    for j in range(grid_cartesian_dimensions[0]):
        for i in range(grid_cartesian_dimensions[1]):
            if (
                i > halo - 1
                and j > halo - 1
                and i < grid_cartesian_dimensions[1] - halo
                and j < grid_cartesian_dimensions[0] - halo
            ):
                for k in range(2):
                    filtered_c2v.append(
                        c2v[(j * grid_cartesian_dimensions[1] + i) * 2 + k]
                    )
    return np.array(filtered_c2v)

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

    original_c2v = torus_grid.get_offset_provider("C2V").table
    filtered_c2v = filter_c2v_vector(
        original_c2v, grid_cartesian_dimensions, args.halo
    )
    
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

    runtimes = {}

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

    print_median_runtimes(runtimes)

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

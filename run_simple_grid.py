import argparse
import numpy as np
from os import path
from pathlib import Path

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize  # type: ignore [import-not-found]

from icon4py.model.common.dimension import E2C2VDim  # type: ignore [import-not-found]

import icon_benchmark  # type: ignore [import-not-found]


def init_grid_manager(fname, num_levels=65):
    grid_manager = GridManager(
        IndexTransformation(), fname, VerticalGridSize(num_levels)
    )
    grid_manager()
    return grid_manager


def get_torus_grid(filename, num_levels):
    grid_manager = init_grid_manager(filename, num_levels)
    grid_manager()
    simple_grid = grid_manager.get_grid()
    return simple_grid


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("grid")
    parser.add_argument("--klevels", type=int, default=65, help="Number of k levels")
    parser.add_argument(
        "--repetitions", type=int, default=101, help="Number of repetitions"
    )
    parser.add_argument(
        "--dry-run", default=False, help="Do a dry run or not", action="store_true"
    )

    return parser.parse_args()


def run_benchmarks():
    args = parse_arguments()

    torus_grid = get_torus_grid(args.grid, args.klevels)

    repetitions = args.repetitions
    dry_runs = 1 if args.dry_run else 0

    print(
        "CellsDim: {} VertexDim: {} EdgeDim: {} KDim: {} E2C2VDim: {}".format(
            torus_grid.num_cells,
            torus_grid.num_vertices,
            torus_grid.num_edges,
            torus_grid.num_levels,
            torus_grid.size[E2C2VDim],
        )
    )

    unstructured_naive_runtimes = icon_benchmark.nabla4_benchmark_unstructured_naive(
        torus_grid.get_offset_provider("E2C2V").table,
        torus_grid.get_offset_provider("E2ECV").table,
        torus_grid.num_cells,
        torus_grid.num_vertices,
        torus_grid.num_edges,
        torus_grid.num_levels,
        torus_grid.size[E2C2VDim],
        repetitions,
        dry_runs,
    )

    print(
        "unstructured naive median runtime: {}".format(
            np.median(unstructured_naive_runtimes)
        )
    )

    structured_simple_naive_runtimes = (
        icon_benchmark.nabla4_benchmark_structured_simple_naive(
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
        "structured SimpleGrid naive median runtime: {}".format(
            np.median(structured_simple_naive_runtimes)
        )
    )

    structured_simple_cpu_ifirst_runtimes = (
        icon_benchmark.nabla4_benchmark_structured_simple_cpu_ifirst(
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
        "structured SimpleGrid cpu_ifirst median runtime: {}".format(
            np.median(structured_simple_cpu_ifirst_runtimes)
        )
    )

    structured_simple_cpu_kfirst_runtimes = (
        icon_benchmark.nabla4_benchmark_structured_simple_cpu_kfirst(
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
        "structured SimpleGrid cpu_kfirst median runtime: {}".format(
            np.median(structured_simple_cpu_kfirst_runtimes)
        )
    )


if __name__ == "__main__":
    run_benchmarks()

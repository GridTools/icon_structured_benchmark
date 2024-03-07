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


def init_grid_manager(fname, num_levels=65, transformation=ToGt4PyTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
    grid_manager()
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    grid_manager()
    simple_grid = grid_manager.get_grid()
    return simple_grid


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("grid")
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
        "--dry-run", default=True, help="Do a dry run or not", action="store_true"
    )

    return parser.parse_args()


def run_benchmarks():
    args = parse_arguments()

    transformation = (
        ToGt4PyTransformation()
        if args.transformation == "gt4py"
        else IndexTransformation()
    )

    torus_grid = get_torus_grid(args.grid, args.klevels, transformation)

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

    ifirst_runtimes = icon_benchmark.nabla4_benchmark_cpu_ifirst(
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

    print("cpu_ifirst mean runtime: {}".format(np.mean(ifirst_runtimes)))

    kfirst_runtimes = icon_benchmark.nabla4_benchmark_cpu_kfirst(
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

    print("cpu_kfirst mean runtime: {}".format(np.mean(kfirst_runtimes)))


if __name__ == "__main__":
    run_benchmarks()

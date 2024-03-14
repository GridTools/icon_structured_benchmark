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

from plot_runtimes import plot_runtimes


def init_grid_manager(fname, num_levels=65, transformation=ToGt4PyTransformation()):
    grid_manager = GridManager(
        transformation, fname, VerticalGridSize(num_levels), True
    )
    grid_manager()
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    grid_manager()
    simple_grid = grid_manager.get_grid()
    return simple_grid


def run_sanity_checks(grid, lon_dim, lat_dim):
    random_validation_data = icon_benchmark.get_nabla4_benchmark_validation_data(
        grid.get_offset_provider("E2C2V").table,
        grid.get_offset_provider("E2ECV").table,
        grid.num_cells,
        grid.num_vertices,
        grid.num_edges,
        grid.num_levels,
        grid.size[E2C2VDim],
    )

    z_nabla4_e2_comp_unstructured = icon_benchmark.nabla4_validate_unstructured_naive(
        grid.get_offset_provider("E2C2V").table,
        grid.get_offset_provider("E2ECV").table,
        random_validation_data.CellDim,
        random_validation_data.VertexDim,
        random_validation_data.EdgeDim,
        random_validation_data.KDim,
        random_validation_data.ECVDim,
        random_validation_data.u_vert,
        random_validation_data.v_vert,
        random_validation_data.primal_normal_vert_v1,
        random_validation_data.primal_normal_vert_v2,
        random_validation_data.z_nabla2_e,
        random_validation_data.inv_vert_vert_length,
        random_validation_data.inv_primal_edge_length,
    )

    assert np.allclose(
        z_nabla4_e2_comp_unstructured,
        random_validation_data.z_nabla4_e2_wp,
    )

    z_nabla4_e2_comp_structured_naive = (
        icon_benchmark.nabla4_validate_structured_torus_naive(
            random_validation_data.CellDim,
            random_validation_data.VertexDim,
            random_validation_data.EdgeDim,
            random_validation_data.KDim,
            random_validation_data.ECVDim,
            lon_dim,
            lat_dim,
            random_validation_data.u_vert,
            random_validation_data.v_vert,
            random_validation_data.primal_normal_vert_v1,
            random_validation_data.primal_normal_vert_v2,
            random_validation_data.z_nabla2_e,
            random_validation_data.inv_vert_vert_length,
            random_validation_data.inv_primal_edge_length,
        )
    )

    assert np.allclose(
        z_nabla4_e2_comp_structured_naive,
        random_validation_data.z_nabla4_e2_wp,
    )

    z_nabla4_e2_comp_structured_cpu_ifirst = (
        icon_benchmark.nabla4_validate_structured_torus_cpu_ifirst(
            random_validation_data.CellDim,
            random_validation_data.VertexDim,
            random_validation_data.EdgeDim,
            random_validation_data.KDim,
            random_validation_data.ECVDim,
            lon_dim,
            lat_dim,
            random_validation_data.u_vert,
            random_validation_data.v_vert,
            random_validation_data.primal_normal_vert_v1,
            random_validation_data.primal_normal_vert_v2,
            random_validation_data.z_nabla2_e,
            random_validation_data.inv_vert_vert_length,
            random_validation_data.inv_primal_edge_length,
        )
    )

    assert np.allclose(
        z_nabla4_e2_comp_structured_cpu_ifirst,
        random_validation_data.z_nabla4_e2_wp,
    )

    z_nabla4_e2_comp_structured_cpu_kfirst = (
        icon_benchmark.nabla4_validate_structured_torus_cpu_kfirst(
            random_validation_data.CellDim,
            random_validation_data.VertexDim,
            random_validation_data.EdgeDim,
            random_validation_data.KDim,
            random_validation_data.ECVDim,
            lon_dim,
            lat_dim,
            random_validation_data.u_vert,
            random_validation_data.v_vert,
            random_validation_data.primal_normal_vert_v1,
            random_validation_data.primal_normal_vert_v2,
            random_validation_data.z_nabla2_e,
            random_validation_data.inv_vert_vert_length,
            random_validation_data.inv_primal_edge_length,
        )
    )

    assert np.allclose(
        z_nabla4_e2_comp_structured_cpu_kfirst,
        random_validation_data.z_nabla4_e2_wp,
    )

    print("Sanity checks pass")


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
        "--dry-run", default=False, help="Do a dry run or not", action="store_true"
    )
    parser.add_argument(
        "--lon-dim", type=int, help="Longitude dimension", required=True
    )
    parser.add_argument("--lat-dim", type=int, help="Latitude dimension", required=True)
    parser.add_argument(
        "--output", type=str, default="output.pdf", help="Output file with plots"
    )
    parser.add_argument(
        "--sanity-checks",
        default=False,
        help="Do a validation with random data between structured and unstructured for the given grid",
        action="store_true",
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

    if args.sanity_checks:
        run_sanity_checks(torus_grid, args.lon_dim, args.lat_dim)

    runtimes = {}

    runtimes[
        "nabla4_benchmark_unstructured_naive"
    ] = icon_benchmark.nabla4_benchmark_unstructured_naive(
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

    runtimes[
        "nabla4_benchmark_structured_torus_naive"
    ] = icon_benchmark.nabla4_benchmark_structured_torus_naive(
        torus_grid.num_cells,
        torus_grid.num_vertices,
        torus_grid.num_edges,
        torus_grid.num_levels,
        torus_grid.size[E2C2VDim],
        args.lon_dim,
        args.lat_dim,
        repetitions,
        dry_runs,
    )

    runtimes[
        "nabla4_benchmark_unstructured_cpu_ifirst"
    ] = icon_benchmark.nabla4_benchmark_unstructured_cpu_ifirst(
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

    runtimes[
        "nabla4_benchmark_structured_torus_cpu_ifirst"
    ] = icon_benchmark.nabla4_benchmark_structured_torus_cpu_ifirst(
        torus_grid.num_cells,
        torus_grid.num_vertices,
        torus_grid.num_edges,
        torus_grid.num_levels,
        torus_grid.size[E2C2VDim],
        args.lon_dim,
        args.lat_dim,
        repetitions,
        dry_runs,
    )

    runtimes[
        "nabla4_benchmark_unstructured_cpu_kfirst"
    ] = icon_benchmark.nabla4_benchmark_unstructured_cpu_kfirst(
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

    runtimes[
        "nabla4_benchmark_structured_torus_cpu_kfirst"
    ] = icon_benchmark.nabla4_benchmark_structured_torus_cpu_kfirst(
        torus_grid.num_cells,
        torus_grid.num_vertices,
        torus_grid.num_edges,
        torus_grid.num_levels,
        torus_grid.size[E2C2VDim],
        args.lon_dim,
        args.lat_dim,
        repetitions,
        dry_runs,
    )

    plot_runtimes(runtimes, args.output)


if __name__ == "__main__":
    run_benchmarks()

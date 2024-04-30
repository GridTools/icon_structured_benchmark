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
        z_nabla4_e2_wp_gtfn = np.zeros(
            shape=(len(filtered_e2c2v), grid.num_levels), dtype=np.float64
        )
        nabla4_gtfn.calculate_nabla4_cpu(
            (
                np.array(random_validation_data.u_vert, dtype=float).T.astype(
                    np.float64
                ),
                (0, 0),
            ),
            (
                np.array(random_validation_data.v_vert, dtype=float).T.astype(
                    np.float64
                ),
                (0, 0),
            ),
            (
                np.array(
                    random_validation_data.primal_normal_vert_v1, dtype=float
                ).astype(np.float64),
                (0,),
            ),
            (
                np.array(
                    random_validation_data.primal_normal_vert_v2, dtype=float
                ).astype(np.float64),
                (0,),
            ),
            (
                np.array(random_validation_data.z_nabla2_e, dtype=float).T.astype(
                    np.float64
                ),
                (0, 0),
            ),
            (
                np.array(
                    random_validation_data.inv_vert_vert_length, dtype=float
                ).astype(np.float64),
                (0,),
            ),
            (
                np.array(
                    random_validation_data.inv_primal_edge_length, dtype=float
                ).astype(np.float64),
                (0,),
            ),
            (z_nabla4_e2_wp_gtfn, (0, 0)),
            0,
            len(filtered_e2c2v),
            0,
            grid.num_levels,
            (
                filtered_e2c2v.astype(np.int64),
                (0, 0),
            ),
            (
                filtered_e2ecv.astype(np.int64),
                (0, 0),
            ),
        )
        assert np.allclose(
            z_nabla4_e2_wp_gtfn.T,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("gtfn_cpu sanity check passed")

    if backend in ["all_gpu", "gtfn_gpu"]:
        print("Running gtfn_gpu sanity check")
        import cupy as cp  # type: ignore [import-not-found]

        z_nabla4_e2_wp_gtfn = cp.zeros(
            shape=(len(filtered_e2c2v), grid.num_levels), dtype=cp.float64
        )
        nabla4_gtfn.calculate_nabla4_gpu(
            (
                cp.array(random_validation_data.u_vert, dtype=float).T.astype(
                    cp.float64
                ),
                (0, 0),
            ),
            (
                cp.array(random_validation_data.v_vert, dtype=float).T.astype(
                    cp.float64
                ),
                (0, 0),
            ),
            (
                cp.array(
                    random_validation_data.primal_normal_vert_v1, dtype=float
                ).astype(cp.float64),
                (0,),
            ),
            (
                cp.array(
                    random_validation_data.primal_normal_vert_v2, dtype=float
                ).astype(cp.float64),
                (0,),
            ),
            (
                cp.array(random_validation_data.z_nabla2_e, dtype=float).T.astype(
                    cp.float64
                ),
                (0, 0),
            ),
            (
                cp.array(
                    random_validation_data.inv_vert_vert_length, dtype=float
                ).astype(cp.float64),
                (0,),
            ),
            (
                cp.array(
                    random_validation_data.inv_primal_edge_length, dtype=float
                ).astype(cp.float64),
                (0,),
            ),
            (z_nabla4_e2_wp_gtfn, (0, 0)),
            0,
            len(filtered_e2c2v),
            0,
            grid.num_levels,
            (
                cp.array(filtered_e2c2v).astype(cp.int64),
                (0, 0),
            ),
            (
                cp.array(filtered_e2ecv).astype(cp.int64),
                (0, 0),
            ),
        )
        assert np.allclose(
            z_nabla4_e2_wp_gtfn.get().T,
            random_validation_data.z_nabla4_e2_wp,
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

    if backend in ["all_gpu", "gpu"]:
        print("Running unstructured gpu sanity check")
        z_nabla4_e2_comp_unstructured_gpu_gridtools = (
            icon_benchmark.nabla4_validate_unstructured_gpu_gridtools(
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
            z_nabla4_e2_comp_unstructured_gpu_gridtools,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("unstructured gpu sanity check passed")

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

    if backend in ["all_gpu", "gpu"]:
        print("Running structured gpu sanity check")
        z_nabla4_e2_comp_structured_torus_gpu_gridtools_halo = (
            icon_benchmark.nabla4_validate_structured_torus_gpu_gridtools_halo(
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
            z_nabla4_e2_comp_structured_torus_gpu_gridtools_halo,
            random_validation_data.z_nabla4_e2_wp,
        )
        print("structured gpu sanity check passed")

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

    filtered_e2c2v = filter_edge_vector(
        torus_grid.get_offset_provider("E2C2V").table,
        grid_cartesian_dimensions,
        args.e2c2v_ordering,
        args.halo,
    )
    filtered_e2ecv = filter_edge_vector(
        torus_grid.get_offset_provider("E2ECV").table,
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
        runtimes["nabla4_benchmark_unstructured_gtfn_cpu"] = []

        for _ in range(repetitions):
            random_validation_data_gtfn = (
                icon_benchmark.get_nabla4_benchmark_validation_data(
                    torus_grid.get_offset_provider("E2C2V").table,
                    torus_grid.get_offset_provider("E2ECV").table,
                    torus_grid.num_cells,
                    torus_grid.num_vertices,
                    torus_grid.num_edges,
                    torus_grid.num_levels,
                    torus_grid.size[E2C2VDim],
                )
            )

            z_nabla4_e2_wp_gtfn = np.zeros(
                shape=(len(filtered_e2c2v), torus_grid.num_levels), dtype=np.float64
            )

            runtimes["nabla4_benchmark_unstructured_gtfn_cpu"].append(
                nabla4_gtfn.calculate_nabla4_cpu(
                    (
                        np.array(
                            random_validation_data_gtfn.u_vert, dtype=float
                        ).T.astype(np.float64),
                        (0, 0),
                    ),
                    (
                        np.array(
                            random_validation_data_gtfn.v_vert, dtype=float
                        ).T.astype(np.float64),
                        (0, 0),
                    ),
                    (
                        np.array(
                            random_validation_data_gtfn.primal_normal_vert_v1,
                            dtype=float,
                        ).astype(np.float64),
                        (0,),
                    ),
                    (
                        np.array(
                            random_validation_data_gtfn.primal_normal_vert_v2,
                            dtype=float,
                        ).astype(np.float64),
                        (0,),
                    ),
                    (
                        np.array(
                            random_validation_data_gtfn.z_nabla2_e, dtype=float
                        ).T.astype(np.float64),
                        (0, 0),
                    ),
                    (
                        np.array(
                            random_validation_data_gtfn.inv_vert_vert_length,
                            dtype=float,
                        ).astype(np.float64),
                        (0,),
                    ),
                    (
                        np.array(
                            random_validation_data_gtfn.inv_primal_edge_length,
                            dtype=float,
                        ).astype(np.float64),
                        (0,),
                    ),
                    (z_nabla4_e2_wp_gtfn, (0, 0)),
                    0,
                    len(filtered_e2c2v),
                    0,
                    torus_grid.num_levels,
                    (
                        filtered_e2c2v.astype(np.int64),
                        (0, 0),
                    ),
                    (
                        filtered_e2ecv.astype(np.int64),
                        (0, 0),
                    ),
                )
            )

    if args.backend in ["all_gpu", "gtfn_gpu"]:
        import cupy as cp  # type: ignore [import-not-found]

        runtimes["nabla4_benchmark_unstructured_gtfn_gpu"] = []

        for _ in range(repetitions):
            random_validation_data_gtfn = (
                icon_benchmark.get_nabla4_benchmark_validation_data(
                    torus_grid.get_offset_provider("E2C2V").table,
                    torus_grid.get_offset_provider("E2ECV").table,
                    torus_grid.num_cells,
                    torus_grid.num_vertices,
                    torus_grid.num_edges,
                    torus_grid.num_levels,
                    torus_grid.size[E2C2VDim],
                )
            )

            z_nabla4_e2_wp_gtfn = cp.zeros(
                shape=(len(filtered_e2c2v), torus_grid.num_levels), dtype=cp.float64
            )

            runtimes["nabla4_benchmark_unstructured_gtfn_gpu"].append(
                nabla4_gtfn.calculate_nabla4_gpu(
                    (
                        cp.array(
                            random_validation_data_gtfn.u_vert, dtype=float
                        ).T.astype(cp.float64),
                        (0, 0),
                    ),
                    (
                        cp.array(
                            random_validation_data_gtfn.v_vert, dtype=float
                        ).T.astype(cp.float64),
                        (0, 0),
                    ),
                    (
                        cp.array(
                            random_validation_data_gtfn.primal_normal_vert_v1,
                            dtype=float,
                        ).astype(cp.float64),
                        (0,),
                    ),
                    (
                        cp.array(
                            random_validation_data_gtfn.primal_normal_vert_v2,
                            dtype=float,
                        ).astype(cp.float64),
                        (0,),
                    ),
                    (
                        cp.array(
                            random_validation_data_gtfn.z_nabla2_e, dtype=float
                        ).T.astype(cp.float64),
                        (0, 0),
                    ),
                    (
                        cp.array(
                            random_validation_data_gtfn.inv_vert_vert_length,
                            dtype=float,
                        ).astype(cp.float64),
                        (0,),
                    ),
                    (
                        cp.array(
                            random_validation_data_gtfn.inv_primal_edge_length,
                            dtype=float,
                        ).astype(cp.float64),
                        (0,),
                    ),
                    (z_nabla4_e2_wp_gtfn, (0, 0)),
                    0,
                    len(filtered_e2c2v),
                    0,
                    torus_grid.num_levels,
                    (
                        cp.array(filtered_e2c2v).astype(cp.int64),
                        (0, 0),
                    ),
                    (
                        cp.array(filtered_e2ecv).astype(cp.int64),
                        (0, 0),
                    ),
                )
            )

    if args.backend in ["all_cpu", "cpu_ifirst"]:
        runtimes[
            "nabla4_benchmark_unstructured_cpu_ifirst_gridtools"
        ] = icon_benchmark.nabla4_benchmark_unstructured_cpu_ifirst_gridtools(
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

    print_median_runtimes(runtimes)

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

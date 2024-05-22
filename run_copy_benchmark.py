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

import copy_gtfn  # type: ignore [import-not-found]

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


def run_gtfn(EdgeDim, KDim, nabla4_data, backend):
    if backend == "gt:gpu":
        import cupy as cp  # type: ignore [import-not-found]

        ds_dtype = cp.float64
    else:
        ds_dtype = np.float64

    from gt4py.storage import zeros, from_array  # type: ignore [import-not-found]

    z_nabla4_e2_wp_gtfn = zeros(shape=(EdgeDim, KDim), dtype=ds_dtype, backend=backend)
    if backend == "gt:gpu":
        runtime = copy_gtfn.calculate_copy_gpu(
            (
                from_array(
                    np.array(nabla4_data.z_nabla2_e).T, dtype=ds_dtype, backend=backend
                ),
                (0, 0),
            ),
            (z_nabla4_e2_wp_gtfn, (0, 0)),
            0,
            EdgeDim,
            0,
            KDim,
        )
    else:
        runtime = copy_gtfn.calculate_copy_cpu(
            (
                from_array(
                    np.array(nabla4_data.z_nabla2_e).T, dtype=ds_dtype, backend=backend
                ),
                (0, 0),
            ),
            (z_nabla4_e2_wp_gtfn, (0, 0)),
            0,
            EdgeDim,
            0,
            KDim,
        )

    def compare_ndarrays(a, b):
        same = True
        if a.shape != b.shape:
            same = False

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if not np.isclose(a[i][j], b[i][j]):
                    print(
                        "Difference at index ({}, {}) is {}".format(
                            i, j, a[i][j] - b[i][j]
                        )
                    )
                    same = False
        if not same:
            print("Arrays are not the same")
            import sys

            sys.exit(1)

    # compare_ndarrays(z_nabla4_e2_wp_gtfn, np.array(nabla4_data.z_nabla2_e).T)
    return z_nabla4_e2_wp_gtfn, runtime


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--edges", type=int, default=902700, help="Number of edges")
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
        "--backend",
        choices=[
            "all_cpu",
            "all_gpu",
            "gtfn_cpu",
            "gtfn_gpu",
            "cpu_ifirst",
            "cpu_kfirst",
            "gpu",
        ],
        default="all_cpu",
        help="Which backend to benchmark",
    )
    args = parser.parse_args()
    return args


def run_benchmarks():
    args = parse_arguments()

    repetitions = args.repetitions
    dry_runs = 10 if args.dry_run else 0

    print(
        "EdgeDim: {} KDim: {}".format(
            args.edges,
            args.klevels,
        )
    )

    runtimes = {}

    if args.backend in ["all_cpu", "gtfn_cpu"]:
        runtimes["copy_benchmark_gtfn_cpu"] = []
        dummy_neighbor_list = np.zeros((args.edges, 4), dtype=np.int32)
        for _ in range(repetitions):
            random_validation_data_gtfn = (
                icon_benchmark.get_nabla4_benchmark_validation_data(
                    dummy_neighbor_list,
                    dummy_neighbor_list,
                    0,
                    0,
                    args.edges,
                    args.klevels,
                    0,
                )
            )

            _, runtime = run_gtfn(
                args.edges,
                args.klevels,
                random_validation_data_gtfn,
                "gt:cpu_ifirst",
            )
            runtimes["copy_benchmark_gtfn_cpu"].append(runtime)
        print("gtfn cpu done")

    if args.backend in ["all_gpu", "gtfn_gpu"]:
        import cupy as cp  # type: ignore [import-not-found]

        runtimes["copy_benchmark_gtfn_gpu"] = []
        dummy_neighbor_list = np.zeros((args.edges, 4), dtype=np.int32)
        for _ in range(repetitions):
            random_validation_data_gtfn = (
                icon_benchmark.get_nabla4_benchmark_validation_data(
                    dummy_neighbor_list,
                    dummy_neighbor_list,
                    0,
                    0,
                    args.edges,
                    args.klevels,
                    0,
                )
            )

            _, runtime = run_gtfn(
                args.edges,
                args.klevels,
                random_validation_data_gtfn,
                "gt:gpu",
            )
            runtimes["copy_benchmark_gtfn_gpu"].append(runtime)
        print("gtfn gpu done")

    if args.backend in ["all_cpu", "cpu_ifirst"]:
        runtimes[
            "copy_benchmark_cpu_ifirst"
        ] = icon_benchmark.copy_benchmark_cpu_ifirst(
            args.edges,
            args.klevels,
            repetitions,
            dry_runs,
        )
        print("cpu_ifirst done")

    if args.backend in ["all_cpu", "cpu_kfirst"]:
        runtimes[
            "copy_benchmark_cpu_kfirst"
        ] = icon_benchmark.copy_benchmark_cpu_kfirst(
            args.edges,
            args.klevels,
            repetitions,
            dry_runs,
        )
        print("cpu_kfirst done")

    if args.backend in ["all_gpu", "gpu"]:
        runtimes["copy_benchmark_gpu"] = icon_benchmark.copy_benchmark_gpu(
            args.edges,
            args.klevels,
            repetitions,
            dry_runs,
        )
        print("gpu done")

    print_median_runtimes(runtimes)

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

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

from run_filtered_torus_grid_int_interpolate import (
    process_v2e_per_orientation,
    process_v2e_per_vertex,
)


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


def filter_c2v_vector(c2v, grid_cartesian_dimensions, halo=3):
    filtered_c2v = []
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


def run_sanity_checks(
    filtered_e2c2v_separate,
    filtered_e2ecv_separate,
    filtered_v2e_separate,
    filtered_c2v_separate,
    filtered_e2c2v_inlined,
    filtered_e2ecv_inlined,
    filtered_v2e_inlined,
    filtered_c2v_inlined,
    grid,
    lon_dim,
    lat_dim,
    backend="all_cpu",
    halo=2,
    e2c2v_ordering="per-vertex",
    combination="separate",
):
    print("Generating separate validation data")
    random_validation_data_separate = (
        icon_benchmark.get_nabla4_benchmark_validation_data(
            filtered_e2c2v_separate,
            filtered_e2ecv_separate,
            grid.num_cells,
            grid.num_vertices,
            grid.num_edges,
            grid.num_levels,
            grid.size[E2C2VDim],
        )
    )
    p_e_in = np.array(random_validation_data_separate.z_nabla4_e2_wp).T
    ptr_coeff_1 = np.random.rand(grid.num_edges, 6)
    ptr_coeff_2 = np.random.rand(grid.num_edges, 6)
    (
        p_u_out_ref_separate,
        p_v_out_ref_separate,
    ) = icon_benchmark.interpolate_validate_unstructured_cpu_ifirst(
        grid.num_vertices,
        (lon_dim - 2 * halo) * (lat_dim - 2 * halo) * 3,
        grid.num_levels,
        filtered_v2e_separate,
        p_e_in,
        ptr_coeff_1,
        ptr_coeff_2,
    )
    inner_cells = (lon_dim - 2 * (halo + 2)) * (lat_dim - 2 * (halo + 2)) * 2
    assert len(filtered_c2v_separate) == inner_cells
    ptr_c_coeff_1 = np.random.rand(inner_cells, 3)
    ptr_c_coeff_2 = np.random.rand(inner_cells, 3)
    p_cell_out_ref = icon_benchmark.verts2cells_validate_unstructured_cpu_kfirst(
        len(p_u_out_ref_separate),
        inner_cells,
        grid.num_levels,
        filtered_c2v_separate,
        p_u_out_ref_separate,
        p_v_out_ref_separate,
        ptr_c_coeff_1,
        ptr_c_coeff_2,
    )
    print("Generated separated validation data")

    print("Generating inlined validation data")
    inv_vert_vert_length_inlined = np.zeros(
        random_validation_data_separate.EdgeDim, dtype=np.float64
    )
    inv_primal_edge_length_inlined = np.zeros(
        random_validation_data_separate.EdgeDim, dtype=np.float64
    )
    z_nabla2_e_inlined = np.zeros(
        (random_validation_data_separate.EdgeDim, grid.num_levels),
        dtype=np.float64,
    )
    z_nabla2_e_ref = np.array(random_validation_data_separate.z_nabla2_e).T
    if e2c2v_ordering == "per-orientation":
        # Generate all edge indices
        edge_indices = np.arange(random_validation_data_separate.EdgeDim)

        # Compute j, i, orientation, and internal_index
        j = (edge_indices % (lon_dim * lat_dim)) // lat_dim
        i = (edge_indices % (lon_dim * lat_dim)) % lat_dim
        orientation = edge_indices // (lon_dim * lat_dim)

        internal_index = (
            i
            - halo
            + (j - halo) * (lat_dim - 2 * halo)
            + orientation * (lon_dim - 2 * halo) * (lat_dim - 2 * halo)
        )

        # Find valid indices
        valid_mask = (
            (i >= halo) & (j >= halo) & (i < lat_dim - halo) & (j < lon_dim - halo)
        )
        valid_edge_indices = edge_indices[valid_mask]
        valid_internal_indices = internal_index[valid_mask]

        # Assign the corresponding values using valid indices
        inv_vert_vert_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_vert_vert_length
        )[valid_internal_indices]
        inv_primal_edge_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_primal_edge_length
        )[valid_internal_indices]
        z_nabla2_e_inlined[valid_edge_indices] = z_nabla2_e_ref[valid_internal_indices]
    else:
        # Generate all edge indices
        edge_indices = np.arange(random_validation_data_separate.EdgeDim)

        # Compute j, i, and orientation
        j = (edge_indices // 3) // lat_dim
        i = (edge_indices // 3) % lat_dim
        orientation = edge_indices % 3

        # Compute internal_index
        internal_index = (
            (i - halo) + (j - halo) * (lat_dim - 2 * halo)
        ) * 3 + orientation

        # Find valid indices
        valid_mask = (
            (i >= halo) & (j >= halo) & (i < lat_dim - halo) & (j < lon_dim - halo)
        )
        valid_edge_indices = np.array(edge_indices[valid_mask], dtype=np.int32)
        # import pdb; pdb.set_trace()
        valid_internal_indices = np.array(internal_index[valid_mask], dtype=np.int32)

        # Assign the corresponding values using valid indices
        inv_vert_vert_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_vert_vert_length
        )[valid_internal_indices]
        inv_primal_edge_length_inlined[valid_edge_indices] = np.array(
            random_validation_data_separate.inv_primal_edge_length
        )[valid_internal_indices]
        z_nabla2_e_inlined[valid_edge_indices] = z_nabla2_e_ref[valid_internal_indices]
    print("Generated inlined validation data")

    if backend in ["all_cpu", "cpu_kfirst"]:
        if combination in ["all", "separate"]:
            print("Running unstructured cpu kfirst separate sanity check")
            p_cell_out_unstructured_cpu_kfirst = icon_benchmark.nabla4_interpolate_verts2cells_validate_unstructured_cpu_kfirst_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                filtered_c2v_separate,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(p_cell_out_unstructured_cpu_kfirst, p_cell_out_ref)
            print("unstructured cpu kfirst separate sanity check passed")

            print("Running structured cpu kfirst separate sanity check")
            p_cell_out_structured_cpu_kfirst = icon_benchmark.nabla4_interpolate_verts2cells_validate_structured_cpu_kfirst_separate(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(p_cell_out_structured_cpu_kfirst, p_cell_out_ref)
            print("structured cpu kfirst separate sanity check passed")

    if backend in ["all_gpu", "gpu_naive"]:
        if combination in ["all", "separate"]:
            print("Running unstructured gpu_naive separate sanity check")
            p_cell_out_unstructured_gpu_naive_separate = icon_benchmark.nabla4_interpolate_verts2cells_validate_unstructured_gpu_naive_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                filtered_c2v_separate,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(
                p_cell_out_unstructured_gpu_naive_separate, p_cell_out_ref
            )
            print("unstructured gpu_naive separate sanity check passed")

            print("Running structured gpu_naive separate sanity check")
            p_cell_out_structured_gpu_naive_separate = icon_benchmark.nabla4_interpolate_verts2cells_validate_structured_gpu_naive_separate(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(p_cell_out_structured_gpu_naive_separate, p_cell_out_ref)
            print("structured gpu_naive separate sanity check passed")

        if combination in ["all", "inlined"]:
            print("Running unstructured gpu_naive inlined sanity check")
            p_cell_out_unstructured_gpu_naive_inlined = icon_benchmark.nabla4_interpolate_verts2cells_validate_unstructured_gpu_naive_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                filtered_c2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(
                p_cell_out_unstructured_gpu_naive_inlined, p_cell_out_ref
            )
            print("unstructured gpu_naive inlined sanity check passed")

            print("Running structured gpu_naive inlined sanity check")
            p_cell_out_structured_gpu_naive_inlined = icon_benchmark.nabla4_interpolate_verts2cells_validate_structured_gpu_naive_inlined(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            if not np.allclose(p_cell_out_structured_gpu_naive_inlined, p_cell_out_ref):
                print("structured gpu_naive inlined sanity check failed")
                print(
                    "p_cell_out_structured_gpu_naive_inlined: {}".format(
                        p_cell_out_structured_gpu_naive_inlined
                    )
                )
                print("p_cell_out_ref: {}".format(p_cell_out_ref))
            assert np.allclose(p_cell_out_structured_gpu_naive_inlined, p_cell_out_ref)
            print("structured gpu_naive inlined sanity check passed")

    if backend in ["all_gpu", "gpu_kloop"]:
        if combination in ["all", "separate"]:
            print("Running unstructured gpu_kloop separate sanity check")
            p_cell_out_unstructured_gpu_kloop_separate = icon_benchmark.nabla4_interpolate_verts2cells_validate_unstructured_gpu_kloop_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                filtered_c2v_separate,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(
                p_cell_out_unstructured_gpu_kloop_separate, p_cell_out_ref
            )
            print("unstructured gpu_kloop separate sanity check passed")

            print("Running structured gpu_kloop separate sanity check")
            p_cell_out_structured_gpu_kloop_separate = icon_benchmark.nabla4_interpolate_verts2cells_validate_structured_gpu_kloop_separate(
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                lon_dim,
                lat_dim,
                halo,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                np.array(random_validation_data_separate.z_nabla2_e).T,
                random_validation_data_separate.inv_vert_vert_length,
                random_validation_data_separate.inv_primal_edge_length,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(p_cell_out_structured_gpu_kloop_separate, p_cell_out_ref)
            print("structured gpu_naive separate sanity check passed")

        if combination in ["all", "inlined"]:
            print("Running unstructured gpu_kloop inlined sanity check")
            p_cell_out_unstructured_gpu_kloop_inlined = icon_benchmark.nabla4_interpolate_verts2cells_validate_unstructured_gpu_kloop_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                filtered_c2v_inlined,
                random_validation_data_separate.CellDim,
                random_validation_data_separate.VertexDim,
                random_validation_data_separate.EdgeDim,
                random_validation_data_separate.KDim,
                random_validation_data_separate.ECVDim,
                np.array(random_validation_data_separate.u_vert).T,
                np.array(random_validation_data_separate.v_vert).T,
                random_validation_data_separate.primal_normal_vert_v1,
                random_validation_data_separate.primal_normal_vert_v2,
                z_nabla2_e_inlined,
                inv_vert_vert_length_inlined,
                inv_primal_edge_length_inlined,
                ptr_coeff_1,
                ptr_coeff_2,
                ptr_c_coeff_1,
                ptr_c_coeff_2,
            )
            assert np.allclose(
                p_cell_out_unstructured_gpu_kloop_inlined, p_cell_out_ref
            )
            print("unstructured gpu_kloop separate sanity check passed")

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
        "--combination",
        choices=["separate", "inlined", "all"],
        default="all",
        help="Which combination of kernels to benchmark",
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
    parser.add_argument(
        "--vertical",
        default=False,
        help="Use nabla4_vertical kernel instead of nabla4 (disabled by default)",
        action="store_true",
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

    def filter_neighbors(
        grid_e2c2v,
        grid_e2ecv,
        grid_v2e,
        e2c2v_ordering="per-vertex",
        combination="separate",
    ):
        filtered_e2c2v = filter_edge_vector(
            grid_e2c2v,
            grid_cartesian_dimensions,
            args.e2c2v_ordering,
            args.halo if combination == "separate" else 0,
        )

        def _get_gpu_coalesced_permuted_e2ecv():
            orientation_permuted_e2ecv = np.zeros_like(grid_e2ecv)
            edges_size = len(grid_e2ecv)
            for i in range(edges_size):
                for j in range(4):
                    orientation_permuted_e2ecv[i][j] = j * edges_size + i
            return orientation_permuted_e2ecv

        permuted_e2ecv = (
            grid_e2ecv
            if args.e2c2v_ordering == "per-vertex"
            else _get_gpu_coalesced_permuted_e2ecv()
        )
        filtered_e2ecv = filter_edge_vector(
            permuted_e2ecv,
            grid_cartesian_dimensions,
            args.e2c2v_ordering,
            args.halo if combination == "separate" else 0,
        )

        filtered_v2e = (
            process_v2e_per_orientation(
                grid_v2e,
                grid_cartesian_dimensions[1],
                grid_cartesian_dimensions[0],
                args.halo + 1,
            )
            if e2c2v_ordering == "per-orientation"
            else process_v2e_per_vertex(
                grid_v2e,
                grid_cartesian_dimensions[1],
                grid_cartesian_dimensions[0],
                args.halo + 1,
            )
        )

        def transform_v2e_to_nabla4_region_per_orientation(v2e, x_dim, y_dim, halo):
            x_dim_inner = x_dim - 2 * (halo + 1)
            y_dim_inner = y_dim - 2 * (halo + 1)
            x_dim_nabla4 = x_dim - 2 * halo
            y_dim_nabla4 = y_dim - 2 * halo
            nabla4_dim = x_dim_nabla4 * y_dim_nabla4
            transformed_v2e = np.zeros_like(v2e)
            assert (x_dim_inner * y_dim_inner, 6) == v2e.shape
            for i in range(x_dim_inner):
                for j in range(y_dim_inner):
                    for k in range(6):
                        global_vertex_i = (
                            v2e[i + j * x_dim_inner][k] % (x_dim * y_dim)
                        ) % x_dim
                        global_vertex_j = (
                            v2e[i + j * x_dim_inner][k] % (x_dim * y_dim)
                        ) // x_dim
                        orientation = v2e[i + j * x_dim_inner][k] // (x_dim * y_dim)
                        nabla4_local_vertex_i = global_vertex_i - halo
                        nabla4_local_vertex_j = global_vertex_j - halo
                        transformed_v2e[i + j * x_dim_inner][k] = (
                            nabla4_local_vertex_i
                            + nabla4_local_vertex_j * x_dim_nabla4
                            + orientation * nabla4_dim
                        )
            return transformed_v2e

        def transform_v2e_to_nabla4_region_per_vertex(v2e, x_dim, y_dim, halo):
            x_dim_inner = x_dim - 2 * (halo + 1)
            y_dim_inner = y_dim - 2 * (halo + 1)
            x_dim_nabla4 = x_dim - 2 * halo
            y_dim_nabla4 = y_dim - 2 * halo
            transformed_v2e = np.zeros_like(v2e)
            assert (x_dim_inner * y_dim_inner, 6) == v2e.shape
            for i in range(x_dim_inner):
                for j in range(y_dim_inner):
                    for k in range(6):
                        global_vertex_i = (v2e[i + j * x_dim_inner][k] // 3) % x_dim
                        global_vertex_j = (v2e[i + j * x_dim_inner][k] // 3) // x_dim
                        orientation = v2e[i + j * x_dim_inner][k] % 3
                        nabla4_local_vertex_i = global_vertex_i - halo
                        nabla4_local_vertex_j = global_vertex_j - halo
                        transformed_v2e[i + j * x_dim_inner][k] = (
                            nabla4_local_vertex_i + nabla4_local_vertex_j * x_dim_nabla4
                        ) * 3 + orientation
            return transformed_v2e

        if combination == "separate":
            filtered_v2e = (
                transform_v2e_to_nabla4_region_per_orientation(
                    filtered_v2e,
                    grid_cartesian_dimensions[1],
                    grid_cartesian_dimensions[0],
                    args.halo,
                )
                if e2c2v_ordering == "per-orientation"
                else transform_v2e_to_nabla4_region_per_vertex(
                    filtered_v2e,
                    grid_cartesian_dimensions[1],
                    grid_cartesian_dimensions[0],
                    args.halo,
                )
            )
        return filtered_e2c2v, filtered_e2ecv, filtered_v2e

    (
        filtered_e2c2v_separate,
        filtered_e2ecv_separate,
        filtered_v2e_separate,
    ) = filter_neighbors(
        torus_grid.get_offset_provider("E2C2V").table,
        torus_grid.get_offset_provider("E2ECV").table,
        torus_grid.get_offset_provider("V2E").table,
        args.e2c2v_ordering,
        "separate",
    )
    # print("Filtered v2e separate size: {}".format(filtered_v2e_separate.shape))
    # for vertex, edges in enumerate(filtered_v2e_separate):
    #     print("Vertex: {} Edges: {}".format(vertex, edges))
    (
        filtered_e2c2v_inlined,
        filtered_e2ecv_inlined,
        filtered_v2e_inlined,
    ) = filter_neighbors(
        torus_grid.get_offset_provider("E2C2V").table,
        torus_grid.get_offset_provider("E2ECV").table,
        torus_grid.get_offset_provider("V2E").table,
        args.e2c2v_ordering,
        "inlined",
    )
    # print("Filtered v2e inlined size: {}".format(filtered_v2e_inlined.shape))
    # for vertex, edges in enumerate(filtered_v2e_inlined):
    #     print("Vertex: {} Edges: {}".format(vertex, edges))

    original_c2v = torus_grid.get_offset_provider("C2V").table
    filtered_c2v_original_indexes = filter_c2v_vector(
        original_c2v, grid_cartesian_dimensions, args.halo + 2
    )
    # print("Filtered C2V inlined size: {}".format(filtered_c2v_inlined.shape))
    # for cell, vertices in enumerate(filtered_c2v_inlined):
    #     print("Cell: {} Vertices: {}".format(cell, vertices))
    filtered_c2v = []
    for cell, vertices in enumerate(filtered_c2v_original_indexes):
        new_vertices = []
        for vertex in vertices:
            i = vertex % grid_cartesian_dimensions[1]
            j = vertex // grid_cartesian_dimensions[1]
            new_vertices.append(
                i - 3 + (j - 3) * (grid_cartesian_dimensions[1] - 2 * 3)
            )
        filtered_c2v.append(new_vertices)
    filtered_c2v = np.array(filtered_c2v)
    # print("Filtered C2V separate size: {}".format(filtered_c2v_separate.shape))
    # for cell, vertices in enumerate(filtered_c2v_separate):
    #     print("Cell: {} Vertices: {}".format(cell, vertices))

    runtimes = {}

    halo = args.halo

    if args.sanity_checks:
        run_sanity_checks(
            filtered_e2c2v_separate,
            filtered_e2ecv_separate,
            filtered_v2e_separate,
            filtered_c2v,
            filtered_e2c2v_inlined,
            filtered_e2ecv_inlined,
            filtered_v2e_inlined,
            filtered_c2v,
            torus_grid,
            grid_cartesian_dimensions[0],
            grid_cartesian_dimensions[1],
            args.backend,
            halo,
            args.e2c2v_ordering,
            args.combination,
        )

    if args.backend in ["all_cpu", "cpu_kfirst"]:
        if args.combination in ["all", "separate"]:
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_unstructured_cpu_kfirst_separate"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_unstructured_cpu_kfirst_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                filtered_c2v,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            print(
                "nabla4_interpolate_verts2cells_benchmark_unstructured_cpu_kfirst_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_unstructured_cpu_kfirst_separate"
                        ]
                    )
                )
            )
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_structured_cpu_kfirst_separate"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_structured_cpu_kfirst_separate(
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
            print(
                "nabla4_interpolate_verts2cells_benchmark_structured_cpu_kfirst_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_structured_cpu_kfirst_separate"
                        ]
                    )
                )
            )

    if args.backend in ["all_gpu", "gpu_naive"]:
        if args.combination in ["all", "separate"]:
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_separate"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                filtered_c2v,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            print(
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_separate"
                        ]
                    )
                )
            )
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_separate"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_separate(
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
            print(
                "nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_separate"
                        ]
                    )
                )
            )

        if args.combination in ["all", "inlined"]:
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_inlined"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                filtered_c2v,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            print(
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_inlined median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_naive_inlined"
                        ]
                    )
                )
            )
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_inlined"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_inlined(
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
            print(
                "nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_inlined median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_structured_gpu_naive_inlined"
                        ]
                    )
                )
            )

    if args.backend in ["all_gpu", "gpu_kloop"]:
        if args.combination in ["all", "separate"]:
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_separate"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_separate(
                filtered_e2c2v_separate,
                filtered_e2ecv_separate,
                filtered_v2e_separate,
                filtered_c2v,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            print(
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_separate"
                        ]
                    )
                )
            )
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_structured_gpu_kloop_separate"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_structured_gpu_kloop_separate(
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
            print(
                "nabla4_interpolate_verts2cells_benchmark_structured_gpu_kloop_separate median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_structured_gpu_kloop_separate"
                        ]
                    )
                )
            )

        if args.combination in ["all", "inlined"]:
            runtimes[
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_inlined"
            ] = icon_benchmark.nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_inlined(
                filtered_e2c2v_inlined,
                filtered_e2ecv_inlined,
                filtered_v2e_inlined,
                filtered_c2v,
                torus_grid.num_cells,
                torus_grid.num_vertices,
                torus_grid.num_edges,
                torus_grid.num_levels,
                torus_grid.size[E2C2VDim],
                repetitions,
                dry_runs,
            )
            print(
                "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_inlined median: {}".format(
                    np.median(
                        runtimes[
                            "nabla4_interpolate_verts2cells_benchmark_unstructured_gpu_kloop_inlined"
                        ]
                    )
                )
            )

    with open(args.output.split(".")[0] + ".json", "w") as file:
        dump(runtimes, file)


if __name__ == "__main__":
    run_benchmarks()

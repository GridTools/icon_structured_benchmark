import numpy as np

from icon4py.model.common.grid.grid_manager import GridManager  # type: ignore [import-not-found]
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
from icon4py.model.common.grid.gridfile import ToZeroBasedIndexTransformation  # type: ignore [import-not-found]

import netCDF4  # type: ignore [import-not-found]


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
    apply_torus_permutation=False,
):
    grid_manager = GridManager(
        grid_file=fname,
        config=VerticalGridConfig(num_levels=num_levels),
        offset_transformation=transformation,
        apply_torus_permutation=apply_torus_permutation,
    )
    grid_manager(allocator=None, keep_skip_values=True)
    return grid_manager


def get_torus_grid(
    filename,
    num_levels,
    transformation,
    e2c2v_ordering="per-vertex",
    apply_torus_permutation=False,
):
    del e2c2v_ordering
    grid_manager = init_grid_manager(
        filename,
        num_levels,
        transformation,
        apply_torus_permutation=apply_torus_permutation,
    )
    return grid_manager.grid


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
                    filtered_vector.append(vector[(i * grid_cartesian_dimensions[1] + j)])
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
                            + grid_cartesian_dimensions[0] * grid_cartesian_dimensions[1]
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
                            + (grid_cartesian_dimensions[0] * grid_cartesian_dimensions[1]) * 2
                        ]
                    )
    return np.array(filtered_vector)


def generate_filtered_e2c2v(
    grid_cartesian_dimensions, e2c2v_ordering="per-vertex", halo=2
):
    x_dim = grid_cartesian_dimensions[1]
    y_dim = grid_cartesian_dimensions[0]

    if e2c2v_ordering == "per-vertex":
        filtered_e2c2v = []
        for j in range(y_dim):
            for i in range(x_dim):
                if i > halo - 1 and j > halo - 1 and i < x_dim - halo and j < y_dim - halo:
                    i_j = j * x_dim + i
                    i_jp1 = (j + 1) * x_dim + i
                    ip1_j = j * x_dim + i + 1
                    i_jm1 = (j - 1) * x_dim + i
                    ip1_jm1 = (j - 1) * x_dim + i + 1
                    im1_jp1 = (j + 1) * x_dim + i - 1

                    filtered_e2c2v.append([i_j, i_jp1, im1_jp1, ip1_j])
                    filtered_e2c2v.append([i_j, ip1_j, i_jp1, ip1_jm1])
                    filtered_e2c2v.append([i_j, ip1_jm1, ip1_j, i_jm1])
    else:
        orientation_0 = []
        orientation_1 = []
        orientation_2 = []
        for j in range(y_dim):
            for i in range(x_dim):
                if i > halo - 1 and j > halo - 1 and i < x_dim - halo and j < y_dim - halo:
                    i_j = j * x_dim + i
                    i_jp1 = (j + 1) * x_dim + i
                    ip1_j = j * x_dim + i + 1
                    i_jm1 = (j - 1) * x_dim + i
                    ip1_jm1 = (j - 1) * x_dim + i + 1
                    im1_jp1 = (j + 1) * x_dim + i - 1

                    orientation_0.append([i_j, i_jp1, im1_jp1, ip1_j])
                    orientation_1.append([i_j, ip1_j, i_jp1, ip1_jm1])
                    orientation_2.append([i_j, ip1_jm1, ip1_j, i_jm1])

        filtered_e2c2v = orientation_0 + orientation_1 + orientation_2

    return np.array(filtered_e2c2v, dtype=np.int32)


def generate_original_e2ecv(original_e2c2v, e2c2v_ordering="per-vertex"):
    if original_e2c2v.ndim != 2 or original_e2c2v.shape[1] != 4:
        raise ValueError(
            "Expected original_e2c2v with shape (n_edges, 4), got {}".format(
                original_e2c2v.shape
            )
        )

    n_edges = original_e2c2v.shape[0]
    linear = np.arange(n_edges * 4, dtype=original_e2c2v.dtype)

    if e2c2v_ordering == "per-vertex":
        return linear.reshape(n_edges, 4)
    if e2c2v_ordering == "per-orientation":
        return linear.reshape(4, n_edges).T

    raise ValueError("Invalid e2c2v_ordering: {}".format(e2c2v_ordering))


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


def filter_c2v_vector(c2v, grid_cartesian_dimensions, halo=3):
    filtered_c2v = []
    for j in range(grid_cartesian_dimensions[0]):
        for i in range(grid_cartesian_dimensions[1]):
            if (
                i > halo - 2
                and j > halo - 2
                and i < grid_cartesian_dimensions[1] - halo
                and j < grid_cartesian_dimensions[0] - halo
            ):
                for k in range(2):
                    filtered_c2v.append(c2v[(j * grid_cartesian_dimensions[1] + i) * 2 + k])
    return np.array(filtered_c2v)

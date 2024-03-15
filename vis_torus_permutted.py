import argparse
from os import path
import numpy as np

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    ToGt4PyTransformation,
)

from plotting import plot_torus  # type: ignore [import-not-found]
from utilities import (
    import_grid,
    get_vertices_coordinates,
    print_torus_file_information,
    print_lat_lon_dimensions,
    get_torus_cartesian_dimensions,
)  # type: ignore [import-not-found]

# grid_file = path.dirname(__file__) + "/tests/data/torus_grid/torus_100000_100000_24576.nc"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("grid")
    return parser.parse_args()


args = parse_arguments()
grid_file = args.grid

grid = import_grid(grid_file, ToGt4PyTransformation())

vertice_coords = get_vertices_coordinates(grid_file)

print_lat_lon_dimensions(grid_file)

print_torus_file_information(grid_file)

# print(grid.get_offset_provider("E2C2V").table)
# for i in range(len(grid.get_offset_provider("E2C2V").table)):
#     print("{} {}".format(i, grid.get_offset_provider("E2C2V").table[i]))


def sort_e2v(e2v_table):
    # print("original e2v_table")
    # for i in range(len(e2v_table)):
    #     print("{} {}".format(i, e2v_table[i]))
    out_of_order_edges = []
    for i in range(2, len(e2v_table), 3):
        out_of_order_edges.append(e2v_table[i])
    # print("out_of_order_edges")
    # print(out_of_order_edges)
    out_of_order_edges.sort(key=lambda x: x[0])
    # print("sorted_out_of_order_edges")
    # print(type(e2v_table))
    # for i in range(len(out_of_order_edges)):
    #     print("{} {}".format(i, out_of_order_edges[i]))
    # e2v_table[i*3] = out_of_order_edges[i]
    sorted_e2v_table = []
    for i in range(len(e2v_table)):
        if i % 3 != 2:
            sorted_e2v_table.append(e2v_table[i].tolist())
        else:
            sorted_e2v_table.append(out_of_order_edges[i // 3].tolist())
    # print("sorted_e2v_table")
    import numpy

    # for i in range(len(sorted_e2v_table)):
    #     print("{} {}".format(i, sorted_e2v_table[i]))
    return numpy.array(sorted_e2v_table)


# def get_permutation_vector(e2v_table):
#     permutation = []
#     for i in range(2, len(e2v_table), 3):
#         permutation.append((e2v_table[i], i))
#     permutation.sort(key=lambda x: x[0][0])
#     permutation_indexes = []
#     j = 0
#     for i in range(len(e2v_table)):
#         if i % 3 != 2:
#             permutation_indexes.append(i)
#         else:
#             permutation_indexes.append(permutation[j][1])
#             j += 1
#     return permutation_indexes

# print(get_permutation_vector(grid.get_offset_provider("E2V").table))
np.set_printoptions(threshold=np.inf)  # type: ignore [arg-type]

sorted_e2v = sort_e2v(grid.get_offset_provider("E2V").table)
print(sorted_e2v[:])
# sorted_e2v = grid.get_offset_provider("E2V").table
print(get_torus_cartesian_dimensions(grid_file))
plot_torus(
    vertice_coords,
    sorted_e2v,
)

# print("E2C2V")
# print(grid.get_offset_provider("E2C2V").table)
# print("E2ECV")
# print(grid.get_offset_provider("E2ECV").table)

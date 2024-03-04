# icon-structured

## Pseudo-code structure for benchmarks

```
@pybind11
std::vector<double> run_nabla4_benchmark_unstructured(E2C2V, VertexDim, KDim, ECVDim, EdgeDim, iterations=101, data_type = double) {
    std::vector<double> runtimes;
    u_vert = init_data<float>(VertexDim, KDim); # initialize with random numbers
    ...
    for (iterations) {
        start = timer();
        z_nabla4_e2_wp = _calculate_nabla4(E2C2V, u_vert, ...);
        end = timer();
        runtimes.push_back(end - start);
    }
    return runtimes;
}
```

```
@pybind11
std::vector<double> run_nabla4_benchmark_structured(E2C2V, VertexDim, KDim, ECVDim, EdgeDim, iterations, data_type = double) {
    std::vector<double> runtimes;
    u_vert = init_data<float>(VertexDim, KDim); # initialize with random numbers
    ...
    for (iterations) {
        start = timer();
        z_nabla4_e2_wp = _calculate_nabla4(E2C2V, u_vert, ...);
        end = timer();
        runtimes.push_back(end - start);
    }
    return runtimes;
}
```

```
@pybind11
std::vector<double> run_nabla4_validation(E2C2V, u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2, z_nabla2_e, inv_vert_vert_length, inv_primal_edge_length, VertexDim, KDim, ECVDim, EdgeDim, iterations, data_type = double) {
    std::vector<double> runtimes;
    z_nabla4_e2_wp = _calculate_nabla4(E2C2V, u_vert, ...);
    return z_nabla4_e2_wp;
}
```

## Instructions

To get the necessary information to run the `nabla4` kernel from a `netcfd` file you can use the following script:

```
from icon4py.model.common.grid.grid_manager import (
    GridManager,
    IndexTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridSize

from icon4py.model.common.dimension import (
    E2C2VDim
)

def init_grid_manager(fname, num_levels=65, transformation=IndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
    grid_manager()
    return grid_manager

grid_manager = init_grid_manager("simple_grid_gridfile.nc")
grid_manager()
simple_grid = grid_manager.get_grid()
print(simple_grid.get_offset_provider("E2C2V").table)
print("CellDim: {}".format(simple_grid.num_cells))
print("VertexDim: {}".format(simple_grid.num_vertices))
print("EdgeDim: {}".format(simple_grid.num_edges))
print("KDim: {}".format(simple_grid.num_levels))
print("E2C2VDim: {}".format(simple_grid.size[E2C2VDim]))
```

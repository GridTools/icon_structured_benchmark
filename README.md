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

## nabla4 kernel

```
@field_operator
def _calculate_nabla4(
    u_vert: Field[[VertexDim, KDim], vpfloat],
    v_vert: Field[[VertexDim, KDim], vpfloat],
    primal_normal_vert_v1: Field[[ECVDim], wpfloat],
    primal_normal_vert_v2: Field[[ECVDim], wpfloat],
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    inv_vert_vert_length: Field[[EdgeDim], wpfloat],
    inv_primal_edge_length: Field[[EdgeDim], wpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    u_vert_wp, v_vert_wp = astype((u_vert, v_vert), wpfloat)

    nabv_tang_vp = astype(
        (
            u_vert_wp(E2C2V[0]) * primal_normal_vert_v1(E2ECV[0])
            + v_vert_wp(E2C2V[0]) * primal_normal_vert_v2(E2ECV[0])
            + u_vert_wp(E2C2V[1]) * primal_normal_vert_v1(E2ECV[1])
            + v_vert_wp(E2C2V[1]) * primal_normal_vert_v2(E2ECV[1])
        ),
        vpfloat,
    )

    nabv_norm_vp = astype(
        (
            u_vert_wp(E2C2V[2]) * primal_normal_vert_v1(E2ECV[2])
            + v_vert_wp(E2C2V[2]) * primal_normal_vert_v2(E2ECV[2])
            + u_vert_wp(E2C2V[3]) * primal_normal_vert_v1(E2ECV[3])
            + v_vert_wp(E2C2V[3]) * primal_normal_vert_v2(E2ECV[3])
        ),
        vpfloat,
    )
    nabv_tang_wp, nabv_norm_wp = astype((nabv_tang_vp, nabv_norm_vp), wpfloat)
    z_nabla4_e2_wp = wpfloat("4.0") * (
        (nabv_norm_wp - wpfloat("2.0") * z_nabla2_e) * (inv_vert_vert_length * inv_vert_vert_length)
        + (nabv_tang_wp - wpfloat("2.0") * z_nabla2_e)
        * (inv_primal_edge_length * inv_primal_edge_length)
    )
    return astype(z_nabla4_e2_wp, vpfloat)

@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla4(
    u_vert: Field[[VertexDim, KDim], vpfloat],
    v_vert: Field[[VertexDim, KDim], vpfloat],
    primal_normal_vert_v1: Field[[ECVDim], wpfloat],
    primal_normal_vert_v2: Field[[ECVDim], wpfloat],
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    inv_vert_vert_length: Field[[EdgeDim], wpfloat],
    inv_primal_edge_length: Field[[EdgeDim], wpfloat],
    z_nabla4_e2: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        out=z_nabla4_e2,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
```

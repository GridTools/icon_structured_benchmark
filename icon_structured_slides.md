---
marp: false
theme: cscs
# class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('../slides-support/common/4k-slide-bg-white.png')
size: 16:9
---

# ICON structured grid benchmark

![bg cover](../slides-support/common/title-bg3.png)

<!-- _paginate: skip  -->
<!-- _class: titlecover -->
<!-- _footer: "" -->

### CSCS

---

# Introduction

## Motivation

<div class="twocolumns">
<div>

- ICON grid is partly structured
- Investigate how expensive are indirect neighbor accesses compared to strided indexes of neighbors
- Investigate optimizations based on strided indexes
- Apply optimizations to current and future `GT4Py` development

</div>
<div>
<img src="../slides-images/icon_grid.png" style="width: 80%" align="right"/>
</div>
</div>

---

# ICON grid handling

## ICON grid input

- `netcdf` file with neighbor lists
- Use of structured torus grid
  - Generated using [grid-generator](https://gitlab.dkrz.de/mpim-sw/grid-generator) from `DKRZ` by `David Strassman`
  - Read using [icon4py GridManager](https://github.com/C2SM/icon4py/blob/main/model/common/src/icon4py/model/common/grid/grid_manager.py)

---

## Manipulation of ICON grid

<div class="twocolumns">
<div>

### Torus grid characteristics

- Torus grid has periodic boundaries
  - Adds complexity to compute strided indexes
  - Not applicable in production scenario of CH

</div>
<div>
<img src="../slides-images/full_torus.png" style="width: 120%" align="right"/>
</div>
</div>

---

## Manipulation of ICON grid

<div class="twocolumns">
<div>

### Selection of computation domain

- Filter torus grid to end up with a cartesian grid with 2 level halo
  - 1 would also be sufficient
  - Discard periodic edges
- Calculate `X` and `Y` dimensions based on the distribution of vertices in space
- Order edges and vertices in memory
  - `per-vertex`: `edges` are ordered in memory based on `vertex` id
  - `per-orientation`: `edges` are ordered in memory based on their orientation

### Computation domain in <span style="color:red">red</span>

</div>
<div>
<img src="../slides-images/torus_computation_domain.png" style="width: 110%" align="right"/>
</div>
</div>

---

## Manipulation of ICON grid

<div class="twocolumns">
<div>

### Selection of computation domain

- Fix `e2v` ordering of edges read by torus grid file to be by default `per-vertex` in [GridManager](https://github.com/iomaganaris/icon4py/blob/0f861306c65e135345aac534a5e0283de6e8554e/model/common/src/icon4py/model/common/grid/grid_manager.py#L499)
  - Some elements in `e2v` didn't follow this ordering by default
- Select between `per-vertex` and `per-orientation` ordering in [GridManager](https://github.com/iomaganaris/icon4py/blob/0f861306c65e135345aac534a5e0283de6e8554e/model/common/src/icon4py/model/common/grid/grid_manager.py#L500)
- Filter halo vertices and edges in [benchmark python script](https://github.com/GridTools/icon_structured_benchmark/blob/main/run_filtered_torus_grid_int_nabla4_interpolate.py#L74)

### Computation domain in <span style="color:red">red</span>

</div>
<div>
<img src="../slides-images/torus_computation_domain.png" style="width: 110%" align="right"/>
</div>
</div>

---

## Selected kernels for exploration

- [Nabla4](https://github.com/C2SM/icon4py/blob/a32640c8bacc65bbafe41ec9a68b772c7a3c1853/model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/stencils/calculate_nabla4.py#L23)
  - Neighbor tables
    - **e2c2v[4]**
    - **e2ecv[4]**
  - Input fields
    - **u_vert[EdgeDim, KDim]**
    - **v_vert[EdgeDim, KDim]**
    - **primal_normal_vert_v1[EdgeDim * ECVDim]**
    - **primal_normal_vert_v2[EdgeDim * ECVDim]**
    - **z_nabla2_e[OutputEdgeDim, KDim]**
    - **inv_vert_vert_length_gt[OutputEdgeDim]**
    - **inv_primal_edge_length_gt[OutputEdgeDim]**
  - Output field
    - **z_nabla4_e2[OutputEdgeDim, KDim]**

---

## Selected kernels for exploration

- Nabla4 vertical (artificial)
  - Similar to `nabla4` but all input fields are vertical
  - Neighbor tables
    - **e2c2v[4]**
    - **e2ecv[4]**
  - Input fields
    - **u_vert[EdgeDim, KDim]**
    - **v_vert[EdgeDim, KDim]**
    - **primal_normal_vert_v1[EdgeDim * ECVDim, KDim]**
    - **primal_normal_vert_v2[EdgeDim * ECVDim, KDim]**
    - **z_nabla2_e[OutputEdgeDim, KDim]**
    - **inv_vert_vert_length_gt[OutputEdgeDim, KDim]**
    - **inv_primal_edge_length_gt[OutputEdgeDim, KDim]**
  - Output field
    - **z_nabla4_e2[OutputEdgeDim, KDim]**

---

## Selected kernels for exploration

- [Interpolate](https://github.com/C2SM/icon4py/blob/main/model/common/src/icon4py/model/common/interpolation/stencils/mo_intp_rbf_rbf_vec_interpol_vertex.py#L19)
  - - Neighbor tables
    - **v2e[6]**
  - Input fields
    - **p_e_in[EdgeDim, KDim]**
    - **ptr_coeff_1[OutputVertexDim, 6]**
    - **ptr_coeff_2[OutputVertexDim, 6]**
  - Output fields
    - **p_u_out[OutputVertexDim, KDim]**
    - **p_v_out[OutputVertexDim, KDim]**

---

# Questions?

![bg cover](../slides-support/common/title-bg2.png)

<!-- _paginate: skip  -->
<!-- _class: titlecover -->
<!-- _footer: "" -->

---

# Resources

- [GitHub Repository](https://github.com/GridTools/icon_structured_benchmark)

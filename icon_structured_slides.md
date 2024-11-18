---
marp: false
theme: cscs
# class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('slides-support/common/4k-slide-bg-white.png')
size: 16:9
---

# ICON structured grid benchmark

![bg cover](slides-support/common/title-bg3.png)

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
<img src="slides-images/icon_grid.png" style="width: 80%" align="right"/>
</div>
</div>

---

# ICON grid handling

## ICON grid input

- `netcdf` file with neighbor lists
- Use of structured torus grid
  - Generated using [grid-generator](https://gitlab.dkrz.de/mpim-sw/grid-generator) from `DKRZ` by **David Strassman**
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
<img src="slides-images/full_torus.png" style="width: 120%" align="right"/>
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
  - `per-vertex`: `edges` are ordered in memory based on `vertex` id and then orientation
  - `per-orientation`: `edges` are ordered in memory based on their orientation and then their corresponding vertex coordinates

### Computation domain in <span style="color:red">red</span>

</div>
<div>
<img src="slides-images/torus_computation_domain.png" style="width: 110%" align="right"/>
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
<img src="slides-images/torus_computation_domain.png" style="width: 110%" align="right"/>
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

## Kernel implementations

- Grid structure
  - Unstructured
    - Indirect accesses via neighbor tables
  - Structred
    - Neighbor accesses via strides
- Iteration strategies
  - `gpu_naive`
    - Default iteration strategy in `GridTools C++`
    - One GPU thread calculates 1 element in horizontal and vertical axis
  - `gpu_kloop`
    - Optional iteration strategy in `GridTools C++`
    - One GPU thread calculates 1 element in horizontal axis but multiple vertical levels
    - Save neighbors and vertically independent fields in registers and iterate over multiple vertical fields

---

## Kernel implementations

- Kernel combinations
  - Separate
    - Execute `nabla4` kernel and then `interpolate`
  - Inlined
    - Compute `nabla4` output for every input of `interpolate` kernel
    - More computations
    - Less writes to device memory
  - Inlined v2v (unstructured only)
    - Compress `v2e[e2c2v]` neighbor accesses to `v2e2c2v`
      - Read fields for 7 vertices instead of (6\*4=) 24 vertices
    - Assumes certain order of vertices in `e2c2v` and edges in `v2e`
  - Inlined_cached
    - Save `nabla4` output in shared memory and then use it in `interpolate` kernel
    - Reduce overcomputations compared to `inlined` implementation

---

## Kernel implementations

- `gtfn`
  - Only `unstructured`
  - Based on `GridTools C++`
  - Improved `GridTools C++`
    - `const` neighbor tables and input fields
    - Memory loads via `__ldg`
    - `gpu_kloop` option
    - Kudos to **Felix Thaler**
- `CUDA`
  - Plain cuda kernels
  - Launched by python script
  - Random input in benchmarks
  - Validated kernels with serialized data from `GT4Py`

---

## General kernel optimizations

- Occupancy
  - All kernels have been optimized for best occupancy
- `__launch_bounds__` and `__maxnreg__`
  - Launch bounds are applied to all kernels except some that were performing better with register limitation to a certain register number
- Thread Block size
  - Especially for `gpu_naive` implementations, increasing the vertical axis thread block size (`ThreadBlockDim.y/z`) was beneficial since there are more chances to find neighbor tables and vertically independent fields in cache
  - 4-8-9 most used number with 80 vertical levels in total
- `gpu_kloop`
  - For this implementation vertical `GridDim` is set to 1. Iteration number is set based on the `ThreadBlockDim.y/z` and `KDim`. Exception are the `inline_cached` versions where the shared memory size is limiting the number of iterations possible
  - Optimized number of iterations per kernel. 8-40 iterations. 8-20 usually have the best performance

---

## Notes for specific kernels

- `structured_gpu_{naive,kloop}_inlined_cached`: Tried 2 different implementations, number of threads same as input but then deactivate for the output some of them and number of threads same as output where each thread caclulates multiple elements. Former is better
- `structured_*_inlined`: calculate only necessary indexes, similar to `v2e2c2v`
- `unstructured_*_inlined_v2v`: pass `v2e2c2v` as input
- `unstructured`: `nabla4` iterates on edges (`per-orientation` - 1 edge per thread)
- `structured`: `nabla4` iterates on vertices (`per-vertex` - 3 edges per vertex/thread)
- Both `structured` and `unstructured` versions operate on data with same ordering in memory
  - No SFC. Vertices are indered per `i` and `j` coordinates and `edges` per `orientation/vertices`
- `structured`: `e2ecv` is also computed

---

## Results

- `Nabla4` & `interpolate`
  - `gpu_naive` & `gpu_kloop`
  - See if `gpu_naive` has anything interesting
    - Probably nice to show improvement over starting point (`gpu_naive` `unstructured` to fastest `gpu_kloop` `inlined` version)
- `Nabla4_vertical` & `interpolate`
  - Compare to normal `nabla4`
- Use `256` torus grid

---

## Conclusions

---

## Notes

- `LDGSTS` for `nabla4` `structured`
  - Initial implementation for loading vertical input fields to memory per vertical level
    - Big slowdown
    - `LDGSTS.BYPASS` can be used only for certain fields
- `TMA` for `nabla4` `structured`
  - Used in a similar way
    - Slowdown and problematic implementation
    - Probably `TMA` is not properly used
  - Alignment requirements need a different approach
    - Aligned loads for `blockDim.x` in `j`, `j-1` and `j+1` to calculate `i`, `j` in `j`
    - Similar to `gpu_kloop_inlined_structured_cached`

---

## Next steps

- Try `cached` approach for `unstructured` implementation
  - Compute border coordinates for each Thread Block (should be the same as `structured` for our grid)
  - Load them to shared memory
- Add another kernel with `c2v` neighbor to see implact in `inlined` versions due to overcomputations
  - `c2v` neighbor will require computing 3 vertices per cell = 3 times the computations
  - Try `c2v2e2c2v` compressed neighbor in `unstructured` version as well
- Improve `TMA` implementation

---

# Questions?

![bg cover](slides-support/common/title-bg2.png)

<!-- _paginate: skip  -->
<!-- _class: titlecover -->
<!-- _footer: "" -->

---

# Resources

- [GitHub Repository](https://github.com/GridTools/icon_structured_benchmark)

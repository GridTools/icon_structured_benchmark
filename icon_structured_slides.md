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
<img src="slides-images/full_torus.png" style="width: 98%" align="right"/>
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
  - `x_dim` equals the number of vertices with `y == 0`

### Computation domain in <span style="color:red">red</span>

</div>
<div>
<img src="slides-images/torus_computation_domain.png" style="width: 98%" align="right"/>
</div>
</div>

---

## Manipulation of ICON grid

### Organization of edges in memory

<div class="twocolumns">
<div>
<img src="slides-images/per-vertex.svg" style="width: 100%" align="center"/>
</div>
<div>
<img src="slides-images/per-orientation.svg" style="width: 100%" align="center"/>
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
<img src="slides-images/torus_computation_domain.png" style="width: 98%" align="right"/>
</div>
</div>

---

## Selected kernels for exploration

- [Nabla4](https://github.com/C2SM/icon4py/blob/a32640c8bacc65bbafe41ec9a68b772c7a3c1853/model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/stencils/calculate_nabla4.py#L23)
  - Neighbor tables
    - **e2c2v[4]**
    - **e2ecv[4]**
  - Input fields
    - **u_vert[<span style="color:SandyBrown">VertexDim</span>, <span style="color:blue">KDim</span>]**
    - **v_vert[<span style="color:SandyBrown">VertexDim</span>, <span style="color:blue">KDim</span>]**
    - **primal_normal_vert_v1[<span style="color:DarkGreen">EdgeDim</span> * <span style="color:Tan">ECVDim</span>]**
    - **primal_normal_vert_v2[<span style="color:DarkGreen">EdgeDim</span> * <span style="color:Tan">ECVDim</span>]**
    - **z_nabla2_e[<span style="color:OliveDrab">OutputEdgeDim</span>, <span style="color:blue">KDim</span>]**
    - **inv_vert_vert_length[<span style="color:OliveDrab">OutputEdgeDim</span>]**
    - **inv_primal_edge_length[<span style="color:OliveDrab">OutputEdgeDim</span>]**
  - Output field
    - **z_nabla4_e2[<span style="color:OliveDrab">OutputEdgeDim</span>, <span style="color:blue">KDim</span>]**

---

## Selected kernels for exploration

- Nabla4 vertical (artificial)
  - Similar to `nabla4` but all input fields are vertical
  - Neighbor tables
    - **e2c2v[4]**
    - **e2ecv[4]**
  - Input fields
    - **u_vert[<span style="color:SandyBrown">VertexDim</span>, <span style="color:blue">KDim</span>]**
    - **v_vert[<span style="color:SandyBrown">VertexDim</span>, <span style="color:blue">KDim</span>]**
    - **primal_normal_vert_v1[<span style="color:DarkGreen">EdgeDim</span> * <span style="color:Tan">ECVDim</span>, <span style="color:DeepSkyBlue">KDim</span>]**
    - **primal_normal_vert_v2[<span style="color:DarkGreen">EdgeDim</span> * <span style="color:Tan">ECVDim</span>, <span style="color:DeepSkyBlue">KDim</span>]**
    - **z_nabla2_e[<span style="color:OliveDrab">OutputEdgeDim</span>, <span style="color:blue">KDim</span>]**
    - **inv_vert_vert_length[<span style="color:OliveDrab">OutputEdgeDim</span>, <span style="color:DeepSkyBlue">KDim</span>]**
    - **inv_primal_edge_length[<span style="color:OliveDrab">OutputEdgeDim</span>, <span style="color:DeepSkyBlue">KDim</span>]**
  - Output field
    - **z_nabla4_e2[<span style="color:OliveDrab">OutputEdgeDim</span>, <span style="color:blue">KDim</span>]**

---

## Selected kernels for exploration

- [Interpolate](https://github.com/C2SM/icon4py/blob/main/model/common/src/icon4py/model/common/interpolation/stencils/mo_intp_rbf_rbf_vec_interpol_vertex.py#L19)
  - - Neighbor tables
    - **v2e[6]**
  - Input fields
    - **p_e_in[<span style="color:DarkGreen">EdgeDim</span>, <span style="color:blue">KDim</span>]**
    - **ptr_coeff_1[<span style="color:BurlyWood">OutputVertexDim</span>, 6]**
    - **ptr_coeff_2[<span style="color:BurlyWood">OutputVertexDim</span>, 6]**
  - Output fields
    - **p_u_out[<span style="color:BurlyWood">OutputVertexDim</span>, <span style="color:blue">KDim</span>]**
    - **p_v_out[<span style="color:BurlyWood">OutputVertexDim</span>, <span style="color:blue">KDim</span>]**

---

## Nabla4 execution

<div style="text-align: center;">
  <img src="slides-images/nabla4_0.svg" style="width: 86%;"/>
</div>

---

## Nabla4 execution

<div style="text-align: center;">
  <img src="slides-images/nabla4_1.svg" style="width: 86%;"/>
</div>

---

## Interpolate execution

<div style="text-align: center;">
  <img src="slides-images/interpolate_0.svg" style="width: 86%;"/>
</div>

---

## Nabla4 & Interpolate inlined v2e[e2c2v] execution

<div style="text-align: center;">
  <img src="slides-images/nabla4_interpolate_inlined.svg" style="width: 86%;"/>
</div>

---

## Nabla4 & Interpolate inlined v2e2c2v execution

<div style="text-align: center;">
  <img src="slides-images/nabla4_interpolate_inlined_v2v.svg" style="width: 86%;"/>
</div>

---

## Kernel implementations

- Neighbor accesses
  - Indirect
    - Indirect accesses via neighbor tables
  - Strided
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

<div class="twocolumns">
<div>

- Separate
  - Execute `nabla4` kernel and then `interpolate`
- Inlined
  - Compute `nabla4` output for every input of `interpolate` kernel
  - More computations
  - Less writes to device memory
- Inlined v2v (`indirect` only)
  - Compress `v2e[e2c2v]` neighbor accesses to `v2e2c2v`
    - Read fields for 7 vertices instead of (6\*4=) 24 vertices
  - Assumes certain order of vertices in `e2c2v` and edges in `v2e`

</div>

<div style="text-align: left;">
  <img src="slides-images/nabla4_interpolate.svg" style="width: 110%"/>
</div>
</div>

---

## Kernel implementations

<div class="twocolumns">
<div>

- Inlined_cached
  - Save `nabla4` output in shared memory and then use it in `interpolate` kernel
  - Reduces overcomputations compared to `inlined` implementation

</div>

<div style="text-align: left;">
  <img src="slides-images/nabla4_interpolate.svg" style="width: 110%"/>
</div>
</div>

---

## Kernel implementations

- `gtfn`
  - Only `indirect`
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
  - Launch bounds are applied to all kernels except some that were performing better with register limitation to a certain number of registers
- Thread Block size
  - Specifically for `gpu_naive` implementations, increasing the vertical axis thread block size (`ThreadBlockDim.y/z`) was beneficial since there are more chances to find neighbor tables and vertically independent fields in cache
  - 4-8-9 most used number with 80 vertical levels in total. For some kernels 12 or 16
- `gpu_kloop`
  - For this implementation vertical `GridDim` is set to 1. Iteration number is controlled by `ThreadBlockDim.y/z` and `KDim`. Exceptions are the `inline_cached` versions where the shared memory size is limiting the number of iterations possible
  - Optimized number of iterations per kernel. 5-40 iterations. 8-20 usually have the best performance

---

## Notes for specific kernels

- `strided_gpu_{naive,kloop}_inlined_cached`: Tried 2 different implementations, number of threads same as input but then deactivate for the output some of them and number of threads same as output where each thread caclulates multiple elements. Former is better
- `strided_*_inlined`: calculate only necessary indexes, similar to `v2e2c2v`
- `indirect_*_inlined_v2v`: pass `v2e2c2v` as input
- `indirect`: `nabla4` iterates on edges (`per-orientation` - 1 edge per thread)
- `strided`: `nabla4` iterates on vertices (`per-vertex` - 3 edges per vertex/thread)
- Both `strided` and `indirect` versions operate on data with same ordering in memory
  - No SFC. Vertices are ordered per `i` and `j` coordinates and `edges` per `orientation/vertices`
- `strided`: `e2ecv` is also computed
- Smaller grids benefit by more threads and less `k level` iterations
- Loop in `k` is done with stride `blockDim.y/z * gridDim.y/z`
  - It would be more beneficial for kernels that read data from adjacent `k` levels to do the looping with stride `1` in each thread
    - Current implementation in `gtfn`, not on the `CUDA` kernels since we don't evaluate such case

---

## Results

- `GH200` GPU
- Median runtime presented
  - 10 dry runs (not taken into account)
  - 101 runs to select median
- 229758 edges
  - Close to the amount of edges that fit in a single GPU for ICON runs
- 915948 edges
  - For exploration

---

## gtfn improvements

<div style="text-align: center;">
  <img src="slides-images/runtimes_torus_accel_gtfn_128_80.png" style="width: 62%;"/>
</div>

---

<div class="twocolumns">
<div style="text-align: center;">
  <img src="slides-images/runtimes_torus_accel_nabla4_inter_256_80_v2.png" style="width: 133%"/>
</div>

<div style="margin-left: 320px;">

- `strided` `separate` kernels **~10%** speed up
- Inlining in `gpu_naive` doesn't help without extra optimizations due to overcomputations
- `cached` approach in `gpu_naive` another **~20%** speedup
- `gpu_kloop` **~10-20%** speedup compared to `gpu_naive` for `separate` kernels
- `indirect` `gpu_kloop` `inlined_v2v` and `strided` `gpu_kloop` `inlined` **~2x** faster
- `cached` implementation not fastest in `gpu_kloop`

</div>
</div>

---

<!-- <div>
<img src="slides-images/runtimes_torus_accel_nabla4_inter_128_80.png" style="width: 70%" align="center"/>
</div>

--- -->

<div class="twocolumns">
<div style="text-align: center;">
  <img src="slides-images/runtimes_torus_accel_nabla4_vertical_inter_256_80_v2.png" style="width: 133%"/>
</div>
<div style="margin-left: 320px;">

- `gpu_kloop` not always beneficial since there are no vertical fields to save to registers
  - Only neighbors are saved in registers which are either loaded from memory or computed
- `strided` `gpu_kloop` `inlined_cached` not better than more expensive overcomputations for other `inlined` implementations

</div>
</div>

---

<!-- <div>
<img src="slides-images/runtimes_torus_accel_nabla4_vertical_inter_128_80.png" style="width: 70%" align="center"/>
</div>

--- -->

## Conclusions

- For `separate` kernels `strided` is faster than `indirect` by **8-10%**
- For `nabla4_interpolate` where there are vertically independent fields `gpu_kloop` is beneficial
  - For `nabla4_vertical_interpolate` where there aren't many vertically independent fields `gpu_kloop` is not necessaarily beneficial due to more register usage and lower occupancy
- `strided` is only slightly faster than `indirect` `inlined_v2v` in `nabla4_interpolate` but much faster than `indirect`
  - In `nabla4_vertical_interpolate` performance is different
    - Probably should understand it better but since `strided_inlined_cached` is not faster than `strided_inlined` maybe not worth it
- `inlined_cached` using shared memory is not better than overcomputations/loading from memory for `inlined` version in `strided` implementations
  - Potentially room for improvement there

---

## Next steps

- Add interpolation stencil with `c2v` neighbor to see implact in `inlined` versions due to overcomputations
  - `c2v` neighbor will require computing 3 vertices per cell = 3 times the computations
  - Try `c2v2e2c2v` compressed neighbor in `indirect` version as well
- Try `cached` approach for `indirect` implementation
  - Compute border coordinates for each Thread Block (should be the same as `strided` for our grid)
    - `per-vertex` ordering should be better due to smaller range of vertices/edges that need to be saved to shared memory
  - Load them to shared memory
- Try `TMA` implementation
  - Tried `cuda::pipeline` to use `LDGSTS`/`LDGSTS.BYPASS` instructions for loading the input fields of `nabla4` kernel
    - Big slowdown
    - Probably because computations are not enough to hide memory latencies
  - Try with `TMA` to see if it improves
  - More complex due to memory alignment requirements

---

## Next steps

- Compile time strides
  - `x_dim` is necessary to calculate the strides in `strided` version
    - Can be given in case of JIT compilation
- Use cache hints for loads and non temporal stores

---

# Questions?

![bg cover](slides-support/common/title-bg2.png)

<!-- _paginate: skip  -->
<!-- _class: titlecover -->
<!-- _footer: "" -->

---

# Resources

- [GitHub Repository](https://github.com/GridTools/icon_structured_benchmark)

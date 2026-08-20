# icon-structured

Standalone benchmark that evaluates the performance of a structured grid compared to an unstructured grid using the [nabla4 kernel from icon4py](https://github.com/C2SM/icon4py/blob/main/model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/stencils/calculate_nabla4.py#L23) and its combination with the [mo_intp_rbf_rbf_vec_interpol_vertex kernel](https://github.com/C2SM/icon4py/blob/main/model/common/src/icon4py/model/common/interpolation/stencils/mo_intp_rbf_rbf_vec_interpol_vertex.py#L19) and [an artificial interpolation kernel with a `c2v` access](https://github.com/GridTools/icon_structured_benchmark/blob/main/tests/test_verts2cells.py#L36).

## Benchmark description

### Grids

The supported grids are `unstructured`, `structured_simple` and `structured_torus`.

- `unstructured`: Can represent any type of grid and is using indirect accesses for accessing the necessary neighbors
- `structured_simple`: Is based on the `SimpleGrid` from [icon4py](https://github.com/C2SM/icon4py/blob/main/model/common/src/icon4py/model/common/grid/simple.py)
- `structured_torus`: Generated using the torus script from [mpim-sw/grid-generator](https://gitlab.dkrz.de/mpim-sw/grid-generator). We have also experimented with benchmarking this grid with halo = 2 so that we can process it without taking care of the periodic boundaries. Instead of a torus we end up with a cartesian plain grid.

### Grids visualization

To visualize the `SimpleGrid` and the various `torus` grids one can use the python scripts `vis_simple.py`, `vis_torus_original.py` and `vis_torus_permutted.py` that has the preferred ordering of the edges (north, east and southeast per vertex).

### nabla4 kernel

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
```

The `nabla4` kernel can be seen above. It is using a neighbor LUT for `e2c2v` or `diamond` vertices and another for `e2ecv` for accessing the sparse inputs.

### interpolate kernel

```
@field_operator
def _mo_intp_rbf_rbf_vec_interpol_vertex(
    p_e_in: Field[[EdgeDim, KDim], wpfloat],
    ptr_coeff_1: Field[[VertexDim, V2EDim], wpfloat],
    ptr_coeff_2: Field[[VertexDim, V2EDim], wpfloat],
) -> tuple[Field[[VertexDim, KDim], wpfloat], Field[[VertexDim, KDim], wpfloat]]:
    p_u_out = neighbor_sum(ptr_coeff_1 * p_e_in(V2E), axis=V2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_e_in(V2E), axis=V2EDim)
    return p_u_out, p_v_out
```

1 input field and 2 output fields. 1 LUT for `v2e`.

The selection of these two kernels came from `icon4py` where the `nabla2` kernel is combined with the `interpolate` kernel for the `z_nabla2` field. `nabla4` is just a simplified version of `nabla2` that only calculates the `z_nabla2` field.

### Implementations

The kernel implementations is using C++ and CUDA and support CPU and GPU execution.
The memory layout and the backends are divided into `cpu_ifirst`, `cpu_kfirst` and `gpu`

- `cpu_ifirst`: the `stride=1` dimension is the dimension of the edges. Edges and vertices are ordered based on the vertex index they correspond to.
- `cpu_kifirst`: the `stride=1` dimension is the dimension of the `k` levels. Edges and vertices are ordered based on the vertex index they correspond to.
- `gpu_naive/gpu`: naive GPU implementation where each thread processes one k-level. Edges and vertices are ordered based on their orientation so that edges with the same orientation are grouped together
- `gpu_kloop`: similar as `gpu_naive/gpu` but each thread processes more than 1 k-level (k blocking)

On top of these handwritten implementation there is a `gtfn` implementation as well. This one was generated automatically by `gt4py` for the `nabla4` kernel and was slightly adapted for the purposes of the benchmark framework.

**For more information regarding the currently implemented experiments have a look at the [slides](https://github.com/GridTools/icon_structured_benchmark/blob/main/icon-structured-slides.pdf).**

## Installation

For running the benchmarking framework with its tests there are the following dependencies:

1. `CMake`
2. `Boost`
3. [icon4py](https://github.com/C2SM/icon4py)
4. [GridTools C++](https://github.com/GridTools/gridtools)
5. Other python packages in `requirements.txt`

### Instructions

First it's helpful to have a Python virtual environment set up with Python 3.11+.

`icon_structured_benchmark` has some Python dependencies to run the Python benchmark and validation scripts as well as the plotting scripts.
Those are defined in the `requirements.txt` file in the root directory of the repository.
By installing the requirements in the following way, `icon4py` from the submodules is also installed.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To install `icon-structured` you need to use `CMake`. `GridTools` will also be cloned in the `CMake` build directory if not found by `CMake`.

Important CMake options:

- `IS_GPU`: Enable GPU execution
- `INDEX_TYPE`: Set the type of the indexes of the neighbor tables (Default is `int`)
- `CMAKE_CUDA_ARCHITECTURES`: Set the correct CUDA compute capability for the GPU being used
- `CMAKE_CUDA_FLAGS="-diag-suppress 177 --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr"`: some extra flags useful for CUDA compilation
- `CMAKE_{CXX,CUDA}_FLAGS="-DNDEBUG"`: needs to be set for CXX and CUDA compilation to avoid GridTools checks and overhead when measuring performance
- `CMAKE_PREFIX_PATH=<GRIDTOOLS_INSTALLATION_PATH>`: useful to avoid cloning GridTools in each build folder. It's important to install `GridTools` with the same compiler and dependencies as `icon-structured-benchmarks`.

```
pip install -r requirements.txt
mkdir build
pushd build
cmake ..
cmake --build . <EXTRA_CMAKE_OPTIONS>
                # all executables at the moment are build inside the `build` folder
                # take into account what C++ compiler flags are needed to get best performance
                # add them with -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS=<cpp_compiler_flags>
popd
```

## Instructions to run benchmark and plot results

### TL;DR

See `build_bench_all.sh` on how to run and plot everything

### Running benchmark

The benchmark drivers are written in `python`.
There are 3 different scripts for executing the different kernels implemented that can be found in the root directory of the repository.

- `run_filtered_torus_grid_int_nabla4.py` is for executing all the standalone `nabla4` implementations
- `run_filtered_torus_grid_int_inteprolate.py` is for executing all the standalone `interpolate` implementations
- `run_filtered_torus_grid_int_nabla4_interpolate.py` is for executing all the combinations of `nabla4`, `nabla4` and `interpolate` kernels. **This is the main script for benchmarking**

```
python3.11 run_filtered_torus_grid_int_nabla4_interpolate.py --help
usage: run_filtered_torus_grid_int_nabla4_interpolate.py [-h] [--transformation {gt4py,index}]
                                                         [--klevels KLEVELS]
                                                         [--repetitions REPETITIONS] [--dry-run]
                                                         [--output OUTPUT] [--sanity-checks]
                                                         [--backend {all_cpu,all_gpu,gtfn_cpu,gtfn_gpu,naive,cpu_ifirst,cpu_kfirst,gpu_kloop,gpu_naive}]
                                                         [--combination {separate,inlined,all}]
                                                         [--e2c2v-ordering {per-vertex,per-orientation}]
                                                         [--halo HALO] [--vertical]
                                                         grid

positional arguments:
  grid                  Torus grid netCDF4 file

options:
  -h, --help            show this help message and exit
  --transformation {gt4py,index}
                        Use either ToGt4PyTransformation or IndexTransformation (gt4py by default)
  --klevels KLEVELS     Number of k levels (80 default)
  --repetitions REPETITIONS
                        Number of repetitions (101 default)
  --dry-run             Enable dry runs (not taken into runtime results) (disabled by default)
  --output OUTPUT       JSON output file name
  --sanity-checks       Do a validation with random data between structured and unstructured for the
                        given grid (disabled by default)
  --backend {all_cpu,all_gpu,gtfn_cpu,gtfn_gpu,naive,cpu_ifirst,cpu_kfirst,gpu_kloop,gpu_naive}
                        Which backend to benchmark (default all_cpu)
  --combination {separate,inlined,all}
                        Which combination of kernels to benchmark
  --e2c2v-ordering {per-vertex,per-orientation}
                        E2C2V ordering (per-vertex in CPU and per-orienteation in GPU by default)
  --halo HALO           Halo size for structured grids (default 2) [Shouldn't be changed]
  --vertical            Use nabla4_vertical kernel instead of nabla4 (disabled by default)
```

For examples on how to execute the python script to gather results for multiple configurations have a look at these scripts: [run_nabla4_interpolate.sh](https://github.com/GridTools/icon_structured_benchmark/blob/main/run_nabla4_interpolate.sh) and [run_nabla4_vertical_interpolate.sh](https://github.com/GridTools/icon_structured_benchmark/blob/main/run_nabla4_vertical_interpolate.sh). These scripts will gather the runtimes for all the experiments executed in `JSON` form in a certain folder set inside them.

### Plotting torus results

After generating the runtime data you can create plots with the median runtimes using the following scripts found on the root directory of this repository:

- `analysis_halo_gpu_nabla4_interpolate.py` for the combinations of `nabla4` and `interpolate` kernels
- `analysis_halo_gpu_nabla4_vertical_interpolate.py` for the combinations of `nabla4-vertical` and `interpolate` kernels

The run scripts and the analysis scripts use the same naming conventions to make it easy to plot the data. By setting the necessary variables inside the scripts it's possible to easily generate and plot the data.

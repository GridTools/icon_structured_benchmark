# icon-structured

Standalone benchmark that evaluates the performance of a structured grid compared to an unstructured grid using the [nabla4 kernel from icon4py](https://github.com/C2SM/icon4py/blob/main/model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/stencils/calculate_nabla4.py#L23) and its combination with the `mo_intp_rbf_rbf_vec_interpol_vertex kernel`[https://github.com/C2SM/icon4py/blob/main/model/common/src/icon4py/model/common/interpolation/stencils/mo_intp_rbf_rbf_vec_interpol_vertex.py#L41].

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

## Installation

For running the benchmarking framework with its tests there are the following dependencies:

1. [serialbox](https://github.com/GridTools/serialbox)
2. [icon4py](https://github.com/C2SM/icon4py)
3. Other python packages in `requirements.txt`

To install `serialbox` you can do the following:

```
git clone https://github.com/GridTools/serialbox.git
cd serialbox
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX=./install -DSERIALBOX_ENABLE_PYTHON=ON
cmake --build . --target install
export PYTHONPATH=$(pwd)/install/python:$PYTHONPATH
popd
```

To install `icon4py`:

```
git submodule update --init --recursive -f
pushd tests/ext/icon4py
pip install -r requirements-dev-opt.txt # ideally in a venv (tested with Python 3.11)
popd
```

To install `icon-structured`:

Important CMake options:

- `IS_GPU`: Enable GPU execution
- `INDEX_TYPE`: Set the type of the indexes of the neighbor tables (Default is `int`
  )
- `CMAKE_CUDA_ARCHITECTURES`: Set the correct CUDA compute capability for the GPU being used
- `CMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr"`: some extra flags useful for CUDA compilation
- `CMAKE_CXX/CUDA_FLAGS="-DNDEBUG"`: needs to be set for CXX and CUDA compilation to avoid GridTools checks and overhead when measuring performance
- `CMAKE_PREFIX_PATH=<GRIDTOOLS_INSTALLATION_PATH>`: useful to avoid cloning GridTools in each build folder

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

### Running benchmark

The benchmark driver is written in `python`.
To run the benchmark and provide different options one can use the `run_simple_grid.py` for benchmarking the `SimpleGrid`, the `run_torus_grid.py` for the `torus` grid and `run_filtered_torus_grid.py` for the torus grid with `halo = 2` (no periodic boundaries) and other flavors of `run_filtered_torus_grid.py` based on each experiment.
To get the available options you can run the following:

```
$ python run_simple_grid.py --help
usage: run_simple_grid.py [-h] [--klevels KLEVELS] [--repetitions REPETITIONS] [--dry-run] grid

positional arguments:
  grid                  SimpleGrid netCDF4 file

options:
  -h, --help            show this help message and exit
  --klevels KLEVELS     Number of k levels
  --repetitions REPETITIONS
                        Number of repetitions
  --dry-run             Do a dry run or not

$ python run_torus_grid.py --help  # (same as run_filtered_torus_grid.py)
usage: run_torus_grid.py [-h] [--transformation {gt4py,index}] [--klevels KLEVELS] [--repetitions REPETITIONS] [--dry-run] [--output OUTPUT]
                         [--sanity-checks]
                         grid

positional arguments:
  grid                  Torus grid netCDF4 file

options:
  -h, --help            show this help message and exit
  --transformation {gt4py,index}
                        Use either ToGt4PyTransformation or IndexTransformation
  --klevels KLEVELS     Number of k levels
  --repetitions REPETITIONS
                        Number of repetitions
  --dry-run             Do a dry run or not
  --output OUTPUT       JSON output file name
  --sanity-checks       Do a validation with random data between structured and unstructured for the given grid
  --backend {all,gtfn,naive,cpu_ifirst,cpu_kfirst}
                        Which backend to benchmark
```

### Plotting torus results

After running `run_torus_grid.py` or `run_filtered_torus_grid.py` the runtimes for every implementation will be dumped into a JSON file named based on the `OUTPUT` argument.

For gathering the data one can use a script like `run_grids.sh` that does a sweep to the different torus files and `k` levels.
After dumping all the JSON files in a directory, one can read and create plots with the acceleration of `structured` over `unstructured` and violin plots with the actual runtimes using `analysis.py` for the torus grid with periodic boundaries or the `analysis_halo.py` for the same grid without periodic boundaries (halo = 2). Other plotting scripts also exist to plot results for the different expleriments (`analysis_halo_*.py`).
For runs with 1 output per rank, one can use `analysis_multi.py` and `analysis_multi_halo.py`.

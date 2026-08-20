#!/bin/bash
#SBATCH --job-name=icon_str
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --uenv=prgenv-gnu/26.6:2584154534@santis
#SBATCH --view=default
#SBATCH -A csstaff

# nvcc version > 13.3 necessary for cutile implemenmtations

# Below are necessary first
# python -m venv .venv
# pip install -r requirements.txt

# Download necessary torus grids using the links below:
# torus_100000_100000_256.nc: https://polybox.ethz.ch/index.php/s/BnGUygPcApKycAv
# torus_100000_100000_1024.nc: https://polybox.ethz.ch/index.php/s/JDYRnEc9Ww64Btr

source .venv/bin/activate

BUILD_DIR="build_all_gpu"
mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -fPIC -DNDEBUG" -DIS_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DNDEBUG" -DINDEX_TYPE=int
cmake --build . --parallel 16
popd

export PYTHONPATH=$(pwd)/${BUILD_DIR}:$PYTHONPATH

GIT_COMMIT="$(git rev-parse --short HEAD)"

SANITY_GRID="torus_100000_100000_1024.nc"
BENCHMARK_GRID="torus_100000_100000_256.nc"
TORUS_FILENAME=$(echo "${BENCHMARK_GRID}" | sed 's/\.nc$//')
TORUS_SIZE=$(echo "${TORUS_FILENAME}" | cut -d'_' -f4)

IMPLEMENTATIONS="all_gpu" # "all_cpu"

for IMPLEMENTATION in ${IMPLEMENTATIONS}; do
    echo "###### Running implementation: ${IMPLEMENTATION} ######"
    echo "###### Running sanity checks on grid: ${SANITY_GRID} ######"
    python run_filtered_torus_grid_int_nabla4.py ${SANITY_GRID} --backend "${IMPLEMENTATION}" --sanity-checks --repetitions 1
    python run_filtered_torus_grid_int_nabla4_interpolate.py ${SANITY_GRID} --backend "${IMPLEMENTATION}" --sanity-checks --repetitions 1
    python run_filtered_torus_grid_int_nabla4_interpolate_c2v.py ${SANITY_GRID} --backend "${IMPLEMENTATION}" --sanity-checks --repetitions 1
    echo "###### Running sanity checks done ######"

    echo "###### Running nabla4 benchmarks on grid: ${BENCHMARK_GRID} ######"
    mkdir -p results/nabla4_output_${GIT_COMMIT}_gpu_int
    python run_filtered_torus_grid_int_nabla4.py ${BENCHMARK_GRID} --backend "${IMPLEMENTATION}" --dry-run --output results/nabla4_output_${GIT_COMMIT}_gpu_int/${TORUS_FILENAME}_k80_${GIT_COMMIT}_gpu_int
    python analysis_halo_gpu_nabla4.py
    echo "###### Running nabla4 benchmarks done ######"

    echo "###### Running nabla4 interpolate benchmarks on grid: ${BENCHMARK_GRID} ######"
    mkdir -p results/nabla4_interpolate_output_${GIT_COMMIT}_gpu_int_${IMPLEMENTATION}
    python run_filtered_torus_grid_int_nabla4_interpolate.py ${BENCHMARK_GRID} --backend "${IMPLEMENTATION}" --dry-run --output results/nabla4_interpolate_output_${GIT_COMMIT}_gpu_int_${IMPLEMENTATION}/ni_${TORUS_FILENAME}_k80_${GIT_COMMIT}_gpu_int
    python analysis_halo_gpu_nabla4_interpolate.py
    echo "###### Running nabla4 interpolate benchmarks done ######"

    echo "###### Running nabla4 interpolate verts2cells benchmarks on grid: ${BENCHMARK_GRID} ######"
    mkdir -p results/nabla4_interpolate_verts2cells_output_${GIT_COMMIT}_gpu_int_${IMPLEMENTATION}
    python run_filtered_torus_grid_int_nabla4_interpolate_c2v.py ${BENCHMARK_GRID} --backend "${IMPLEMENTATION}" --dry-run --output results/nabla4_interpolate_verts2cells_output_${GIT_COMMIT}_gpu_int_${IMPLEMENTATION}/niv_${TORUS_FILENAME}_k80_${GIT_COMMIT}_gpu_int
    python analysis_halo_gpu_nabla4_interpolate_verts2cells.py
    echo "###### Running nabla4 interpolate verts2cells benchmarks done ######"

    echo "###### Running ncu profiling for nabla4 on grid: ${BENCHMARK_GRID} ######"
    ncu --set full -f --import-source yes -o icon_structured_${TORUS_SIZE}_${IMPLEMENTATION} python run_filtered_torus_grid_int_nabla4.py ${BENCHMARK_GRID} --backend ${IMPLEMENTATION} --repetitions 1
    echo "###### Running ncu profiling for nabla4 done ######"
    echo "###### Running ncu profiling for nabla4 interpolate on grid: ${BENCHMARK_GRID} ######"
    ncu --set full -f --import-source yes -o icon_structured_${TORUS_SIZE}_${IMPLEMENTATION} python run_filtered_torus_grid_int_nabla4_interpolate.py ${BENCHMARK_GRID} --backend ${IMPLEMENTATION} --repetitions 1
    echo "###### Running ncu profiling for nabla4 interpolate done ######"
    echo "###### Running ncu profiling for nabla4 interpolate verts2cells on grid: ${BENCHMARK_GRID} ######"
    ncu --set full -f --import-source yes -o icon_structured_${TORUS_SIZE}_${IMPLEMENTATION} python run_filtered_torus_grid_int_nabla4_interpolate_c2v.py ${BENCHMARK_GRID} --backend ${IMPLEMENTATION} --repetitions 1
    echo "###### Running ncu profiling for nabla4 interpolate verts2cells done ######"
done

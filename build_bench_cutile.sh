#!/bin/bash
#SBATCH --job-name=icon_str_cutile
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --uenv=prgenv-gnu/26.6:2584154534@santis,/capstor/scratch/cscs/ioannmag/cycle37/icon_structured_benchmark/py_venv.squashfs:/capstor/scratch/cscs/ioannmag/cycle37/icon_structured_benchmark/.venv
#SBATCH --view=default
#SBATCH -A csstaff

source .venv/bin/activate

pushd build_cutile
cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -fPIC -DNDEBUG" -DIS_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DNDEBUG" -DINDEX_TYPE=int
cmake --build . --parallel 4
popd

export PYTHONPATH=$(pwd)/build_cutile:$PYTHONPATH

SANITY_GRID="torus_100000_100000_1024.nc"
BENCHMARK_GRID="torus_100000_100000_128.nc"

IMPLEMENTATION="all_gpu"

# python run_filtered_torus_grid_int_nabla4.py ${SANITY_GRID} --backend "${IMPLEMENTATION}" --sanity-checks --repetitions 1

python run_filtered_torus_grid_int_nabla4.py ${BENCHMARK_GRID} --backend "${IMPLEMENTATION}" --dry-run

ncu --set full -f --import-source yes -o icon_structured_$(echo "${BENCHMARK_GRID##*_}" | sed 's/\.nc$//')_${IMPLEMENTATION} python run_filtered_torus_grid_int_nabla4.py ${BENCHMARK_GRID} --backend "${IMPLEMENTATION}" --repetitions 1

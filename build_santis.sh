#!/usr/bin/env bash

set -e

source venv_py311_icon4py/bin/activate

COMMIT="6ad3dc2"

BACKENDS="cpu gpu"

INDEX_TYPES="sizet int64 uint32 int"

for backend in ${BACKENDS}; do
    for index_type in ${INDEX_TYPES}; do
        BUILD_DIR="build_${COMMIT}_${backend}_${index_type}"
        echo "Building ${BUILD_DIR}"
        mkdir ${BUILD_DIR} || true
        pushd ${BUILD_DIR}
        if [[ ${backend} == "cpu" ]]; then
            cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -msve-vector-bits=128 -fopt-info-vec-missed -fvect-cost-model=unlimited -fPIC -fopenmp-simd -DNDEBUG --param vect-max-version-for-alias-checks=50" -DINDEX_TYPE=${index_type}
        else
            cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -fPIC -DNDEBUG" -DIS_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DNDEBUG" -DINDEX_TYPE=${index_type}
        fi
        cmake --build . --parallel 1 2>&1 | tee build_santis.log
        popd
    done
done

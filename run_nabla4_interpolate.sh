#!/bin/bash -l

GRIDS_DIRECTORY="" # Set directory with grid files

KLEVELS="80"
REPETITIONS=101

source venv_py311/bin/activate # Set virtual environment path properly

COMMITS="979c2b9"

BACKEND="gpu"

INDEX_TYPE="int"

SUFFIX="_all_gpu_101"

ORIGINAL_PYTHONPATH=$PYTHONPATH

for commit in $COMMITS; do

    BUILD_DIR="<BUILD_DIR>/build_${commit}" # Set build directory. Currently based on the COMMIT

    if [[ ! -d $BUILD_DIR ]]; then
        echo "Build directory ${BUILD_DIR} does not exist"
        exit 1
    fi

    export PYTHONPATH=${BUILD_DIR}:$ORIGINAL_PYTHONPATH

    GRID_FILES="${GRIDS_DIRECTORY}/torus_100000_100000_256.nc ${GRIDS_DIRECTORY}/torus_100000_100000_128.nc"

    OUTPUT_DIR="nabla4_interpolate_output_${commit}_${BACKEND}_${INDEX_TYPE}${SUFFIX}"

    echo "Launching benchmark with build ${BUILD_DIR}"

    set -e
    mkdir $OUTPUT_DIR
    set +e

    for grid_file in ${GRID_FILES}; do
        echo "Running grid ${grid_file}"
        grid_name=$(echo $grid_file | grep -E "torus_100000_100000_[0-9]*" -o)
        for k in $KLEVELS; do
            FILE_NAME="${grid_name}_k${k}_${commit}_${BACKEND}_${INDEX_TYPE}"
            # Make sure that libnvrtc.so is in the LD_LIBRARY_PATH
            LD_LIBRARY_PATH="<PATH_TO_libnvrtc.so>":$LD_LIBRARY_PATH python3.11 run_filtered_torus_grid_int_nabla4_interpolate.py ${grid_file} --klevels $k --repetitions $REPETITIONS --output ${FILE_NAME}.json --backend all_gpu --dry-run
            if [ $? -ne 0 ]; then
                echo "Grid ${grid_file} failed with kevels $k"
            fi
            mv ${FILE_NAME}.json $OUTPUT_DIR
        done
    done
done

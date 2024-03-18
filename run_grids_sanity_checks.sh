#!/usr/bin/env bash

GRIDS_DIRECTORY="/Users/ioannmag/cscs_repos/cycle20"

KLEVELS=1
REPETITIONS=1

source venv_py311_icon4py/bin/activate

export PYTHONPATH=/Users/ioannmag/cscs_repos/cycle20/icon-structured/build_santis:/Users/ioannmag/cscs_repos/serialbox/build/install/python:$PYTHONPATH

for grid_file in ${GRIDS_DIRECTORY}/torus_100000_100000_*.nc; do
    echo "Running grid ${grid_file}"
    grid_name=$(echo $grid_file | grep -E "torus_100000_100000_[0-9]*" -o)
    python3.11 run_torus_grid.py ${grid_file} --klevels $KLEVELS --repetitions $REPETITIONS --sanity-checks --output $grid_name.pdf
    if [ $? -ne 0 ]; then
        echo "Grid ${grid_file} failed"
    else
        echo "Grid ${grid_file} passed"
    fi
done

#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make st_vs_bdst -j$(nproc)

echo "Running st_vs_bdst..."
taskset -c 0 ./st_vs_bdst

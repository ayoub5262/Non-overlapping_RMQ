#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make run_benchmark3 -j$(nproc)

echo "Running run_benchmark3..."
taskset -c 0 ./run_benchmark3

#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make run_benchmark2 -j$(nproc)

echo "Running run_benchmark2..."
taskset -c 0 ./run_benchmark2

#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make run_benchmark1 -j$(nproc)

echo "Running run_benchmark1..."
taskset -c 0 ./run_benchmark1

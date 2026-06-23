#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make run_benchmark -j$(nproc)

echo "Running run_benchmark..."
./run_benchmark

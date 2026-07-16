#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make run_benchmark3 -j$(nproc)

echo "Running run_benchmark3..."
./run_benchmark2

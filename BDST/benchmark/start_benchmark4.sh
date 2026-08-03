#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make run_benchmark4 -j$(nproc)

echo "Running run_benchmark4..."
./run_benchmark4

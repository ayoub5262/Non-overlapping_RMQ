#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make run_test -j$(nproc)

echo "Running run_test..."
taskset -c 6 ./run_test

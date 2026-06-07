#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make main_bench -j$(nproc)
./main_bench --benchmark_out=result2.csv --benchmark_out_format=csv

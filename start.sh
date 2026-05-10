#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./rmq_benchmarks --benchmark_format=csv | tee results.csv
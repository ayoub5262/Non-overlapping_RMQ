#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./rmq_bench | tee results_console.txt
echo ""
echo "CSV saved to: build/results.csv"

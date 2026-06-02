#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) rmq_bench4
./rmq_bench4 | tee result4_console.txt
echo ""
echo "CSV saved to: build/result4.csv"

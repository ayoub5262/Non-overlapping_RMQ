#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) rmq_bench3
./rmq_bench3 | tee result3_console.txt
echo ""
echo "CSV saved to: build/result3.csv"

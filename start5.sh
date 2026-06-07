#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make rmq_bench5 -j$(nproc)

echo "Running rmq_bench5..."
./rmq_bench5

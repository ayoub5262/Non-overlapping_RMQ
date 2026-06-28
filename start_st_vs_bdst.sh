#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make st_vs_bddst -j$(nproc)

echo "Running st_vs_bddst..."
./st_vs_bddst

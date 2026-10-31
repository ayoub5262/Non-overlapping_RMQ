# Space-Efficient and Cache-Optimized Range Minimum Query (RMQ) Data Structures

This repository contains the source code, experimental implementations, and benchmarking suite developed for a Bachelor's Thesis on static Range Minimum Query (RMQ) data structures. 

The primary contribution of this work is the design, implementation, and empirical analysis of **Blocked Disjoint Sparse Tables (BDST)** and their 1-bit decision vector variants. The project explores space-time trade-offs between macro/mini-block hierarchical decompositions, micro-architectural bit-parallel word scans, succinct rank/select dictionaries, and comparisons against established succinct, learned, and classical RMQ baselines.

---

## Table of Contents

- [Overview](#overview)
- [Key Features and Theoretical Background](#key-features-and-theoretical-background)
- [Repository Structure](#repository-structure)
- [Implemented Data Structures](#implemented-data-structures)
  - [Blocked Disjoint Sparse Tables](#blocked-disjoint-sparse-tables)
  - [1-Bit Decision Vector BDST Variants](#1-bit-decision-vector-bdst-variants)
  - [Blocked Segment Trees](#blocked-segment-trees)
  - [Compressed Sparse Table](#compressed-sparse-table)
  - [Intra-Block Scanning Kernels](#intra-block-scanning-kernels)
- [Compared Baselines](#compared-baselines)
- [Benchmarking and Evaluation](#benchmarking-and-evaluation)
  - [Evaluation Datasets](#evaluation-datasets)
  - [Measured Metrics](#measured-metrics)
  - [Benchmark Programs](#benchmark-programs)
- [Prerequisites and Dependencies](#prerequisites-and-dependencies)
- [Build and Usage](#build-and-usage)
  - [Compiling the Suite](#compiling-the-suite)
  - [Running Correctness Tests](#running-correctness-tests)
  - [Running Benchmarks](#running-benchmarks)
- [License and References](#license-and-references)

---

## Overview

Given an array $A[0 \dots n-1]$ of comparable elements, a Range Minimum Query $\text{RMQ}(l, r)$ returns the index of the minimum value in the subarray $A[l \dots r]$:

$$\text{RMQ}_A(l, r) = \arg\min_{l \le i \le r} A[i]$$

While classical Disjoint Sparse Tables (DST) achieve $O(1)$ query time with $O(n \log n)$ construction time and space, block decomposition divides the array into blocks of size $B$, reducing the top-level table size to $O((n/B) \log(n/B))$. 

This repository explores two major optimization directions:
1. **1-Bit Decision Vectors**: Replacing multi-bit indices in the DST levels with single-bit flags indicating prefix and suffix running minimum updates.
2. **Two-Level Hierarchical Decompositions**: Introducing mini-blocks within macro-blocks to limit linear scan penalties at query boundaries while maintaining low memory footprints.

---

## Key Features and Theoretical Background

- **Hardware Bit Manipulation**: Utilizes modern x86-64 bitwise operations (`std::countr_zero`, `std::countl_zero` / POPCNT / LZCNT) to scan 64-bit words in $O(1)$ time within CPU L1 cache, avoiding pointer-chasing rank/select structures.
- **Succinct Rank/Select Interfacing**: Implements adapters using SDSL rank and select dictionaries (`rank_support_v5`, `select_support_mcl`, and interleaved `bit_vector_il`) to compare theoretical $O(1)$ dictionary structures against hardware word scanning.
- **Ragged Boundary Scanners**: Employs two-phase mini-block scans that avoid costly random data lookups by first checking compact cached summary values.
- **Micro-Profiling**: Integrates `malloc_count` to capture precise runtime heap allocations and peak memory usage during table construction and queries.

---

## Repository Structure

```
.
|-- BDST/
|   |-- include/                         # Header-only RMQ implementations
|   |   |-- BDST.hpp                     # Standard Blocked Disjoint Sparse Table
|   |   |-- BDST1.hpp                    # Compact single-level BDST
|   |   |-- BDST2.hpp                    # Two-level (macro/mini-block) BDST
|   |   |-- BDST_1Bit_wordscan.hpp       # 1-bit BDST with 64-bit hardware word scanning
|   |   |-- BDST_1Bit_Fast.hpp           # Optimized two-level 1-bit BDST with word scan
|   |   |-- BDST_1Bit_select_mcl.hpp     # 1-bit BDST using SDSL Clark's select_support_mcl
|   |   |-- BDST_1Bit_select_il.hpp      # 1-bit BDST using SDSL interleaved bitvector (il)
|   |   |-- Blocked_SegmentTree.hpp      # Blocked iterative segment tree over block minima
|   |   |-- Blocked_SegmentTree_Fast.hpp # Two-level blocked segment tree with mini-blocks
|   |   |-- ST_IntVec.hpp                # SDSL bit-packed integer vector Sparse Table
|   |   `-- ScanStrategies.hpp           # Low-level scanning kernels (SIMD, WordScan, MiniBlock)
|   `-- benchmark/                       # Testing and benchmarking suite
|       |-- CMakeLists.txt               # CMake configuration
|       |-- benchmark_common.hpp         # Framework, data generators, and CSV writers
|       |-- test.cpp                     # Correctness verification against SDSL ground truth
|       |-- ST_vs_BDST.cpp               # Comparative study: ST_IntVec vs BDST_1Bit
|       |-- benchmark1.cpp               # Out-of-block scanning: WordScan vs select_il vs select_mcl
|       |-- benchmark2.cpp               # BDST_1Bit vs succinct RMQ structures
|       |-- benchmark3.cpp               # Blocked structures vs learned FL-RMQ
|       `-- benchmark4.cpp               # Two-level structures (BDST_Fast, BbST2, SDSL ST)
|-- BbST/                                # Blocked-based Sparse Table (Kowalski & Grabowski)
|-- FL-RMQ/                              # Fully-Learned RMQ baseline (Ferragina & Lari)
|-- rmq/                                 # Balanced Parentheses RMQ (BP_RMQ64)
|-- malloc_count/                        # Runtime memory profiling library
`-- sdsl-lite/                           # Succinct Data Structure Library v2
```

---

## Implemented Data Structures

### Blocked Disjoint Sparse Tables

- **`BDST`** (`BDST.hpp`):
  Divides input into blocks of size $B$. Precomputes intra-block minima in a base row and builds a classic Disjoint Sparse Table over block minima. Query resolves interior blocks with $O(1)$ DST lookups and scans the fringe blocks linearly.
- **`BDST1`** (`BDST1.hpp`):
  A compact single-level block variant that stores relative intra-block offsets and caches base block values to minimize dereference latency.
- **`BDST2`** (`BDST2.hpp`):
  A hierarchical two-level block decomposition using macro-blocks and mini-blocks. Queries use ragged boundary scans over mini-blocks to avoid scanning full macro-blocks.

### 1-Bit Decision Vector BDST Variants

Rather than storing $O(\frac{n}{B} \log \frac{n}{B} \log n)$ bits of index data, 1-bit variants represent the Disjoint Sparse Table levels using binary vectors where a `1` denotes the arrival of a new prefix or suffix running minimum:

- **`BDST_1Bit`** (`BDST_1Bit_wordscan.hpp`):
  Uses 64-bit integer words directly. Prefix/suffix queries are resolved using `WordScan` with `std::countr_zero` and `std::countl_zero` instructions.
- **`BDST_1Bit_Fast`** (`BDST_1Bit_Fast.hpp`):
  Combines the 1-bit `WordScan` approach with two-level mini-block decomposition, value caching, and early exit short-circuits.
- **`BDST_1Bit_select_mcl`** (`BDST_1Bit_select_mcl.hpp`):
  Backs decision vectors with SDSL `rank_support_v5` and `select_support_mcl` to evaluate theoretical constant-time dictionary structures.
- **`BDST_1Bit_select_il`** (`BDST_1Bit_select_il.hpp`):
  Uses interleaved bitvectors (`bit_vector_il<512>`) to trade space for improved cache locality in rank/select operations.

### Blocked Segment Trees

- **`Blocked_SegmentTree`** (`Blocked_SegmentTree.hpp`):
  Replaces the Disjoint Sparse Table over block minima with an iterative, heap-indexed segment tree.
- **`Blocked_SegmentTree_Fast`** (`Blocked_SegmentTree_Fast.hpp`):
  Augments the blocked segment tree with two-level mini-block structures and early query termination checks.

### Compressed Sparse Table

- **`ST_IntVec`** (`ST_IntVec.hpp`):
  A classical Sparse Table whose levels are compressed using SDSL variable bit-width integer vectors (`sdsl::int_vector<0>`), encoding relative offsets instead of absolute 64-bit indices.

### Intra-Block Scanning Kernels

Defined in `ScanStrategies.hpp`:
- **`LinearScan`**: Baseline sequential scan.
- **`LinearSkipScan`**: Early-stopping scan that halts upon encountering a known lower bound.
- **`OneLookUpLinearScan`**: Caches the current minimum in a register to reduce memory reads.
- **`MiniBlockScanner`**: Two-stage ragged boundary scanner for hierarchical mini-blocks.
- **`WordScan`**: Direct bitwise traversal over 64-bit integers using hardware count-trailing/leading zeros.

---

## Compared Baselines

The suite compares custom implementations against established state-of-the-art RMQ architectures:

| Baseline | Category | Source / Paper |
| :--- | :--- | :--- |
| **SDSL Sparse Table** | Classical Index | SDSL (`sdsl::rmq_support_sparse_table`) |
| **SDSL Succinct Rec** | Succinct Tree | Fischer-Heun 2-level recursive (`rmq_succinct_rec_new`) |
| **BP_RMQ64** | Succinct (BP) | Balanced Parentheses with 64-bit RMM trees |
| **BbST / BbST2** | Block-based ST | Kowalski & Grabowski (Blocked-based Sparse Tables) |
| **FL-RMQ** | Learned Index | Ferragina & Lari (CPM 2025: Fully-Learned RMQ) |

---

## Benchmarking and Evaluation

### Evaluation Datasets

The benchmarking harness generates four distinct data profiles across scales up to $10^9$ elements:
- **`RAND`**: Uniformly distributed random permutations of integers $[0, n-1]$.
- **`LCP`**: Longest Common Prefix arrays constructed from real-world text corpora (`english.1024MB`) or random character distributions.
- **`INC`**: Monotonically increasing trends with uniform bounded noise.
- **`DEC`**: Monotonically decreasing trends with uniform bounded noise.

### Measured Metrics

Each benchmark run captures:
- **Construction Time**: Time in nanoseconds/milliseconds to build the data structure.
- **Query Latency**: Execution time per query (ns) averaged across $10^4$ lookups, tested across multiple range length buckets ($10^1$ to $10^7$).
- **Bits per Element**: Final heap memory consumption divided by $n$.
- **Peak Bits per Element**: Peak transient allocation during construction divided by $n$ (via `malloc_count`).

### Benchmark Programs

- **`run_test`**: Verifies correctness across all structures against SDSL RMQ ground truth.
- **`st_vs_bdst`**: Analyzes space and query scaling between `ST_IntVec` and `BDST_1Bit`.
- **`run_benchmark1`**: Evaluates 1-bit BDST scanning strategies (`WordScan` vs `select_il` vs `select_mcl`).
- **`run_benchmark2`**: Compares `BDST_1Bit` against succinct baselines (`SDSL_rec_new` and `BP_RMQ64`).
- **`run_benchmark3`**: Analyzes blocked structures (`BDST`, `Blocked_SegmentTree`) against learned `FL-RMQ` across varying block sizes $\epsilon \in \{256, 512, 1024, 2048\}$.
- **`run_benchmark4`**: Profiles two-level hierarchical structures (`BDST_1Bit_Fast`, `BDST2`, `Blocked_SegmentTree_Fast`, `BbST2`) and SDSL Sparse Table.

---

## Prerequisites and Dependencies

- **Operating System**: Linux (Ubuntu 20.04+ or WSL2 recommended)
- **Compiler**: GCC 11+ or Clang 13+ with C++20 support (`-std=c++20`, `-march=native`)
- **Build System**: CMake 3.14 or newer
- **Required Libraries**:
  - `libdivsufsort` and `libdivsufsort64` (required by SDSL)
  - `libdl` (standard POSIX dynamic linker)
  - SDSL-lite (included as submodule/subfolder in `sdsl-lite/`)

On Debian/Ubuntu-based systems:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libdivsufsort-dev
```

---

## Build and Usage

### Compiling the Suite

All executables are managed via CMake inside `BDST/benchmark/`:

```bash
cd BDST/benchmark
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

This compiles:
- `run_test`
- `st_vs_bdst`
- `run_benchmark1`
- `run_benchmark2`
- `run_benchmark3`
- `run_benchmark4`

### Running Correctness Tests

Run the test suite to verify all implementations against the ground truth:

```bash
./run_test
```

### Running Benchmarks

Each benchmark outputs progress to `stdout` and writes detailed results to a CSV file in the current working directory:

```bash
# Compare 1-bit scan strategies (outputs benchmark1.csv)
./run_benchmark1

# Compare BDST against succinct RMQ baselines (outputs benchmark2.csv)
./run_benchmark2

# Compare against learned FL-RMQ (outputs benchmark3.csv)
./run_benchmark3

# Benchmark two-level mini-block structures (outputs benchmark4.csv)
./run_benchmark4
```

CSV files contain the following schema:
```csv
algorithm,block_size,size,data_type,scenario,query_range,cpu_time_ns,bits_per_element,bits_per_element_peak
```

---


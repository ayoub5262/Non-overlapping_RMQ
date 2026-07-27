/**
 * @file dst.hpp
 * @brief Classical Disjoint Sparse Table (DST).
 *
 * Provides a canonical O(N log N) space, O(1) query latency Disjoint Sparse Table
 * implementation used as an uncompressed, highly stable latency baseline.
 */

#pragma once

#include <vector>
#include <bit>
#include <algorithm>
#include <cstdint>
#include <sdsl/int_vector.hpp>

class DisjointSparseTable {
    const std::vector<int>& data;
    std::vector<sdsl::int_vector<>> table;
    size_t n;

public:
    DisjointSparseTable(const std::vector<int>& arr) : data(arr){
        n = data.size();
        if (n <= 1) return;

        int levels { static_cast<int>(std::bit_width(static_cast<unsigned>(n - 1)) + 1) };
        
        table.resize(levels);

        for (int h {1}; h < levels; ++h) {
            table[h] = sdsl::int_vector<>(n, 0, h);
            size_t range {1ULL << h};
            size_t half {1ULL << (h - 1)};
            uint64_t mask {(1ULL << h) - 1};

            for (size_t center {half}; center < n; center += range) {
                // Compute the prefix minima to left using forward loop on offset k
                size_t min_idx { center - 1 };
                table[h][center - 1] = min_idx & mask;
                for (size_t k {1}; k < half; ++k) {
                    size_t i {center - 1 - k};
                    if (data[i] < data[min_idx]) {
                        min_idx = i;
                    }
                    table[h][i] = min_idx & mask;
                }
                // Compute the suffix minima to right
                size_t limit { std::min(n, center + half) };
                if (center < limit) {
                    min_idx = center;
                    table[h][center] = min_idx & mask;
                    for (size_t i {center + 1}; i < limit; ++i) {
                        if (data[i] < data[min_idx]) {
                            min_idx = i;
                        }
                        table[h][i] = min_idx & mask;
                    }
                }
            }
        }
    }

    uint32_t query(uint32_t l, uint32_t r) const {
        if (l == r) return l;
        // Find the highest level where l and r lie in different blocks
        int h {std::bit_width(l ^ r)};

        uint64_t mask {(1ULL << h) - 1};
        uint32_t l_min { static_cast<uint32_t>((static_cast<uint64_t>(l) & ~mask) | table[h][l]) };
        uint32_t r_min { static_cast<uint32_t>((static_cast<uint64_t>(r) & ~mask) | table[h][r]) };

        if (data[l_min] < data[r_min])
            return l_min;
        
        return r_min;
    }
};

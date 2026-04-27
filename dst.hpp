#pragma once

#include <vector>
#include <algorithm>
#include <bit>

class DisjointSparseTable {
    // Flat 1-D storage: row-major, table[h * n + i]
    // Avoids pointer-chasing from vector-of-vectors
    std::vector<int> table;
    int n, levels;

public:
    // Build  O(n log n) time │ O(n log n) space 
    explicit DisjointSparseTable(const std::vector<int>& data) {
        n = static_cast<int>(data.size());

        // Number of bits needed to cover n-1 elements, plus 1 for the 0-level
        levels = n <= 1 ? 1 : std::bit_width(static_cast<unsigned>(n - 1)) + 1;

        table.resize(levels * n);

        std::copy(data.begin(), data.end(), table.begin());

        for (int h {1}; h < levels; ++h) {
            int half { 1 << (h - 1)};
            int range { half << 1};
            int* row  { &table[h * n]}; // pointer to current row — no repeated multiply

            // The blocks (centers) are independent of each other perfect for multithreading.
            // #pragma omp parallel for schedule(dynamic) // Deaktiviert für reine Single-Thread Messungen
            for (int center = half; center < n; center += range) {
                // Left prefix-minima
                row[center - 1] = data[center - 1];
                for (int i {center - 2}; i >= center - half && i >= 0; --i)
                    row[i] = std::min(data[i], row[i + 1]);

                // Right suffix-minima 
                int rend { std::min(n, center + half)};
                row[center] = data[center];
                for (int i { center + 1}; i < rend; ++i)
                    row[i] = std::min(data[i], row[i - 1]);
            }
        }
    }

    int query(int l, int r) const {
        if (l == r) return table[l];

        // Extracted bit_width automatically corresponds to our shifted memory layout (1-indexed level)
        int h { std::bit_width(static_cast<unsigned>(l ^ r)) };

        const int* row { &table[h * n]};
        return std::min(row[l], row[r]);
    }

    size_t size_in_bytes() const {
        return table.size() * sizeof(int);
    }
};

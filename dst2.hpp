#pragma once

#include <vector>
#include <bit>
#include <algorithm>
#include <cmath>

class DisjointSparseTable {
    std::vector<std::vector<int>> table;
    int n;

public:
    DisjointSparseTable(const std::vector<int>& data) {
        n = data.size();
        int levels = 1;
        while ((1 << levels) < n) ++levels;
        ++levels;
        
        table.assign(levels, std::vector<int>(n));
        table[0] = data;

        for (int h = 1; h < levels; ++h) {
            int range = 1 << h;
            int half = 1 << (h - 1);
            for (int center = half; center < n; center += range) {
                // Berechne Präfix-Minima nach links
                table[h][center - 1] = data[center - 1];
                for (int i = center - 2; i >= center - half; --i) {
                    table[h][i] = std::min(data[i], table[h][i + 1]);
                }
                // Berechne Suffix-Minima nach rechts
                table[h][center] = data[center];
                for (int i = center + 1; i < std::min(n, center + half); ++i) {
                    table[h][i] = std::min(data[i], table[h][i - 1]);
                }
            }
        }
    }

    int query(int l, int r) const {
        if (l == r) return table[0][l];
        // Finde die höchste Ebene, in der l und r in verschiedenen Blöcken liegen
        int h = 31 - std::countl_zero(static_cast<unsigned>(l ^ r));

        return std::min(table[h][l], table[h][r]);
    }

    size_t size_in_bytes() const {
        return table.size() * n * sizeof(int);
    }
};

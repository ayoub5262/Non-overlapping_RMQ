#pragma once

#include <vector>
#include <bit>
#include <algorithm>

class DisjointSparseTable {
    std::vector<std::vector<int>> table;
    std::vector<int> raw_data;
    int n;

public:
    DisjointSparseTable(const std::vector<int>& data) : raw_data(data) {
        n = data.size();
        int levels = 1;
        while ((1 << levels) < n) ++levels;
        ++levels;
        
        table.assign(levels, std::vector<int>(n));

        for (int h {1}; h < levels; ++h) {
            int range = 1 << h;
            int half = 1 << (h - 1);
            for (int center {half}; center < n; center += range) {
                // Compute the prefix minima to left
                table[h][center - 1] = center - 1;
                for (int i {center - 2}; i >= center - half; --i) {
                    if (data[i] < data[table[h][i + 1]]){
                        table[h][i] = i;
                    } else {
                        table[h][i] = table[h][i + 1];
                    }
                }
                // Compute the suffix minima to right
                table[h][center] = center;
                for (int i {center + 1}; i < std::min(n, center + half); ++i) {
                    if (data[i] < data[table[h][i - 1]]){
                        table[h][i] = i;
                    } else {
                        table[h][i] = table[h][i - 1];
                    }
                }
            }
        }
    }

    int query(int l, int r) const {
        if (l == r) return l;
        // Find the highest level where l and r in different blocks lay
        int h = std::bit_width(static_cast<unsigned>(l ^ r));

        if (raw_data[table[h][l]] < raw_data[table[h][r]]){
            return table[h][l];
        } else {
            return table[h][r];
        }
    }

};

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>

#include "BlockScanStrategies.hpp"

template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class ST {
    std::unique_ptr<Tidx[]> table; 
    const std::vector<T>& data;
    Tidx BLOCK_SIZE, blocks, levels;

public:
    ST(const std::vector<T>& arr, const Tidx B_S) : data(arr), BLOCK_SIZE(B_S) {
        Tidx n { static_cast<Tidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks));
        table = std::make_unique_for_overwrite<Tidx[]>(levels * blocks);

        // Calculate minima per block (Base level h=0)
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx { i * BLOCK_SIZE };
            Tidx end { std::min(n, (i + 1) * BLOCK_SIZE) };
            if (end > i * BLOCK_SIZE + 1) {
                Tidx temp {QueryStrategy::scan_block(data, i * BLOCK_SIZE + 1, end - 1)};
                if (data[temp] < data[min_idx]) min_idx = temp;
            }
            table[i] = min_idx; 
        }

        // Build Sparse Table
        for (Tidx h { 1 }; h < levels; ++h) {
            Tidx step { Tidx(1) << (h - 1) };
            Tidx len { Tidx(1) << h };
            Tidx* row_curr { &table[h * blocks] };
            const Tidx* row_prev { &table[(h - 1) * blocks] };

            for (Tidx i { 0 }; i + len <= blocks; ++i) {
                Tidx left_min = row_prev[i];
                Tidx right_min = row_prev[i + step];
                row_curr[i] = data[left_min] < data[right_min] ? left_min : right_min;
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        Tidx b_l { l / BLOCK_SIZE };
        Tidx b_r { r / BLOCK_SIZE };
        Tidx res { l };

        // Case 1: Query completely within the same block : linear scan
        if (b_l == b_r) return QueryStrategy::scan_block(data, l, r);

        // Case 2: Query spans multiple blocks
        // Intermediate full blocks via Sparse Table
        if (Tidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            Tidx k { static_cast<Tidx>(std::bit_width(static_cast<unsigned>(qr - ql + 1)) - 1) };
            const Tidx* row { &table[k * blocks] };
            
            Tidx left_min = row[ql];
            Tidx right_min = row[qr - (1 << k) + 1];

            if (data[left_min] < data[res]) res = left_min;
            if (data[right_min] < data[res]) res = right_min;
        }

        // Left boundary 
        if (data[res] >= data[table[b_l]]){
            Tidx end_l { (b_l + 1) * BLOCK_SIZE };
            if(l + 1 < end_l){
                Tidx left_min {QueryStrategy::scan_block(data, l + 1, end_l - 1)};
                if(data[res] > data[left_min]) res = left_min;
            }
        }
        
        // Right boundary
        if (data[res] >= data[table[b_r]]){
            Tidx start_r { b_r * BLOCK_SIZE };
            if(start_r <= r){
                Tidx right_min {QueryStrategy::scan_block(data, start_r, r)};
                if(data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }

    size_t table_bits() const {
        return static_cast<size_t>(levels) * blocks * sizeof(Tidx) * 8;
    }
};

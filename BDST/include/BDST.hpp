/**
 * @file BDST.hpp
 * @brief Blocked Disjoint Sparse Table.
 *
 * Implements a static Range Minimum Query (RMQ) structure combining macro-block
 * partition decomposition with a standard internal Disjoint Sparse Table layout
 */

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>

#include "ScanStrategies.hpp"

template <typename T = int , typename Tidx = std::uint64_t>
class BDST {
    std::unique_ptr<Tidx[]> table; 
    const std::vector<T>& data;
    Tidx block_size, blocks, levels;

public:
    BDST(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        Tidx n { static_cast<Tidx>(data.size()) };
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));
        table = std::make_unique_for_overwrite<Tidx[]>(levels * blocks);

        // Calculate minima per block
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx { i * block_size };
            Tidx end { std::min(n, (i + 1) * block_size) };
            if (end > i * block_size + 1) {
                Tidx temp {LinearScan::scan_block(data, i * block_size + 1, end - 1)};
                if (data[temp] < data[min_idx]) min_idx = temp;
            }
            // Base level of the Sparse Table
            table[i] = min_idx; 
        }

        // Build Disjoint Sparse Table only over the block minima
        for (Tidx h { 2 }; h <= levels; ++h) {
            Tidx half { Tidx(1) << (h - 1) };
            Tidx range { half << 1 };
            Tidx h_idx { static_cast<Tidx>(h - 1) };
            Tidx* row { &table[h_idx * blocks] };

            for (Tidx center { half }; center < blocks; center += range) {
                row[center - 1] = table[center - 1];
                for (Tidx j { 1 }; j < half; ++j) {
                    Tidx i { center - 1 - j };
                    row[i] = data[table[i]] < data[row[i + 1]] ? table[i] : row[i + 1];
                }

                Tidx rend { std::min(blocks, center + half) };
                row[center] = table[center];
                for (Tidx i { center + 1 }; i < rend; ++i)
                    row[i] = data[table[i]] < data[row[i - 1]] ? table[i] : row[i - 1];
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        assert(l <= r);
        assert( r <= data.size());
        if (l == r) return l;
        
        Tidx b_l { l / block_size };
        Tidx b_r { r / block_size };
        Tidx res { l };

        // Case 1: Query completely within the same block : linear scan
        if (b_l == b_r) return LinearScan::scan_block(data, l, r);

        // Case 2: Query spans multiple blocks
        // Intermediate full blocks via DST
        if (Tidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            int bw { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            Tidx h_idx { static_cast<Tidx>(bw == 0 ? 0 : bw - 1) };
            const Tidx* row { &table[h_idx * blocks] };
            
            if (data[row[ql]] < data[res]) res = row[ql];
            if (data[row[qr]] < data[res]) res = row[qr];
        }

        // Left boundary 
        if (data[res] >= data[table[b_l]]){
            Tidx end_bl { (b_l + 1) * block_size};
            if(l + 1 < end_bl){
                Tidx left_min {LinearScan::scan_block(data, l + 1, end_bl - 1)};
                if(data[res] > data[left_min]) res = left_min;
            }
        }
        
        // Right boundary
        if (data[res] >= data[table[b_r]]){
            Tidx start_br { b_r * block_size };
            if(start_br <= r){
                Tidx right_min {LinearScan::scan_block(data, start_br, r)};
                if(data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }
};

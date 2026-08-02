#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>

template <typename T, typename Tidx = std::uint32_t>
class BDDST {
    Tidx BLOCK_SIZE;
    
    // Uninitialized memory prevents expensive zeroing
    std::unique_ptr<Tidx[]> table; 
    const std::vector<T>& raw_data;
    Tidx blocks, levels;

public:
    // Build O(n) Time | O(n / BLOCK_SIZE * log(n / BLOCK_SIZE)) Space
    explicit BDDST(const std::vector<T>& data, const Tidx B_S) : raw_data(data), BLOCK_SIZE(B_S) {
        Tidx n { static_cast<Tidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

        // Allocates memory without overwriting it with zeros
        table = std::make_unique_for_overwrite<Tidx[]>(levels * blocks);

        // 1. Calculate minima per block
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx { i * BLOCK_SIZE };
            Tidx end { std::min(n, (i + 1) * BLOCK_SIZE) };
            for (Tidx j { i * BLOCK_SIZE + 1 }; j < end; ++j) {
                if (data[j] < data[min_idx]) min_idx = j;
            }
            table[i] = min_idx; // Base level of the Sparse Table
        }

        // 2. Build Disjoint Sparse Table only over the block minima
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

    // Query O(BLOCK_SIZE) Time
    Tidx query(Tidx l, Tidx r) const {
        Tidx b_l { l / BLOCK_SIZE };
        Tidx b_r { r / BLOCK_SIZE };
        Tidx res { l };

        // Case 1: Query completely within the same block : linear scan
        if (b_l == b_r) {
            for (Tidx i = l + 1; i <= r; ++i) {
                if (raw_data[i] < raw_data[res]) res = i;
            }
            return res;
        }

        // Case 2: Query spans multiple blocks
        // Left boundary (intra block)
        Tidx end_bl { (b_l + 1) * BLOCK_SIZE };
        for (Tidx i { l + 1 }; i < end_bl; ++i) {
            if (raw_data[i] < raw_data[res]) res = i;
        }

        // Right boundary (intra block)
        Tidx start_br { b_r * BLOCK_SIZE };
        for (Tidx i { start_br }; i <= r; ++i) {
            if (raw_data[i] < raw_data[res]) res = i;
        }

        // Intermediate full blocks via DST
        if (Tidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            int bw { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            Tidx h_idx { static_cast<Tidx>(bw == 0 ? 0 : bw - 1) };
            const Tidx* row { &table[h_idx * blocks] };
            
            if (raw_data[row[ql]] < raw_data[res]) res = row[ql];
            if (raw_data[row[qr]] < raw_data[res]) res = row[qr];
        }

        return res;
    }
};

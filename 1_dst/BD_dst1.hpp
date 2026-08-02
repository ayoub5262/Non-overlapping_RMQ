#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>

template <typename DT, typename DTidx = std::uint32_t>
class BigDataDST {
    DTidx BLOCK_SIZE;
    
    // Uninitialized memory prevents expensive zeroing for Giga-arrays
    std::unique_ptr<DTidx[]> table; 
    const std::vector<DT>& raw_data;
    DTidx blocks, levels;

public:
    // Build O(n) Time | O(n / BLOCK_SIZE * log(n / BLOCK_SIZE)) Space
    explicit BigDataDST(const std::vector<DT>& data, const DTidx B_S) : raw_data(data), BLOCK_SIZE(B_S) {
        DTidx n { static_cast<DTidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1)) + 1;

        // C++20: Allocates memory without overwriting it with zeros
        table = std::make_unique_for_overwrite<DTidx[]>(levels * blocks);

        // 1. Calculate minima per block
        // #pragma omp parallel for schedule(dynamic)
        for (DTidx i { 0 }; i < blocks; ++i) {
            DTidx min_idx { i * BLOCK_SIZE };
            DTidx end { std::min(n, (i + 1) * BLOCK_SIZE) };
            for (DTidx j { i * BLOCK_SIZE + 1 }; j < end; ++j) {
                if (data[j] < data[min_idx]) min_idx = j;
            }
            table[i] = min_idx; // Base level of the Sparse Table
        }

        // 2. Build Disjoint Sparse Table only over the block minima
        for (DTidx h { 1 }; h < levels; ++h) {
            DTidx half { DTidx(1) << (h - 1) };
            DTidx range { half << 1 };
            DTidx* row { &table[h * blocks] };

            // #pragma omp parallel for schedule(dynamic)
            for (DTidx center { half }; center < blocks; center += range) {
                row[center - 1] = table[center - 1];
                for (DTidx j { 1 }; j < half; ++j) {
                    DTidx i { center - 1 - j };
                    row[i] = data[table[i]] < data[row[i + 1]] ? table[i] : row[i + 1];
                }

                DTidx rend { std::min(blocks, center + half) };
                row[center] = table[center];
                for (DTidx i { center + 1 }; i < rend; ++i)
                    row[i] = data[table[i]] < data[row[i - 1]] ? table[i] : row[i - 1];
            }
        }
    }

    // Query O(BLOCK_SIZE) Time
    DTidx query(DTidx l, DTidx r) const {
        DTidx b_l { l / BLOCK_SIZE };
        DTidx b_r { r / BLOCK_SIZE };
        DTidx res { l };

        // Case 1: Query completely within the same block : linear scan
        if (b_l == b_r) {
            for (DTidx i = l + 1; i <= r; ++i) {
                if (raw_data[i] < raw_data[res]) res = i;
            }
            return res;
        }

        // Case 2: Query spans multiple blocks
        // Left boundary (intra block)
        DTidx end_bl { (b_l + 1) * BLOCK_SIZE };
        for (DTidx i { l + 1 }; i < end_bl; ++i) {
            if (raw_data[i] < raw_data[res]) res = i;
        }

        // Right boundary (intra block)
        DTidx start_br { b_r * BLOCK_SIZE };
        for (DTidx i { start_br }; i <= r; ++i) {
            if (raw_data[i] < raw_data[res]) res = i;
        }

        // Intermediate full blocks via DST
        if (DTidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            int h { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            const DTidx* row { &table[h * blocks] };
            
            if (raw_data[row[ql]] < raw_data[res]) res = row[ql];
            if (raw_data[row[qr]] < raw_data[res]) res = row[qr];
        }

        return res;
    }
};

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>
#include <cstddef>

template <typename DT, typename DTidx = std::uint32_t>
class MinItBigDataDST {
    DTidx BLOCK_SIZE;
    
    // Uninitialized memory prevents expensive zeroing for Giga-arrays
    std::unique_ptr<DT[]> table; 
    const std::vector<DT>& raw_data;
    DTidx blocks, levels;

public:
    // Build O(n) Time | O(n / BLOCK_SIZE * log(n / BLOCK_SIZE)) Space
    explicit MinItBigDataDST(const std::vector<DT>& data, const DTidx B_S) : raw_data(data), BLOCK_SIZE(B_S) {
        DTidx n { static_cast<DTidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1)) + 1;

        // C++20: Allocates memory without overwriting it with zeros
        table = std::make_unique_for_overwrite<DT[]>(levels * blocks);

        // 1. Calculate minima per block
        // #pragma omp parallel for schedule(dynamic)
        for (DTidx i { 0 }; i < blocks; ++i) {
            DT min_val { data[i * BLOCK_SIZE] };
            DTidx end { std::min(n, (i + 1) * BLOCK_SIZE) };
            for (DTidx j { i * BLOCK_SIZE + 1 }; j < end; ++j) {
                min_val = std::min(min_val, data[j]);
            }
            table[i] = min_val; // Base level of the Sparse Table
        }

        // 2. Build Disjoint Sparse Table only over the block minima
        for (DTidx h { 1 }; h < levels; ++h) {
            DTidx half { DTidx(1) << (h - 1) };
            DTidx range { half << 1 };
            DT* row { &table[h * blocks] };

            // #pragma omp parallel for schedule(dynamic)
            for (DTidx center { half }; center < blocks; center += range) {
                row[center - 1] = table[center - 1];
                for (DTidx j { 1 }; j < half; ++j) {
                    DTidx i { center - 1 - j };
                    row[i] = std::min(table[i], row[i + 1]);
                }

                DTidx rend { std::min(blocks, center + half) };
                row[center] = table[center];
                for (DTidx i { center + 1 }; i < rend; ++i)
                    row[i] = std::min(table[i], row[i - 1]);
            }
        }
    }

    // Query O(BLOCK_SIZE) Time
    DT query(DTidx l, DTidx r) const {
        DTidx b_l { l / BLOCK_SIZE };
        DTidx b_r { r / BLOCK_SIZE };
        DT res { raw_data[l] };

        // Case 1: Query completely within the same block : linear scan
        if (b_l == b_r) {
            for (DTidx i = l + 1; i <= r; ++i) {
                res = std::min(res, raw_data[i]);
            }
            return res;
        }

        // Case 2: Query spans multiple blocks
        // Left boundary (intra block)
        DTidx end_bl { (b_l + 1) * BLOCK_SIZE };
        for (DTidx i { l + 1 }; i < end_bl; ++i) {
            res = std::min(res, raw_data[i]);
        }

        // Right boundary (intra block)
        DTidx start_br { b_r * BLOCK_SIZE };
        for (DTidx i { start_br }; i <= r; ++i) {
            res = std::min(res, raw_data[i]);
        }

        // Intermediate full blocks via DST
        if (DTidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            int h { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            const DT* row { &table[h * blocks] };
            
            res = std::min(res, row[ql]);
            res = std::min(res, row[qr]);
        }

        return res;
    }
};

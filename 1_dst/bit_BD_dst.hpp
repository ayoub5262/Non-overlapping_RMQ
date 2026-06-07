#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>

template <typename DT, typename DTidx = std::uint32_t>
class BitBigDataDST {
    static constexpr DTidx BLOCK_SIZE { 32 };
    
    // Uninitialized memory prevents expensive zeroing for Giga-arrays
    std::unique_ptr<DTidx[]> table; 
    std::unique_ptr<uint32_t[]> masks; // Bitmasks for intra-block RMQ
    const std::vector<DT>& raw_data;
    DTidx blocks, levels;

public:
    // Build O(n) Time | O(n / BLOCK_SIZE * log(n / BLOCK_SIZE)) Space
    explicit BitBigDataDST(const std::vector<DT>& data) : raw_data(data) {
        
        DTidx n { static_cast<DTidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1)) + 1;

        // C++20: Allocates memory without overwriting it with zeros
        table = std::make_unique_for_overwrite<DTidx[]>(levels * blocks);
        masks = std::make_unique_for_overwrite<uint32_t[]>(n);

        // 1. Calculate minima per block and build bitmasks
        for (DTidx i { 0 }; i < blocks; ++i) {
            DTidx start { i * BLOCK_SIZE };
            DTidx end { std::min(n, start + BLOCK_SIZE) };
            
            DTidx min_idx { start };
            uint32_t current_mask = 0;

            for (DTidx j { start }; j < end; ++j) {
                // Find global min for DST
                if (data[j] < data[min_idx]) min_idx = j;

                // Build bitmask for intra-block RMQ
                while (current_mask > 0) {
                    int top_bit = 31 - std::countl_zero(current_mask);
                    if (data[start + top_bit] > data[j]) {
                        current_mask &= ~(1U << top_bit);
                    } else {
                        break;
                    }
                }
                current_mask |= (1U << (j - start));
                masks[j] = current_mask;
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

    // Query O(1) Time
    DTidx query(DTidx l, DTidx r) const {
        DTidx b_l { l / BLOCK_SIZE };
        DTidx b_r { r / BLOCK_SIZE };

        auto intra_block_rmq = [&](DTidx left, DTidx right) -> DTidx {
            DTidx block_start = (left / BLOCK_SIZE) * BLOCK_SIZE;
            int bit_l = left - block_start;
            uint32_t valid_bits = masks[right] & (~0U << bit_l);
            return block_start + std::countr_zero(valid_bits);
        };

        // Case 1: Query completely within the same block
        if (b_l == b_r) {
            return intra_block_rmq(l, r);
        }

        // Case 2: Query spans multiple blocks
        DTidx res_l = intra_block_rmq(l, (b_l + 1) * BLOCK_SIZE - 1);
        DTidx res_r = intra_block_rmq(b_r * BLOCK_SIZE, r);
        DTidx res = raw_data[res_l] < raw_data[res_r] ? res_l : res_r;

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

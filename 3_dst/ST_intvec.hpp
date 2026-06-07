#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

// Sparse Table with relative offset encoding + per-level sdsl::int_vector<> bit-packing.
template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class ST_IntVec {
    std::vector<sdsl::int_vector<0>> level_rows;
    const std::vector<T>& data;
    Tidx BLOCK_SIZE, BLOCK_SHIFT, blocks, levels;

public:
    ST_IntVec(const std::vector<T>& arr, const Tidx B_S) : data(arr), BLOCK_SIZE(B_S) {
        BLOCK_SHIFT = std::bit_width(static_cast<unsigned>(BLOCK_SIZE)) - 1;
        Tidx n { static_cast<Tidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks));
        
        level_rows.resize(levels);

        // ── Level 0 : per-block minima ─ ABSOLUTE indices ─────────────────
        const uint8_t base_width { static_cast<uint8_t>(
            n <= 1 ? 1 : std::bit_width(static_cast<unsigned>(n - 1))
        )};
        level_rows[0] = sdsl::int_vector<0>(blocks, 0, base_width);

        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx { i << BLOCK_SHIFT };
            Tidx end { std::min(n, (i + 1) << BLOCK_SHIFT) };
            if (end > (i << BLOCK_SHIFT) + 1) {
                Tidx temp { QueryStrategy::scan_block(data, (i << BLOCK_SHIFT) + 1, end - 1) };
                if (data[temp] < data[min_idx]) min_idx = temp;
            }
            level_rows[0][i] = min_idx;
        }

        // ── Levels h=1..levels-1 : RELATIVE offsets ─────────────────────────
        if (levels > 1) {
            // Temporary arrays to hold the absolute indices of the previous and current levels
            std::vector<Tidx> prev_abs(blocks);
            std::vector<Tidx> curr_abs(blocks);

            for (Tidx i = 0; i < blocks; ++i) {
                prev_abs[i] = level_rows[0][i];
            }

            for (Tidx h { 1 }; h < levels; ++h) {
                Tidx step { Tidx(1) << (h - 1) };
                Tidx len  { Tidx(1) << h };

                // Maximum relative index at level h is (2^h * BLOCK_SIZE) - 1
                // E.g., for h=1, len=2, max offset is 2B-1.
                Tidx max_offset = (len << BLOCK_SHIFT) - 1;
                uint8_t req_bits = static_cast<uint8_t>(std::bit_width(static_cast<unsigned>(max_offset)));
                
                // We strictly only need elements up to blocks - len
                Tidx num_elements = (blocks >= len) ? (blocks - len + 1) : 0;
                level_rows[h] = sdsl::int_vector<0>(num_elements, 0, req_bits);

                for (Tidx i { 0 }; i + len <= blocks; ++i) {
                    Tidx left_min = prev_abs[i];
                    Tidx right_min = prev_abs[i + step];
                    Tidx min_idx = data[left_min] < data[right_min] ? left_min : right_min;
                    
                    curr_abs[i] = min_idx;
                    // Store offset relative to the START of the interval (i * BLOCK_SIZE)
                    level_rows[h][i] = min_idx - (i << BLOCK_SHIFT);
                }
                
                // Copy current absolute indices to prev for the next level
                for (Tidx i { 0 }; i + len <= blocks; ++i) {
                    prev_abs[i] = curr_abs[i];
                }
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        Tidx b_l { l >> BLOCK_SHIFT };
        Tidx b_r { r >> BLOCK_SHIFT };
        Tidx res { l };

        Tidx bl_min { static_cast<Tidx>(level_rows[0][b_l]) };
        Tidx br_min { static_cast<Tidx>(level_rows[0][b_r]) };

        if (b_l == b_r) return QueryStrategy::scan_block(data, l, r);

        if (Tidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            Tidx k { static_cast<Tidx>(std::bit_width(static_cast<unsigned>(qr - ql + 1)) - 1) };
            
            Tidx left_min, right_min;

            if (k == 0) {
                // Base level: absolute indices
                left_min = static_cast<Tidx>(level_rows[0][ql]);
                right_min = static_cast<Tidx>(level_rows[0][qr]); // Note: qr == ql
            } else {
                // Offset-encoded level
                // Actual Index = Offset + (StartIndex * BLOCK_SIZE)
                left_min = static_cast<Tidx>(level_rows[k][ql]) + (ql << BLOCK_SHIFT);
                
                Tidx right_idx = qr - (1 << k) + 1;
                right_min = static_cast<Tidx>(level_rows[k][right_idx]) + (right_idx << BLOCK_SHIFT);
            }

            if (data[left_min] < data[res]) res = left_min;
            if (data[right_min] < data[res]) res = right_min;
        }

        // Left boundary 
        if (data[res] >= data[bl_min]){
            Tidx end_l { (b_l + 1) << BLOCK_SHIFT };
            if(l + 1 < end_l){
                Tidx left_min {QueryStrategy::scan_block(data, l + 1, end_l - 1)};
                if(data[res] > data[left_min]) res = left_min;
            }
        }
        
        // Right boundary
        if (data[res] >= data[br_min]){
            Tidx start_r { b_r << BLOCK_SHIFT };
            if(start_r <= r){
                Tidx right_min {QueryStrategy::scan_block(data, start_r, r)};
                if(data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }

    size_t table_bits() const {
        size_t total = 0;
        for (const auto& row : level_rows)
            total += row.bit_size();
        return total;
    }
};

/**
 * @file BDST_1Bit.hpp
 * @brief 1-Bit Decision Vector Blocked Disjoint Sparse Table.
 *
 * Replaces traditional explicit multi-bit index tables with succinct 1-bit decision
 * vectors per hierarchy level, leveraging rank and select structures to significantly
 * compress index footprint.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "ScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"
#include "sdsl-lite/include/sdsl/bit_vector_il.hpp"



template <typename T = int, typename Tidx = std::uint64_t>
class BDST_1Bit_il {
    sdsl::int_vector<0> base_row;
    
    std::vector<sdsl::bit_vector_il<512>> level_rows;
    std::vector<sdsl::rank_support_il<1, 512>> level_rank;
    std::vector<sdsl::select_support_il<1, 512>> level_select;

    const std::vector<T>& data;
    Tidx n, block_size, log_block,  blocks, levels;

public:
    BDST_1Bit_il(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        n = static_cast<Tidx>(data.size());
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

        // Level 0 Per block absolute minimums as intra-block offsets 
        base_row = sdsl::int_vector<0>(blocks, 0, log_block);
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx {i << log_block};
            Tidx end {std::min(n, (i + 1) << log_block)};
            min_idx = OneLookUpLinearScan::scan_block(data, min_idx, end - 1);
            base_row[i] = min_idx - (i << log_block);
        }

        if (levels <= 1) return;
        level_rows.resize(levels);
        level_rank.resize(levels);
        level_select.resize(levels);

        // Build the table
        for (Tidx level {1}; level < levels; ++level) {
            const Tidx half {Tidx(1) << level};
            if (blocks <= half) continue;
            
            const Tidx num_centers {(blocks -1 - half) / (half * 2) + 1};
            
            // Exactly 1 bit per block, block index is  the bit index
            sdsl::bit_vector cur_level(blocks, 0);

            for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
                const Tidx center {range_idx * (half * 2) + half};

                // Build Left
                Tidx left_min_block {center - 1};
                cur_level[center - 1] = 1; // Start of running min is always a new minimum
                
                for (Tidx dist {1}; dist < half && center - 1 >= dist; ++dist) {
                    const Tidx i {center - 1 - dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx left_min_idx {(left_min_block << log_block) + base_row[left_min_block]};
                    
                    if (data[cur_min_idx] < data[left_min_idx]) {
                        left_min_block = i;
                        cur_level[i] = 1;
                    }
                }

                // Build Right
                Tidx right_min_block {center} ;
                cur_level[center] = 1; // Start of running min is always a new minimum
                
                for (Tidx dist {1}; dist < half && center + dist < blocks; ++dist) {
                    const Tidx i {center + dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx right_min_idx {(right_min_block << log_block) + base_row[right_min_block]};
                    
                    if (data[cur_min_idx] < data[right_min_idx]) {
                        right_min_block = i;
                        cur_level[i] = 1;
                    }
                }
            }
            
            level_rows[level] = sdsl::bit_vector_il<512>(cur_level);
            sdsl::util::init_support(level_rank[level], &level_rows[level]);
            sdsl::util::init_support(level_select[level], &level_rows[level]);
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        assert(l <= r);
        assert( r < n);

        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};

        if (b_l == b_r) {
            Tidx bl_min_idx {(b_l << log_block) +  base_row[b_l]};
            if (bl_min_idx >= l && bl_min_idx <= r) return bl_min_idx;
            return LinearScan::scan_block(data, l, r);
        }

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};
        Tidx res {l};

        if (inner_b_l <= inner_b_r) {
            
            if (inner_b_l == inner_b_r) {
                res = (inner_b_l << log_block) + static_cast<Tidx>(base_row[inner_b_l]);
            } else if (inner_b_r == inner_b_l + 1) {
                const Tidx idx_l {(inner_b_l << log_block) + base_row[inner_b_l]};
                const Tidx idx_r {(inner_b_r << log_block) + base_row[inner_b_r]};
                res = (data[idx_l] > data[idx_r]) ? idx_r : idx_l;
            } else {
                const int bw {std::bit_width(inner_b_l ^ inner_b_r)};
                const Tidx level {static_cast<Tidx>(bw - 1)};

                // Left lookup using Rank/Select
                const Tidx actual_block_l {static_cast<Tidx>(level_select[level](level_rank[level](inner_b_l) + 1))};

                // Right lookup using Rank/Select
                const Tidx actual_block_r {static_cast<Tidx>(level_select[level](level_rank[level](inner_b_r + 1)))};
                
                const Tidx idx_l {(actual_block_l << log_block) + base_row[actual_block_l]};
                const Tidx idx_r {(actual_block_r << log_block) + base_row[actual_block_r]};
                res = (data[idx_l] > data[idx_r]) ? idx_r : idx_l;
            }
        }

        // Left Fringe Block 
        const Tidx start_bl { b_l << log_block};
        const Tidx bl_min_idx {start_bl + base_row[b_l]};
        if (data[bl_min_idx] <= data[res]) {
            if (bl_min_idx >= l) {
                // Precomputed minimum lies within query bounds we avoid the scan
                res = bl_min_idx;
            } else {
                const Tidx end_bl {((b_l + 1) << log_block) - 1};
                const Tidx left_min {LinearScan::scan_block(data, l, end_bl)};
                if (data[left_min] <= data[res]) {
                    res = left_min;
                }
            }
        }


        
        // Right Fringe Block
        const Tidx start_br {b_r << log_block};
        const Tidx br_min_idx {start_br + base_row[b_r]};
        if (data[br_min_idx] < data[res]) {
            if (br_min_idx <= r) {
                // Precomputed minimum lies within query bounds we avoid the scan
                return br_min_idx;
            } else {
                const Tidx right_min {LinearScan::scan_block(data, start_br, r)};
                if (data[right_min] < data[res]) return right_min;
            }
        }

        return res;
    }
};

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"
#include "sdsl-lite/include/sdsl/rank_support.hpp"
#include "sdsl-lite/include/sdsl/select_support.hpp"



template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST_1Bit {
    sdsl::int_vector<0> base_row;
    
    std::vector<sdsl::bit_vector> level_buf;
    std::vector<sdsl::rank_support_v5<>> level_rank;
    std::vector<sdsl::select_support_mcl<1>> level_select;

    const std::vector<T>& data;
    Tidx block_size, log_block, blocks, levels;

public:
    BDDST_1Bit(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        Tidx n {static_cast<Tidx>(data.size())};
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

        // Level 0 Per block absolute minimums as intra-block offsets 
        base_row = sdsl::int_vector<0>(blocks, 0, log_block);
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx {i << log_block};
            Tidx end {std::min(n, (i + 1) << log_block)};
            min_idx = QueryStrategy::scan_block(data, min_idx, end - 1);
            base_row[i] = min_idx - (i << log_block);
        }

        if (levels <= 1) return;
        level_buf.resize(levels);
        level_rank.resize(levels);
        level_select.resize(levels);

        // Build the tree 
        for (Tidx level {1}; level < levels; ++level) {
            const Tidx half {Tidx(1) << level};
            if (blocks <= half) continue;
            
            const Tidx num_centers {(blocks - 1 - half) / (half * 2) + 1};
            
            // Exactly 1 bit per block, block index is the bit index
            level_buf[level] = sdsl::bit_vector(blocks, 0);

            for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
                const Tidx center {range_idx * (half * 2) + half};

                // Build Left
                Tidx left_min_block {center - 1};
                level_buf[level][center - 1] = 1; // Start of running min is always a new minimum
                
                for (Tidx dist {1}; dist < half && center - 1 >= dist; ++dist) {
                    const Tidx i {center - 1 - dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx left_min_idx {(left_min_block << log_block) + base_row[left_min_block]};
                    
                    if (data[cur_min_idx] < data[left_min_idx]) {
                        left_min_block = i;
                        level_buf[level][i] = 1;
                    }
                }

                // Build Right
                Tidx right_min_block {center};
                level_buf[level][center] = 1; // Start of running min is always a new minimum
                
                for (Tidx dist {1}; dist < half && center + dist < blocks; ++dist) {
                    const Tidx i {center + dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx right_min_idx {(right_min_block << log_block) + base_row[right_min_block]};
                    
                    if (data[cur_min_idx] < data[right_min_idx]) {
                        right_min_block = i;
                        level_buf[level][i] = 1;
                    }
                }
            }

            // Initialize O(1) Rank and Select supports
            sdsl::util::init_support(level_rank[level], &level_buf[level]);
            sdsl::util::init_support(level_select[level], &level_buf[level]);
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};

        if (b_l == b_r) return QueryStrategy::scan_block(data, l, r);

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};

        if (inner_b_l <= inner_b_r) {
            const int bw = std::bit_width(static_cast<unsigned>(inner_b_l ^ inner_b_r));

            if (bw <= 1) {
                const Tidx inner_b_l_abs {(inner_b_l << log_block) + base_row[inner_b_l]};
                if (data[inner_b_l_abs] < data[res]) res = inner_b_l_abs;
                
                if (inner_b_l != inner_b_r) {
                    const Tidx inner_b_r_abs {(inner_b_r << log_block) + base_row[inner_b_r]};
                    if (data[inner_b_r_abs] < data[res]) res = inner_b_r_abs;
                }
            } else {
                const Tidx level {static_cast<Tidx>(bw - 1)};

                // Left lookup using Rank/Select in O(1)
                // We want the first 1 at or to the right of inner_b_l
                const Tidx actual_block_l {level_select[level](level_rank[level](inner_b_l) + 1)};
                const Tidx inner_b_l_abs {(actual_block_l << log_block) + base_row[actual_block_l]};

                // Right lookup using Rank/Select in O(1)
                // We want the last 1 at or to the left of inner_b_r
                const Tidx actual_block_r {level_select[level](level_rank[level](inner_b_r + 1))};
                const Tidx inner_b_r_abs {(actual_block_r << log_block) + base_row[actual_block_r]};

                if (data[inner_b_l_abs] < data[res]) res = inner_b_l_abs;
                if (data[inner_b_r_abs] < data[res]) res = inner_b_r_abs;
            }
        }

        // Left Boundary Scan
        const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
        if (data[res] >= data[bl_min]) {
            const Tidx end_l {(b_l + 1) << log_block};
            if (l + 1 < end_l) {
                const Tidx left_min {QueryStrategy::scan_block(data, l + 1, end_l - 1)};
                if (data[res] > data[left_min]) res = left_min;
            }
        }

        // Right Boundary Scan
        const Tidx br_min {(b_r << log_block) + base_row[b_r]};
        if (data[res] >= data[br_min]) {
            const Tidx start_r {b_r << log_block};
            if (start_r <= r) {
                const Tidx right_min {QueryStrategy::scan_block(data, start_r, r)};
                if (data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }
    
    size_t table_bits() const {
        size_t total {sdsl::size_in_bytes(base_row) * 8};
        for (const auto& vec : level_buf) total += sdsl::size_in_bytes(vec) * 8;
        for (const auto& rs : level_rank) total += sdsl::size_in_bytes(rs) * 8;
        for (const auto& ss : level_select) total += sdsl::size_in_bytes(ss) * 8;
        return total;
    }
};

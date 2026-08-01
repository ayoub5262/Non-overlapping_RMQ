/**
 * @file BDST_1Bit_Fast.hpp
 * @brief Fast 1-Bit Blocked Disjoint Sparse Table with Two-Level Decomposition.
 *
 * A space efficient RMQ structure integrating 1-bit decision vectors,
 * macro/mini-block hierarchical scanning, and global early return query short circuits
 * for optimal trade-offs between RAM capacity and execution speed.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"
#include "sdsl-lite/include/sdsl/rank_support.hpp"
#include "sdsl-lite/include/sdsl/select_support.hpp"

template <typename T = int, typename Tidx = std::uint64_t>
class BDST_1Bit_Fast {
    sdsl::int_vector<0> base_row;
    sdsl::int_vector<0> mini_base_row;
    std::vector<T> block_base_val;  // data[base_row[i]] cached per block
    std::vector<T> mini_base_val;   // data[mini_base_row[i]] cached per mini-block
    
    std::vector<sdsl::bit_vector> level_rows;
    std::vector<sdsl::rank_support_v5<1>> level_rank;
    std::vector<sdsl::select_support_mcl<1>> level_select;

    const std::vector<T>& data;
    Tidx block_size, log_block, blocks, levels;
    Tidx log_mini_block, mini_block_size, mini_blocks;




public:
    BDST_1Bit_Fast(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        Tidx n {static_cast<Tidx>(data.size())};
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

        // Determine mini-block size (cap at 256 for linear scan performance)
        log_mini_block = (log_block - 3 > 8) ? 8 : log_block - 3;
        mini_block_size = Tidx(1) << log_mini_block;
        mini_blocks = (n + mini_block_size - 1) / mini_block_size;

        if (log_mini_block < log_block) {
            mini_base_row = sdsl::int_vector<0>(mini_blocks, 0, log_mini_block);
            mini_base_val.resize(mini_blocks);
            for (Tidx i {0}; i < mini_blocks; ++i) {
                Tidx min_idx {i << log_mini_block};
                Tidx end {std::min(n, (i + 1) << log_mini_block)};
                min_idx = LinearScan::scan_block(data, min_idx, end - 1);
                mini_base_row[i] = min_idx - (i << log_mini_block);
                mini_base_val[i] =  data[min_idx];
            }
        }

        // Level 0 Per block absolute minimums as intra-block offsets 
        base_row = sdsl::int_vector<0>(blocks, 0, log_block);
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx {i << log_block};
            Tidx end {std::min(n, (i + 1) << log_block)};
            
            // We use mini_base_row to speedup the construction
            if (log_mini_block < log_block) {
                Tidx mb_start {i << (log_block - log_mini_block)};
                Tidx mb_end {std::min( mini_blocks, (i + 1) << (log_block - log_mini_block))};
                min_idx = (mb_start << log_mini_block) + mini_base_row[mb_start];
                for (Tidx mb {mb_start + 1}; mb < mb_end; ++mb) {
                    Tidx candidate {(mb << log_mini_block) + static_cast<Tidx>(mini_base_row[mb])};
                    if (data[candidate] < data[min_idx]) {
                        min_idx = candidate;
                    }
                }
            } else {
                min_idx = LinearScan::scan_block(data, min_idx, end - 1);
            }
            base_row[i] = min_idx - (i << log_block);
        }
        block_base_val.resize(blocks);
        for (Tidx i {0}; i < blocks; ++i)
            block_base_val[i] = data[(i << log_block) + base_row[i]];

        if (levels <= 1) return;
        level_rows.resize(levels);
        level_rank.resize(levels);
        level_select.resize(levels);

        // Build the tree 
        for (Tidx level {1}; level < levels; ++level) {
            const Tidx half {Tidx(1) << level};
            if (blocks <= half) continue;
            
            const Tidx num_centers {(blocks - 1 - half) / (half * 2) + 1};
            
            level_rows[level] = sdsl::bit_vector(blocks, 0);

            for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
                const Tidx center {range_idx * (half * 2) + half};

                // Build Left
                Tidx left_min_block {center - 1};
                level_rows[level][center - 1] = 1; 
                
                for (Tidx dist {1}; dist < half && center - 1 >= dist; ++dist) {
                    const Tidx i {center - 1 - dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + Tidx( base_row[i])};
                    const Tidx left_min_idx {(left_min_block << log_block) + Tidx(base_row[left_min_block])};
                    
                    if (data[cur_min_idx] < data[left_min_idx]) {
                        left_min_block = i;
                        level_rows[level][i] = 1;
                    }
                }

                // Build Right
                Tidx right_min_block {center};
                level_rows[level][center] = 1; 
                
                for (Tidx dist {1}; dist < half && center + dist < blocks; ++dist) {
                    const Tidx i {center + dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + Tidx(base_row[i])};
                    const Tidx right_min_idx {(right_min_block << log_block) + Tidx(base_row[ right_min_block])};
                    
                    if (data[cur_min_idx] < data[right_min_idx]) {
                        right_min_block = i;
                        level_rows[level][i] = 1;
                    }
                }
            }

            // Initialize Rank and Select supports
            sdsl::util::init_support(level_rank[level], &level_rows[level]);
            sdsl::util::init_support(level_select[level], &level_rows[level]);
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};
        T best_val;

        if (b_l == b_r) {
            const Tidx bl_min {(b_l << log_block) + static_cast<Tidx>(base_row[b_l])};
            if (bl_min >= l && bl_min <= r) {
                return bl_min;
            }
            best_val = data[l];
            return MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, 
                log_mini_block, l, r, l, best_val);
        }

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};

        if (inner_b_l <= inner_b_r) {
            const int bw {std::bit_width(static_cast<unsigned>(inner_b_l ^ inner_b_r))};
            Tidx inner_best;

            if (bw <= 1) {
                inner_best = inner_b_l;
                if (inner_b_l != inner_b_r && block_base_val[inner_b_r] < block_base_val[inner_b_l]) {
                    inner_best = inner_b_r;
                }
            } else {
                const Tidx level {static_cast<Tidx>(bw - 1)};
                // Left lookup using Rank/Select 
                const Tidx actual_block_l {static_cast<Tidx>(level_select[level](level_rank[level](inner_b_l) + 1))};
                // Right lookup using Rank/Select 
                const Tidx actual_block_r {static_cast<Tidx>(level_select[level](level_rank[level](inner_b_r + 1)))};

                if (block_base_val[actual_block_r] < block_base_val[actual_block_l]) {
                    inner_best = actual_block_r;
                } else {
                    inner_best = actual_block_l;
                }
            }

            // Global Early Return Short-Circuit:
            Tidx best_block {inner_best};
            if (block_base_val[b_l] < block_base_val[best_block]) best_block = b_l;
            if (block_base_val[b_r] < block_base_val[best_block]) best_block = b_r;

            const Tidx best_min_idx {(best_block << log_block) + static_cast<Tidx>(base_row[best_block])};
            if (l <= best_min_idx && best_min_idx <= r) {
                return best_min_idx;
            }

            res = (inner_best << log_block) + static_cast<Tidx>(base_row[inner_best]);
            best_val = block_base_val[inner_best];
        } else {
            // Adjacent blocks (b_r == b_l + 1): Check if either block minimum is inside [l, r]
            const Tidx bl_min_idx {(b_l << log_block) + static_cast<Tidx>(base_row[b_l])};
            const Tidx br_min_idx {(b_r << log_block) + static_cast<Tidx>(base_row[b_r])};
            if (bl_min_idx >= l && br_min_idx <= r) {
                return (block_base_val[b_l] <= block_base_val[b_r]) ? bl_min_idx : br_min_idx;
            }
            if (bl_min_idx >= l && block_base_val[b_l] <= block_base_val[b_r]) return bl_min_idx;
            if (br_min_idx <= r && block_base_val[b_r] < block_base_val[b_l]) return br_min_idx;

            best_val = data[res];
        }

        // Left Boundary Scan
        if (block_base_val[b_l] <= best_val) {
            const Tidx start_l { b_l << log_block};
            const Tidx bl_min_idx {start_l + static_cast<Tidx>(base_row[b_l])};
            if (l == start_l || bl_min_idx >= l) {
                res = bl_min_idx;
                best_val = block_base_val[b_l];
            } else {
                const Tidx end_l {(b_l + 1) << log_block};
                const Tidx bound {std::min(static_cast<Tidx>( data.size()), end_l) - 1};
                res = MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, 
                    log_block, log_mini_block, l, bound, res, best_val);
                best_val = data[res];
            }
        }

        // Right Boundary Scan
        if (block_base_val[b_r] < best_val) {
            const Tidx start_r {b_r << log_block};
            const Tidx end_r {std::min(static_cast<Tidx>(data.size()) - 1, ((b_r + 1) << log_block) - 1)};
            const Tidx br_min_idx {start_r + static_cast<Tidx>(base_row[b_r])};
            if (r == end_r || br_min_idx <= r) {
                return br_min_idx;
            } else {
                return MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, 
                    log_mini_block, start_r, r, res, best_val);
            }
        }

        return res;
    }
};

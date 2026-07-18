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
                mini_base_val[i] = data[min_idx];
            }
        }

        // Level 0 Per block absolute minimums as intra-block offsets 
        base_row = sdsl::int_vector<0>(blocks, 0, log_block);
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx {i << log_block};
            Tidx end {std::min(n, (i + 1) << log_block)};
            
            // Speedup construction: if we have mini_base_row, use it!
            if (log_mini_block < log_block) {
                Tidx mb_start = i << (log_block - log_mini_block);
                Tidx mb_end = std::min(mini_blocks, (i + 1) << (log_block - log_mini_block));
                min_idx = (mb_start << log_mini_block) + mini_base_row[mb_start];
                for (Tidx mb = mb_start + 1; mb < mb_end; ++mb) {
                    Tidx candidate = (mb << log_mini_block) + mini_base_row[mb];
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
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx left_min_idx {(left_min_block << log_block) + base_row[left_min_block]};
                    
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
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx right_min_idx {(right_min_block << log_block) + base_row[right_min_block]};
                    
                    if (data[cur_min_idx] < data[right_min_idx]) {
                        right_min_block = i;
                        level_rows[level][i] = 1;
                    }
                }
            }

            // Initialize O(1) Rank and Select supports
            sdsl::util::init_support(level_rank[level], &level_rows[level]);
            sdsl::util::init_support(level_select[level], &level_rows[level]);
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};

        if (b_l == b_r) {
            const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
            if (bl_min >= l && bl_min <= r) {
                return bl_min;
            }
            T best_val = data[l];
            return MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, log_mini_block, l, r, l, best_val);
        }

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};

        if (inner_b_l <= inner_b_r) {
            const int bw = std::bit_width(static_cast<unsigned>(inner_b_l ^ inner_b_r));

            if (bw <= 1) {
                if (block_base_val[inner_b_l] < data[res]) {
                    const Tidx inner_b_l_abs {(inner_b_l << log_block) + base_row[inner_b_l]};
                    res = inner_b_l_abs;
                }
                if (inner_b_l != inner_b_r) {
                    if (block_base_val[inner_b_r] < data[res]) {
                        const Tidx inner_b_r_abs {(inner_b_r << log_block) + base_row[inner_b_r]};
                        res = inner_b_r_abs;
                    }
                }
            } else {
                const Tidx level {static_cast<Tidx>(bw - 1)};

                // Left lookup using Rank/Select in O(1)
                const Tidx actual_block_l {level_select[level](level_rank[level](inner_b_l) + 1)};
                // Right lookup using Rank/Select in O(1)
                const Tidx actual_block_r {level_select[level](level_rank[level](inner_b_r + 1))};

                // Use block_base_val to compare without touching large data[]
                if (block_base_val[actual_block_l] < data[res]) {
                    res = (actual_block_l << log_block) + base_row[actual_block_l];
                }
                if (block_base_val[actual_block_r] < data[res]) {
                    res = (actual_block_r << log_block) + base_row[actual_block_r];
                }
            }
        }

        T best_val = data[res];

        // Left Boundary Scan
        const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
        if (l == (b_l << log_block) || bl_min >= l) {
            if (block_base_val[b_l] < best_val) {
                res = bl_min;
                best_val = block_base_val[b_l];
            }
        } else if (block_base_val[b_l] < best_val) {
            const Tidx end_l {(b_l + 1) << log_block};
            const Tidx bound = std::min(static_cast<Tidx>(data.size()), end_l) - 1;
            res = MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, log_mini_block, l, bound, res, best_val);
        }

        // Right Boundary Scan
        const Tidx start_r {b_r << log_block};
        const Tidx br_min {(b_r << log_block) + base_row[b_r]};
        if (r == std::min(static_cast<Tidx>(data.size()) - 1, ((b_r + 1) << log_block) - 1) || br_min <= r) {
            if (block_base_val[b_r] < best_val) {
                res = br_min;
            }
        } else if (block_base_val[b_r] < best_val) {
            res = MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, log_mini_block, start_r, r, res, best_val);
        }

        return res;
    }
public:
    size_t table_bits() const {
        size_t total {sdsl::size_in_bytes(base_row) * 8};
        total += block_base_val.size() * sizeof(T) * 8;
        if (log_mini_block < log_block) {
            total += sdsl::size_in_bytes(mini_base_row) * 8;
            total += mini_base_val.size() * sizeof(T) * 8;
        }
        for (const auto& vec : level_rows) total += sdsl::size_in_bytes(vec) * 8;
        for (const auto& rs : level_rank) total += sdsl::size_in_bytes(rs) * 8;
        for (const auto& ss : level_select) total += sdsl::size_in_bytes(ss) * 8;
        return total;
    }
};

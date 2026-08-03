/**
 * @file BDST_1Bit_Fast.hpp
 * @brief Fast 1-Bit Blocked Disjoint Sparse Table with Two-Level Decomposition.
 *
 * A space efficient RMQ structure integrating 1-bit decision vectors,
 * macro/mini-block hierarchical scanning, and global early return query short circuits
 * for optimal trade-offs between RAM capacity and execution speed.
 */

#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "ScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

template <typename T = int, typename Tidx = std::uint64_t>
class BDST_1Bit_Fast {
    sdsl::int_vector<0> base_row;
    sdsl::int_vector<0> mini_base_row;
    std::vector<T> block_base_val;  // data[base_row[i]] cached per block
    std::vector<T> mini_base_val;   // data[mini_base_row[i]] cached per mini-block
    
    std::vector<sdsl::bit_vector> level_rows;

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

        }
    }

    Tidx query(Tidx l, Tidx r) const {
        assert(l <= r);
        assert( r <= data.size());
        if (l == r) return l;

        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};
        

        if (b_l == b_r) {
            const Tidx bl_min {(b_l << log_block) + static_cast<Tidx>(base_row[b_l])};
            if (bl_min >= l && bl_min <= r) {
                return bl_min;
            }
            return LinearScan::scan_block(data, l, r);
        }

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};
        T best_val {std::numeric_limits<T>::max()};

        if (inner_b_l <= inner_b_r) {
            Tidx inner_best;
            
            if (inner_b_l == inner_b_r) {
                inner_best = inner_b_l;
            } else if (inner_b_r == inner_b_l + 1) {
                inner_best = (block_base_val[inner_b_r] < block_base_val[inner_b_l]) ? inner_b_r : inner_b_l;
            } else {
                const int bw {std::bit_width(static_cast<unsigned>(inner_b_l ^ inner_b_r))};
                const Tidx level {static_cast<Tidx>(bw - 1)};
                
                const uint64_t* words = level_rows[level].data();
                
                // Left lookup using raw word scan 
                const Tidx actual_block_l {static_cast<Tidx>(WordScan::find_next_one(words, inner_b_l))};
                // Right lookup using raw word scan 
                const Tidx actual_block_r {static_cast<Tidx>(WordScan::find_prev_one(words, inner_b_r))};

                if (block_base_val[actual_block_r] < block_base_val[actual_block_l]) {
                    inner_best = actual_block_r;
                } else {
                    inner_best = actual_block_l;
                }
            }



            res = (inner_best << log_block) + static_cast<Tidx>(base_row[inner_best]);
            best_val = block_base_val[inner_best];
        } else {
            // Adjacent fringe blocks (b_l + 1 == b_r)
            const Tidx bl_min_idx {(b_l << log_block) + static_cast<Tidx>(base_row[b_l])};
            const Tidx br_min_idx {(b_r << log_block) + static_cast<Tidx>(base_row[b_r])};
            const bool l_in {bl_min_idx >= l};
            const bool r_in {br_min_idx <= r};

            if (l_in && r_in) {
                return (block_base_val[b_l] <= block_base_val[b_r]) ? bl_min_idx : br_min_idx;
            }
            
            if (l_in) {
                if (block_base_val[b_l] <= block_base_val[b_r]) return bl_min_idx;
                res = bl_min_idx;
                best_val = block_base_val[b_l];
            } else if (r_in) {
                if (block_base_val[b_r] < block_base_val[b_l]) return br_min_idx;
                res = br_min_idx;
                best_val = block_base_val[b_r];
            }
        }

        // Left Boundary Scan
        if (block_base_val[b_l] < best_val) {
            const Tidx start_bl { b_l << log_block};
            const Tidx bl_min_idx {start_bl + static_cast<Tidx>(base_row[b_l])};
            if (bl_min_idx >= l) {
                res = bl_min_idx;
                best_val = block_base_val[b_l];
            } else {
                const Tidx end_bl {(b_l + 1) << log_block};
                const Tidx bound {std::min(static_cast<Tidx>( data.size()), end_bl) - 1};
                res = MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, 
                    log_block, log_mini_block, l, bound, res, best_val);
                best_val = data[res];
            }
        }

        // Right Boundary Scan
        if (block_base_val[b_r] < best_val) {
            const Tidx start_br {b_r << log_block};
            const Tidx br_min_idx {start_br + static_cast<Tidx>(base_row[b_r])};
            if (br_min_idx <= r) {
                return br_min_idx;
            } else {
                return MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, 
                    log_mini_block, start_br, r, res, best_val);
            }
        }

        return res;
    }
};

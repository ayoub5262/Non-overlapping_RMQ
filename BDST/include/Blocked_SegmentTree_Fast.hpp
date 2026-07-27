/**
 * @file Blocked_SegmentTree_Fast.hpp
 * @brief Accelerated Blocked Segment Tree with Mini-Block Decomposition.
 *
 * Optimizes blocked segment tree query evaluation by utilizing two-level miniblock
 * partitions and early global short circuit checks to bypass repetitive array scans.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

template <typename T = int, typename Tidx = std::uint64_t>
class Blocked_SegmentTree_Fast {
    sdsl::int_vector<0> base_row;
    sdsl::int_vector<0> mini_base_row;
    std::vector<T> block_base_val;  // data[base_row[i]] cached per block
    std::vector<T> mini_base_val;   // data[mini_base_row[i]] cached per mini-block
    
    std::vector<Tidx> seg_tree;
    Tidx n_pow2;

    const std::vector<T>& data;
    Tidx block_size, log_block, blocks;
    Tidx log_mini_block, mini_block_size, mini_blocks;



public:
    Blocked_SegmentTree_Fast(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        Tidx n {static_cast<Tidx>(data.size())};
        blocks = (n + block_size - 1) / block_size;

        // Determine mini-block size and cap at 256 for linear scan performance
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
            
            // Speedup construction using mini_base_row
            if (log_mini_block < log_block) {
                Tidx mb_start {i << (log_block - log_mini_block)};
                Tidx mb_end {std::min(mini_blocks, (i + 1) << (log_block - log_mini_block))};
                min_idx = (mb_start << log_mini_block) + mini_base_row[mb_start];
                for (Tidx mb {mb_start + 1}; mb < mb_end; ++mb) {
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

        block_base_val.resize(blocks + 1);
        
        for (Tidx i {0}; i < blocks; ++i)
            block_base_val[i] = data[(i << log_block) + base_row[i]];
        
        block_base_val[blocks] = std::numeric_limits<T>::max(); // Dummy node for out-of-bounds

        n_pow2 = std::bit_ceil(static_cast<unsigned>(blocks));
    
        if (n_pow2 > 0) {
            seg_tree.assign(2 * n_pow2, blocks);
            for (Tidx i {0}; i < blocks; ++i) {
                seg_tree[n_pow2 + i] = i;
            }
            for (Tidx i {n_pow2 - 1}; i > 0; --i) {
                Tidx l {seg_tree[2 * i]};
                Tidx r {seg_tree[2 * i + 1]};
                seg_tree[i] = (block_base_val[l] < block_base_val[r]) ? l : r;
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};
        T best_val;

        if (b_l == b_r) {
            const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
            if (bl_min >= l && bl_min <= r) {
                return bl_min;
            }
            best_val = data[l];
            return MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, log_mini_block, l, r, l, best_val);
        }

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};

        if (inner_b_l <= inner_b_r) {
            Tidx l_idx {inner_b_l + n_pow2};
            Tidx r_idx {inner_b_r + n_pow2};
            Tidx best_b {blocks}; // padded to +inf

            while (l_idx <= r_idx) {
                if (l_idx & 1) {
                    Tidx b { seg_tree[l_idx++] };
                    if (block_base_val[b] < block_base_val[best_b]) best_b = b;
                }
                if (!(r_idx & 1)) {
                    Tidx b { seg_tree[r_idx--] };
                    if (block_base_val[b] < block_base_val[best_b]) best_b = b;
                }
                l_idx >>= 1;
                r_idx >>= 1;
            }
            
            if (best_b != blocks) {
                // Global Early Return Short-Circuit:
                Tidx best_block {best_b};
                if (block_base_val[b_l] < block_base_val[best_block]) best_block = b_l;
                if (block_base_val[b_r] < block_base_val[best_block]) best_block = b_r;

                const Tidx best_min_idx {(best_block << log_block) + base_row[best_block]};
                if (l <= best_min_idx && best_min_idx <= r) {
                    return best_min_idx;
                }

                res = (best_b << log_block) + base_row[best_b];
                best_val = block_base_val[best_b];
            } else {
                best_val = data[res];
            }
        } else {
            // Adjacent blocks (b_r == b_l + 1): Check if either block minimum is inside [l, r]
            const Tidx bl_min_idx {(b_l << log_block) + base_row[b_l]};
            const Tidx br_min_idx {(b_r << log_block) + base_row[b_r]};
            if (bl_min_idx >= l && br_min_idx <= r) {
                return (block_base_val[b_l] <= block_base_val[b_r]) ? bl_min_idx : br_min_idx;
            }
            if (bl_min_idx >= l && block_base_val[b_l] <= block_base_val[b_r]) return bl_min_idx;
            if (br_min_idx <= r && block_base_val[b_r] < block_base_val[b_l]) return br_min_idx;

            best_val = data[res];
        }

        // Left Boundary Scan
        if (block_base_val[b_l] <= best_val) {
            const Tidx start_l {b_l << log_block};
            const Tidx bl_min_idx {start_l + base_row[b_l]};
            if (l == start_l || bl_min_idx >= l) {
                res = bl_min_idx;
                best_val = block_base_val[b_l];
            } else {
                const Tidx end_l {(b_l + 1) << log_block};
                const Tidx bound {std::min(static_cast<Tidx>(data.size()), end_l) - 1};
                res = MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, log_mini_block, l, bound, res, best_val);
                best_val = data[res];
            }
        }

        // Right Boundary Scan
        if (block_base_val[b_r] < best_val) {
            const Tidx start_r {b_r << log_block};
            const Tidx end_r {std::min(static_cast<Tidx>(data.size()) - 1, ((b_r + 1) << log_block) - 1)};
            const Tidx br_min_idx {start_r + base_row[b_r]};
            if (r == end_r || br_min_idx <= r) {
                return br_min_idx;
            } else {
                return MiniBlockScanner::scan_ragged(data, mini_base_row, mini_base_val, log_block, log_mini_block, start_r, r, res, best_val);
            }
        }

        return res;
    }

};

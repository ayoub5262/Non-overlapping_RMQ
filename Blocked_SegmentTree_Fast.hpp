#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
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
                min_idx = QueryStrategy::scan_block(data, min_idx, end - 1);
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
                min_idx = QueryStrategy::scan_block(data, min_idx, end - 1);
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
            for (Tidx i = 0; i < blocks; ++i) {
                seg_tree[n_pow2 + i] = i;
            }
            for (Tidx i = n_pow2 - 1; i > 0; --i) {
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

        if (b_l == b_r) {
            const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
            if (bl_min >= l && bl_min <= r) {
                return bl_min;
            }
            return scan_ragged(l, r, l);
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
            
            if (best_b != blocks && block_base_val[best_b] < data[res]) {
                res = (best_b << log_block) + base_row[best_b];
            }
        }

        // Left Boundary Scan
        const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
        if (l == (b_l << log_block) || bl_min >= l) {
            if (block_base_val[b_l] < data[res]) res = bl_min;
        } else if (block_base_val[b_l] < data[res]) {
            const Tidx end_l {(b_l + 1) << log_block};
            const Tidx bound = std::min(static_cast<Tidx>(data.size()), end_l) - 1;
            res = scan_ragged(l, bound, res);
        }

        // Right Boundary Scan
        const Tidx start_r {b_r << log_block};
        const Tidx br_min {(b_r << log_block) + base_row[b_r]};
        if (r == std::min(static_cast<Tidx>(data.size()) - 1, ((b_r + 1) << log_block) - 1) || br_min <= r) {
            if (block_base_val[b_r] < data[res]) res = br_min;
        } else if (block_base_val[b_r] < data[res]) {
            res = scan_ragged(start_r, r, res);
        }

        return res;
    }
    
private:
    Tidx scan_ragged(Tidx ql, Tidx qr, Tidx current_res) const {
        if (ql > qr) return current_res;
        
        if (log_mini_block == log_block) {
            Tidx scan_res = LinearSkipScan::scan_block(data, ql, qr, data[current_res]);
            return (data[scan_res] < data[current_res]) ? scan_res : current_res;
        }
        
        Tidx mb_l {ql >> log_mini_block};
        Tidx mb_r {qr >> log_mini_block};
        
        if (mb_l == mb_r) {
            Tidx min_idx = (mb_l << log_mini_block) + mini_base_row[mb_l];
            if (min_idx >= ql && min_idx <= qr) {
                return (data[min_idx] < data[current_res]) ? min_idx : current_res;
            }
            Tidx scan_res = LinearSkipScan::scan_block(data, ql, qr, data[current_res]);
            return (data[scan_res] < data[current_res]) ? scan_res : current_res;
        }
        
        Tidx res = current_res;
        
        // Left ragged mini-block
        Tidx l_min_idx = (mb_l << log_mini_block) + mini_base_row[mb_l];
        if (ql == (mb_l << log_mini_block) || l_min_idx >= ql) {
            if (mini_base_val[mb_l] < data[res]) res = l_min_idx;
        } else if (mini_base_val[mb_l] < data[res]) {
            Tidx end_l = (mb_l + 1) << log_mini_block;
            Tidx scan_res = LinearSkipScan::scan_block(data, ql, end_l - 1, data[res]);
            if (data[scan_res] < data[res]) res = scan_res;
        }
        
        // Full mini-blocks: two-step to minimize random data[] accesses
        // Step 1: scan mini_base_val sequentially 
        T best_val { data[res] };
        Tidx best_mb { Tidx(-1) };
        for (Tidx i = mb_l + 1; i < mb_r; ++i) {
            if (mini_base_val[i] < best_val) {
                best_val = mini_base_val[i];
                best_mb = i;
            }
        }
        
        // Step 2: only one random jump into large data[] if needed
        if (best_mb != Tidx(-1))
            res = (best_mb << log_mini_block) + mini_base_row[best_mb];
        
        // Right ragged mini-block
        Tidx n {static_cast<Tidx>(data.size())};
        Tidx r_min_idx { (mb_r << log_mini_block) + mini_base_row[mb_r] };
        if (qr == std::min(n - 1, ((mb_r + 1) << log_mini_block) - 1) || r_min_idx <= qr) {
            if (mini_base_val[mb_r] < data[res]) res = r_min_idx;
        } else if (mini_base_val[mb_r] < data[res]) {
            Tidx start_r {mb_r << log_mini_block};
            Tidx scan_res {LinearSkipScan::scan_block(data, start_r, qr, data[res])};
            if (data[scan_res] < data[res]) res = scan_res;
        }
        
        return res;
    }

};

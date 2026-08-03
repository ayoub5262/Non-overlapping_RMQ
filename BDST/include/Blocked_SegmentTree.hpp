/**
 * @file Blocked_SegmentTree.hpp
 * @brief Blocked Segment Tree RMQ Index.
 *
 * Implements a hybrid RMQ data structure building an interative heap-based
 * segment tree over block minima
 */

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

#include "ScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

template <typename T = int, typename Tidx = std::uint64_t>
class Blocked_SegmentTree {
    sdsl::int_vector<0> base_row;
    std::vector<T> block_base_val;  // data[base_row[i]] cached per block
    
    std::vector<Tidx> seg_tree;
    Tidx n_pow2;

    const std::vector<T>& data;
    Tidx block_size, log_block, blocks;

public:
    Blocked_SegmentTree(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        Tidx n {static_cast<Tidx>(data.size())};
        blocks = (n + block_size - 1) / block_size;

        // Level 0 Per block absolute minimums as intra-block offsets 
        base_row = sdsl::int_vector<0>(blocks, 0, log_block);
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx {i << log_block};
            Tidx end {std::min(n, (i + 1) << log_block)};
            
            min_idx = LinearScan::scan_block(data, min_idx, end - 1);
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
        assert(l <= r);
        assert( r <= data.size());
        if (l == r) return l;
        
        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};

        if (b_l == b_r) return LinearScan::scan_block(data, l, r);

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
        if (data[res] >= data[bl_min]) {
            const Tidx end_bl {(b_l + 1) << log_block};
            if (l + 1 < end_bl) {
                const Tidx left_min {LinearScan::scan_block(data, l + 1, end_bl - 1)};
                if (data[res] > data[left_min]) res = left_min;
            }
        }

        // Right Boundary Scan 
        const Tidx br_min {(b_r << log_block) + base_row[b_r]};
        if (data[res] >= data[br_min]) {
            const Tidx start_br {b_r << log_block};
            if (start_br <= r) {
                const Tidx right_min {LinearScan::scan_block(data, start_br, r)};
                if (data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }
};

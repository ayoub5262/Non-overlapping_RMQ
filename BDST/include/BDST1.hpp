/**
 * @file BDST1.hpp
 * @brief Compact Single-Level Blocked Disjoint Sparse Table.
 *
 * Implements an optimized single-level block decomposition scheme with concise
 * table indexing and direct intra-block scanning for space-efficient RMQ evaluation.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <bit>
#include <algorithm>

template<typename T = int, typename Tidx = uint64_t>
class BDST1 {
    const std::vector<T>& data;
    Tidx block_size, log_block_size, blocks, n, levels;
    std::vector<Tidx> table; 
    std::vector<T> base_row_values;

public:
    BDST1(const std::vector<T>& arr, Tidx b) : data(arr), block_size(b) {
        log_block_size = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        n = static_cast<Tidx>(data.size());
        blocks = (n + block_size - 1) / block_size;
        levels = std::bit_width(static_cast<unsigned>(blocks - 1));

        table.resize((levels + 1) * blocks);
        base_row_values.resize(blocks);

        // Storing the block minimums in Row 0
        Tidx* row_0 = &table[0];
        for (Tidx i {0}; i < blocks; ++i) {
            Tidx start {i << log_block_size};
            Tidx end {std::min(n, ((i + 1) << log_block_size) - 1)};
            Tidx min_idx {scan_block(data, start, end)};
            row_0[i] = min_idx - start;
            base_row_values[i] = data[min_idx];
        }

        // Computing Disjoint Sparse Table over block minimums
        for (Tidx level {2}; level <= levels; ++level) {
            Tidx half_range {Tidx(1) << (level - 1)};
            Tidx range {half_range << 1};
            Tidx* row = &table[(level - 1) * blocks];

            for (Tidx center {half_range}; center < blocks; center += range) {
                row[center - 1] = center - 1;
                for (Tidx step {1}; step < half_range; ++step) {
                    Tidx i {(center - 1) - step};
                    if (base_row_values[row[i + 1]] > base_row_values[i])
                        row[i] = i;
                    else
                        row[i] = row[i + 1];
                }
                row[center] = center;
                for (Tidx i {center + 1}; i < std::min(center + half_range, blocks); ++i) {
                    if (base_row_values[row[i - 1]] > base_row_values[i])
                        row[i] = i;
                    else
                        row[i] = row[i - 1];
                }
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        assert(l <= r);
        assert( r <= data.size());
        if (l == r) return l;
        
        const Tidx b_l {l >> log_block_size};
        const Tidx b_r {r >> log_block_size};
        const Tidx* row_0 {&table[0]};

        // Case 1: Query inside a single block
        if (b_l == b_r) {
            Tidx b_min_abs {(b_l << log_block_size) + row_0[b_l]};
            return scan_block(data, l, r);
        }

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};
        Tidx res {l};

        // Case 2: Query spans multiple blocks
        if (inner_b_l <= inner_b_r) {
            int bw {std::bit_width(static_cast<unsigned>(inner_b_l ^ inner_b_r))};
            Tidx left_idx, right_idx;
            if (bw <= 1) {
                left_idx = inner_b_l;
                right_idx = inner_b_r;
            } else {
                const Tidx* row_h {&table[(bw - 1) * blocks]};
                left_idx = row_h[inner_b_l];
                right_idx = row_h[inner_b_r];
            }
            if (base_row_values[left_idx] > base_row_values[right_idx]) {
                res = (right_idx << log_block_size) + row_0[right_idx];
            } else {
                res = (left_idx << log_block_size) + row_0[left_idx];
            }
        }

        // Left Fringe Scan 
        if (base_row_values[b_l] < data[res]) {
            Tidx l_min_abs {(b_l << log_block_size) + row_0[b_l]};
            
            // Entire block min is within [l, end], skip scan
            if (l_min_abs >= l) {
                res = l_min_abs; 
            } else {
                Tidx end_bl {(inner_b_l << log_block_size) - 1};
                Tidx l_min_idx {scan_block(data, l, end_bl)};
                if (data[l_min_idx] < data[res]) res = l_min_idx;
            }
        }

        // Right Fringe Scan 
        if (base_row_values[b_r] < data[res]) {
            Tidx r_min_abs {(b_r << log_block_size) + row_0[b_r]};
            // Entire block min is within [start, r], skip scan
            if (r_min_abs <= r) {
                res = r_min_abs; 
            } else {
                Tidx start_br {b_r << log_block_size};
                Tidx r_min_idx {scan_block(data, start_br, r)};
                if (data[r_min_idx] < data[res]) res = r_min_idx;
            }
        }

        return res;
    }

    static Tidx scan_block(const std::vector<T>& arr, Tidx l, Tidx r) {
        Tidx min_idx {l};
        for (Tidx i {l + 1}; i <= r; ++i) {
            if (arr[i] < arr[min_idx]) min_idx = i;
        }
        return min_idx;
    }
};

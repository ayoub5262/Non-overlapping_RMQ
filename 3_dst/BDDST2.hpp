#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

// ============================================================================
// BDDST2: The Ultimate Synthesis
// 
// This structure combines TWO massive compression techniques:
// 1. Position-Dependent Bit-Widths: Elements closer to the center use fewer bits.
//    (A distance of `dist` only requires `log2(dist)` bits).
// 2. Block-Index Encoding: Instead of storing the absolute intra-block offset
//    (which requires an unavoidable `log_block` penalty per element), we only
//    store the *relative block index*.
//
// By combining these, the bit-width of an element at distance `dist` is 
// strictly `std::bit_width(dist)`. The `log_block` penalty is completely 
// annihilated from the tree, drastically shrinking the total memory footprint 
// to rival or beat standard sliding-window Sparse Tables!
// ============================================================================
template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST2 {
    sdsl::int_vector<0> base_row;
    std::vector<sdsl::int_vector<0>> level_buf;

    const std::vector<T>& data;
    Tidx block_size, log_block, blocks, levels;

    // Math Helpers (Now only storing Block Offsets, NO log_block added!)
    static Tidx prefix_bits(Tidx dist) {
        if (dist <= 1) return 0;
        const Tidx p = std::bit_width(dist - 1) - 1;
        return dist + 1 - (1ull << (p + 1)) + p * dist;
    }

public:
    BDDST2(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        Tidx n {static_cast<Tidx>(data.size())};
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks));

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

        // Build the tree 
        for (Tidx level {1}; level < levels; ++level) {
            const Tidx half {Tidx(1) << level};
            if (blocks <= half) continue;
            
            const Tidx num_centers {(blocks - 1 - half) / (half * 2) + 1};
            
            const Tidx half_bits {prefix_bits(half)};
            const Tidx bits_per_range {2 * half_bits};

            const Tidx last_center {(num_centers - 1) * (half * 2) + half};
            const Tidx max_dist {blocks - 1 - last_center};
            const Tidx actual_dist {std::min(max_dist, half - 1)};
            const Tidx right_bits {prefix_bits(actual_dist + 1)};
            const Tidx exact_bits {(num_centers - 1) * bits_per_range + half_bits + right_bits};

            level_buf[level] = sdsl::int_vector<0>(exact_bits, 0, 1);

            for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
                const Tidx center {range_idx * (half * 2) + half};
                const Tidx range_base {range_idx * bits_per_range};
                
                Tidx bit_idx {range_base};

                // Build Left
                Tidx left_min_block {center - 1};
                
                for (Tidx dist {1}; dist < half && center - 1 >= dist; ++dist) {
                    const Tidx i {center - 1 - dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx left_min_idx {(left_min_block << log_block) + base_row[left_min_block]};
                    
                    if (data[cur_min_idx] < data[left_min_idx]) left_min_block = i;
                    
                    const uint8_t width {static_cast<uint8_t>(std::bit_width(dist))};
                    level_buf[level].set_int(bit_idx, left_min_block - i, width);
                    bit_idx += width;
                }

                // Build Right
                bit_idx = range_base + half_bits;
                Tidx right_min_block {center};
                
                for (Tidx dist {1}; dist < half && center + dist < blocks; ++dist) {
                    const Tidx i {center + dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx right_min_idx {(right_min_block << log_block) + base_row[right_min_block]};
                    
                    if (data[cur_min_idx] < data[right_min_idx]) right_min_block = i;
                    
                    const uint8_t width {static_cast<uint8_t>(std::bit_width(dist))};
                    level_buf[level].set_int(bit_idx, right_min_block - center, width);
                    bit_idx += width;
                }
            }
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
                const Tidx half { Tidx(1) << level};
                const Tidx range_idx {inner_b_l >> (level + 1)};
                const Tidx center {range_idx * (half * 2) + half};
                
                const Tidx half_bits {prefix_bits(half)};
                const Tidx bits_per_range {2 * half_bits};
                const Tidx range_base {range_idx * bits_per_range};

                // Left lookup
                const Tidx dist_inner_b_l {center - 1 - inner_b_l};
                Tidx inner_b_l_abs;
                if (dist_inner_b_l == 0) {
                    inner_b_l_abs = (inner_b_l << log_block) + base_row[inner_b_l];
                } else {
                    const uint8_t width {static_cast<uint8_t>(std::bit_width(dist_inner_b_l))};
                    const Tidx bit_idx {range_base + prefix_bits(dist_inner_b_l)};
                    const Tidx block_offset {level_buf[level].get_int(bit_idx, width)};
                    const Tidx actual_block {inner_b_l + block_offset};
                    inner_b_l_abs = (actual_block << log_block) + base_row[actual_block];
                }

                // Right lookup
                const Tidx dist_inner_b_r {inner_b_r - center};
                Tidx inner_b_r_abs;
                if (dist_inner_b_r == 0) {
                    inner_b_r_abs = (inner_b_r << log_block) + base_row[inner_b_r];
                } else {
                    const uint8_t width {static_cast<uint8_t>(std::bit_width(dist_inner_b_r))};
                    const Tidx bit_idx {range_base + half_bits + prefix_bits(dist_inner_b_r)};
                    const Tidx block_offset {level_buf[level].get_int(bit_idx, width)};
                    const Tidx actual_block {center + block_offset};
                    inner_b_r_abs = (actual_block << log_block) + base_row[actual_block];
                }

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
        size_t total {base_row.bit_size()};
        for (const auto& vec : level_buf) total += vec.bit_size();
        return total;
    }
};

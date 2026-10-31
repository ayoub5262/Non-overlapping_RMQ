#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

// An implementation of Position-Dependent Bit-Widths.
// Strips away all caching, branchless hacks, and complex scan strategies.
// Focuses entirely on the core mathematical beauty of the flat structure.

template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST0 {
    sdsl::int_vector<0> base_row;
    
    // Simple 2D structure: one bit-buffer per level
    std::vector<sdsl::int_vector<0>> level_buf;

    const std::vector<T>& data;
    Tidx block_size, log_block, blocks, levels;

    // Math Helpers
    static uint8_t bits_for_dist(Tidx dist, uint8_t log_block) {
        if (dist == 0) return log_block;
        return log_block + static_cast<uint8_t>(std::bit_width(dist));
    }

    static Tidx prefix_bits(Tidx dist, uint8_t log_block) {
        if (dist <= 1) return dist * log_block;
        const Tidx p = std::bit_width(dist - 1) - 1;
        return dist * log_block + dist + 1 - (1ull << (p + 1)) + p * dist;
    }

public:
    BDDST0(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
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
            const Tidx half_bits {prefix_bits(half, log_block)};
            const Tidx bits_per_range {2 * half_bits};

            // Allocate the 2D bit buffer for this level dynamically
            level_buf[level] = sdsl::int_vector<0>(num_centers * bits_per_range, 0, 1);

            for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
                const Tidx center {range_idx * (half * 2) + half};
                const Tidx center_data {center << log_block};
                const Tidx range_base {range_idx * bits_per_range};

                // Build Left
                Tidx left_min {((center - 1) << log_block) + base_row[center - 1]};
                level_buf[level].set_int(range_base, base_row[center - 1], log_block);
                Tidx bit_idx {range_base + log_block};

                for (Tidx dist {1}; dist < half && center - 1 >= dist; ++dist) {
                    const Tidx i {center - 1 - dist};
                    const Tidx cur {(i << log_block) + base_row[i]};
                    if (data[cur] < data[left_min]) left_min = cur;
                    
                    const uint8_t width {bits_for_dist(dist, log_block)};
                    level_buf[level].set_int(bit_idx, left_min - (i << log_block), width);
                    bit_idx += width;
                }

                // Build Right
                const Tidx right_base {range_base + half_bits};
                Tidx right_min {(center << log_block) + base_row[center]};
                level_buf[level].set_int(right_base, base_row[center], log_block);
                bit_idx = right_base + log_block;

                for (Tidx dist {1}; dist < half && center + dist < blocks; ++dist) {
                    const Tidx i {center + dist};
                    const Tidx cur {(i << log_block) + base_row[i]};
                    if (data[cur] < data[right_min]) right_min = cur;
                    
                    const uint8_t width {bits_for_dist(dist, log_block)};
                    level_buf[level].set_int(bit_idx, right_min - center_data, width);
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
                // If bw <= 1, they are in the same block (0) or adjacent (1)
                // Only Level 0 is needed.
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
                
                const Tidx half_bits {prefix_bits(half, log_block)};
                const Tidx bits_per_range {2 * half_bits};
                const Tidx range_base {range_idx * bits_per_range};

                // Left lookup
                const Tidx dist_inner_b_l {center - 1 - inner_b_l};
                const uint8_t width_inner_b_l {bits_for_dist(dist_inner_b_l, log_block)};
                const Tidx bit_inner_b_l {range_base + prefix_bits(dist_inner_b_l, log_block)};
                const Tidx inner_b_l_abs {(inner_b_l << log_block) + level_buf[level].get_int(bit_inner_b_l, width_inner_b_l)};

                // Right lookup
                const Tidx dist_inner_b_r {inner_b_r - center};
                const uint8_t width_inner_b_r {bits_for_dist(dist_inner_b_r, log_block)};
                const Tidx bit_inner_b_r {range_base + half_bits + prefix_bits(dist_inner_b_r, log_block)};
                const Tidx inner_b_r_abs {(center << log_block) + level_buf[level].get_int(bit_inner_b_r, width_inner_b_r)};

                if (data[inner_b_l_abs] < data[res]) res = inner_b_l_abs;
                if (data[inner_b_r_abs] < data[res]) res = inner_b_r_abs;
            }
        }

        // Left Boundary Scan
        const Tidx b_l_min {(b_l << log_block) + base_row[b_l]};
        if (data[res] >= data[b_l_min]) {
            const Tidx end_bl {(b_l + 1) << log_block};
            if (l + 1 < end_bl) {
                const Tidx left_min {QueryStrategy::scan_block(data, l + 1, end_bl - 1)};
                if (data[res] > data[left_min]) res = left_min;
            }
        }

        // Right Boundary Scan
        const Tidx b_r_min {(b_r << log_block) + base_row[b_r]};
        if (data[res] >= data[b_r_min]) {
            const Tidx start_br {b_r << log_block};
            if (start_br <= r) {
                const Tidx right_min {QueryStrategy::scan_block(data, start_br, r)};
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

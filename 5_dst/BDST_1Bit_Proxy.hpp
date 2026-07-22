#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies_Proxy.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"
#include "sdsl-lite/include/sdsl/rank_support.hpp"
#include "sdsl-lite/include/sdsl/select_support.hpp"

template <typename T = int, typename Tidx = std::uint64_t, typename Container = std::vector<T>>
class BDST_1Bit_Proxy {
    std::vector<uint16_t> base_row;
    
    // Flattened contiguous storage for all level bit-rows (eliminates pointer chasing)
    std::vector<uint64_t> flat_rows;     // all bit data in one contiguous block
    std::vector<size_t>   row_offsets;   // flat_rows word offset for level[i]

    const Container& data;
    Tidx block_size, log_block, blocks, levels;

public:
    BDST_1Bit_Proxy(const Container& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<unsigned>(block_size)) - 1;
        Tidx n {static_cast<Tidx>(data.size())};
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

        // Level 0 Per block absolute minimums as intra-block offsets 
        base_row.assign(blocks, 0);
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx {i << log_block};
            Tidx end {std::min(n, (i + 1) << log_block)};
            min_idx = LinearScanProxy::scan_block<Container, Tidx>(data, min_idx, end - 1);
            base_row[i] = min_idx - (i << log_block);
        }

        if (levels <= 1) return;
        row_offsets.resize(levels, 0);

        // Compute word-offsets and allocate flat storage
        size_t total_words = 0;
        for (Tidx level = 1; level < levels; ++level) {
            const Tidx half {Tidx(1) << level};
            if (blocks <= half) { row_offsets[level] = total_words; continue; }
            row_offsets[level] = total_words;
            total_words += (blocks + 63) / 64;
        }
        flat_rows.assign(total_words, 0);

        // Build the tree 
        for (Tidx level {1}; level < levels; ++level) {
            const Tidx half {Tidx(1) << level};
            if (blocks <= half) continue;
            
            const Tidx num_centers {(blocks - 1 - half) / (half * 2) + 1};
            
            // Raw word pointer for this level
            uint64_t* row = flat_rows.data() + row_offsets[level];

            for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
                const Tidx center {range_idx * (half * 2) + half};

                // Build Left
                Tidx left_min_block {center - 1};
                row[(center - 1) / 64] |= (1ULL << ((center - 1) % 64));
                
                for (Tidx dist {1}; dist < half && center - 1 >= dist; ++dist) {
                    const Tidx i {center - 1 - dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx left_min_idx {(left_min_block << log_block) + base_row[left_min_block]};
                    
                    if (data[cur_min_idx] < data[left_min_idx]) {
                        left_min_block = i;
                        row[i / 64] |= (1ULL << (i % 64));
                    }
                }

                // Build Right
                Tidx right_min_block {center};
                row[center / 64] |= (1ULL << (center % 64));
                
                for (Tidx dist {1}; dist < half && center + dist < blocks; ++dist) {
                    const Tidx i {center + dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx right_min_idx {(right_min_block << log_block) + base_row[right_min_block]};
                    
                    if (data[cur_min_idx] <= data[right_min_idx]) {
                        right_min_block = i;
                        row[i / 64] |= (1ULL << (i % 64));
                    }
                }
            }
        }
    }

    inline Tidx next_1(const uint64_t* row, size_t row_words, Tidx idx) const {
        size_t word_idx = idx / 64;
        size_t bit_idx = idx % 64;
        uint64_t w = row[word_idx] >> bit_idx;
        if (w) return idx + __builtin_ctzll(w);
        
        word_idx++;
        while (word_idx < row_words && (w = row[word_idx]) == 0) word_idx++;
        return word_idx * 64 + __builtin_ctzll(w);
    }

    inline Tidx prev_1(const uint64_t* row, Tidx idx) const {
        size_t word_idx = idx / 64;
        size_t bit_idx = idx % 64;
        uint64_t w = row[word_idx] << (63 - bit_idx);
        if (w) return idx - __builtin_clzll(w);
        
        word_idx--;
        while ((w = row[word_idx]) == 0) word_idx--;
        return word_idx * 64 + 63 - __builtin_clzll(w);
    }

    Tidx query(Tidx l, Tidx r) const {
        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};

        if (b_l == b_r) return LinearScanProxy::scan_block<Container, Tidx>(data, l, r);

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};

        if (inner_b_l <= inner_b_r) {
            const int bw = std::bit_width(static_cast<unsigned>(inner_b_l ^ inner_b_r));

            if (bw <= 1) {
                const Tidx inner_b_l_abs {(inner_b_l << log_block) + base_row[inner_b_l]};
                if (data[inner_b_l_abs] <= data[res]) res = inner_b_l_abs;
                
                if (inner_b_l != inner_b_r) {
                    const Tidx inner_b_r_abs {(inner_b_r << log_block) + base_row[inner_b_r]};
                    if (data[inner_b_r_abs] <= data[res]) res = inner_b_r_abs;
                }
            } else {
                const Tidx level {static_cast<Tidx>(bw - 1)};
                const size_t row_words = (blocks + 63) / 64;
                const uint64_t* row = flat_rows.data() + row_offsets[level];

                // Left lookup using blazing fast native bitwise scan
                const Tidx actual_block_l {next_1(row, row_words, inner_b_l)};
                const Tidx inner_b_l_abs {(actual_block_l << log_block) + base_row[actual_block_l]};

                // Right lookup using blazing fast native bitwise scan
                const Tidx actual_block_r {prev_1(row, inner_b_r)};
                const Tidx inner_b_r_abs {(actual_block_r << log_block) + base_row[actual_block_r]};

                if (data[inner_b_l_abs] <= data[res]) res = inner_b_l_abs;
                if (data[inner_b_r_abs] <= data[res]) res = inner_b_r_abs;
            }
        }

        // Left Boundary Scan
        const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
        if (data[bl_min] <= data[res]) {
            const Tidx end_l {(b_l + 1) << log_block};
            if (l < end_l - 1) {
                const Tidx left_min {OneLookUpLinearScanProxy::scan_block<Container, Tidx>(data, l, end_l - 1)};
                if (data[left_min] <= data[res]) res = left_min;
            }
        }

        // Right Boundary Scan
        const Tidx br_min {(b_r << log_block) + base_row[b_r]};
        if (data[br_min] <= data[res]) {
            const Tidx start_r {b_r << log_block};
            if (start_r <= r) {
                const Tidx right_min {OneLookUpLinearScanProxy::scan_block<Container, Tidx>(data, start_r, r)};
                if (data[right_min] <= data[res]) res = right_min;
            }
        }

        return res;
    }
    
    size_t table_bits() const {
        size_t total {base_row.size() * sizeof(uint16_t) * 8};
        total += flat_rows.size() * 64;
        return total;
    }
};

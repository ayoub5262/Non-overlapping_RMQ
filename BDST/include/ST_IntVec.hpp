/**
 * @file ST_IntVec.hpp
 * @brief Succinct Integer Vector Sparse Table.
 *
 * A memory-compressed Sparse Table variant leveraging SDSL bit-packed integer vectors
 * to store block offsets rather than absolute raw pointers.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

// Sparse Table with relative offset encoding and per level sdsl::int_vector<> bit-packing.
template <typename T = int, typename Tidx = std::uint64_t>
class ST_IntVec {
    std::vector<sdsl::int_vector<0>> level_rows;
    const std::vector<T>& data;
    Tidx block_size, log_block, blocks, levels;

public:
    ST_IntVec(const std::vector<T>& arr, const Tidx B_S) : data(arr), block_size(B_S) {
        log_block = std::bit_width(static_cast<uint64_t>(block_size)) - 1;
        Tidx n { static_cast<Tidx>(data.size()) };
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<uint64_t>(blocks));
        
        level_rows.resize(levels);

        // Level 0 store relative offsets of the minimum within block
        // log_block bits to represent [0, block_size-1].
        level_rows[0] = sdsl::int_vector<0>(blocks, 0, log_block);

        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx { i << log_block };
            Tidx end { std::min(n, (i + 1) << log_block) };
            if (end > (i << log_block) + 1) {
                Tidx temp { LinearScan::scan_block(data, (i << log_block) + 1, end - 1) };
                if (data[temp] < data[min_idx]) min_idx = temp;
            }
            level_rows[0][i] = min_idx - (i << log_block);
        }

        // Store at each level the relative offsets 
        if (levels > 1) {
            for (Tidx h { 1 }; h < levels; ++h) {
                Tidx step { Tidx(1) << (h - 1) };
                Tidx len  { Tidx(1) << h };

                // The bit width of (2^h - 1) is exactly h
                uint8_t req_bits = static_cast<uint8_t>(h);
                
                // We strictly only need elements up to [blocks - len]
                Tidx num_elements {(blocks >= len) ? (blocks - len + 1) : 0};
                level_rows[h] = sdsl::int_vector<0>(num_elements, 0, req_bits);

                for (Tidx i { 0 }; i + len <= blocks; ++i) {
                    Tidx left_min, right_min;
                    
                    // Reconstruct the absolute minimum indices directly from the previous level
                    if (h == 1) {
                        left_min = (i << log_block) + static_cast<Tidx>(level_rows[0][i]);
                        right_min = ((i + step) << log_block) + static_cast<Tidx>(level_rows[0][i + step]);
                    } else {
                        Tidx l_block {i + static_cast<Tidx>(level_rows[h - 1][i])};
                        left_min = (l_block << log_block) + static_cast<Tidx>(level_rows[0][l_block]);
                        
                        Tidx r_block {(i + step) + static_cast<Tidx>(level_rows[h - 1][i + step])};
                        right_min = (r_block << log_block) + static_cast<Tidx>(level_rows[0][r_block]);
                    }

                    Tidx min_idx {data[left_min] < data[right_min] ? left_min : right_min};
                    
                    // Store block offset relative to the start block of the interval
                    Tidx min_block {min_idx >> log_block};
                    level_rows[h][i] = min_block - i;
                }
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        assert(l <= r);
        assert( r <= data.size());
        if (l == r) return l;
        
        Tidx b_l { l >> log_block };
        Tidx b_r { r >> log_block };
        Tidx res { l };

        Tidx bl_min { (b_l << log_block) + static_cast<Tidx>(level_rows[0][b_l]) };
        Tidx br_min { (b_r << log_block) + static_cast<Tidx>(level_rows[0][b_r]) };


        if (b_l == b_r) return LinearScan::scan_block(data, l, r);

        Tidx ql { b_l + 1 }, qr { b_r - 1 }; 
        if (ql <= qr) {
            Tidx bw { static_cast<Tidx>(std::bit_width(static_cast<uint64_t>(qr - ql + 1)) - 1) };
            
            Tidx left_min, right_min;

            if (bw == 0) {
                // Base level, relative offsets to reconstruct absolute
                left_min = (ql << log_block) + static_cast<Tidx>(level_rows[0][ql]);
                right_min = (qr << log_block) + static_cast<Tidx>(level_rows[0][qr]);
            } else {
                // Offset encoded level
                // Read block offset and resolve via Level 0
                Tidx left_block {ql + static_cast<Tidx>(level_rows[bw][ql])};
                left_min = (left_block << log_block) + static_cast<Tidx>(level_rows[0][left_block]);
                
                Tidx right_idx {qr - (1 << bw) + 1};
                Tidx right_block {right_idx + static_cast<Tidx>(level_rows[bw][right_idx])};
                right_min = (right_block << log_block) + static_cast<Tidx>(level_rows[0][right_block]);
            }



            if (data[left_min] < data[res]) res = left_min;
            if (data[right_min] < data[res]) res = right_min;
        }

        // Left boundary 
        if (data[res] >= data[bl_min]){
            Tidx end_bl { (b_l + 1) << log_block };
            if(l + 1 < end_bl){
                Tidx left_min {LinearScan::scan_block(data, l + 1, end_bl - 1)};
                if(data[res] > data[left_min]) res = left_min;
            }
        }
        
        // Right boundary
        if (data[res] >= data[br_min]){
            Tidx start_br { b_r << log_block };
            if(start_br <= r){
                Tidx right_min {LinearScan::scan_block(data, start_br, r)};
                if(data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }

};

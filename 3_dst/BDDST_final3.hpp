#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

// -- BDDST_Final3: Native Position-Dependent Bit-Width BDDST -----------------
//
// This version completely eliminates distance classes and descriptor tables.
// It uses an exact $O(1)$ mathematical formula to calculate the prefix sum 
// of log-based bit-widths, allowing us to map a 2D variable-width structure 
// into a perfectly flat 1D bit-array per level.
//
// The number of bits required for an entry at distance j from its center is:
//   bits_for_j(j) = BLOCK_SHIFT + floor(log2(j)) + 1   (for j >= 1)
//                 = BLOCK_SHIFT                        (for j == 0)
//
// Memory Layout:
// For each level h_idx, we store all groups sequentially in a flat bit-array.
//   Group 0: [j=0, j=1, ..., j=half-1]
//   Group 1: [j=0, j=1, ..., j=half-1]
//   ...
//
// Time Complexity: O(1) query via closed-form prefix sum math.
// Space Complexity: Mathematically optimal position-dependent compression.

template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST_Final3 {

    sdsl::int_vector<0> base_row;
    
    // One monolithic bit-buffer for ALL levels (left and right)
    sdsl::int_vector<0> left_buf;
    sdsl::int_vector<0> right_buf;
    std::vector<uint64_t> level_offset; // Starting bit index for each level

    const std::vector<T>& data;
    Tidx BLOCK_SIZE, BLOCK_SHIFT, blocks, levels;

    // ── Pure Math Helpers ───────────────────────────────────────────────────

    // Returns exact bit-width needed to store an entry at distance j
    static uint8_t bits_for_j(Tidx j, uint8_t BLOCK_SHIFT) {
        if (j == 0) return BLOCK_SHIFT;
        return BLOCK_SHIFT + static_cast<uint8_t>(std::bit_width(j));
    }

    // Exact O(1) closed-form sum of bits_for_j(x) for x = 0 to j-1.
    // Derived from the series sum of floor(log2(x)).
    static uint64_t prefix_bits(Tidx j, uint8_t BLOCK_SHIFT) {
        if (j <= 1) return j * BLOCK_SHIFT;
        
        const Tidx p = std::bit_width(j - 1) - 1;
        const uint64_t extra_bits = j + 1 - (1ull << (p + 1)) + p * j;
        
        return j * BLOCK_SHIFT + extra_bits;
    }

    // ── Bit-Level Access ────────────────────────────────────────────────────

    // Read `w` bits from the flat buffer at `bit_idx`
    uint64_t read_bit(const sdsl::int_vector<0>& buf, uint64_t bit_idx, uint8_t w) const {
        const uint64_t wi = bit_idx >> 6;
        const uint64_t bp = bit_idx & 63;
        const uint64_t mask = (1ull << w) - 1;
        
        uint64_t val = buf.data()[wi] >> bp;
        if (bp + w > 64) {
            val |= (buf.data()[wi + 1] << (64 - bp));
        }
        return val & mask;
    }

    // Write `w` bits into the flat buffer at `bit_idx`
    void write_bit(sdsl::int_vector<0>& buf, uint64_t bit_idx, uint64_t val, uint8_t w) const {
        const uint64_t wi = bit_idx >> 6;
        const uint64_t bp = bit_idx & 63;
        const uint64_t mask = (1ull << w) - 1;
        
        buf.data()[wi] = (buf.data()[wi] & ~(mask << bp)) | ((val & mask) << bp);
        if (bp + w > 64) {
            const uint64_t done = 64 - bp;
            buf.data()[wi + 1] = (buf.data()[wi + 1] & ~(mask >> done)) | ((val & mask) >> done);
        }
    }

public:
    BDDST_Final3(const std::vector<T>& arr, const Tidx B_S) : data(arr), BLOCK_SIZE(B_S) {
        BLOCK_SHIFT = std::bit_width(static_cast<unsigned>(BLOCK_SIZE)) - 1;
        const Tidx n = static_cast<Tidx>(data.size());
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

        // -- Level 0: per-block minima as intra-block offsets -----------------
        base_row = sdsl::int_vector<0>(blocks, 0, BLOCK_SHIFT);
        for (Tidx i = 0; i < blocks; ++i) {
            const Tidx start = i << BLOCK_SHIFT;
            const Tidx end   = std::min(n, start + BLOCK_SIZE);
            Tidx min_idx = start;
            if (end > start + 1) {
                Tidx temp = QueryStrategy::scan_block(data, start + 1, end - 1);
                if (data[temp] < data[min_idx]) min_idx = temp;
            }
            base_row[i] = min_idx - start;
        }

        if (levels <= 1) return;

        level_offset.resize(levels, 0);

        // -- Pre-calculate absolute minimums for level 0 to avoid bit-extraction overhead in hot loops
        std::vector<Tidx> base_abs(blocks);
        for (Tidx i = 0; i < blocks; ++i) {
            base_abs[i] = (i << BLOCK_SHIFT) + static_cast<Tidx>(base_row[i]);
        }

        // -- Pass 1: Calculate total bit budget and level offsets -------------
        uint64_t total_bits = 0;
        for (Tidx h_idx = 1; h_idx < levels; ++h_idx) {
            level_offset[h_idx] = total_bits;
            const Tidx half  = Tidx(1) << h_idx;
            const Tidx range = half << 1;
            if (blocks > half) {
                const Tidx num_centers = (blocks - 1 - half) / range + 1;
                total_bits += num_centers * prefix_bits(half, BLOCK_SHIFT);
            }
        }

        // Allocate the two monolithic buffers for all levels
        left_buf  = sdsl::int_vector<0>((total_bits + 63) / 64, 0, 64);
        right_buf = sdsl::int_vector<0>((total_bits + 63) / 64, 0, 64);

        // -- Pass 2: Build DST Levels -----------------------------------------
        for (Tidx h_idx = 1; h_idx < levels; ++h_idx) {
            const Tidx half  = Tidx(1) << h_idx;
            const Tidx range = half << 1;

            if (blocks <= half) continue;
            
            const Tidx num_centers = (blocks - 1 - half) / range + 1;
            const uint64_t BPG = prefix_bits(half, BLOCK_SHIFT);
            const uint64_t base_offset = level_offset[h_idx];

            for (Tidx g = 0; g < num_centers; ++g) {
                const Tidx center      = g * range + half;
                const Tidx center_data = center << BLOCK_SHIFT;
                const uint64_t g_base  = base_offset + g * BPG;

                // ---------------- Right side ----------------
                Tidx right_prev_abs = base_abs[center];
                write_bit(right_buf, g_base, base_row[center], BLOCK_SHIFT);
                
                uint64_t current_bit = g_base + BLOCK_SHIFT;
                for (Tidx j = 1; j < half; ++j) {
                    const Tidx i = center + j;
                    if (i >= blocks) break;
                    
                    const Tidx cur = base_abs[i];
                    if (data[cur] < data[right_prev_abs]) right_prev_abs = cur;
                    
                    const uint8_t w = bits_for_j(j, BLOCK_SHIFT);
                    write_bit(right_buf, current_bit, right_prev_abs - center_data, w);
                    current_bit += w;
                }

                // ---------------- Left side ----------------
                Tidx left_prev_abs = base_abs[center - 1];
                write_bit(left_buf, g_base, base_row[center - 1], BLOCK_SHIFT);
                
                current_bit = g_base + BLOCK_SHIFT;
                for (Tidx j = 1; j < half; ++j) {
                    if (center <= j) break;
                    
                    const Tidx i   = center - 1 - j;
                    const Tidx cur = base_abs[i];
                    if (data[cur] < data[left_prev_abs]) left_prev_abs = cur;
                    
                    const uint8_t w = bits_for_j(j, BLOCK_SHIFT);
                    write_bit(left_buf, current_bit, left_prev_abs - (i << BLOCK_SHIFT), w);
                    current_bit += w;
                }
            }
        }
    }

    // ── Query ────────────────────────────────────────────────────────────────
    Tidx query(Tidx l, Tidx r) const {
        const Tidx b_l = l >> BLOCK_SHIFT;
        const Tidx b_r = r >> BLOCK_SHIFT;
        Tidx res = l;

        if (b_l == b_r) return QueryStrategy::scan_block(data, l, r);

        if (Tidx ql = b_l + 1, qr = b_r - 1; ql <= qr) {
            const int bw = std::bit_width(static_cast<unsigned>(ql ^ qr));

            if (bw == 0) {
                const Tidx abs = (ql << BLOCK_SHIFT) + static_cast<Tidx>(base_row[ql]);
                if (data[abs] < data[res]) res = abs;
            } else {
                const Tidx h_idx = static_cast<Tidx>(bw - 1);

                if (h_idx == 0) {
                    const Tidx ql_abs = (ql << BLOCK_SHIFT) + static_cast<Tidx>(base_row[ql]);
                    const Tidx qr_abs = (qr << BLOCK_SHIFT) + static_cast<Tidx>(base_row[qr]);
                    if (data[ql_abs] < data[res]) res = ql_abs;
                    if (data[qr_abs] < data[res]) res = qr_abs;
                } else {
                    const Tidx half   = Tidx(1) << h_idx;
                    const Tidx g      = ql >> (h_idx + 1);
                    const Tidx center = (g << (h_idx + 1)) | half;
                    const uint64_t BPG = prefix_bits(half, BLOCK_SHIFT);

                    // ── LEFT lookup ──────────────────────────────────────────
                    const Tidx j_ql  = center - 1 - ql;
                    const uint8_t w_ql = bits_for_j(j_ql, BLOCK_SHIFT);
                    const uint64_t bit_ql = level_offset[h_idx] + g * BPG + prefix_bits(j_ql, BLOCK_SHIFT);
                    
                    const Tidx ql_stored = read_bit(left_buf, bit_ql, w_ql);
                    const Tidx ql_abs = (ql << BLOCK_SHIFT) + ql_stored;

                    // ── RIGHT lookup ─────────────────────────────────────────
                    const Tidx j_qr  = qr - center;
                    const uint8_t w_qr = bits_for_j(j_qr, BLOCK_SHIFT);
                    const uint64_t bit_qr = level_offset[h_idx] + g * BPG + prefix_bits(j_qr, BLOCK_SHIFT);
                    
                    const Tidx qr_stored = read_bit(right_buf, bit_qr, w_qr);
                    const Tidx qr_abs = (center << BLOCK_SHIFT) + qr_stored;

                    if (data[ql_abs] < data[res]) res = ql_abs;
                    if (data[qr_abs] < data[res]) res = qr_abs;
                }
            }
        }

        const Tidx bl_min = (b_l << BLOCK_SHIFT) + static_cast<Tidx>(base_row[b_l]);
        const Tidx br_min = (b_r << BLOCK_SHIFT) + static_cast<Tidx>(base_row[b_r]);

        // Left boundary scan.
        if (data[res] >= data[bl_min]) {
            const Tidx end_l = (b_l + 1) << BLOCK_SHIFT;
            if (l + 1 < end_l) {
                if (BLOCK_SIZE >= 1024) { 
                    const Tidx left_min = LinearSkipScan::scan_block(data, l + 1, end_l - 1, data[bl_min]);
                    if (data[res] > data[left_min]) res = left_min;
                } else {
                     const Tidx left_min = QueryStrategy::scan_block(data, l + 1, end_l - 1);
                    if (data[res] > data[left_min]) res = left_min;
                }
            }
        }

        // Right boundary scan.
        if (data[res] >= data[br_min]) {
            const Tidx start_r = b_r << BLOCK_SHIFT;
            if (start_r <= r) {
                if (BLOCK_SIZE >= 1024) { 
                    const Tidx right_min = LinearSkipScan::scan_block(data, start_r, r, data[br_min]);
                    if (data[res] > data[right_min]) res = right_min;
                } else {
                     const Tidx right_min = QueryStrategy::scan_block(data, start_r, r);
                    if (data[res] > data[right_min]) res = right_min;
                }
            }
        }

        return res;
    }

    // Returns exact total bits used by all internal tables.
    size_t table_bits() const {
        return base_row.bit_size() + left_buf.bit_size() + right_buf.bit_size();
    }
};

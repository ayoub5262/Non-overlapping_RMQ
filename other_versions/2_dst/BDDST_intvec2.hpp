#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

// BDDST with relative offset encoding + per-level sdsl::int_vector<> bit-packing.
//
// Space per level h_idx (0-indexed):
//   h_idx == 0 (base):    width = BLOCK_SHIFT bits  (relative offset within block, 0..BLOCK_SIZE-1)
//   h_idx >= 1 (DST):     width = h_idx + BLOCK_SHIFT bits
//
// Example: N=10^9, B=256 (BLOCK_SHIFT=8), levels=12, blocks=N/B
//   Base row:     8 bits/entry  (was 30 bits with absolute indices!)
//   Level 1:      9 bits/entry
//   Level 2:     10 bits/entry
//   Level 11:    19 bits/entry
//
// Reconstruction: absolute_idx = (block_id << BLOCK_SHIFT) + level_rows[0][block_id]
// vs BDDST (absolute, uint64_t): 64 bits/entry at every level.
template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST_IntVec2 {
    // One bit-packed int_vector<> per DST level.
    // level_rows[0]     = base level (relative offsets within block, width = BLOCK_SHIFT)
    // level_rows[h_idx] = DST level  (relative offsets from center,  width = h_idx + BLOCK_SHIFT)
    std::vector<sdsl::int_vector<0>> level_rows;
    const std::vector<T>& data;
    Tidx BLOCK_SIZE, BLOCK_SHIFT, blocks, levels;

public:
    BDDST_IntVec2(const std::vector<T>& arr, const Tidx B_S) : data(arr), BLOCK_SIZE(B_S) {
        // BLOCK_SIZE must be a power of 2.
        BLOCK_SHIFT = std::bit_width(static_cast<unsigned>(BLOCK_SIZE)) - 1;
        Tidx n { static_cast<Tidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));
        level_rows.resize(levels);

        // ── Level 0 : per-block minima ─ RELATIVE offsets within block ─────
        // Width = BLOCK_SHIFT bits (enough to encode 0..BLOCK_SIZE-1).
        // Reconstruction: absolute_idx = (i << BLOCK_SHIFT) + level_rows[0][i]
        level_rows[0] = sdsl::int_vector<0>(blocks, 0, BLOCK_SHIFT);

        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx block_start { i << BLOCK_SHIFT };
            Tidx min_idx { block_start };
            Tidx end { std::min(n, block_start + BLOCK_SIZE) };
            if (end > block_start + 1) {
                Tidx temp { QueryStrategy::scan_block(data, block_start + 1, end - 1) };
                if (data[temp] < data[min_idx]) min_idx = temp;
            }
            level_rows[0][i] = min_idx - block_start;  // stored as relative offset
        }

        // ── Levels h=2..levels : RELATIVE offsets ─────────────────────────
        // At level h (h_idx = h-1), half = 2^(h-1):
        //   max offset = half * BLOCK_SIZE - 1 = 2^(h_idx + BLOCK_SHIFT) - 1
        //   → exactly h_idx + BLOCK_SHIFT bits needed.
        for (Tidx h { 2 }; h <= levels; ++h) {
            Tidx h_idx { h - 1 };
            Tidx half   { Tidx(1) << (h - 1) };
            Tidx range  { half << 1 };

            const uint8_t row_width { static_cast<uint8_t>(h_idx + BLOCK_SHIFT) };
            level_rows[h_idx] = sdsl::int_vector<0>(blocks, 0, row_width);

            for (Tidx center { half }; center < blocks; center += range) {
                const Tidx center_data { center << BLOCK_SHIFT };

                // Level 0 now stores relative offsets: reconstruct absolute via (i << BLOCK_SHIFT) + level_rows[0][i]

                // ── Left side: row[i] = actual_idx - i * BLOCK_SIZE ──────
                level_rows[h_idx][center - 1] =
                    static_cast<Tidx>(level_rows[0][center - 1]);  // offset within block == offset from (center-1)<<SHIFT

                for (Tidx j { 1 }; j < half; ++j) {
                    Tidx i { center - 1 - j };
                    Tidx right_abs { static_cast<Tidx>(level_rows[h_idx][i + 1]) + ((i + 1) << BLOCK_SHIFT) };
                    Tidx cur_abs   { (i << BLOCK_SHIFT) + static_cast<Tidx>(level_rows[0][i]) };
                    level_rows[h_idx][i] = (data[cur_abs] < data[right_abs])
                                             ? cur_abs   - (i << BLOCK_SHIFT)
                                             : right_abs - (i << BLOCK_SHIFT);
                }

                // ── Right side: row[i] = actual_idx - center * BLOCK_SIZE ─
                level_rows[h_idx][center] =
                    static_cast<Tidx>(level_rows[0][center]);  // offset within block == offset from center<<SHIFT

                Tidx rend { std::min(blocks, center + half) };
                for (Tidx i { center + 1 }; i < rend; ++i) {
                    Tidx prev_abs { static_cast<Tidx>(level_rows[h_idx][i - 1]) + center_data };
                    Tidx cur_abs  { (i << BLOCK_SHIFT) + static_cast<Tidx>(level_rows[0][i]) };
                    level_rows[h_idx][i] = (data[cur_abs] < data[prev_abs])
                                             ? cur_abs   - center_data
                                             : static_cast<Tidx>(level_rows[h_idx][i - 1]);
                }
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        Tidx b_l { l >> BLOCK_SHIFT };
        Tidx b_r { r >> BLOCK_SHIFT };
        Tidx res  { l };

        // Prefetch boundary block minima as early as possible
        // Level 0 stores relative offsets: reconstruct absolute via (block_id << BLOCK_SHIFT) + level_rows[0][block_id]
        Tidx bl_min { (b_l << BLOCK_SHIFT) + static_cast<Tidx>(level_rows[0][b_l]) };
        Tidx br_min { (b_r << BLOCK_SHIFT) + static_cast<Tidx>(level_rows[0][b_r]) };

        // Case 1: same block — pure scan
        if (b_l == b_r) return QueryStrategy::scan_block(data, l, r);

        // Case 2: intermediate full blocks via DST
        if (Tidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            int  bw    { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            Tidx h_idx { static_cast<Tidx>(bw == 0 ? 0 : bw - 1) };
            const sdsl::int_vector<0>& row { level_rows[h_idx] };

            if (h_idx == 0) {
                // Base level: relative offsets → reconstruct absolute
                Tidx ql_abs { (ql << BLOCK_SHIFT) + static_cast<Tidx>(row[ql]) };
                Tidx qr_abs { (qr << BLOCK_SHIFT) + static_cast<Tidx>(row[qr]) };
                if (data[ql_abs] < data[res]) res = ql_abs;
                if (data[qr_abs] < data[res]) res = qr_abs;
            } else {
                // Offset-encoded level (bw >= 2, so bw-1 >= 1 is safe)
                // ql is on the LEFT  → actual = offset + ql * BLOCK_SIZE
                Tidx ql_abs { static_cast<Tidx>(row[ql]) + (ql << BLOCK_SHIFT) };
                // qr is on the RIGHT → actual = offset + center_qr * BLOCK_SIZE
                Tidx center_qr { (qr >> (bw - 1)) << (bw - 1) };
                Tidx qr_abs    { static_cast<Tidx>(row[qr]) + (center_qr << BLOCK_SHIFT) };

                if (data[ql_abs] < data[res]) res = ql_abs;
                if (data[qr_abs] < data[res]) res = qr_abs;
            }
        }

        // Left boundary scan
        if (data[res] >= data[bl_min]) {
            Tidx end_bl { (b_l + 1) << BLOCK_SHIFT };
            if (l + 1 < end_bl) {
                Tidx left_min { QueryStrategy::scan_block(data, l + 1, end_bl - 1) };
                if (data[res] > data[left_min]) res = left_min;
            }
        }

        // Right boundary scan
        if (data[res] >= data[br_min]) {
            Tidx start_br { b_r << BLOCK_SHIFT };
            if (start_br <= r) {
                Tidx right_min { QueryStrategy::scan_block(data, start_br, r) };
                if (data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }

    // Returns total bits used by the DST table.
    size_t table_bits() const {
        size_t total = 0;
        for (const auto& row : level_rows)
            total += row.bit_size();
        return total;
    }
};

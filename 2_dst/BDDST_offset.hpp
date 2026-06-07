#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>

#include "BlockScanStrategies.hpp"

// BDDST with relative offset encoding:
//   - Level 0 (base): stores absolute indices into data[].
//   - Levels h >= 2 (h_idx >= 1): stores RELATIVE offsets.
//       * Left-side entry row[i]: stores  (actual_idx - i * BLOCK_SIZE)
//           → offset ∈ [0, half * BLOCK_SIZE), needs (h-1 + log2 B) bits
//       * Right-side entry row[i]: stores (actual_idx - center * BLOCK_SIZE)
//           → offset ∈ [0, half * BLOCK_SIZE), needs (h-1 + log2 B) bits
//   vs. absolute encoding which always needs log2(N) bits.
//   Space savings are largest at low levels, and shrink toward the top.
template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST_Offset {
    std::unique_ptr<Tidx[]> table;
    const std::vector<T>& data;
    Tidx BLOCK_SIZE, BLOCK_SHIFT, blocks, levels;

public:
    BDDST_Offset(const std::vector<T>& arr, const Tidx B_S) : data(arr), BLOCK_SIZE(B_S) {
        // BLOCK_SIZE must be a power of 2 for BLOCK_SHIFT to be correct.
        BLOCK_SHIFT = std::bit_width(static_cast<unsigned>(BLOCK_SIZE)) - 1;
        Tidx n { static_cast<Tidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));
        table = std::make_unique_for_overwrite<Tidx[]>(levels * blocks);

        // ── Level 0: per-block minima ─ stored as ABSOLUTE indices ──────────
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx { i << BLOCK_SHIFT };
            Tidx end { std::min(n, (i + 1) << BLOCK_SHIFT) };
            if (end > (i << BLOCK_SHIFT) + 1) {
                Tidx temp { QueryStrategy::scan_block(data, (i << BLOCK_SHIFT) + 1, end - 1) };
                if (data[temp] < data[min_idx]) min_idx = temp;
            }
            table[i] = min_idx;  // absolute index
        }

        // ── Levels h=2..levels: stored as RELATIVE offsets ──────────────────
        //   Left  entry row[i]: offset = actual_idx - i * BLOCK_SIZE
        //   Right entry row[i]: offset = actual_idx - center * BLOCK_SIZE
        for (Tidx h { 2 }; h <= levels; ++h) {
            Tidx half  { Tidx(1) << (h - 1) };
            Tidx range { half << 1 };
            Tidx h_idx { h - 1 };
            Tidx* row  { &table[h_idx * blocks] };

            for (Tidx center { half }; center < blocks; center += range) {
                const Tidx center_data { center << BLOCK_SHIFT };  // center * BLOCK_SIZE

                // ── Left side ────────────────────────────────────────────────
                // row[i] = actual_idx - i * BLOCK_SIZE
                // actual_idx ∈ [i*BS, center*BS - 1]  →  offset always ≥ 0
                row[center - 1] = table[center - 1] - ((center - 1) << BLOCK_SHIFT);
                for (Tidx j { 1 }; j < half; ++j) {
                    Tidx i      { center - 1 - j };
                    Tidx base_i { i << BLOCK_SHIFT };
                    // Reconstruct absolute from the already-encoded right neighbor
                    Tidx right_abs { row[i + 1] + ((i + 1) << BLOCK_SHIFT) };
                    Tidx cur_abs   { table[i] };
                    row[i] = (data[cur_abs] < data[right_abs])
                               ? cur_abs   - base_i   // new minimum is cur block
                               : right_abs - base_i;  // carry over right neighbor
                }

                // ── Right side ───────────────────────────────────────────────
                // row[i] = actual_idx - center * BLOCK_SIZE
                // actual_idx ∈ [center*BS, (i+1)*BS - 1]  →  offset always ≥ 0
                row[center] = table[center] - center_data;
                Tidx rend { std::min(blocks, center + half) };
                for (Tidx i { center + 1 }; i < rend; ++i) {
                    Tidx prev_abs { row[i - 1] + center_data };
                    Tidx cur_abs  { table[i] };
                    // If cur is smaller, store its offset; otherwise carry prev (same value)
                    row[i] = (data[cur_abs] < data[prev_abs])
                               ? cur_abs - center_data
                               : row[i - 1];
                }
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        Tidx b_l { l >> BLOCK_SHIFT };
        Tidx b_r { r >> BLOCK_SHIFT };
        Tidx res  { l };

        // Fire prefetches for boundary block-minima before branching
        __builtin_prefetch(&data[table[b_l]], 0, 0);
        __builtin_prefetch(&data[table[b_r]], 0, 0);

        // Case 1: same block — pure scan
        if (b_l == b_r) return QueryStrategy::scan_block(data, l, r);

        // Case 2: intermediate full blocks via DST
        if (Tidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            int  bw    { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            Tidx h_idx { static_cast<Tidx>(bw == 0 ? 0 : bw - 1) };
            const Tidx* row { &table[h_idx * blocks] };

            if (h_idx == 0) {
                // Base level: absolute indices — no reconstruction needed
                __builtin_prefetch(&data[row[ql]], 0, 0);
                __builtin_prefetch(&data[row[qr]], 0, 0);
                if (data[row[ql]] < data[res]) res = row[ql];
                if (data[row[qr]] < data[res]) res = row[qr];
            } else {
                // Offset-encoded level (h_idx >= 1, so bw >= 2)
                // ql is on the LEFT  of its DST zone center
                //   → reconstruct: actual = offset + ql * BLOCK_SIZE
                Tidx ql_abs { row[ql] + (ql << BLOCK_SHIFT) };

                // qr is on the RIGHT of its DST zone center
                //   center_qr = floor(qr / 2^(bw-1)) * 2^(bw-1)
                //   → clear the lower (bw-1) bits of qr
                Tidx center_qr { (qr >> (bw - 1)) << (bw - 1) };
                Tidx qr_abs { row[qr] + (center_qr << BLOCK_SHIFT) };

                __builtin_prefetch(&data[ql_abs], 0, 0);
                __builtin_prefetch(&data[qr_abs], 0, 0);

                if (data[ql_abs] < data[res]) res = ql_abs;
                if (data[qr_abs] < data[res]) res = qr_abs;
            }
        }

        // Left boundary scan
        if (data[res] >= data[table[b_l]]) {
            Tidx end_l { (b_l + 1) << BLOCK_SHIFT };
            if (l + 1 < end_l) {
                Tidx left_min { QueryStrategy::scan_block(data, l + 1, end_l - 1) };
                if (data[res] > data[left_min]) res = left_min;
            }
        }

        // Right boundary scan
        if (data[res] >= data[table[b_r]]) {
            Tidx start_r { b_r << BLOCK_SHIFT };
            if (start_r <= r) {
                Tidx right_min { QueryStrategy::scan_block(data, start_r, r) };
                if (data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }
};

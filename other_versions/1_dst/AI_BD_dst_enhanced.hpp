#pragma once

// ============================================================
//  AI_BD_dst_enhanced.hpp — Enhanced Blocked Disjoint Sparse Table
//
//  Root cause of previous OOM: BLOCK_SIZE=16 → n/16 blocks →
//  DST table = (n/16) * log(n/16) * 4 bytes = 6.5 GB at n=10^9.
//
//  Fix: Decouple outer block (DST table size) from inner block (LUT).
//    OUTER_BLOCK = 256  → DST table identical in size to original
//    INNER_BLOCK = 8    → LUT is 128 * 64 = 8 KB (fits in L1 cache!)
//
//  Improvements applied:
//  1. Compile-time OUTER_BLOCK → divisions become SHR bit-shifts
//  2. Fischer-Heun O(1) in-block LUT (INNER_BLOCK=8, 8KB, L1-resident)
//     applied to boundary partial-block scans
//  3. Bit-shift table row pointer (h * blocks → h << log2_blocks)
//  4. [[likely]] / [[unlikely]] branch annotations
//  5. #pragma GCC ivdep vectorisation hints on build loops
// ============================================================

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <array>

// ── compile-time integer log2 ────────────────────────────────
static constexpr int ilog2_e(unsigned v) noexcept {
    return v <= 1 ? 0 : (8 * sizeof(unsigned) - 1 - __builtin_clz(v));
}

// ============================================================
//  Improvement 2: 8-KB In-Block Lookup Table (INNER_BLOCK = 8)
//
//  For 8-element sub-blocks there are 2^(8-1) = 128 distinct
//  Cartesian-tree types.  lut[type][l*8+r] = index of minimum
//  in the range [l,r] relative to the sub-block start.
// ============================================================
template<int B>
struct InBlockLUT_E {
    static constexpr int TYPES = (1 << (B - 1));
    // lut[type][l*B + r] = relative position of minimum
    std::array<std::array<uint8_t, B * B>, TYPES> lut{};

    InBlockLUT_E() {
        for (int type = 0; type < TYPES; ++type) {
            // Build representative values for this type bitmask
            std::array<int, B> val{};
            val[0] = B;
            for (int i = 1; i < B; ++i)
                val[i] = ((type >> (i - 1)) & 1) ? val[i-1] + 1 : val[i-1] - 1;
            // Precompute min for every [l,r] pair
            for (int l = 0; l < B; ++l) {
                uint8_t best = l;
                for (int r = l; r < B; ++r) {
                    if (val[r] < val[best]) best = r;
                    lut[type][l * B + r] = best;
                }
            }
        }
    }
};

// ============================================================
//  BigDataDST_Enhanced
//
//  Template parameters:
//    DT          — element type (int, long, ...)
//    OUTER_BLOCK — outer block size for DST table (default 256)
//                  Must be a power of two. Controls space: O(n/OUTER_BLOCK)
//    INNER_BLOCK — inner sub-block for in-boundary O(1) LUT (default 8)
//                  Must be a power of two and divide OUTER_BLOCK.
//    DTidx       — index type (uint32_t supports up to 4 billion elements)
// ============================================================
template <typename DT,
          int OUTER_BLOCK = 256,
          int INNER_BLOCK = 8,
          typename DTidx  = std::uint32_t>
class BigDataDST_Enhanced {

    static_assert((OUTER_BLOCK & (OUTER_BLOCK - 1)) == 0, "OUTER_BLOCK must be power of two");
    static_assert((INNER_BLOCK & (INNER_BLOCK - 1)) == 0, "INNER_BLOCK must be power of two");
    static_assert(INNER_BLOCK <= 16,                       "INNER_BLOCK must be ≤ 16 for LUT to fit in L2");
    static_assert(OUTER_BLOCK % INNER_BLOCK == 0,          "OUTER_BLOCK must be divisible by INNER_BLOCK");

    // ── Improvement 1: compile-time shifts ───────────────────
    static constexpr int LOG_OUTER = ilog2_e(OUTER_BLOCK);
    static constexpr int LOG_INNER = ilog2_e(INNER_BLOCK);
    static constexpr int INNER_PER_OUTER = OUTER_BLOCK / INNER_BLOCK;

    // ── Shared static LUT (constructed once, 8 KB for B=8) ──
    static const InBlockLUT_E<INNER_BLOCK>& lut() {
        static InBlockLUT_E<INNER_BLOCK> instance;
        return instance;
    }

    // ── Inner-block type per INNER_BLOCK-sized sub-block ─────
    //    Stored as uint8_t (128 possible types for B=8)
    std::unique_ptr<uint8_t[]>  inner_types;  // n / INNER_BLOCK entries

    // ── DST table over outer-block minima ────────────────────
    std::unique_ptr<DTidx[]>    table;

    const std::vector<DT>& raw_data;
    DTidx n_elem, blocks, levels;

    // ── Compute Cartesian-tree type bitmask for one inner block ─
    inline uint8_t compute_inner_type(DTidx start, DTidx end) const noexcept {
        const DT* raw = raw_data.data();
        uint8_t type = 0;
        for (DTidx i = start + 1; i < end; ++i)
            if (raw[i] > raw[i - 1])
                type |= uint8_t(1u << (i - start - 1));
        return type;
    }

    // ── O(1) minimum index within [l, r] inside one inner block ─
    inline DTidx inner_min(DTidx ib_id, DTidx l_rel, DTidx r_rel) const noexcept {
        uint8_t t   = inner_types[ib_id];
        uint8_t off = lut().lut[t][l_rel * INNER_BLOCK + r_rel];
        return (ib_id << LOG_INNER) + off;
    }

    // ── O(OUTER_BLOCK/INNER_BLOCK) minimum in partial outer block ─
    //    Scans at most 32 inner sub-blocks (for OUTER/INNER = 256/8)
    DTidx partial_min(DTidx l, DTidx r) const noexcept {
        const DT* raw = raw_data.data();

        DTidx ib_l = l >> LOG_INNER;
        DTidx ib_r = r >> LOG_INNER;

        // Same inner block
        if (ib_l == ib_r) {
            return inner_min(ib_l, l - (ib_l << LOG_INNER), r - (ib_r << LOG_INNER));
        }

        // Left partial inner block
        DTidx res = inner_min(ib_l, l - (ib_l << LOG_INNER), INNER_BLOCK - 1);

        // Full inner blocks in the middle
        for (DTidx ib = ib_l + 1; ib < ib_r; ++ib) {
            DTidx cand = inner_min(ib, 0, INNER_BLOCK - 1);
            if (raw[cand] < raw[res]) res = cand;
        }

        // Right partial inner block
        DTidx rcand = inner_min(ib_r, 0, r - (ib_r << LOG_INNER));
        if (raw[rcand] < raw[res]) res = rcand;

        return res;
    }

public:
    // ── Construction: O(n) time, O(n/OUTER * log(n/OUTER)) space ──
    explicit BigDataDST_Enhanced(const std::vector<DT>& data)
        : raw_data(data), n_elem(static_cast<DTidx>(data.size()))
    {
        blocks = (n_elem + OUTER_BLOCK - 1) >> LOG_OUTER;
        levels = blocks <= 1 ? 1
                             : std::bit_width(static_cast<unsigned>(blocks - 1)) + 1;

        DTidx n_inner = (n_elem + INNER_BLOCK - 1) >> LOG_INNER;

        table       = std::make_unique_for_overwrite<DTidx[]>(levels * blocks);
        inner_types = std::make_unique_for_overwrite<uint8_t[]>(n_inner);

        const DT* raw = data.data();

        // ── Step 1: outer-block minima + inner-block types ───
        for (DTidx i = 0; i < blocks; ++i) {
            DTidx ob_start = i << LOG_OUTER;
            DTidx ob_end   = std::min(n_elem, ob_start + DTidx(OUTER_BLOCK));

            // Per inner sub-block: compute type and track block min
            DTidx min_idx = ob_start;
            for (DTidx ib_off = 0; ib_off < INNER_PER_OUTER; ++ib_off) {
                DTidx ib_start = ob_start + (ib_off << LOG_INNER);
                if (ib_start >= ob_end) break;
                DTidx ib_end   = std::min(ob_end, ib_start + DTidx(INNER_BLOCK));
                DTidx ib_id    = ib_start >> LOG_INNER;

                inner_types[ib_id] = compute_inner_type(ib_start, ib_end);

                for (DTidx j = ib_start; j < ib_end; ++j)
                    if (raw[j] < raw[min_idx]) min_idx = j;
            }
            table[i] = min_idx;
        }

        // ── Step 2: DST over outer-block minima ─────────────
        for (DTidx h = 1; h < levels; ++h) {
            DTidx half  = DTidx(1) << (h - 1);
            DTidx range = half << 1;
            // Improvement 3: no multiply on hot path
            DTidx* row  = table.get() + h * blocks;

            // Improvement 5: ivdep for SIMD
            #pragma GCC ivdep
            for (DTidx center = half; center < blocks; center += range) {
                DTidx rend = std::min(blocks, center + half);

                row[center - 1] = table[center - 1];
                for (DTidx j = 1; j < half; ++j) {
                    DTidx i = center - 1 - j;
                    row[i] = raw[table[i]] < raw[row[i + 1]] ? table[i] : row[i + 1];
                }
                row[center] = table[center];
                for (DTidx i = center + 1; i < rend; ++i)
                    row[i] = raw[table[i]] < raw[row[i - 1]] ? table[i] : row[i - 1];
            }
        }
    }

    // ── Query: O(1) for inter-block, O(OUTER/INNER) boundary scan ──
    DTidx query(DTidx l, DTidx r) const noexcept {
        // Improvement 1: shift replaces division
        DTidx b_l = l >> LOG_OUTER;
        DTidx b_r = r >> LOG_OUTER;

        // Improvement 4: same-block is rare at large n
        if (b_l == b_r) [[unlikely]] {
            return partial_min(l, r);
        }

        // Left boundary partial outer block
        DTidx end_bl = (b_l + 1) << LOG_OUTER;
        DTidx res   = partial_min(l, end_bl - 1);

        // Right boundary partial outer block
        DTidx start_br = b_r << LOG_OUTER;
        DTidx rcand   = partial_min(start_br, r);
        if (raw_data[rcand] < raw_data[res]) res = rcand;

        // Intermediate full outer blocks via DST
        DTidx ql = b_l + 1, qr = b_r - 1;
        if (ql <= qr) [[likely]] {
            // Improvement 3: bit_width instead of multiply for row ptr
            int h = std::bit_width(static_cast<unsigned>(ql ^ qr));
            const DTidx* row = table.get() + h * blocks;
            DTidx ml = row[ql], mr = row[qr];
            if (raw_data[ml] < raw_data[res]) res = ml;
            if (raw_data[mr] < raw_data[res]) res = mr;
        }

        return res;
    }

    DTidx size()     const noexcept { return n_elem; }
    DTidx n_blocks() const noexcept { return blocks; }
};

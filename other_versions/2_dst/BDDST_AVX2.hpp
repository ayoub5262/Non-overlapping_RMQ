#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>

#ifdef __AVX2__
#include <immintrin.h>
#endif

template <typename DT, typename DTidx = std::uint32_t>
class BDDST {
    DTidx BLOCK_SIZE;

    // Uninitialized memory prevents expensive zeroing
    std::unique_ptr<DTidx[]> table;
    const std::vector<DT>& raw_data;
    DTidx blocks, levels;

    // ----------------------------------------------------------------
    // Finds the index of the minimum element in raw_data[l..r] (inclusive).
    //
    // Dispatch priority:
    //   1. AVX2 SIMD path  — 8 int32 elements/cycle (requires -mavx2)
    //   2. Scalar fallback — branchless + 4-way loop unrolling
    // ----------------------------------------------------------------
    DTidx intra_block_min(DTidx l, DTidx r) const {
        const DT* d = raw_data.data();

#ifdef __AVX2__
        if constexpr (std::is_same_v<DT, int> || std::is_same_v<DT, std::int32_t>) {
            DTidx i = l;

            if (r - l + 1 >= 8) {
                // --- Initialise with the first 8 elements ---
                __m256i min_vals = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&d[i]));
                // Lane k holds index (l + k)
                __m256i min_idxs = _mm256_set_epi32(
                    (int)(i+7), (int)(i+6), (int)(i+5), (int)(i+4),
                    (int)(i+3), (int)(i+2), (int)(i+1), (int)(i+0));
                __m256i cur_idxs = min_idxs;
                __m256i step     = _mm256_set1_epi32(8);
                i += 8;

                // --- Main SIMD loop: 8 elements per iteration ---
                for (; i + 7 <= r; i += 8) {
                    cur_idxs = _mm256_add_epi32(cur_idxs, step);
                    __m256i vals = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(&d[i]));
                    // mask[k] = 0xFFFFFFFF where vals[k] < min_vals[k]
                    __m256i mask = _mm256_cmpgt_epi32(min_vals, vals);
                    min_vals = _mm256_blendv_epi8(min_vals, vals,     mask);
                    min_idxs = _mm256_blendv_epi8(min_idxs, cur_idxs, mask);
                }

                // --- Horizontal reduction over 8 lanes ---
                alignas(32) int v[8], idx[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(v),   min_vals);
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx), min_idxs);

                for (int k = 1; k < 8; ++k) {
                    if (v[k] < v[0]) { v[0] = v[k]; idx[0] = idx[k]; }
                }
                DTidx res = static_cast<DTidx>(idx[0]);

                // --- Branchless scalar tail (< 8 remaining elements) ---
                for (; i <= r; ++i)
                    res = (d[i] < d[res]) ? i : res;

                return res;
            }
            // Fall through to scalar if range < 8
        }
#endif

        // ---- Scalar path: branchless comparisons + 4-way unrolling ----
        // Four independent accumulators break the sequential dependency chain,
        // allowing the CPU to execute comparisons in parallel (ILP).
        DTidx res1 = l, res2 = l, res3 = l, res4 = l;
        DTidx i = l + 1;

        for (; i + 3 <= r; i += 4) {
            res1 = (d[i  ] < d[res1]) ? i   : res1;
            res2 = (d[i+1] < d[res2]) ? i+1 : res2;
            res3 = (d[i+2] < d[res3]) ? i+2 : res3;
            res4 = (d[i+3] < d[res4]) ? i+3 : res4;
        }

        // Merge the 4 accumulators (branchless)
        res1 = (d[res2] < d[res1]) ? res2 : res1;
        res1 = (d[res3] < d[res1]) ? res3 : res1;
        res1 = (d[res4] < d[res1]) ? res4 : res1;

        // Tail (0–3 remaining elements)
        for (; i <= r; ++i)
            res1 = (d[i] < d[res1]) ? i : res1;

        return res1;
    }

public:
    // Build O(n) Time | O(n / BLOCK_SIZE * log(n / BLOCK_SIZE)) Space
    explicit BDDST(const std::vector<DT>& data, const DTidx B_S) : raw_data(data), BLOCK_SIZE(B_S) {
        DTidx n { static_cast<DTidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

        // Allocates memory without overwriting it with zeros
        table = std::make_unique_for_overwrite<DTidx[]>(levels * blocks);

        // 1. Calculate minima per block via fast intra_block_min
        for (DTidx i { 0 }; i < blocks; ++i) {
            DTidx start { i * BLOCK_SIZE };
            DTidx end   { std::min(n, (i + 1) * BLOCK_SIZE) - 1 }; // inclusive
            table[i] = intra_block_min(start, end);
        }

        // 2. Build Disjoint Sparse Table only over the block minima
        for (DTidx h { 2 }; h <= levels; ++h) {
            DTidx half  { DTidx(1) << (h - 1) };
            DTidx range { half << 1 };
            DTidx h_idx { static_cast<DTidx>(h - 1) };
            DTidx* row  { &table[h_idx * blocks] };

            for (DTidx center { half }; center < blocks; center += range) {
                row[center - 1] = table[center - 1];
                for (DTidx j { 1 }; j < half; ++j) {
                    DTidx i { center - 1 - j };
                    row[i] = data[table[i]] < data[row[i + 1]] ? table[i] : row[i + 1];
                }

                DTidx rend { std::min(blocks, center + half) };
                row[center] = table[center];
                for (DTidx i { center + 1 }; i < rend; ++i)
                    row[i] = data[table[i]] < data[row[i - 1]] ? table[i] : row[i - 1];
            }
        }
    }

    // Query O(BLOCK_SIZE) Time
    DTidx query(DTidx l, DTidx r) const {
        DTidx b_l { l / BLOCK_SIZE };
        DTidx b_r { r / BLOCK_SIZE };

        // Case 1: Query completely within the same block
        if (b_l == b_r)
            return intra_block_min(l, r);

        // Case 2: Query spans multiple blocks
        // Left partial block
        DTidx end_bl { (b_l + 1) * BLOCK_SIZE - 1 }; // inclusive
        DTidx res = intra_block_min(l, end_bl);

        // Right partial block
        DTidx start_br { b_r * BLOCK_SIZE };
        DTidx res_r = intra_block_min(start_br, r);
        res = (raw_data[res_r] < raw_data[res]) ? res_r : res; // branchless merge

        // Intermediate full blocks via DST
        if (DTidx ql { b_l + 1 }, qr { b_r - 1 }; ql <= qr) {
            int bw { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            DTidx h_idx { static_cast<DTidx>(bw == 0 ? 0 : bw - 1) };
            const DTidx* row { &table[h_idx * blocks] };

            if (raw_data[row[ql]] < raw_data[res]) res = row[ql];
            if (raw_data[row[qr]] < raw_data[res]) res = row[qr];
        }

        return res;
    }
};

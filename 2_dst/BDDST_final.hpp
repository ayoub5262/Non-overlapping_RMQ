#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

// -- BDDST_Final: position-dependent bit-width BDDST --------------------------
//
// Unique to non-overlapping BDDST:
//   An entry at distance j from its center has a stored value bounded by
//   (j+1)*BLOCK_SIZE - 1 (because the minimum can never be more than j+1
//   blocks away from the base block).  This bound is MONOTONICALLY TIGHTER
//   for entries close to the center -- a property that Sparse Tables cannot
//   exploit, since their entries have no ordering guarantee across positions.
//
// Implementation: within each DST level h_idx, entries are split into
//   width-classes k = 0, 1, ..., h_idx, where class k covers distances
//   j in [class_start(k), class_start(k) + class_size(k) - 1].
//
//   k=0 : j=0         -> bit-width = BLOCK_SHIFT          (1 entry / side / center)
//   k=1 : j=1         -> bit-width = 1 + BLOCK_SHIFT      (1 entry / side / center)
//   k>=2: j in [2^(k-1), 2^k - 1] -> bit-width = k + BLOCK_SHIFT  (2^(k-1) entries)
//
// Storage: all classes for all levels are packed into exactly TWO monolithic
//   sdsl::int_vector<0> objects (left_buf / right_buf).  A small descriptor
//   table (class_desc) stores, for each (h_idx, k) pair, the bit-offset into
//   the buffer and the bit-width.  This eliminates the per-object heap overhead
//   that would inflate malloc_count for small N.
//
// Space savings vs. uniform (h_idx + BLOCK_SHIFT) bit encoding: ~6-8% per level.

template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST_Final {

    // -- Level-0 base row (per-block minima as intra-block offsets) -----------
    sdsl::int_vector<0> base_row;

    // -- Two monolithic buffers for all DST levels ----------------------------
    sdsl::int_vector<0> left_buf;   // all left  entries, packed consecutively
    sdsl::int_vector<0> right_buf;  // all right entries, packed consecutively

    // Descriptor for one (h_idx, k) class in the buffer.
    struct ClassDesc {
        uint64_t bit_offset;    // start bit inside left_buf / right_buf
        uint32_t num_entries;   // total number of entries in this class
        uint8_t  width;         // bit-width of each entry
    };

    // class_desc[(h_idx-1)*(h_idx+2)/2 + k] describes class k at level h_idx.
    std::vector<ClassDesc> class_desc;
    size_t total_class_slots = 0;  // (levels-1)*(levels+2)/2

    const std::vector<T>& data;
    Tidx BLOCK_SIZE, BLOCK_SHIFT, blocks, levels;

    // -- Distance-class helpers -----------------------------------------------

    static int j_class(Tidx j) noexcept {
        return (j == 0) ? 0 : static_cast<int>(std::bit_width(static_cast<unsigned>(j)));
    }

    static Tidx class_start(int k) noexcept {
        return (k <= 1) ? Tidx(k) : (Tidx(1) << (k - 1));
    }

    static int class_shift(int k) noexcept {
        return (k <= 1) ? 0 : (k - 1);
    }

    static Tidx class_size(int k) noexcept {
        return Tidx(1) << class_shift(k);
    }

    uint8_t class_bits(int k) const noexcept {
        return static_cast<uint8_t>(k + BLOCK_SHIFT);
    }

    size_t flat_idx(Tidx h_idx, int k) const noexcept {
        // sum_{i=1}^{h_idx-1} (i+1) + k  =  (h_idx-1)*(h_idx+2)/2 + k
        return (h_idx - 1) * (h_idx + 2) / 2 + k;
    }

    // Read entry g, class k at level h_idx from buf.
    uint64_t buf_get(const sdsl::int_vector<0>& buf,
                     Tidx h_idx, int k, Tidx g, Tidx j) const noexcept {
        const ClassDesc& d = class_desc[flat_idx(h_idx, k)];
        const Tidx local   = (g << class_shift(k)) + (j - class_start(k));
        const uint64_t bit = d.bit_offset + local * d.width;
        // manual extraction from the underlying 64-bit word storage
        const uint64_t word_idx = bit >> 6;
        const uint64_t bit_pos  = bit & 63;
        const uint64_t mask     = (d.width == 64) ? ~uint64_t(0)
                                                   : ((uint64_t(1) << d.width) - 1);
        // sdsl stores in little-endian word order, low bits first within each word
        uint64_t val = (buf.data()[word_idx] >> bit_pos);
        if (bit_pos + d.width > 64)
            val |= (buf.data()[word_idx + 1] << (64 - bit_pos));
        return val & mask;
    }

    // Write entry g, class k at level h_idx into buf.
    void buf_set(sdsl::int_vector<0>& buf,
                 Tidx h_idx, int k, Tidx g, Tidx j, uint64_t val) const noexcept {
        const ClassDesc& d = class_desc[flat_idx(h_idx, k)];
        const Tidx local   = (g << class_shift(k)) + (j - class_start(k));
        // Use sdsl's bit-level access through the proxy
        const uint64_t start_bit = d.bit_offset + local * d.width;
        const uint64_t mask      = (d.width == 64) ? ~uint64_t(0)
                                                    : ((uint64_t(1) << d.width) - 1);
        uint64_t word_idx = start_bit >> 6;
        uint64_t bit_pos  = start_bit & 63;
        buf.data()[word_idx] &= ~(mask << bit_pos);
        buf.data()[word_idx] |=  (val  << bit_pos);
        if (bit_pos + d.width > 64) {
            const uint64_t bits_written = 64 - bit_pos;
            buf.data()[word_idx + 1] &= ~(mask >> bits_written);
            buf.data()[word_idx + 1] |=  (val  >> bits_written);
        }
    }

public:
    BDDST_Final(const std::vector<T>& arr, const Tidx B_S) : data(arr), BLOCK_SIZE(B_S) {
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

        // -- Pass 1: build class descriptors and total bit budget -------------
        total_class_slots = (levels - 1) * (levels + 2) / 2;
        class_desc.resize(total_class_slots);

        uint64_t left_total_bits  = 0;
        uint64_t right_total_bits = 0;

        for (Tidx h_idx = 1; h_idx < levels; ++h_idx) {
            const Tidx half  = Tidx(1) << h_idx;
            const Tidx range = half << 1;
            const int  num_classes = static_cast<int>(h_idx) + 1;

            Tidx num_centers = 0;
            if (blocks > half)
                num_centers = (blocks - 1 - half) / range + 1;

            for (int k = 0; k < num_classes; ++k) {
                const size_t fi = flat_idx(h_idx, k);
                const uint8_t w  = class_bits(k);
                const uint64_t cnt = (num_centers == 0) ? 0
                                   : (num_centers << class_shift(k));

                class_desc[fi].width       = w;
                class_desc[fi].num_entries = static_cast<uint32_t>(cnt);

                // left and right share the same layout
                class_desc[fi].bit_offset = left_total_bits;  // will be set again in pass 2

                left_total_bits  += cnt * w;
                right_total_bits += cnt * w;
            }
        }

        // -- Allocate the two monolithic buffers (bit-addressable) ------------
        // sdsl::int_vector<0> with width=1 gives us a plain bit array; we
        // re-interpret it as a raw word buffer via .data().
        // We use width=64 so sdsl allocates ceil(bits/64) 64-bit words.
        const uint64_t left_words  = (left_total_bits  + 63) / 64;
        const uint64_t right_words = (right_total_bits + 63) / 64;
        left_buf  = sdsl::int_vector<0>(left_words,  0, 64);
        right_buf = sdsl::int_vector<0>(right_words, 0, 64);

        // -- Pass 2: assign bit_offsets (left and right share same layout) ----
        uint64_t left_off = 0, right_off = 0;
        for (Tidx h_idx = 1; h_idx < levels; ++h_idx) {
            const int num_classes = static_cast<int>(h_idx) + 1;
            for (int k = 0; k < num_classes; ++k) {
                const size_t fi = flat_idx(h_idx, k);
                class_desc[fi].bit_offset = left_off;   // same for left and right
                left_off  += uint64_t(class_desc[fi].num_entries) * class_desc[fi].width;
                right_off += uint64_t(class_desc[fi].num_entries) * class_desc[fi].width;
            }
        }

        // -- Pass 3: fill entries ---------------------------------------------
        // Loop order: level h_idx -> group g -> class k -> j
        // This ensures we do a single linear scan over the `data` array per group,
        // maximizing cache locality, while still hoisting class descriptor lookups.
        uint64_t* __restrict__ lw = left_buf.data();
        uint64_t* __restrict__ rw = right_buf.data();

        auto write_bit = [](uint64_t* buf, uint64_t bit, uint64_t val, uint8_t w, uint64_t mask) {
            const uint64_t wi  = bit >> 6;
            const uint64_t bp  = bit & 63;
            buf[wi] = (buf[wi] & ~(mask << bp)) | ((val & mask) << bp);
            if (bp + w > 64) {
                const uint64_t done = 64 - bp;
                buf[wi+1] = (buf[wi+1] & ~(mask >> done)) | ((val & mask) >> done);
            }
        };

        for (Tidx h_idx = 1; h_idx < levels; ++h_idx) {
            const Tidx half       = Tidx(1) << h_idx;
            const Tidx range      = half << 1;
            const int  num_classes = static_cast<int>(h_idx) + 1;

            if (blocks <= half) continue;
            const Tidx num_centers = (blocks - 1 - half) / range + 1;

            for (Tidx g = 0; g < num_centers; ++g) {
                const Tidx center      = g * range + half;
                const Tidx center_data = center << BLOCK_SHIFT;

                // ---------------- Right side ----------------
                Tidx right_prev_abs = center_data + static_cast<Tidx>(base_row[center]);
                
                // k = 0 (j = 0)
                {
                    const uint8_t w = class_bits(0);
                    const uint64_t rbase = class_desc[flat_idx(h_idx, 0)].bit_offset;
                    const uint64_t mask = (w == 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
                    write_bit(rw, rbase + g * w, base_row[center], w, mask);
                }

                Tidx j_r = 1;
                for (int k = 1; k < num_classes; ++k) {
                    const Tidx j_start = class_start(k);
                    const Tidx j_end   = j_start + class_size(k);
                    const int sh       = class_shift(k);
                    const uint8_t w    = class_bits(k);
                    const uint64_t rbase = class_desc[flat_idx(h_idx, k)].bit_offset;
                    const uint64_t mask = (w == 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);

                    for (; j_r < j_end; ++j_r) {
                        const Tidx i = center + j_r;
                        if (i >= blocks) goto done_right;
                        const Tidx cur = (i << BLOCK_SHIFT) + static_cast<Tidx>(base_row[i]);
                        if (data[cur] < data[right_prev_abs]) right_prev_abs = cur;
                        
                        const uint64_t local = (uint64_t(g) << sh) + (j_r - j_start);
                        write_bit(rw, rbase + local * w, right_prev_abs - center_data, w, mask);
                    }
                }
                done_right:;

                // ---------------- Left side ----------------
                Tidx left_prev_abs = ((center - 1) << BLOCK_SHIFT) + static_cast<Tidx>(base_row[center - 1]);
                
                // k = 0 (j = 0)
                {
                    const uint8_t w = class_bits(0);
                    const uint64_t lbase = class_desc[flat_idx(h_idx, 0)].bit_offset;
                    const uint64_t mask = (w == 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
                    write_bit(lw, lbase + g * w, base_row[center - 1], w, mask);
                }

                Tidx j_l = 1;
                for (int k = 1; k < num_classes; ++k) {
                    const Tidx j_start = class_start(k);
                    const Tidx j_end   = j_start + class_size(k);
                    const int sh       = class_shift(k);
                    const uint8_t w    = class_bits(k);
                    const uint64_t lbase = class_desc[flat_idx(h_idx, k)].bit_offset;
                    const uint64_t mask = (w == 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);

                    for (; j_l < j_end; ++j_l) {
                        if (center <= j_l) goto done_left;
                        const Tidx i = center - 1 - j_l;
                        const Tidx i_data = i << BLOCK_SHIFT;
                        const Tidx cur = i_data + static_cast<Tidx>(base_row[i]);
                        if (data[cur] < data[left_prev_abs]) left_prev_abs = cur;
                        
                        const uint64_t local = (uint64_t(g) << sh) + (j_l - j_start);
                        write_bit(lw, lbase + local * w, left_prev_abs - i_data, w, mask);
                    }
                }
                done_left:;
            }
        }
    }

    // -- Query ----------------------------------------------------------------
    Tidx query(Tidx l, Tidx r) const {
        const Tidx b_l = l >> BLOCK_SHIFT;
        const Tidx b_r = r >> BLOCK_SHIFT;
        Tidx res = l;

        const Tidx bl_min = (b_l << BLOCK_SHIFT) + static_cast<Tidx>(base_row[b_l]);
        const Tidx br_min = (b_r << BLOCK_SHIFT) + static_cast<Tidx>(base_row[b_r]);

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

                    const Tidx j_ql   = center - 1 - ql;
                    const int  k_ql   = j_class(j_ql);
                    const Tidx ql_stored = static_cast<Tidx>(
                        buf_get(left_buf, h_idx, k_ql, g, j_ql));
                    const Tidx ql_abs = (ql << BLOCK_SHIFT) + ql_stored;

                    const Tidx j_qr   = qr - center;
                    const int  k_qr   = j_class(j_qr);
                    const Tidx qr_stored = static_cast<Tidx>(
                        buf_get(right_buf, h_idx, k_qr, g, j_qr));
                    const Tidx qr_abs = (center << BLOCK_SHIFT) + qr_stored;

                    if (data[ql_abs] < data[res]) res = ql_abs;
                    if (data[qr_abs] < data[res]) res = qr_abs;
                }
            }
        }

        // Left boundary scan.
        if (data[res] >= data[bl_min]) {
            const Tidx end_l = (b_l + 1) << BLOCK_SHIFT;
            if (l + 1 < end_l) {
                const Tidx left_min = QueryStrategy::scan_block(data, l + 1, end_l - 1);
                if (data[res] > data[left_min]) res = left_min;
            }
        }

        // Right boundary scan.
        if (data[res] >= data[br_min]) {
            const Tidx start_r = b_r << BLOCK_SHIFT;
            if (start_r <= r) {
                const Tidx right_min = QueryStrategy::scan_block(data, start_r, r);
                if (data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }

    // Returns total bits used by all internal tables (exact, no overhead).
    size_t table_bits() const {
        size_t total = base_row.bit_size();
        for (const auto& d : class_desc)
            total += 2 * uint64_t(d.num_entries) * d.width;   // left + right
        return total;
    }
};

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "BlockScanStrategies.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"

template <typename T = int, typename QueryStrategy = LinearScan, typename Tidx = std::uint64_t>
class BDDST_Final2 {

    sdsl::int_vector<0> base_row;
    std::vector<std::vector<sdsl::int_vector<0>>> left_data;
    std::vector<std::vector<sdsl::int_vector<0>>> right_data;

    const std::vector<T>& data;
    Tidx BLOCK_SIZE, BLOCK_SHIFT, blocks, levels;

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

    Tidx entry_idx(Tidx g, int k, Tidx j) const noexcept {
        return (g << class_shift(k)) + (j - class_start(k));
    }

public:
    BDDST_Final2(const std::vector<T>& arr, const Tidx B_S) : data(arr), BLOCK_SIZE(B_S) {
        BLOCK_SHIFT = std::bit_width(static_cast<unsigned>(BLOCK_SIZE)) - 1;
        const Tidx n = static_cast<Tidx>(data.size());
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1));

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

        left_data.resize(levels - 1);
        right_data.resize(levels - 1);

        for (Tidx h_idx = 1; h_idx < levels; ++h_idx) {
            const Tidx half  = Tidx(1) << h_idx;
            const Tidx range = half << 1;
            const int num_classes = static_cast<int>(h_idx) + 1;

            left_data [h_idx-1].resize(num_classes);
            right_data[h_idx-1].resize(num_classes);

            if (blocks <= half) {
                for (int k = 0; k < num_classes; ++k) {
                    left_data [h_idx-1][k] = sdsl::int_vector<0>(0, 0, class_bits(k));
                    right_data[h_idx-1][k] = sdsl::int_vector<0>(0, 0, class_bits(k));
                }
                continue;
            }

            const Tidx num_centers = (blocks - 1 - half) / range + 1;

            for (int k = 0; k < num_classes; ++k) {
                const Tidx sz = num_centers << class_shift(k);
                left_data [h_idx-1][k] = sdsl::int_vector<0>(sz, 0, class_bits(k));
                right_data[h_idx-1][k] = sdsl::int_vector<0>(sz, 0, class_bits(k));
            }

            for (Tidx g = 0; g < num_centers; ++g) {
                const Tidx center      = g * range + half;
                const Tidx center_data = center << BLOCK_SHIFT;

                // Right side
                {
                    right_data[h_idx-1][0][entry_idx(g, 0, 0)] = base_row[center];
                    Tidx right_prev_abs = center_data + static_cast<Tidx>(base_row[center]);

                    for (Tidx j = 1; j < half; ++j) {
                        const Tidx i = center + j;
                        if (i >= blocks) break;
                        const int  k   = j_class(j);
                        const Tidx cur = (i << BLOCK_SHIFT) + static_cast<Tidx>(base_row[i]);
                        Tidx stored;
                        if (data[cur] < data[right_prev_abs]) {
                            stored = cur - center_data;
                            right_prev_abs = cur;
                        } else {
                            stored = right_prev_abs - center_data;
                        }
                        right_data[h_idx-1][k][entry_idx(g, k, j)] = stored;
                    }
                }

                // Left side
                {
                    left_data[h_idx-1][0][entry_idx(g, 0, 0)] = base_row[center - 1];
                    Tidx left_prev_abs = ((center - 1) << BLOCK_SHIFT) + static_cast<Tidx>(base_row[center - 1]);

                    for (Tidx j = 1; j < half; ++j) {
                        if (center <= j) break;
                        const Tidx i      = center - 1 - j;
                        const Tidx i_data = i << BLOCK_SHIFT;
                        const int  k      = j_class(j);
                        const Tidx cur    = i_data + static_cast<Tidx>(base_row[i]);
                        Tidx stored;
                        if (data[cur] < data[left_prev_abs]) {
                            stored = cur - i_data;
                            left_prev_abs = cur;
                        } else {
                            stored = left_prev_abs - i_data;
                        }
                        left_data[h_idx-1][k][entry_idx(g, k, j)] = stored;
                    }
                }
            }
        }
    }

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
                        left_data[h_idx-1][k_ql][entry_idx(g, k_ql, j_ql)]);
                    const Tidx ql_abs = (ql << BLOCK_SHIFT) + ql_stored;

                    const Tidx j_qr   = qr - center;
                    const int  k_qr   = j_class(j_qr);
                    const Tidx qr_stored = static_cast<Tidx>(
                        right_data[h_idx-1][k_qr][entry_idx(g, k_qr, j_qr)]);
                    const Tidx qr_abs = (center << BLOCK_SHIFT) + qr_stored;

                    if (data[ql_abs] < data[res]) res = ql_abs;
                    if (data[qr_abs] < data[res]) res = qr_abs;
                }
            }
        }

        if (data[res] >= data[bl_min]) {
            const Tidx end_l = (b_l + 1) << BLOCK_SHIFT;
            if (l + 1 < end_l) {
                const Tidx left_min = QueryStrategy::scan_block(data, l + 1, end_l - 1);
                if (data[res] > data[left_min]) res = left_min;
            }
        }

        if (data[res] >= data[br_min]) {
            const Tidx start_r = b_r << BLOCK_SHIFT;
            if (start_r <= r) {
                const Tidx right_min = QueryStrategy::scan_block(data, start_r, r);
                if (data[res] > data[right_min]) res = right_min;
            }
        }

        return res;
    }

    size_t table_bits() const {
        size_t total = base_row.bit_size();
        for (const auto& lv : left_data)
            for (const auto& arr : lv) total += arr.bit_size();
        for (const auto& lv : right_data)
            for (const auto& arr : lv) total += arr.bit_size();
        return total;
    }
};

#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <memory>
#include <cstdint>
#include <cstddef>

template <typename DT, typename DTidx = std::size_t>
class BigDataDST {
    static constexpr DTidx BLOCK_SIZE { 64 };
    
    // Uninitialisierter Speicher verhindert teures Zeroing bei Giga-Arrays
    std::unique_ptr<DT[]> table; 
    const std::vector<DT>& raw_data; // Referenz für Intra-Block Queries
    DTidx blocks, levels;

public:
    // Build O(n) Time | O(n / BLOCK_SIZE * log(n / BLOCK_SIZE)) Space
    explicit BigDataDST(const std::vector<DT>& data) : raw_data(data) {
        DTidx n { static_cast<DTidx>(data.size()) };
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<unsigned>(blocks - 1)) + 1;

        // C++20: Allokiert Speicher ohne ihn mit Nullen zu überschreiben
        table = std::make_unique_for_overwrite<DT[]>(levels * blocks);

        // 1. Minima pro Block berechnen
        // #pragma omp parallel for schedule(dynamic)
        for (DTidx i { 0 }; i < blocks; ++i) {
            DT min_val { data[i * BLOCK_SIZE] };
            DTidx end { std::min(n, (i + 1) * BLOCK_SIZE) };
            for (DTidx j { i * BLOCK_SIZE + 1 }; j < end; ++j) {
                min_val = std::min(min_val, data[j]);
            }
            table[i] = min_val; // Base-Level der Sparse Table
        }

        // 2. Disjoint Sparse Table NUR über die Block-Minima bauen
        for (DTidx h { 1 }; h < levels; ++h) {
            DTidx half { DTidx(1) << (h - 1) };
            DTidx range { half << 1 };
            DT* row { &table[h * blocks] };

            // #pragma omp parallel for schedule(dynamic)
            for (DTidx center { half }; center < blocks; center += range) {
                row[center - 1] = table[center - 1];
                // Vermeidung von Unsigned Underflow (i >= 0 wird problematisch)
                for (DTidx j { 1 }; j < half; ++j) {
                    DTidx i { center - 1 - j };
                    row[i] = std::min(table[i], row[i + 1]);
                }

                DTidx rend { std::min(blocks, center + half) };
                row[center] = table[center];
                for (DTidx i { center + 1 }; i < rend; ++i)
                    row[i] = std::min(table[i], row[i - 1]);
            }
        }
    }

    // Query O(BLOCK_SIZE) Time
    DT query(DTidx l, DTidx r) const {
        DTidx b_l { l / BLOCK_SIZE };
        DTidx b_r { r / BLOCK_SIZE };

        // Fall 1: Query komplett im selben Block (linearer Scan)
        if (b_l == b_r) {
            DT res { raw_data[l] };
            for (DTidx i = l + 1; i <= r; ++i) res = std::min(res, raw_data[i]);
            return res;
        }

        // Fall 2: Query überspannt mehrere Blöcke
        // Rand links (Intra-Block)
        DT res { raw_data[l] };
        DTidx end_l { (b_l + 1) * BLOCK_SIZE };
        for (DTidx i { l + 1 }; i < end_l; ++i) res = std::min(res, raw_data[i]);

        // Rand rechts (Intra-Block)
        DTidx start_r { b_r * BLOCK_SIZE };
        for (DTidx i { start_r }; i <= r; ++i) res = std::min(res, raw_data[i]);

        // Dazwischenliegende Voll-Blöcke (Inter-Block O(1) via Sparse Table)
        if (b_l + 1 <= b_r - 1) {
            DTidx ql { b_l + 1 };
            DTidx qr { b_r - 1 };
            int h { std::bit_width(static_cast<unsigned>(ql ^ qr)) };
            const DT* row { &table[h * blocks] };
            res = std::min({res, row[ql], row[qr]});
        }

        return res;
    }
};

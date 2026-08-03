/**
 * @file ScanStrategies.hpp
 * @brief Micro-Architectural Intra-Block Scanning Strategies.
 *
 * Provides specialized linear scanning kernels (LinearScan, LinearSkipScan, and
 * MiniBlockScanner) optimized for cache density and SIMD vectorization during
 * fringe block query resolution.
 */

#pragma once
#include <vector>
#include <cstdint>


struct LinearScan {
    template <typename T, typename Tidx>
    static inline Tidx scan_block(const std::vector<T>& data, Tidx l, Tidx r) {
        Tidx min_idx { l };
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] < data[min_idx]) min_idx = i;
        }
        return min_idx;
    }
};

struct LinearSkipScan {
    template <typename T, typename Tidx>
    static inline Tidx scan_block(const T* data, Tidx l, Tidx r, T target_min = std::numeric_limits<T>::lowest()) {
        T min_val {data[l]};
        Tidx min_idx { l };
        if (min_val == target_min) return min_idx;
        for (Tidx i { l + 1 }; i <= r; ++i) {
            T curr {data[i]};
            if (curr < min_val) {
                min_val = curr;
                min_idx = i;
                if (min_val == target_min) return min_idx;
            }
        }
        return min_idx;
    }
};


struct OneLookUpLinearScan {

    // Caches the min_val to avoid double array lookups
    template <typename T, typename Tidx>
    static inline Tidx scan_block(const std::vector<T>& data, Tidx l, Tidx r) {
        T min_val {data[l]};
        Tidx min_idx { l };
        
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
        return min_idx;
    }
};


/*
 * This mini-block ragged boundary scanner is a rewrite of the 
 * BbST query handling logic originally proposed by Kowalski and Grabowski.
 */
struct MiniBlockScanner {
    template <typename T, typename Tidx, typename MiniBaseRow, typename MiniBaseVal>
    static inline Tidx scan_ragged(
        const std::vector<T>& data,
        const MiniBaseRow& mini_base_row,
        const MiniBaseVal& mini_base_val,
        unsigned log_block, unsigned log_mini_block,
        Tidx ql, Tidx qr, Tidx current_res, T& best_val) 
    {
        if (ql > qr) return current_res;
        
        if (log_mini_block == log_block) {
            // If no mini-blocks, we don't have a local minimum available here, so we just scan
            Tidx scan_res {LinearScan::scan_block(data, ql, qr)};
            if (data[scan_res] <= best_val) {
                best_val = data[scan_res];
                return scan_res;
            }
            return current_res;
        }
        
        Tidx mb_l {ql >> log_mini_block};
        Tidx mb_r {qr >> log_mini_block};
        
        if (mb_l == mb_r) {
            Tidx min_idx {(mb_l << log_mini_block) + Tidx(mini_base_row[mb_l])};
            if (min_idx >= ql && min_idx <= qr) {
                if (data[min_idx] <= best_val) {
                    best_val = data[min_idx];
                    return min_idx;
                }
                return current_res;
            }
            Tidx scan_res {LinearScan::scan_block(data, ql, qr)};
            if (data[scan_res] <= best_val) {
                best_val = data[scan_res];
                return scan_res;
            }
            return current_res;
        }
        
        Tidx res { current_res};
        
        // Left ragged mini-block
        Tidx l_min_idx {(mb_l << log_mini_block) + Tidx(mini_base_row[mb_l])};
        if (ql == (mb_l << log_mini_block) || l_min_idx >= ql) {
            if (mini_base_val[mb_l] <= best_val) {
                res = l_min_idx;
                best_val = mini_base_val[mb_l];
            }
        } else if (mini_base_val[mb_l] <= best_val) {
            Tidx end_bl {(mb_l + 1) << log_mini_block};
            Tidx scan_res {LinearScan::scan_block(data, ql, end_bl - 1)};
            if (data[scan_res] <= best_val) {
                res = scan_res;
                best_val = data[scan_res];
            }
        }
        
        // Full mini-blocks: two-step to minimize random data[] accesses
        // Step 1: scan mini_base_val sequentially
        Tidx best_mb {Tidx(-1)};
        for (Tidx i {mb_l + 1}; i < mb_r; ++i) {
            if (mini_base_val[i] < best_val) {
                best_val = mini_base_val[i];
                best_mb = i;
            }
        }
        // Step 2: only one random jump into large data[] if needed
        if (best_mb != Tidx(-1))
            res = (best_mb << log_mini_block) + mini_base_row[best_mb];
        
        // Right ragged mini-block
        Tidx n {static_cast<Tidx>(data.size())};
        Tidx r_min_idx {(mb_r << log_mini_block) + Tidx(mini_base_row[mb_r])};
        if (qr == std::min(n - 1, ((mb_r + 1) << log_mini_block) - 1) || r_min_idx <= qr) {
            if (mini_base_val[mb_r] < best_val) {
                res = r_min_idx;
                best_val = mini_base_val[mb_r];
            }
        } else if (mini_base_val[mb_r] < best_val) {
            Tidx start_br {mb_r << log_mini_block};
            Tidx scan_res {LinearScan::scan_block(data, start_br, qr)};
            if (data[scan_res] < best_val) {
                res = scan_res;
                best_val = data[scan_res];
            }
        }
        
        return res;
    }
};

/*
 Bitwise scanner replacing SDSL rank and select overhead 
 since bit_vector is an array 64bit integers
 */
struct WordScan {
    template <typename Tidx>
    static inline Tidx find_next_one(const uint64_t* words, Tidx pos) {
        Tidx word_idx {pos >> 6}; // pos / 64
        int bit_in_word {(int)pos & 63}; // pos % 64
        
        // Shift right to erase all bits before our target position
        uint64_t w {words[word_idx] >> bit_in_word};
        
        // If there are any 1-bits left, std::countr_zero gives the exact distance to the first one
        if (w) return (word_idx << 6) + bit_in_word + std::countr_zero(w);
        
        // Fallback loop forward until we find a non-zero word
        do { ++word_idx; } while (!(w = words[word_idx]));
        return (word_idx << 6) + std::countr_zero(w);
    }

    template <typename Tidx>
    static inline Tidx find_prev_one(const uint64_t* words, Tidx pos) {
        Tidx word_idx {pos >> 6}; // pos / 64
        int bit_in_word {(int)pos & 63}; // pos % 64
        
        // Shift left to push our target bit to the absolute ceiling (index 63)
        // This erases all bits that came after our target position
        uint64_t w {words[word_idx] << (63 - bit_in_word)};
        
        // If there are any 1-bits, std::countl_zero counts how many zeros exist before the first 1
        if (w) return (word_idx << 6) + bit_in_word - std::countl_zero(w);
        
        // Fallback loop backward until we find a non-zero word
        do { --word_idx; } while (!(w = words[word_idx]));
        return (word_idx << 6) + 63 - std::countl_zero(w);
    }
};

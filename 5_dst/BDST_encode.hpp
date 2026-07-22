#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "sdsl-lite/include/sdsl/bit_vectors.hpp"
#include "BDST_1Bit_Proxy.hpp"
#include "SegTree_Proxy.hpp"

#if defined(__BMI2__)
#include <immintrin.h>
#endif

// Configuration based on final architectural trace
const size_t BP_BLOCK_SIZE_BITS = 2048;
constexpr size_t SELECT_SAMPLE_RATE = 512;

struct ByteLUT {
    int8_t net[256];
    int8_t min_rel[256];
    uint8_t argmin[256];

    ByteLUT() {
        for (int i = 0; i < 256; i++) {
            int excess = 0;
            int min_ex = 127; // Use max to ensure it gets updated by the first bit
            uint8_t min_pos = 0;
            
            for (int b = 0; b < 8; b++) {
                int bit = (i >> b) & 1;
                excess += (bit == 1) ? 1 : -1;
                
                if (excess <= min_ex) {
                    min_ex = excess;
                    min_pos = b;
                }
            }
            net[i] = excess;
            min_rel[i] = min_ex;
            argmin[i] = min_pos;
        }
    }

    static const ByteLUT& get() {
        static ByteLUT lut;
        return lut;
    }
};

template <typename Tidx = std::uint64_t, bool UseSegTree = false>
class BDST_Encode_ApproachC {
    std::vector<uint64_t> bp_words;
    size_t num_bits;
    
    std::vector<uint32_t> select_samples;
    
    struct MacroBlockMeta {
        int32_t start_excess;
        int16_t min_rel;
        uint16_t argmin;
    };
    std::vector<MacroBlockMeta> macro_blocks;
    
    struct BlockMinProxy {
        const std::vector<MacroBlockMeta>& blocks;
        inline int operator[](size_t i) const {
            return blocks[i].start_excess + blocks[i].min_rel;
        }
        size_t size() const { return blocks.size(); }
    };
    
    using ProxyTree = std::conditional_t<UseSegTree,
        SegTree_Proxy<int, Tidx, BlockMinProxy>,
        BDST_1Bit_Proxy<int, Tidx, BlockMinProxy>>;
    
    std::unique_ptr<BlockMinProxy> proxy;
    std::unique_ptr<ProxyTree> dst;

    inline uint8_t get_byte(size_t bit_pos) const {
        size_t word_idx = bit_pos / 64;
        size_t bit_offset = bit_pos % 64;
        if (bit_offset <= 56) {
            return (bp_words[word_idx] >> bit_offset) & 0xFF;
        } else {
            uint8_t low = bp_words[word_idx] >> bit_offset;
            uint8_t high = bp_words[word_idx + 1] & ((1 << (8 - (64 - bit_offset))) - 1);
            return low | (high << (64 - bit_offset));
        }
    }
    
    inline int get_bit(size_t bit_pos) const {
        size_t word_idx = bit_pos / 64;
        size_t bit_offset = bit_pos % 64;
        return (bp_words[word_idx] >> bit_offset) & 1;
    }

    void build_bp(const std::vector<int>& data) {
        num_bits = 2 * data.size() + 2;
        bp_words.assign((num_bits + 63) / 64, 0);
        std::vector<Tidx> stack;
        size_t bp_idx = 0;
        
        auto push_bit = [&](int bit) {
            if (bit) {
                bp_words[bp_idx / 64] |= (1ULL << (bp_idx % 64));
            }
            bp_idx++;
        };
        
        push_bit(1); // Sentinel '('
        stack.push_back(-1ULL); 
        
        for (size_t i = 0; i < data.size(); ++i) {
            while (stack.size() > 1 && data[stack.back()] > data[i]) {
                stack.pop_back();
                push_bit(0); // ')'
            }
            stack.push_back(i);
            push_bit(1); // '('
        }
        
        while (!stack.empty()) {
            stack.pop_back();
            push_bit(0); // ')'
        }
        
        // Build Select samples
        size_t total_ones = data.size() + 1;
        select_samples.resize(total_ones / SELECT_SAMPLE_RATE + 1);
        
        uint32_t current_ones = 0;
        for (size_t i = 0; i < num_bits; ++i) {
            if (get_bit(i)) {
                if (current_ones % SELECT_SAMPLE_RATE == 0) {
                    select_samples[current_ones / SELECT_SAMPLE_RATE] = i;
                }
                current_ones++;
            }
        }
    }

    void precompute_blocks() {
        size_t num_blocks = (num_bits + BP_BLOCK_SIZE_BITS - 1) / BP_BLOCK_SIZE_BITS;
        macro_blocks.resize(num_blocks);

        int32_t current_excess = 0;
        const auto& lut = ByteLUT::get();

        for (size_t blk = 0; blk < num_blocks; ++blk) {
            macro_blocks[blk].start_excess = current_excess;
            
            int32_t blk_min_rel = 1e9;
            uint16_t blk_argmin_pos = 0;
            int32_t local_excess = 0;

            size_t bytes_to_process = BP_BLOCK_SIZE_BITS / 8;
            
            for (size_t byte_idx = 0; byte_idx < bytes_to_process; ++byte_idx) {
                size_t global_bit_pos = blk * BP_BLOCK_SIZE_BITS + byte_idx * 8;
                uint8_t b = get_byte(global_bit_pos);
                
                int32_t cand_min = local_excess + lut.min_rel[b];
                
                if (cand_min <= blk_min_rel) {
                    blk_min_rel = cand_min;
                    blk_argmin_pos = byte_idx * 8 + lut.argmin[b]; 
                }
                local_excess += lut.net[b];
            }

            macro_blocks[blk].min_rel = blk_min_rel;
            macro_blocks[blk].argmin = blk_argmin_pos;
            current_excess += local_excess;
        }
    }

    // Fast sparse select using __builtin_ctzll
    inline size_t select_1(size_t k) const {
        size_t idx = k - 1;
        size_t sample_idx = idx / SELECT_SAMPLE_RATE;
        size_t pos = select_samples[sample_idx];
        size_t target_ones_remaining = idx % SELECT_SAMPLE_RATE;
        
        if (target_ones_remaining == 0) return pos;
        
        size_t word_idx = pos / 64;
        size_t bit_offset = pos % 64;
        
        uint64_t word = (bit_offset == 63) ? 0ULL : (bp_words[word_idx] >> (bit_offset + 1));
        int ones_in_word = __builtin_popcountll(word);
        
        if (target_ones_remaining <= (size_t)ones_in_word) {
#if defined(__BMI2__)
            pos += __builtin_ctzll(_pdep_u64(1ULL << (target_ones_remaining - 1), word)) + 1;
            return pos;
#else
            while (target_ones_remaining > 0) {
                int tz = __builtin_ctzll(word);
                pos += tz + 1;
                word >>= tz;
                word >>= 1;
                target_ones_remaining--;
            }
            return pos;
#endif
        }
        
        target_ones_remaining -= ones_in_word;
        pos += (64 - (bit_offset + 1));
        word_idx++;
        
        while (true) {
            word = bp_words[word_idx];
            ones_in_word = __builtin_popcountll(word);
            if (target_ones_remaining <= (size_t)ones_in_word) {
#if defined(__BMI2__)
                pos = word_idx * 64 + __builtin_ctzll(_pdep_u64(1ULL << (target_ones_remaining - 1), word));
                return pos;
#else
                pos = word_idx * 64 - 1;
                while (target_ones_remaining > 0) {
                    int tz = __builtin_ctzll(word);
                    pos += tz + 1;
                    word >>= tz;
                    word >>= 1;
                    target_ones_remaining--;
                }
                return pos;
#endif
            }
            target_ones_remaining -= ones_in_word;
            word_idx++;
        }
    }

    // Blazing fast relative select_1: bypasses select_samples global memory lookup 
    // by scanning forward directly from start_pos. Used when r - l is small.
    inline size_t relative_select_1(size_t start_pos, size_t target_ones_remaining) const {
        size_t pos = start_pos;
        size_t word_idx = pos / 64;
        size_t bit_offset = pos % 64;
        
        uint64_t word = (bit_offset == 63) ? 0ULL : (bp_words[word_idx] >> (bit_offset + 1));
        int ones_in_word = __builtin_popcountll(word);
        
        if (target_ones_remaining <= (size_t)ones_in_word) {
#if defined(__BMI2__)
            return pos + __builtin_ctzll(_pdep_u64(1ULL << (target_ones_remaining - 1), word)) + 1;
#else
            while (target_ones_remaining > 0) {
                int tz = __builtin_ctzll(word);
                pos += tz + 1;
                word >>= tz;
                word >>= 1;
                target_ones_remaining--;
            }
            return pos;
#endif
        }
        
        target_ones_remaining -= ones_in_word;
        pos += (64 - (bit_offset + 1));
        word_idx++;
        
        while (true) {
            word = bp_words[word_idx];
            ones_in_word = __builtin_popcountll(word);
            if (target_ones_remaining <= (size_t)ones_in_word) {
#if defined(__BMI2__)
                return word_idx * 64 + __builtin_ctzll(_pdep_u64(1ULL << (target_ones_remaining - 1), word));
#else
                pos = word_idx * 64 - 1;
                while (target_ones_remaining > 0) {
                    int tz = __builtin_ctzll(word);
                    pos += tz + 1;
                    word >>= tz;
                    word >>= 1;
                    target_ones_remaining--;
                }
                return pos;
#endif
            }
            target_ones_remaining -= ones_in_word;
            word_idx++;
        }
    }

    inline std::pair<int, size_t> scan_fringe(size_t start_pos, size_t end_pos, int start_excess) const {
        if (start_pos > end_pos) return {1e9, start_pos - 1};

        int current_excess = start_excess;
        int min_excess = 1e9;
        size_t min_pos = start_pos;
        
        const auto& lut = ByteLUT::get();
        size_t curr_pos = start_pos;

        // 1. Align to 8 bits
        while (curr_pos <= end_pos && (curr_pos % 8 != 0)) {
            int bit = get_bit(curr_pos);
            current_excess += (bit == 1) ? 1 : -1;
            if (current_excess <= min_excess) {
                min_excess = current_excess;
                min_pos = curr_pos;
            }
            curr_pos++;
        }

        // 2a. Align to 64 bits (process bytes until curr_pos is a multiple of 64)
        while (curr_pos + 7 <= end_pos && (curr_pos % 64 != 0)) {
            int byte_val = get_byte(curr_pos);
            int min_candidate = current_excess + lut.min_rel[byte_val];
            if (min_candidate <= min_excess) {
                min_excess = min_candidate;
                min_pos = curr_pos + lut.argmin[byte_val];
            }
            current_excess += lut.net[byte_val];
            curr_pos += 8;
        }

        // 2b. Process full words (64 bits) at a time
        while (curr_pos + 63 <= end_pos) {
            size_t word_idx = curr_pos / 64;
            uint64_t w_val = bp_words[word_idx];
            int word_net = 2 * __builtin_popcountll(w_val) - 64;
            
            // If the lowest possible excess in this word cannot strictly EQUAL OR BEAT
            // the current minimum, we can skip the entire word.
            if (current_excess - 64 > min_excess) {
                current_excess += word_net;
                curr_pos += 64;
            } else {
                // Unrolled byte processing
#define PROCESS_BYTE(b) \
                do { \
                    int byte_val = (w_val >> (b * 8)) & 0xFF; \
                    int min_candidate = current_excess + lut.min_rel[byte_val]; \
                    if (min_candidate <= min_excess) { \
                        min_excess = min_candidate; \
                        min_pos = curr_pos + lut.argmin[byte_val]; \
                    } \
                    current_excess += lut.net[byte_val]; \
                    curr_pos += 8; \
                } while (0)

                PROCESS_BYTE(0);
                PROCESS_BYTE(1);
                PROCESS_BYTE(2);
                PROCESS_BYTE(3);
                PROCESS_BYTE(4);
                PROCESS_BYTE(5);
                PROCESS_BYTE(6);
                PROCESS_BYTE(7);
#undef PROCESS_BYTE
            }
        }

        // 2c. Process remaining full bytes
        while (curr_pos + 7 <= end_pos) {
            int byte_val = get_byte(curr_pos);
            int min_candidate = current_excess + lut.min_rel[byte_val];
            if (min_candidate <= min_excess) {
                min_excess = min_candidate;
                min_pos = curr_pos + lut.argmin[byte_val];
            }
            current_excess += lut.net[byte_val];
            curr_pos += 8;
        }

        // 3. Tail bits
        while (curr_pos <= end_pos) {
            int bit = get_bit(curr_pos);
            current_excess += (bit == 1) ? 1 : -1;
            if (current_excess <= min_excess) {
                min_excess = current_excess;
                min_pos = curr_pos;
            }
            curr_pos++;
        }
        
        return {min_excess, min_pos};
    }


public:
    BDST_Encode_ApproachC(const std::vector<int>& data, Tidx B = 16) 
    {
        build_bp(data);
        precompute_blocks();
        
        proxy = std::make_unique<BlockMinProxy>(BlockMinProxy{macro_blocks});
        // Build ProxyTree over the block minimums
        dst = std::make_unique<ProxyTree>(*proxy, B);
    }

    Tidx query(Tidx l, Tidx r) const {
        if (l == r) return l;
        
        Tidx k_x = l + 2;
        Tidx k_y = r + 2;
        
        Tidx x = select_1(k_x);
        
        // HUGE cache optimization: if query length is small, y is close to x.
        // Bypassing a global select_1 avoids hitting the 15MB select_samples array again.
        Tidx y;
        if (r - l < 512) {
            y = relative_select_1(x, r - l);
        } else {
            y = select_1(k_y);
        }
        
        Tidx b_l = x / BP_BLOCK_SIZE_BITS;
        Tidx b_r = y / BP_BLOCK_SIZE_BITS;
        
        int32_t min_excess = 1e9;
        Tidx w = 0;
        
        // O(1) Exact mathematical initial excess before x
        int32_t excess_x = 2 * k_x - x - 2;
        
        if (b_l == b_r) {
            auto res = scan_fringe(x + 1, y, excess_x + 1);
            min_excess = res.first;
            w = res.second;
        } else {
            auto res_l = scan_fringe(x + 1, (b_l + 1) * BP_BLOCK_SIZE_BITS - 1, excess_x + 1);
            min_excess = res_l.first;
            w = res_l.second;
            
            // Middle blocks via BDST
            // Because we updated BDST_1Bit_Proxy to use <=, dst->query is guaranteed
            // to return the rightmost block that achieves the minimum excess!
            if (b_l + 1 <= b_r - 1) {
                Tidx min_block = dst->query(b_l + 1, b_r - 1);
                int32_t cand_min = macro_blocks[min_block].start_excess + macro_blocks[min_block].min_rel;
                
                if (cand_min <= min_excess) {
                    min_excess = cand_min;
                    w = min_block * BP_BLOCK_SIZE_BITS + macro_blocks[min_block].argmin;
                }
            }
            
            // Right fringe
            auto res_r = scan_fringe(b_r * BP_BLOCK_SIZE_BITS, y, macro_blocks[b_r].start_excess);
            if (res_r.first <= min_excess) {
                min_excess = res_r.first;
                w = res_r.second;
            }
        }
        
        if (min_excess >= excess_x + 1) return l;

        // Mathematical algebraic formula mapping min_pos to array index
        Tidx rank1 = (w + 1 + min_excess) / 2;
        return rank1 - 1;
    }

    size_t table_bits() const {
        return bp_words.size() * 64 + 
               select_samples.size() * 32 + 
               macro_blocks.size() * 64 +
               dst->table_bits();
    }
};

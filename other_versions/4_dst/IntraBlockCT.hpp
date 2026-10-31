#pragma once
#include <vector>
#include <cstdint>
#include <bit>
#include <algorithm>

struct CT_Tables {
    int8_t T_SUM[256];
    int8_t T_MIN[256];
    uint8_t T_MIN_POS[256];
    
    CT_Tables() {
        for (int i = 0; i < 256; ++i) {
            int current_excess = 0;
            int min_excess = 0; 
            int min_pos = 0; 
            
            for (int b = 0; b < 8; ++b) {
                if ((i >> b) & 1) current_excess++;
                else current_excess--;
                
                if (b == 0 || current_excess <= min_excess) {
                    min_excess = current_excess;
                    min_pos = b;
                }
            }
            T_SUM[i] = current_excess;
            T_MIN[i] = min_excess;
            T_MIN_POS[i] = min_pos;
        }
    }
};
static CT_Tables ct_tables;

template <typename T = int, typename Tidx = std::uint64_t>
class IntraBlockCT {
    std::vector<uint64_t> bp;
    size_t B;
    size_t words_per_block;

public:
    IntraBlockCT() : B(0), words_per_block(0) {}

    IntraBlockCT(const std::vector<T>* data_ptr, size_t B_S) : B(B_S) {
        if (!data_ptr || data_ptr->empty() || B == 0) return;
        const auto& data = *data_ptr;
        size_t n = data.size();
        size_t num_blocks = (n + B - 1) / B;
        
        words_per_block = (2 * B + 63) / 64;
        bp.resize(num_blocks * words_per_block, 0);

        std::vector<uint32_t> stack;
        stack.reserve(B);

        for (size_t k = 0; k < num_blocks; ++k) {
            size_t start = k * B;
            size_t end = std::min(n, start + B);
            stack.clear();
            
            size_t bit_pos = 0;
            auto push_bit = [&](int bit) {
                if (bit) {
                    bp[k * words_per_block + (bit_pos / 64)] |= (1ULL << (bit_pos % 64));
                }
                bit_pos++;
            };

            for (size_t i = start; i < end; ++i) {
                while (!stack.empty() && data[stack.back()] > data[i]) {
                    stack.pop_back();
                    push_bit(0);
                }
                stack.push_back(i);
                push_bit(1);
            }
            while (!stack.empty()) {
                stack.pop_back();
                push_bit(0);
            }
        }
    }

    uint32_t query_local(size_t block_idx, uint32_t l, uint32_t r) const {
        size_t word_offset = block_idx * words_per_block;
        
        // Pass 1: Find pos_l and pos_r
        uint32_t pos_l = -1, pos_r = -1;
        uint32_t ones_seen = 0;
        
        for (uint32_t w = 0; w < words_per_block; ++w) {
            uint64_t word = bp[word_offset + w];
            uint32_t pop = std::popcount(word);
            
            if (pos_l == (uint32_t)-1 && ones_seen + pop > l) {
                uint32_t needed = l - ones_seen;
                for (int b = 0; b < 64; ++b) {
                    if ((word >> b) & 1) {
                        if (needed == 0) { pos_l = w * 64 + b; break; }
                        needed--;
                    }
                }
            }
            if (pos_r == (uint32_t)-1 && ones_seen + pop > r) {
                uint32_t needed = r - ones_seen;
                for (int b = 0; b < 64; ++b) {
                    if ((word >> b) & 1) {
                        if (needed == 0) { pos_r = w * 64 + b; break; }
                        needed--;
                    }
                }
                break;
            }
            ones_seen += pop;
        }

        // Pass 2: Find rightmost minimum excess in [pos_l, pos_r]
        int current_excess = 0;
        int min_excess = 0;
        uint32_t min_m = pos_l;
        uint32_t curr = pos_l;
        
        while (curr <= pos_r && (curr % 8) != 0) {
            uint64_t word = bp[word_offset + (curr / 64)];
            int bit = (word >> (curr % 64)) & 1;
            if (bit) current_excess++; else current_excess--;
            
            if (current_excess <= min_excess) {
                min_excess = current_excess;
                min_m = curr;
            }
            curr++;
        }

        while (curr + 7 <= pos_r) {
            uint64_t word = bp[word_offset + (curr / 64)];
            uint8_t byte = (word >> (curr % 64)) & 0xFF;
            
            int b_min = current_excess + ct_tables.T_MIN[byte];
            if (b_min <= min_excess) {
                min_excess = b_min;
                min_m = curr + ct_tables.T_MIN_POS[byte];
            }
            current_excess += ct_tables.T_SUM[byte];
            curr += 8;
        }

        while (curr <= pos_r) {
            uint64_t word = bp[word_offset + (curr / 64)];
            int bit = (word >> (curr % 64)) & 1;
            if (bit) current_excess++; else current_excess--;
            
            if (current_excess <= min_excess) {
                min_excess = current_excess;
                min_m = curr;
            }
            curr++;
        }

        // Pass 3: Calculate rank_1 up to min_m
        uint32_t rank_1 = 0;
        for (uint32_t w = 0; w < min_m / 64; ++w) {
            rank_1 += std::popcount(bp[word_offset + w]);
        }
        uint64_t last_word = bp[word_offset + (min_m / 64)];
        uint64_t mask = ~0ULL;
        if (min_m % 64 != 63) {
            mask = (1ULL << ((min_m % 64) + 1)) - 1;
        }
        rank_1 += std::popcount(last_word & mask);

        int bit_m = (last_word >> (min_m % 64)) & 1;
        return rank_1 - (bit_m == 1 ? 1 : 0);
    }

    Tidx operator()(Tidx l, Tidx r) const {
        if (l > r) std::swap(l, r);
        if (l == r) return l;
        
        size_t block_idx = l / B;
        uint32_t local_l = l % B;
        uint32_t local_r = r % B;
        
        return block_idx * B + query_local(block_idx, local_l, local_r);
    }
};

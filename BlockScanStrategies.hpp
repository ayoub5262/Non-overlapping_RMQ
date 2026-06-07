#pragma once
#include <vector>
#include <cstdint>
#include <immintrin.h>




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
    static inline Tidx scan_block(const std::vector<T>& data, Tidx l, Tidx r, T target_min = std::numeric_limits<T>::lowest()) {
        Tidx min_idx { l };
        if (data[min_idx] == target_min) return min_idx;
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] < data[min_idx]) {
                min_idx = i;
                if (data[min_idx] == target_min) return min_idx;
                
            }
        }
        return min_idx;
    }
};

// 2. Sequential Range Minimum Query
struct OneLookUpLinearScan {

    // Caches the min_val to avoid double array lookups
    template <typename T, typename Tidx>
    static inline Tidx scan_block(const std::vector<T>& data, Tidx l, Tidx r) {
        T min_val = data[l];
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

// 3. Branchless Range Minimum Query, Recommended for std::vector
struct CmovScan {

    // Emulates the SWAR philosophy: it uses math/logic instead of branching.
    // Compilers translate this into fast `cmov` (conditional move) instructions.
    template <typename T, typename Tidx>
    static inline Tidx scan_block(const std::vector<T>& data, Tidx l, Tidx r) {
        T min_val = data[l];
        Tidx min_idx { l };
        
        for (Tidx i { l + 1 }; i <= r; ++i) {
            T curr = data[i];
            bool is_less = (curr < min_val);
            
            min_val = is_less ? curr : min_val;
            min_idx = is_less ? i : min_idx;
        }
        return min_idx;
    }
};

// 4. Explicit Hardware SIMD (AVX2 / AVX-512) Scan
struct AVX2Scan32b {
    // Vectorized RMQ using 256-bit AVX2 instructions.
    // Processes 8 32-bit integers simultaneously.
    template <typename T, typename Tidx>
    static inline Tidx scan_block(const std::vector<T>& data, Tidx l, Tidx r) {
        T min_val = data[l];
        Tidx min_idx { l };
        Tidx i = l + 1;

#if defined(__AVX2__)
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
            // Note: indices are stored as int32_t inside SIMD lanes.
            // Safe as long as array size fits in int32_t (n < 2^31).
            static_assert(sizeof(Tidx) <= sizeof(int32_t),
                "AVX2Scan: Tidx must be at most 32-bit to avoid index overflow");
            __m256i v_min     = _mm256_set1_epi32(static_cast<int32_t>(min_val));
            __m256i v_min_idx = _mm256_set1_epi32(static_cast<int32_t>(l)); // use l, not min_idx
            __m256i v_idx_offset = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            
            for (; i + 7 <= r; i += 8) {
                __m256i v_data = _mm256_loadu_si256((const __m256i*)&data[i]);
                __m256i v_curr_idx = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int32_t>(i)), v_idx_offset);
                
                // Compare: v_min > v_data (equivalent to v_data < v_min)
                __m256i cmp = _mm256_cmpgt_epi32(v_min, v_data);
                
                // Blend: Update min values and their indices where cmp is true (0xFFFFFFFF)
                v_min = _mm256_blendv_epi8(v_min, v_data, cmp);
                v_min_idx = _mm256_blendv_epi8(v_min_idx, v_curr_idx, cmp);
            }
            
            // Extract the horizontal minimum from the 8 SIMD lanes
            int32_t mins[8];
            int32_t idxs[8];
            _mm256_storeu_si256((__m256i*)mins, v_min);
            _mm256_storeu_si256((__m256i*)idxs, v_min_idx);
            
            for(int k = 0; k < 8; ++k) {
                if (mins[k] < min_val) {
                    min_val = mins[k];
                    min_idx = idxs[k];
                } else if (mins[k] == min_val && idxs[k] < min_idx) {
                    min_idx = idxs[k]; // Maintain leftmost index for stability
                }
            }
        }
#endif

        // Tail processing for remaining elements (or fallback if not AVX2/int)
        for (; i <= r; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
        
        return min_idx;
    }
};

// 5. Explicit Hardware SIMD (AVX2 / AVX-512) Scan for 64-bit Indices
struct AVX2Scan64b {
    // Vectorized RMQ using 256-bit AVX2 instructions.
    // Processes 8 32-bit data elements simultaneously, but tracks 64-bit indices.
    template <typename T, typename Tidx>
    static inline Tidx scan_block(const std::vector<T>& data, Tidx l, Tidx r) {
        T min_val = data[l];
        Tidx min_idx { l };
        Tidx i = l + 1;

#if defined(__AVX2__)
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
            __m256i v_min = _mm256_set1_epi32(static_cast<int32_t>(min_val));
            
            // We need two registers for the 8 64-bit indices
            __m256i v_min_idx_lo = _mm256_set1_epi64x(static_cast<int64_t>(l));
            __m256i v_min_idx_hi = _mm256_set1_epi64x(static_cast<int64_t>(l));
            
            // Current indices offsets:
            __m256i v_idx_offset_lo = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i v_idx_offset_hi = _mm256_setr_epi64x(4, 5, 6, 7);

            for (; i + 7 <= r; i += 8) {
                __m256i v_data = _mm256_loadu_si256((const __m256i*)&data[i]);
                
                __m256i v_base_idx = _mm256_set1_epi64x(static_cast<int64_t>(i));
                __m256i v_curr_idx_lo = _mm256_add_epi64(v_base_idx, v_idx_offset_lo);
                __m256i v_curr_idx_hi = _mm256_add_epi64(v_base_idx, v_idx_offset_hi);
                
                // Compare: v_min > v_data (equivalent to v_data < v_min)
                __m256i cmp = _mm256_cmpgt_epi32(v_min, v_data);
                
                // Update minimum values
                v_min = _mm256_blendv_epi8(v_min, v_data, cmp);
                
                // To update indices, sign-extend the 32-bit mask to 64-bit masks
                __m128i cmp_lo_128 = _mm256_castsi256_si128(cmp);
                __m128i cmp_hi_128 = _mm256_extracti128_si256(cmp, 1);
                
                __m256i cmp_lo = _mm256_cvtepi32_epi64(cmp_lo_128);
                __m256i cmp_hi = _mm256_cvtepi32_epi64(cmp_hi_128);
                
                v_min_idx_lo = _mm256_blendv_epi8(v_min_idx_lo, v_curr_idx_lo, cmp_lo);
                v_min_idx_hi = _mm256_blendv_epi8(v_min_idx_hi, v_curr_idx_hi, cmp_hi);
            }
            
            // Extract the horizontal minimum from the 8 SIMD lanes
            int32_t mins[8];
            int64_t idxs[8];
            _mm256_storeu_si256((__m256i*)mins, v_min);
            _mm256_storeu_si256((__m256i*)idxs, v_min_idx_lo);
            _mm256_storeu_si256((__m256i*)(idxs + 4), v_min_idx_hi);
            
            for(int k = 0; k < 8; ++k) {
                if (mins[k] < min_val) {
                    min_val = mins[k];
                    min_idx = static_cast<Tidx>(idxs[k]);
                } else if (mins[k] == min_val && static_cast<Tidx>(idxs[k]) < min_idx) {
                    min_idx = static_cast<Tidx>(idxs[k]); // Maintain leftmost index for stability
                }
            }
        }
#endif

        // Tail processing for remaining elements (or fallback if not AVX2/int)
        for (; i <= r; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
        
        return min_idx;
    }
};

#include <vector>
#include <utility>
#include <cstdint>

// Macro for hardware popcount depending on the compiler
#if defined(__GNUC__) || defined(__clang__)
#define HW_POPCOUNT __builtin_popcountll
#else
#include <intrin.h>
#define HW_POPCOUNT __popcnt64
#endif

struct NaiveLinearScan {
    
    // 1. Your Original Range Minimum Query (RMQ)
    template <typename T, typename Tidx>
    static inline Tidx block_scan(const std::vector<T>& data, Tidx l, Tidx r) {
        Tidx res { l };
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] < data[res]) res = i;
        }
        return res;
    }
};

// 2. Sequential Range Minimum Count (RMC)
struct LinearScan{

    // Note: The 'else if' introduces a branch misprediction penalty in the CPU pipeline.
    template <typename T, typename Tidx>
    static inline std::pair<Tidx, Tidx> block_scan_rmc_sequential(const std::vector<T>& data, Tidx l, Tidx r) {
        T min_val = data[l];
        Tidx min_idx { l };
        Tidx count { 1 };
        
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
                count = 1;         // Reset count
            } else if (data[i] == min_val) {
                count++;           // Branching penalty here
            }
        }
        return {min_idx, count};
    }
};

// 3. Branchless Range Minimum Count (RMC) - Recommended for std::vector
struct BranchlessScan{

    // Emulates the SWAR philosophy: it uses math/logic instead of branching.
    // Compilers translate this into fast `cmov` (conditional move) instructions.
    template <typename T, typename Tidx>
    static inline std::pair<Tidx, Tidx> block_scan_rmc_branchless(const std::vector<T>& data, Tidx l, Tidx r) {
        T min_val = data[l];
        Tidx min_idx { l };
        Tidx count { 1 };
        
        for (Tidx i { l + 1 }; i <= r; ++i) {
            T curr = data[i];
            bool is_less = (curr < min_val);
            bool is_equal = (curr == min_val);
            
            min_val = is_less ? curr : min_val;
            min_idx = is_less ? i : min_idx;
            
            // If new min, reset to 1. If equal, increment. If greater, add 0.
            count = is_less ? 1 : (count + is_equal);
        }
        return {min_idx, count};
    }
};

// 4. Broadword (SWAR) Concept for a 64-bit Balanced Parentheses Chunk
struct BroadwordScan{
    private:
    // Abstracted helper: True implementation requires broadword byte-multiplication
    static inline uint64_t generate_swar_mask(uint64_t word, int target_min) {
        return 0ULL; // Replace with actual SWAR prefix sum mask generation
    }

    // This is the implementation blueprint for the succinct library approach.
    static inline std::pair<int, int> broadword_bp_rmc(uint64_t bp_word, int exact_min_excess) {
        // Step 1: A SWAR function calculates parallel prefix sums and returns a 64-bit 
        // mask where bits are '1' ONLY at indices where current_excess == exact_min_excess.
        uint64_t min_mask = generate_swar_mask(bp_word, exact_min_excess); 
        
        // Step 2: Math-based extraction (Zero loops, zero branches)
        int min_idx = __builtin_ctzll(min_mask); // Trailing zeros finds the leftmost index
        int count = HW_POPCOUNT(min_mask);       // Popcount finds the exact frequency instantly
        
        return {min_idx, count};
    }


};
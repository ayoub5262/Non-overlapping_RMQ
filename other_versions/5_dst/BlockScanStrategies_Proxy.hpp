#pragma once
#include <vector>
#include <limits>
#include <algorithm>

struct LinearScanProxy {
    template <typename Container, typename Tidx>
    static inline Tidx scan_block(const Container& data, Tidx l, Tidx r) {
        Tidx min_idx { l };
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] <= data[min_idx]) min_idx = i;
        }
        return min_idx;
    }
};

struct LinearSkipScanProxy {
    template <typename Container, typename Tidx, typename T = int>
    static inline Tidx scan_block(const Container& data, Tidx l, Tidx r, T target_min = std::numeric_limits<T>::lowest()) {
        Tidx min_idx { l };
        // We cannot early return on target_min because we want the rightmost!
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] <= data[min_idx]) {
                min_idx = i;
            }
        }
        return min_idx;
    }
};

struct OneLookUpLinearScanProxy {
    // Caches the min_val to avoid double array lookups
    template <typename Container, typename Tidx>
    static inline Tidx scan_block(const Container& data, Tidx l, Tidx r) {
        auto min_val = data[l];
        Tidx min_idx { l };
        
        for (Tidx i { l + 1 }; i <= r; ++i) {
            if (data[i] <= min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
        return min_idx;
    }
};

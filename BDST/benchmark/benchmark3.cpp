
/**
 * @file benchmark3.cpp
 * @brief Compares optimized two-level structures (BDST_1Bit_Fast, BDST2, Blocked_SegmentTree_Fast, BbST2) and sparse tables (SDSL Sparse Table, Disjoint Sparse Table).
 */

#define MINI_BLOCKS
#include "BbST/bbst.h"

#include "benchmark_common.hpp"

#include "BDST/include/BDST_1Bit_Fast.hpp"
#include "BDST/include/BDST2.hpp"
#include "BDST/include/Blocked_SegmentTree_Fast.hpp"
#include "sdsl-lite/include/sdsl/rmq_support_sparse_table.hpp"

int main() {
    ofstream csv("benchmark3.csv");
    write_header(csv);
    Timer timer;

    int rotation {0};
    for_each_scenario(array_sizes, [&](uint32_t n, const string& data_type, const vector<int>& data) {
        // BbST author use unsigned int
        using T1 = BDST_1Bit_Fast<int, uint32_t>;
        using T2 = BDST2<int, uint32_t>;
        using T3 = Blocked_SegmentTree_Fast<int, uint32_t>;

        // BbST block/miniblock parameters for B=4096
        const int k1 {12}; // 2^12 = 4096
        const int k2 {8}; // 2^8 = 256

        for (int rotate {0}; rotate < 5; ++rotate) {
            switch ((rotate + rotation) % 5) {
                case 0:
                    run_generic(csv, "BDST(4096-256)", 4096, n, data_type, timer,
                        [&]() { return std::make_unique<T1>(data, 4096); },
                        [](T1& rmq, uint32_t l, uint32_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 1:
                    run_generic(csv, "BDST2(4096-256)", 4096, n, data_type, timer,
                        [&]() { return std::make_unique<T2>(data, 4096); },
                        [](T2& rmq, uint32_t l, uint32_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 2:
                    run_generic(csv, "B_SegTree(4096-256)", 4096, n, data_type, timer,
                        [&]() { return std::make_unique<T3>(data, 4096); },
                        [](T3& rmq, uint32_t l, uint32_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 3:
                    run_generic(csv, "BbST2(4096-256)", 4096, n, data_type, timer,
                        [&, k1, k2]() {
                            return std::make_unique<BbST>(
                                data.data(), data.size(), k1, k2);
                        },
                        [](BbST& rmq, uint32_t l, uint32_t r) -> size_t {
                            return (size_t)rmq.rmq(l, r);
                        });
                    break;
                case 4:
                    if (n > 100'000'000) {
                        cout << "    -> Skipping SDSL Sparse Table for n=" << n << endl;
                        break;
                    }
                    run_generic(csv, "SDSL_Sparse_Table", 0, n, data_type, timer,
                        [&]() {
                            return std::make_unique<sdsl::rmq_support_sparse_table<true, true, vector<int>>>(&data);
                        },
                        [](sdsl::rmq_support_sparse_table<true, true, vector<int>>& rmq, uint32_t l, uint32_t r) -> size_t {
                            return (size_t)rmq(l, r);
                        });
                    break;
            
            }
        }
        ++rotation;
    });

    csv.close();
    cout << "Benchmarking complete." << endl;
    return 0;
}
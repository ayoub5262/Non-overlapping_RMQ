
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

    int rotation = 0;
    for_each_scenario(array_sizes, [&](uint64_t n, const string& data_type, const vector<int>& data) {
        using T1 = BDST_1Bit_Fast<int, uint64_t>;
        using T2 = BDST2<int, uint64_t>;
        using T3 = Blocked_SegmentTree_Fast<int, uint64_t>;

        // BbST block/miniblock parameters for B=4096
        const int kExp4096  = 12; // 2^12 = 4096
        const int l2_4096   = std::max(0, kExp4096 - 4); // miniblock exponent
        // BbST block parameter for B=512
        const int kExp512   = 9;  // 2^9  = 512
        const int l2_512    = std::max(0, kExp512 - 4);

        for (int rotate = 0; rotate < 6; ++rotate) {
            switch ((rotate + rotation) % 6) {
                case 0:
                    run_generic(csv, "BDST_1Bit(4096-256)", 4096, n, data_type, timer,
                        [&]() { return std::make_unique<T1>(data, 4096); },
                        [](T1& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 1:
                    run_generic(csv, "BDST2(4096-256)", 4096, n, data_type, timer,
                        [&]() { return std::make_unique<T2>(data, 4096); },
                        [](T2& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 2:
                    run_generic(csv, "B_SegTree(4096-256)", 4096, n, data_type, timer,
                        [&]() { return std::make_unique<T3>(data, 4096); },
                        [](T3& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 3:
                    run_generic(csv, "BbST2(4096-256)", 4096, n, data_type, timer,
                        [&, kExp4096, l2_4096]() {
                            return std::make_unique<BbST>(
                                data.data(), data.size(), kExp4096, l2_4096);
                        },
                        [](BbST& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.rmq(l, r);
                        });
                    break;
                case 4:
                    run_generic(csv, "BbST(512)", 512, n, data_type, timer,
                        [&, kExp512, l2_512]() {
                            return std::make_unique<BbST>(
                                data.data(), data.size(), kExp512, l2_512);
                        },
                        [](BbST& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.rmq(l, r);
                        });
                    break;
                case 5:
                    if (n > 100'000'000ULL) {
                        cout << "    -> Skipping SDSL Sparse Table for n=" << n << endl;
                        break;
                    }
                    run_generic(csv, "SDSL_Sparse_Table", 0, n, data_type, timer,
                        [&]() {
                            return std::make_unique<sdsl::rmq_support_sparse_table<true, true, vector<int>>>(&data);
                        },
                        [](sdsl::rmq_support_sparse_table<true, true, vector<int>>& rmq, uint64_t l, uint64_t r) -> size_t {
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

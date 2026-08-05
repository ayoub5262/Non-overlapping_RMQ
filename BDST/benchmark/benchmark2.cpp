/**
 * @file benchmark2.cpp
 * @brief Compares BDST_1Bit against succinct RMQ structures (SDSL_new_rec and BP_RMQ64).
 */

#include "benchmark_common.hpp"

#include "BDST/include/BDST_1Bit_wordscan.hpp"
#include "rmq/includes/RMQRMM64.h"
#include "sdsl-lite/include/sdsl/rmq_succinct_rec_new.hpp"

int main() {
    ofstream csv("benchmark2.csv");
    write_header(csv);
    Timer timer;

    int rotation { 0 };
    for_each_scenario(array_sizes, [&](uint64_t n, const string& data_type, const vector<int>& data) {
        using T1 = BDST_1Bit<int, uint64_t>;
        // We use the parameters recommended by the authors, block_size = 1024 with 2 recursion levels
        using T2 = sdsl::rmq_succinct_rec_new<true, 0, 1024, 1024>;
        // we have changed the block size in the source code from 128 to 1024
        using T3 = RMQRMM64;

        for (int rotate {0}; rotate < 3; ++rotate) {
            switch ((rotate + rotation) % 3) {
                case 0:
                    run_generic(csv, "BDST", 1024, n, data_type, timer,
                        [&]() { return std::make_unique<T1>(data, 1024); },
                        [](T1& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;

                case 1:
                    run_generic(csv, "SDSL_rec_new", 1024, n, data_type, timer,
                        [&]() {
                            return std::make_unique<T2>(&data);
                        },
                        [](T2& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq(l, r);
                        });
                    break;

                case 2:
                    run_generic(csv, "BP_RMQ64", 1024, n, data_type, timer,
                        [&]() {
                            return std::make_unique<T3>(
                                const_cast<int*>(data.data()), data.size());
                        },
                        [](T3& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.queryRMQ(l, r);
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

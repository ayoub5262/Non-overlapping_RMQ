
#include "benchmark_common.hpp"

#include "BDST/include/BDST_1Bit.hpp"
#include "rmq/includes/RMQRMM64.h"
#include "sdsl-lite/include/sdsl/rmq_succinct_rec_new.hpp"

int main() {
    ofstream csv("benchmark1.csv");
    write_header(csv);
    Timer timer;

    int rotation { 0 };
    for_each_scenario(array_sizes, [&](uint64_t n, const string& data_type, const vector<int>& data) {
        using T1 = BDST_1Bit<int, uint64_t>;

        for (int rotate {0}; rotate < 3; ++rotate) {
            switch ((rotate + rotation) % 3) {
                case 0:
                    run_generic(csv, "BDST_1Bit(512)", 512, n, data_type, timer,
                        [&]() { return std::make_unique<T1>(data, 512); },
                        [](T1& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;

                case 1:
                    run_generic(csv, "SDSL_new_rec", 1024, n, data_type, timer,
                        [&]() {
                            return std::make_unique<
                                sdsl::rmq_succinct_rec_new<true, 0, 1024, 1024>>(&data);
                        },
                        [](sdsl::rmq_succinct_rec_new<true, 0, 1024, 1024>& rmq,
                           uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq(l, r);
                        });
                    break;

                case 2:
                    run_generic(csv, "RMQ64", 0, n, data_type, timer,
                        [&]() {
                            return std::make_unique<RMQRMM64>(
                                const_cast<int*>(data.data()), data.size());
                        },
                        [](RMQRMM64& rmq, uint64_t l, uint64_t r) -> size_t {
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

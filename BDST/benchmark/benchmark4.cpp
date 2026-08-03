/**
 * @file benchmark4.cpp
 * @brief Compares BDST_1Bit different out-of-block implementations .
 */

#include "benchmark_common.hpp"

#include "BDST/include/BDST_1Bit_wordscan.hpp"
#include "BDST/include/BDST_1Bit_select_il.hpp"
#include "BDST/include/BDST_1Bit_select_mcl.hpp"


int main() {
    ofstream csv("benchmark4.csv");
    write_header(csv);
    Timer timer;

    int rotation { 0 };
    for_each_scenario(array_sizes, [&](uint64_t n, const string& data_type, const vector<int>& data) {
        using T1 = BDST_1Bit<int, uint64_t>;
        using T2 = BDST_1Bit_select_il<int, uint64_t>;
        using T3 = BDST_1Bit_select_mcl<int, uint64_t>;
        

        for (int rotate {0}; rotate < 3; ++rotate) {
            switch ((rotate + rotation) % 3) {
                case 0:
                    run_generic(csv, "BDST_wordscan", 1024, n, data_type, timer,
                        [&]() { return std::make_unique<T1>(data, 1024); },
                        [](T1& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 1:
                    run_generic(csv, "BDST_select_il", 1024, n, data_type, timer,
                        [&]() { return std::make_unique<T2>(data, 1024); },
                        [](T2& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
                        });
                    break;
                case 2:
                    run_generic(csv, "BDST_select_mcl", 1024, n, data_type, timer,
                        [&]() { return std::make_unique<T3>(data, 1024); },
                        [](T3& rmq, uint64_t l, uint64_t r) -> size_t {
                            return (size_t)rmq.query(l, r);
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

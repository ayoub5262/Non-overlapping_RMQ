
#include "benchmark_common.hpp"

#include "BDST/include/BDST_1Bit.hpp"
#include "BDST/include/Blocked_SegmentTree.hpp"
#include "FL-RMQ/include/fl_rmq.hpp"
#include <cstdint>
#include <vector>

// Helper dispatches run_generic for a given compile-time Epsilon.
// FL-RMQ query takes data as a first argument, so we capture it in the lambda.
template<size_t Epsilon, size_t Samples>
void run_fl(ofstream& csv, const string& algo_name,
            uint64_t n, const string& data_type,
            const vector<int>& data, Timer& timer)
{
    using FLType = FLRMQ<int, int64_t, int64_t, float, Samples, Epsilon>;
    run_generic(csv, algo_name, static_cast<int>(Epsilon), n, data_type, timer,
        [&]() { return std::make_unique<FLType>(data); },
        [&data](FLType& rmq, uint64_t l, uint64_t r) -> size_t {
            return (size_t)rmq.query(data, l, r);
        });
}


int main() {
    ofstream csv("benchmark2.csv");
    write_header(csv);
    Timer timer;
    std::vector<uint64_t> arr_sizes {100'000, 1'000'000, 10'000'000, 100'000'000};

    int rotation = 0;
    for_each_scenario(arr_sizes, [&](uint64_t n, const string& data_type, const vector<int>& data) {

        using T1 = BDST_1Bit<int, uint64_t>;
        using T2 = Blocked_SegmentTree<int, uint64_t>;

        // The paper uses different parameters based on whether the data is pseudo-monotone or random
        bool is_pseudo = (data_type == "INC" || data_type == "DEC");

        // Epsilon dictates the maximum number of elements FL_RMQ has to linearly scan at the very end
        // Since we use block decomposition for our Data Structures, we treat block size the same as
        // Epsilon since both have to linearly scan the same number of elements as a fallback 
        std::vector<int> epsilons {256, 512, 1024, 2048};

        for (int B : epsilons) {
            for (int rotate = 0; rotate < 3; ++rotate) {
                switch ((rotate + rotation) % 3) {
                    case 0:
                        run_generic(csv, "BDST_1Bit", B, n, data_type, timer,
                            [&]() { return std::make_unique<T1>(data, B); },
                            [](T1& rmq, uint64_t l, uint64_t r) -> size_t {
                                return (size_t)rmq.query(l, r);
                            });
                        break;
                    case 1:
                        run_generic(csv, "B_SegTree", B, n, data_type, timer,
                            [&]() { return std::make_unique<T2>(data, B); },
                            [](T2& rmq, uint64_t l, uint64_t r) -> size_t {
                                return (size_t)rmq.query(l, r);
                            });
                        break;
                    case 2:
                        if (is_pseudo) {
                            // Paper uses Samples = 0 for pseudo-monotone
                            if (B == 256) run_fl<256, 0>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 512) run_fl<512, 0>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 1024) run_fl<1024, 0>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 2048) run_fl<2048, 0>(csv, "FL_RMQ", n, data_type, data, timer);
                        } else {
                            // Paper uses Samples = Epsilon / 2 for other distributions
                            if (B == 256) run_fl<256, 128>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 512) run_fl<512, 256>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 1024) run_fl<1024, 512>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 2048) run_fl<2048, 1024>(csv, "FL_RMQ", n, data_type, data, timer);
                        }
                        break;
                }
            }
        }
        ++rotation;
    });

    csv.close();
    cout << "Benchmarking complete." << endl;
    return 0;

}

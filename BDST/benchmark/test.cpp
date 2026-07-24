#include "benchmark_common.hpp"

#include "BDST/include/BDST_1Bit.hpp"
#include "BDST/include/BDST_1Bit_Fast.hpp"
#include "BDST/include/BDST.hpp"
#include "BDST/include/BDST1.hpp"
#include "BDST/include/BDST2.hpp"
#include "BDST/include/Blocked_SegmentTree.hpp"
#include "BDST/include/Blocked_SegmentTree_Fast.hpp"
#include "BDST/include/ST_IntVec.hpp"

#include "sdsl-lite/include/sdsl/int_vector.hpp"
#include "sdsl-lite/include/sdsl/rmq_support_sparse_table.hpp"

#include <iostream>
#include <vector>
#include <string>

using namespace std;

void run_tests() {
    // Array sizes to test
    vector<uint64_t> test_sizes = {10'000, 100'000};
    
    // Total queries per scenario
    const int NUM_TEST_QUERIES = 5000;
    
    // Block sizes
    const int B = 4096;
    const int B_BDST1 = 512;

    for_each_scenario(test_sizes, [&](uint64_t n, const string& data_type, const vector<int>& data) {
        cout << "\n=======================================\n";
        cout << "Testing Scenario: " << data_type << " (n=" << n << ")" << endl;

        // Shift data to be strictly positive so sdsl::int_vector can handle it correctly
        // SDSL works with unsigned integers.
        long long min_elem = *std::min_element(data.begin(), data.end());
        long long offset = min_elem < 0 ? -min_elem + 1 : 0;

        sdsl::int_vector<> sdsl_data(n);
        for (uint64_t i = 0; i < n; ++i) {
            sdsl_data[i] = static_cast<uint64_t>(data[i] + offset);
        }

        // Build baseline (SDSL RMQ Sparse Table)
        cout << "Building SDSL RMQ Sparse Table baseline..." << flush;
        sdsl::rmq_support_sparse_table<1> ground_truth(&sdsl_data);
        cout << " Done" << endl;

        // Build data structures
        cout << "Building BDST_1Bit_Fast..." << flush;
        BDST_1Bit_Fast<int, uint64_t> bdst_1bit(data, B);
        cout << " Done" << endl;

        cout << "Building BDST1..." << flush;
        BDST1<int, uint64_t> bdst1(data, B_BDST1);
        cout << " Done" << endl;

        cout << "Building BDST2..." << flush;
        BDST2<int, uint64_t> bdst2(data, B);
        cout << " Done" << endl;

        cout << "Building Blocked_SegmentTree_Fast..." << flush;
        Blocked_SegmentTree_Fast<int, uint64_t> b_segtree(data, B);
        cout << " Done" << endl;

        cout << "Building BDST..." << flush;
        BDST<int, uint64_t> bdst_slow(data, B);
        cout << " Done" << endl;

        cout << "Building BDST_1Bit..." << flush;
        BDST_1Bit<int, uint64_t> bdst_1bit_slow(data, B);
        cout << " Done" << endl;

        cout << "Building Blocked_SegmentTree..." << flush;
        Blocked_SegmentTree<int, uint64_t> b_segtree_slow(data, B);
        cout << " Done" << endl;

        cout << "Building ST_IntVec..." << flush;
        ST_IntVec<int, uint64_t> st_intvec(data, B);
        cout << " Done" << endl;

        // Generate Random Queries
        mt19937 rng(1337);
        uniform_int_distribution<uint64_t> dist_len(1, n);

        int errors_1bit = 0;
        int errors_bdst1 = 0;
        int errors_bdst2 = 0;
        int errors_b_segtree = 0;
        int errors_bdst = 0;
        int errors_1bit_slow = 0;
        int errors_b_segtree_slow = 0;
        int errors_st_intvec = 0;

        cout << "Running " << NUM_TEST_QUERIES << " random queries..." << flush;
        for (int i = 0; i < NUM_TEST_QUERIES; ++i) {
            uint64_t len = dist_len(rng);
            uniform_int_distribution<uint64_t> dist_l(0, n - len);
            uint64_t l = dist_l(rng);
            uint64_t r = l + len - 1;

            // Baseline
            uint64_t expected_idx = ground_truth(l, r);
            int expected_val = data[expected_idx];

            // Evaluate
            auto check = [&](auto& rmq, int& error_count, const string& name) {
                uint64_t res_idx = rmq.query(l, r);
                int res_val = data[res_idx];
                
                // Compare by VALUE rather than index, as there could be multiple minimums
                if (res_val != expected_val) {
                    if (error_count < 3) {
                        cout << "\n[" << name << " ERROR] query(" << l << ", " << r << ") -> "
                             << "Expected val " << expected_val << " at idx " << expected_idx
                             << ", but got val " << res_val << " at idx " << res_idx << endl;
                    }
                    error_count++;
                }
            };

            check(bdst_1bit, errors_1bit, "BDST_1Bit_Fast");
            check(bdst1, errors_bdst1, "BDST1");
            check(bdst2, errors_bdst2, "BDST2");
            check(b_segtree, errors_b_segtree, "Blocked_SegmentTree_Fast");
            check(bdst_slow, errors_bdst, "BDST");
            check(bdst_1bit_slow, errors_1bit_slow, "BDST_1Bit");
            check(b_segtree_slow, errors_b_segtree_slow, "Blocked_SegmentTree");
            check(st_intvec, errors_st_intvec, "ST_IntVec");
        }
        cout << " Done\n";

        auto print_result = [](const string& name, int errs) {
            cout << "  " << name << " -> ";
            if (errs == 0) cout << "PASS\n";
            else cout << "FAIL (" << errs << " mismatches)\n";
        };

        print_result("BDST_1Bit_Fast", errors_1bit);
        print_result("BDST1", errors_bdst1);
        print_result("BDST2", errors_bdst2);
        print_result("Blocked_SegmentTree_Fast", errors_b_segtree);
        print_result("BDST", errors_bdst);
        print_result("BDST_1Bit", errors_1bit_slow);
        print_result("Blocked_SegmentTree", errors_b_segtree_slow);
        print_result("ST_IntVec", errors_st_intvec);
    });
}

int main() {
    cout << "Starting RMQ Output Tests...\n";
    run_tests();
    cout << "\nAll test scenarios completed." << endl;
    return 0;
}

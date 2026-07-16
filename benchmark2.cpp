
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>


#include "malloc_count/malloc_count.h"

#include "sdsl-lite/include/sdsl/cst_sct3.hpp"
#include "sdsl-lite/include/sdsl/construct.hpp"

#include "BlockScanStrategies.hpp"
#include "BDST_1Bit.hpp"
#include "Blocked_SegmentTree.hpp"
#include "FL-RMQ/include/fl_rmq.hpp"




using namespace std;

template<class T>
void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

//  Data & Query Generators
vector<int> gen_random_perm(uint64_t n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
    return v;
}

vector<int> gen_lcp_array(uint64_t n) {
    string text(n, 'A');
    mt19937 rng(42);
    uniform_int_distribution<char> dist('A', 'Z');
    for (uint64_t i = 0; i < n; ++i) {
        text[i] = dist(rng);
    }
    
    sdsl::cst_sct3<> cst;
    sdsl::construct_im(cst, text, 1);
    
    vector<int> v(n);
    for (uint64_t i = 0; i < n; ++i) {
        v[i] = cst.lcp[i];
    }
    return v;
}

vector<int> gen_increasing_trend(uint64_t n, int delta = 10000) {
    vector<int> v(n);
    mt19937 rng(42);
    for (uint64_t i = 0; i < n; ++i) {
        uniform_int_distribution<int> dist(-delta, delta);
        v[i] = static_cast<int>(i) + dist(rng);
    }
    return v;
}

vector<int> gen_decreasing_trend(uint64_t n, int delta = 10000) {
    vector<int> v(n);
    mt19937 rng(42);
    for (uint64_t i = 0; i < n; ++i) {
        uniform_int_distribution<int> dist(-delta, delta);
        v[i] = static_cast<int>(n - i) + dist(rng);
    }
    return v;
}

struct QueryPair { uint64_t l, r; };

// Generate k queries all having exactly range_len elements ([l, l+range_len-1]).
// Returns an empty vector if range_len > n.
vector<QueryPair> gen_queries_with_range(uint64_t n, uint64_t range_len, int k) {
    if (range_len > n) return {};
    vector<QueryPair> queries;
    queries.reserve(k);
    mt19937 rng(789);
    uniform_int_distribution<uint64_t> dist(0, n - range_len);
    for (int i = 0; i < k; ++i) {
        uint64_t l = dist(rng);
        queries.push_back({l, l + range_len - 1});
    }
    return queries;
}

//  High-Precision Timer
struct Timer {
    std::chrono::high_resolution_clock::time_point cpu_start;
    void start() { cpu_start = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto cpu_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::nano>(cpu_end - cpu_start).count();
    }
};

// CSV Output 
void write_header(ofstream& csv) {
    csv << "algorithm,block_size,size,data_type,scenario,query_range,cpu_time_ns,bits_per_element,bits_per_element_peak\n";
}

void write_row(ofstream& csv, const string& algo, int block_size,
               uint64_t size, const string& data_type, const string& scenario, uint64_t query_range,
               double cpu_time_ns, double bits, double bits_peak = -1.0)
{
    csv << algo << "," << block_size << "," << size << "," << data_type << ","
        << scenario << "," << query_range << ","
        << fixed << setprecision(3)
        << cpu_time_ns << "," << bits << "," << bits_peak << "\n";
    csv.flush();
}

// Constants 
const vector<uint64_t> array_sizes  = {100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};
const vector<uint64_t> query_ranges = {10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000};
const int NUM_QUERIES = 10'000;
const int NUM_WARMUP  = 500;

template<typename STType>
void run_BDST(ofstream& csv, const string& algo_name, uint64_t n, const string& data_type,
            const vector<int>& data, int B, Timer& timer)
{
    cout << "    -> Building " << algo_name << " with B=" << B << "... " << flush;
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    STType rmq(data, B);
    double build_cpu  = timer.stop();
    size_t mem_after  = malloc_count_current();
    size_t mem_peak   = malloc_count_peak();
    double bits       = (mem_after - mem_before) * 8.0 / n;
    double bits_peak  = (mem_peak  - mem_before) * 8.0 / n;
    cout << "Done" << endl;

    write_row(csv, algo_name, B, n, data_type, "build", 0, build_cpu, bits, bits_peak);

    size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;

        for (int i = 0; i < NUM_WARMUP; ++i)
            sink ^= rmq.query(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);
            
        do_not_optimize(sink);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink ^= rmq.query(queries[i].l, queries[i].r);
            
        do_not_optimize(sink);
        double query_cpu = timer.stop();

        write_row(csv, algo_name, B, n, data_type, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

template<size_t Epsilon>
void run_FL(ofstream& csv, const string& algo_name, uint64_t n, const string& data_type,
            const vector<int>& data, Timer& timer)
{
    using FLType = FLRMQ<int, int64_t, int64_t, float, 0, Epsilon>;
    cout << "    -> Building " << algo_name << " with B=" << Epsilon << "... " << flush;
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    FLType rmq(data);
    double build_cpu  = timer.stop();
    size_t mem_after  = malloc_count_current();
    size_t mem_peak   = malloc_count_peak();
    double bits       = (mem_after - mem_before) * 8.0 / n;
    double bits_peak  = (mem_peak  - mem_before) * 8.0 / n;
    cout << "Done" << endl;

    write_row(csv, algo_name, Epsilon, n, data_type, "build", 0, build_cpu, bits, bits_peak);

    size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;

        for (int i = 0; i < NUM_WARMUP; ++i)
            sink ^= rmq.query(data, queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);
            
        do_not_optimize(sink);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink ^= rmq.query(data, queries[i].l, queries[i].r);
            
        do_not_optimize(sink);
        double query_cpu = timer.stop();

        write_row(csv, algo_name, Epsilon, n, data_type, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}




int main() {
    ofstream csv("benchmark2.csv");
    write_header(csv);
    Timer timer;

    int rotation = 0;
    for (uint64_t n : array_sizes) {
        for (const string& data_type : {"RAND", "LCP", "INC", "DEC"}) {
            if (data_type == "LCP" && n >= 1'000'000'000ULL) {
                cout << "Skipping LCP for n=" << n << endl;
                continue;
            }
            cout << "Generating " << data_type << " data for n=" << n << "..." << flush;
            vector<int> data;
            if (data_type == "RAND") data = gen_random_perm(n);
            else if (data_type == "LCP") data = gen_lcp_array(n);
            else if (data_type == "INC") data = gen_increasing_trend(n);
            else if (data_type == "DEC") data = gen_decreasing_trend(n);
            cout << " Done" << endl;
            
            cout << "Running Scenario: " << data_type << " (n=" << n << ")" << endl;

            using T1 = BDST_1Bit<int, LinearScan, uint64_t>;
            using T2 = Blocked_SegmentTree<int, LinearScan, uint64_t>;

            for (int B : {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
                for (int rotate = 0; rotate < 3; ++rotate) {
                    switch ((rotate + rotation) % 3) {
                        case 0:
                            run_BDST<T1>(csv, "BDST_1Bit", n, data_type, data, B, timer);
                            break;
                        case 1:
                            run_BDST<T2>(csv, "B_SegTree", n, data_type, data, B, timer);
                            break;
                        case 2:
                            if (B == 32) run_FL<32>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 64) run_FL<64>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 128) run_FL<128>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 256) run_FL<256>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 512) run_FL<512>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 1024) run_FL<1024>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 2048) run_FL<2048>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 4096) run_FL<4096>(csv, "FL_RMQ", n, data_type, data, timer);
                            else if (B == 8192) run_FL<8192>(csv, "FL_RMQ", n, data_type, data, timer);
                            break;
                    }
                }
            }

            ++rotation;
        }
    }

    csv.close();
    cout << "Benchmarking complete." << endl;
    return 0;
}

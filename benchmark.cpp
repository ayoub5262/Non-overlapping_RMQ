#include <chrono>
#include <ctime>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include "BDST_1Bit.hpp"
#include "ST_IntVec.hpp"
#include "FL-RMQ/include/fl_rmq.hpp"
#include "BlockScanStrategies.hpp"
#include "malloc_count/malloc_count.h"
#include <emmintrin.h>

using namespace std;

//  Data & Query Generators
vector<int> gen_random_perm(uint64_t n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
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
        queries.push_back({(uint64_t)l, (uint64_t)(l + range_len - 1)});
    }
    return queries;
}

//  Simple Timer
struct Timer {
    clock_t cpu_start;

    void start() {
        cpu_start  = clock();
    }

    double stop() {
        clock_t cpu_end = clock();
        return static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1e9;
    }
};

//  CSV Output
void write_header(ofstream& csv) {
    csv << "algorithm,block_size,size,scenario,query_range,cpu_time_ns,bits_per_element,bits_per_element_peak\n";
}

void write_row(ofstream& csv, const string& algo, int block_size,
               uint64_t size, const string& scenario, uint64_t query_range,
               double cpu_time_ns, double bits, double bits_peak = -1.0)
{
    csv << algo << "," << block_size << "," << size << ","
        << scenario << "," << query_range << ","
        << fixed << setprecision(3)
        << cpu_time_ns << "," << bits << "," << bits_peak  << "\n";
    csv.flush();
}

const vector<uint64_t> array_sizes  = {100'000,1'000'000, 10'000'000, 100'000'000};
const vector<uint64_t> query_ranges = {10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000};
const int NUM_QUERIES = 10'000;
const int NUM_WARMUP  = 500;   // discarded warm-up iterations before each timed loop

template<typename STType>
void run_ST(ofstream& csv, const string& algo_name, uint64_t n, const vector<int>& data, int B, Timer& timer) {
    cout << "    -> Building " << algo_name << " with B=" << B << "... " << flush;
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    STType dst(data, B);
    double build_cpu = timer.stop();
    size_t mem_after = malloc_count_current();
    size_t mem_peak  = malloc_count_peak();
    double bits      = (mem_after - mem_before) * 8.0 / n;
    double bits_peak = (mem_peak  - mem_before) * 8.0 / n;
    
    cout << "Done" << endl;

    // Emit one build row 
    write_row(csv, algo_name, B, n, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;   // range_len > n

        // Warm-up: prime caches, result discarded
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = dst.query(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = dst.query(queries[i].l, queries[i].r);
        double query_cpu = timer.stop();

        write_row(csv, algo_name, B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

template<size_t Epsilon>
void run_FL(ofstream& csv, const string& algo_name, uint64_t n, const vector<int>& data, Timer& timer) {
    cout << "    -> Building " << algo_name << " with Epsilon=" << Epsilon << "... " << flush;
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    FLRMQ<int, int64_t, int64_t, float, 0, Epsilon> fl_rmq(data);
    double build_cpu = timer.stop();
    size_t mem_after = malloc_count_current();
    size_t mem_peak  = malloc_count_peak();
    double bits      = (mem_after - mem_before) * 8.0 / n;
    double bits_peak = (mem_peak  - mem_before) * 8.0 / n;

    cout << "Done" << endl;

    // Emit one build row 
    write_row(csv, algo_name, Epsilon, n, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;   // range_len > n

        // Warm-up: prime caches, result discarded
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = fl_rmq.query(data, queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = fl_rmq.query(data, queries[i].l, queries[i].r);
        double query_cpu = timer.stop();

        write_row(csv, algo_name, Epsilon, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

int main() {
    ofstream csv("benchmark.csv");
    write_header(csv);
    Timer timer;

    int rotation = 0;
    for (uint64_t n : array_sizes) {
        cout << "Generating RAND data for n=" << n << "..." << flush;
        auto data = gen_random_perm(n);
        cout << " Done" << endl;

        auto run_all = [&]() {
            using T1 = BDST_1Bit<int, LinearScan, uint64_t>;
            using T2 = ST_IntVec<int, LinearScan, uint64_t>;
            
            auto run_BDST = [&]() {
                run_ST<T1>(csv, "BDST_1Bit", n, data, 32, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 64, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 128, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 256, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 512, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 1024, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 2048, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 4096, timer);
                run_ST<T1>(csv, "BDST_1Bit", n, data, 8192, timer);
            };
            auto run_STInt = [&]() {
                run_ST<T2>(csv, "ST_IntVec", n, data, 32, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 64, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 128, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 256, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 512, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 1024, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 2048, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 4096, timer);
                run_ST<T2>(csv, "ST_IntVec", n, data, 8192, timer);
            };
            auto run_FLRMQ = [&]() {
                run_FL<32>(csv, "FL_RMQ", n, data, timer);
                run_FL<64>(csv, "FL_RMQ", n, data, timer);
                run_FL<128>(csv, "FL_RMQ", n, data, timer);
                run_FL<256>(csv, "FL_RMQ", n, data, timer);
                run_FL<512>(csv, "FL_RMQ", n, data, timer);
                run_FL<1024>(csv, "FL_RMQ", n, data, timer);
                run_FL<2048>(csv, "FL_RMQ", n, data, timer);
                run_FL<4096>(csv, "FL_RMQ", n, data, timer);
                run_FL<8192>(csv, "FL_RMQ", n, data, timer);
            };

            switch (rotation % 3) {
                case 0: run_BDST(); run_STInt(); run_FLRMQ(); break;
                case 1: run_STInt(); run_FLRMQ(); run_BDST(); break;
                case 2: run_FLRMQ(); run_BDST(); run_STInt(); break;
            }
        };
        run_all();
        ++rotation;
    }

    csv.close();
    cout << "Benchmarking complete." << endl;
    return 0;
}

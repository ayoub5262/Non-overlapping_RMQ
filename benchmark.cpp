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
#include <functional>

#include "BDST_1Bit.hpp"
#include "BDST_1Bit_Fast.hpp"
#include "ST_IntVec.hpp"
#include "BDST.hpp"

#include "FL-RMQ/include/fl_rmq.hpp"
#include <sdsl/rmq_support.hpp>
#include <sdsl/cst_sct3.hpp>
#include <sdsl/construct.hpp>

#define MINI_BLOCKS
#include "BbST/bbst.h"
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

//  Simple Timer
struct Timer {
    clock_t cpu_start;
    void start() { cpu_start = clock(); }
    double stop() {
        return static_cast<double>(clock() - cpu_start) / CLOCKS_PER_SEC * 1e9;
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
const vector<uint64_t> array_sizes  = {100'000, 1'000'000, 10'000'000, 100'000'000};
const vector<uint64_t> query_ranges = {10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000};
const int NUM_QUERIES = 10'000;
const int NUM_WARMUP  = 500;

// Generic Runner 
// build_fn()   – constructs the RMQ structure, returns nothing (captures by ref)
// query_fn(l,r) – answers a single query, returns the result index
template<typename BuildFn, typename QueryFn>
void run_generic(ofstream& csv, const string& algo_name, int block_size,
                 uint64_t n, const string& data_type, Timer& timer,
                 BuildFn build_fn, QueryFn query_fn)
{
    cout << "    -> Building " << algo_name;
    if (block_size > 0) cout << " with B=" << block_size;
    cout << "... " << flush;

    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    build_fn();
    double build_cpu  = timer.stop();
    size_t mem_after  = malloc_count_current();
    size_t mem_peak   = malloc_count_peak();
    double bits       = (mem_after - mem_before) * 8.0 / n;
    double bits_peak  = (mem_peak  - mem_before) * 8.0 / n;

    cout << "Done" << endl;
    write_row(csv, algo_name, block_size, n, data_type, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;   // range_len > n

        // Warm-up: prime caches, result discarded
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = query_fn(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = query_fn(queries[i].l, queries[i].r);
        double query_cpu = timer.stop();

        write_row(csv, algo_name, block_size, n, data_type, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

// Typed ST runner (constructs with (data, B)) 

template<typename STType>
void run_ST(ofstream& csv, const string& algo_name, uint64_t n, const string& data_type,
            const vector<int>& data, int B, Timer& timer)
{
    STType* rmq = nullptr;
    run_generic(csv, algo_name, B, n, data_type, timer,
        [&]() { rmq = new STType(data, B); },
        [&](uint64_t l, uint64_t r) -> size_t { return rmq->query(l, r); }
    );
    delete rmq;
}

// FL-RMQ runner (Epsilon is a compile-time constant) 

template<size_t Epsilon>
void run_FL(ofstream& csv, const string& algo_name, uint64_t n, const string& data_type,
            const vector<int>& data, Timer& timer)
{
    using FLType = FLRMQ<int, int64_t, int64_t, float, 0, Epsilon>;
    FLType* rmq = nullptr;
    run_generic(csv, algo_name, static_cast<int>(Epsilon), n, data_type, timer,
        [&]() { rmq = new FLType(data); },
        [&](uint64_t l, uint64_t r) -> size_t { return rmq->query(data, l, r); }
    );
    delete rmq;
}


int main() {
    ofstream csv("benchmark.csv");
    write_header(csv);
    Timer timer;

    int rotation = 0;
    for (uint64_t n : array_sizes) {
        for (const string& data_type : {"RAND", "LCP", "INC", "DEC"}) {
            cout << "Generating " << data_type << " data for n=" << n << "..." << flush;
            vector<int> data;
            if (data_type == "RAND") data = gen_random_perm(n);
            else if (data_type == "LCP") data = gen_lcp_array(n);
            else if (data_type == "INC") data = gen_increasing_trend(n);
            else if (data_type == "DEC") data = gen_decreasing_trend(n);
            cout << " Done" << endl;
            
            cout << "=== Running Scenario: " << data_type << " (n=" << n << ") ===" << endl;

            using T1 = BDST_1Bit<int, LinearScan, uint64_t>;
            using T2 = ST_IntVec<int, LinearScan, uint64_t>;
            using T3 = BDST<int, LinearScan, uint64_t>;
            using T4 = BDST_1Bit_Fast<int, LinearScan, uint64_t>;

            auto run_BDST = [&]() {
                for (int B : {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
                    run_ST<T1>(csv, "BDST_1Bit", n, data_type, data, B, timer);
            };
            auto run_BDST_Fast = [&]() {
                for (int B : {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
                    run_ST<T4>(csv, "BDST_1Bit_Fast", n, data_type, data, B, timer);
            };
            auto run_STInt = [&]() {
                for (int B : {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
                    run_ST<T2>(csv, "ST_IntVec", n, data_type, data, B, timer);
            };
            auto run_BDST_old = [&]() {
                for (int B : {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
                    run_ST<T3>(csv, "BDST_old", n, data_type, data, B, timer);
            };
            auto run_FLRMQ = [&]() {
                run_FL<32>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<64>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<128>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<256>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<512>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<1024>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<2048>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<4096>(csv, "FL_RMQ", n, data_type, data, timer);
                run_FL<8192>(csv, "FL_RMQ", n, data_type, data, timer);
            };

            // BbST2: two-level BbST (MINI_BLOCKS), k1=4096 k2=256 
            auto run_BbST2 = [&]() {
                BbST* rmq = nullptr;
                run_generic(csv, "BbST2", 4096, n, data_type, timer,
                    [&]() { rmq = new BbST(data.data(), data.size(), 12, 8); },
                    [&](uint64_t l, uint64_t r) -> size_t { return rmq->rmq(l, r); }
                );
                delete rmq;
            };

            // SDSL classical Sparse Table 
            auto run_SDSL_ST = [&]() {
                sdsl::rmq_support_sparse_table<true, true, vector<int>>* rmq = nullptr;
                run_generic(csv, "SDSL_ST", 0, n, data_type, timer,
                    [&]() { rmq = new sdsl::rmq_support_sparse_table<true, true, vector<int>>(&data); },
                    [&](uint64_t l, uint64_t r) -> size_t { return (*rmq)(l, r); }
                );
                delete rmq;
            };

            // Rotation to avoid systematic ordering bias 
            auto runners = vector<function<void()>>{
                run_BDST, run_BDST_Fast, run_STInt, run_FLRMQ, run_BDST_old, run_BbST2, run_SDSL_ST
            };
            rotate(runners.begin(), runners.begin() + (rotation % runners.size()), runners.end());
            for (auto& r : runners) r();

            ++rotation;
        }
    }

    csv.close();
    cout << "Benchmarking complete." << endl;
    return 0;
}

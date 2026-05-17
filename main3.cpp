// ============================================================
//  RMQ Benchmark — Block Size Sweep
//  Outputs results to results_block_sweep.csv.
// ============================================================

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

#include <sdsl/rmq_succinct_rec_new.hpp>
#include "BD_dst.hpp"
#include "malloc_count/malloc_count.h"

using namespace std;

// ============================================================
//  Data & Query Generators
// ============================================================

vector<int> gen_random_perm(int n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
    return v;
}

struct QueryPair { int l, r; };

// Generate k queries all having exactly range_len elements ([l, l+range_len-1]).
// Returns an empty vector if range_len > n.
vector<QueryPair> gen_queries_with_range(long long n, long long range_len, int k) {
    if (range_len > n) return {};
    vector<QueryPair> queries;
    queries.reserve(k);
    mt19937 rng(789);
    uniform_int_distribution<long long> dist(0, n - range_len);
    for (int i = 0; i < k; ++i) {
        long long l = dist(rng);
        queries.push_back({(int)l, (int)(l + range_len - 1)});
    }
    return queries;
}

// ============================================================
//  Simple Timer (wall clock + CPU clock)
// ============================================================

struct Timer {
    using Clock = chrono::high_resolution_clock;
    Clock::time_point wall_start;
    clock_t cpu_start;

    void start() {
        wall_start = Clock::now();
        cpu_start  = clock();
    }

    pair<double, double> stop() {
        clock_t cpu_end = clock();
        auto wall_end = Clock::now();

        double wall_ns = chrono::duration<double, nano>(wall_end - wall_start).count();
        double cpu_ns  = static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1e9;
        return {wall_ns, cpu_ns};
    }
};

// ============================================================
//  CSV & Console Output
// ============================================================

void write_header(ofstream& csv) {
    // Note: block_size is the element-level block size for BD_DST and the
    // BP-sequence super-block size for SDSL — the parameters are not directly
    // comparable, but the same values are swept for both to explore their
    // independent space-time trade-offs.
    csv << "algorithm,block_size,size,scenario,query_range,cpu_time_ns,bits_per_element\n";
}

void write_row(ofstream& csv, const string& algo, int block_size,
               long long size, const string& scenario, long long query_range,
               double cpu_time_ns, double bits)
{
    csv << algo << "," << block_size << "," << size << ","
        << scenario << "," << query_range << ","
        << fixed << setprecision(2)
        << cpu_time_ns << "," << bits << "\n";
    csv.flush();
}

const vector<long long> full         = {10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};
const vector<long long> query_ranges = {10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000};
const int NUM_QUERIES = 10'000;
const int NUM_WARMUP  = 500;   // discarded warm-up iterations before each timed loop

// SDSL requires block size as a template parameter
template<int B>
void run_sdsl_benchmark(ofstream& csv, long long n, const vector<int>& data, Timer& timer) {
    size_t mem_before = malloc_count_current();
    timer.start();
    sdsl::rmq_succinct_rec_new<true, 0, B> rmq(&data);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    double bits = (mem_after - mem_before) * 8.0 / n;

    // Emit one build row (query_range = 0 means N/A)
    write_row(csv, "SDSL_new_rec", B, n, "build", 0, build_cpu, bits);

    volatile size_t sink = 0;
    for (long long range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;   // range_len > n

        // Warm-up: prime caches, result discarded
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = rmq(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = rmq(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "SDSL_new_rec", B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

void run_bd_dst_benchmark(ofstream& csv, int B, long long n, const vector<int>& data, Timer& timer) {
    size_t mem_before = malloc_count_current();
    timer.start();
    BigDataDST<int, uint32_t> dst(data, B);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    double bits = (mem_after - mem_before) * 8.0 / n;

    // Emit one build row (query_range = 0 means N/A)
    write_row(csv, "BD_DST", B, n, "build", 0, build_cpu, bits);

    volatile size_t sink = 0;
    for (long long range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;   // range_len > n

        // Warm-up: prime caches, result discarded
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = dst.query(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = dst.query(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "BD_DST", B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

int main() {
    ofstream csv("result3.csv");
    write_header(csv);
    Timer timer;

    vector<int> block_sizes = {64, 128, 256, 512, 1024, 2048};

    for (long long n : full) {
        auto data = gen_random_perm(n);

        // SDSL runs first so it does not inherit a warm query-cache from BD_DST.
        // 1. Benchmark SDSL for each block size (queries generated per range inside)
        run_sdsl_benchmark<64>(csv, n, data, timer);
        run_sdsl_benchmark<128>(csv, n, data, timer);
        run_sdsl_benchmark<256>(csv, n, data, timer);
        run_sdsl_benchmark<512>(csv, n, data, timer);
        run_sdsl_benchmark<1024>(csv, n, data, timer);
        run_sdsl_benchmark<2048>(csv, n, data, timer);

        // 2. Benchmark BD_DST for each block size (queries generated per range inside)
        for (int B : block_sizes) {
            run_bd_dst_benchmark(csv, B, n, data, timer);
        }

    }

    csv.close();
    return 0;
}

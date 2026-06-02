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
#include <emmintrin.h>

#include <sdsl/rmq_succinct_rec_new.hpp>
#include "BDDST.hpp"
#include "BlockScanStrategies.hpp"
#include "malloc_count/malloc_count.h"

using namespace std;

// ============================================================
//  Cache Flushing
// ============================================================
void flush_data_cache(const vector<int>& data) {
    const char* p = reinterpret_cast<const char*>(data.data());
    size_t size = data.size() * sizeof(int);
    for (size_t i = 0; i < size; i += 64) {
        _mm_clflush(p + i);
    }
}


// ============================================================
//  Data & Query Generators
// ============================================================

vector<int> gen_random_perm(uint64_t n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
    return v;
}

struct QueryPair { uint64_t l, r; };

// Generate k queries all having exactly range_len elements ([l, l+range_len-1]).
// Returns an empty vector if range_len > n.
vector<QueryPair> gen_queries_with_range(uint64_t n, uint64_t range_len, uint64_t k) {
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
//  CSV Output
// ============================================================

void write_header(ofstream& csv) {
    // block_size is the element level block size for BD_DST and the
    // BP-sequence super-block size for SDSL, the parameters are not directly
    // comparable, but the same values are swept for both to explore their
    // independent space-time trade-offs.
    csv << "algorithm,block_size,size,scenario,query_range,cpu_time_ns,bits_per_element\n";
}

void write_row(ofstream& csv, const string& algo, int block_size,
               uint64_t size, const string& scenario, uint64_t query_range,
               double cpu_time_ns, double bits)
{
    csv << algo << "," << block_size << "," << size << ","
        << scenario << "," << query_range << ","
        << fixed << setprecision(2)
        << cpu_time_ns << "," << bits << "\n";
    csv.flush();
}

const vector<uint64_t> full         = {10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};
const vector<uint64_t> query_ranges = {10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000};
const int NUM_QUERIES = 10'000;
const int NUM_WARMUP  = 500;   // discarded warm-up iterations before each timed loop

// SDSL requires block size as a template parameter
template<int B>
void run_sdsl_benchmark(ofstream& csv, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    size_t mem_before = malloc_count_current();
    timer.start();
    sdsl::rmq_succinct_rec_new<true, 0, B> rmq(&data);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    double bits = (mem_after - mem_before) * 8.0 / n;

    // Emit one build row (query_range = 0 means N/A)
    write_row(csv, "SDSL_new_rec", B, n, "build", 0, build_cpu, bits);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;   // range_len > n

        flush_data_cache(data);
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

void run_bd_dst_benchmark(ofstream& csv, int B, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    size_t mem_before = malloc_count_current();
    timer.start();
    BDDST<int, LinearScan, uint64_t> dst(data, B);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    double bits = (mem_after - mem_before) * 8.0 / n;

    // Emit one build row (query_range = 0 means N/A)
    write_row(csv, "BD_DST", B, n, "build", 0, build_cpu, bits);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;   // range_len > n

        flush_data_cache(data);
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

    int rotation = 0;
    for (uint64_t n : full) {
        auto data = gen_random_perm(n);

        // Rotate the strategy order per (n, B) combination to cancel warm-cache bias.
        for (int B : block_sizes) {
            auto run_sdsl = [&]() {
                if (B == 64) run_sdsl_benchmark<64>(csv, n, data, timer);
                else if (B == 128) run_sdsl_benchmark<128>(csv, n, data, timer);
                else if (B == 256) run_sdsl_benchmark<256>(csv, n, data, timer);
                else if (B == 512) run_sdsl_benchmark<512>(csv, n, data, timer);
                else if (B == 1024) run_sdsl_benchmark<1024>(csv, n, data, timer);
                else if (B == 2048) run_sdsl_benchmark<2048>(csv, n, data, timer);
            };

            if (rotation % 2 == 0) {
                run_sdsl();
                run_bd_dst_benchmark(csv, B, n, data, timer);
            } else {
                run_bd_dst_benchmark(csv, B, n, data, timer);
                run_sdsl();
            }
            rotation++;
        }
    }

    csv.close();
    return 0;
}

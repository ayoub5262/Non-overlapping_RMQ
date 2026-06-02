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

#include "BDDST.hpp"
#include "BlockScanStrategies.hpp"
#include "malloc_count/malloc_count.h"
#include <emmintrin.h>

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


template<typename Strategy>
void run_bd_dst_benchmark(ofstream& csv, int B, const string& strategy_name, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    size_t mem_before = malloc_count_current();
    timer.start();
    BDDST<int, Strategy, uint64_t> dst(data, B);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    double bits = (mem_after - mem_before) * 8.0 / n;

    // Emit one build row (query_range = 0 means N/A)
    write_row(csv, "BD_DST_" + strategy_name, B, n, "build", 0, build_cpu, bits);

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

        write_row(csv, "BD_DST_" + strategy_name, B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

int main() {
    ofstream csv("result4.csv");
    write_header(csv);
    Timer timer;

    vector<int> block_sizes = {64, 128, 256, 512, 1024, 2048};

    // Rotate the strategy order per (n, B) combination to cancel warm-cache bias.
    // Each strategy takes turns running first (cold cache) and last (warm cache).
    int rotation = 0;
    for (uint64_t n : full) {
        auto data = gen_random_perm(n);

        for (int B : block_sizes) {
            // Use the rotation index to decide the order of strategies
            auto run_all = [&]() {
                switch (rotation % 4) {
                    case 0:
                        run_bd_dst_benchmark<LinearScan>(csv, B, "LinearScan", n, data, timer);
                        run_bd_dst_benchmark<NaiveLinearScan>(csv, B, "NaiveLinearScan", n, data, timer);
                        run_bd_dst_benchmark<BranchlessScan>(csv, B, "BranchlessScan", n, data, timer);
#if defined(__AVX2__)
                        run_bd_dst_benchmark<AVX2Scan64b>(csv, B, "AVX2Scan", n, data, timer);
#endif
                        break;
                    case 1:
                        run_bd_dst_benchmark<NaiveLinearScan>(csv, B, "NaiveLinearScan", n, data, timer);
                        run_bd_dst_benchmark<BranchlessScan>(csv, B, "BranchlessScan", n, data, timer);
#if defined(__AVX2__)
                        run_bd_dst_benchmark<AVX2Scan64b>(csv, B, "AVX2Scan", n, data, timer);
#endif
                        run_bd_dst_benchmark<LinearScan>(csv, B, "LinearScan", n, data, timer);
                        break;
                    case 2:
                        run_bd_dst_benchmark<BranchlessScan>(csv, B, "BranchlessScan", n, data, timer);
#if defined(__AVX2__)
                        run_bd_dst_benchmark<AVX2Scan64b>(csv, B, "AVX2Scan", n, data, timer);
#endif
                        run_bd_dst_benchmark<LinearScan>(csv, B, "LinearScan", n, data, timer);
                        run_bd_dst_benchmark<NaiveLinearScan>(csv, B, "NaiveLinearScan", n, data, timer);
                        break;
                    case 3:
#if defined(__AVX2__)
                        run_bd_dst_benchmark<AVX2Scan64b>(csv, B, "AVX2Scan", n, data, timer);
#endif
                        run_bd_dst_benchmark<LinearScan>(csv, B, "LinearScan", n, data, timer);
                        run_bd_dst_benchmark<NaiveLinearScan>(csv, B, "NaiveLinearScan", n, data, timer);
                        run_bd_dst_benchmark<BranchlessScan>(csv, B, "BranchlessScan", n, data, timer);
                        break;
                }
            };
            run_all();
            ++rotation;
        }
    }

    csv.close();
    return 0;
}

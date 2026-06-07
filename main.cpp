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

#include "sdsl-lite/include/sdsl/rmq_succinct_rec_new.hpp"
#include "BDDST_1Bit.hpp"
#include "BDDST_CT.hpp"
#include "ST_IntVec.hpp"
#include "BlockScanStrategies.hpp"
#include "rmq/includes/RMQRMM64.h"
#include "malloc_count/malloc_count.h"

using namespace std;

//  Cache Flushing
void flush_data_cache(const vector<int>& data) {
    const char* p = reinterpret_cast<const char*>(data.data());
    size_t size = data.size() * sizeof(int);
    for (size_t i = 0; i < size; i += 64) {
        _mm_clflush(p + i);
    }
}

//  Data & Query Generators
vector<int> gen_random_perm(uint64_t n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
    return v;
}

struct QueryPair { uint64_t l, r; };

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

//  Timer
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
        << fixed << setprecision(2)
        << cpu_time_ns << "," << bits << "," << bits_peak << "\n";
    csv.flush();
}

const vector<uint64_t> full         = {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};
const vector<uint64_t> query_ranges = {10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000};
const int NUM_QUERIES = 10'000;
const int NUM_WARMUP  = 500;

template<int B>
void run_sdsl_benchmark(ofstream& csv, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    sdsl::rmq_succinct_rec_new<true, 0, B> rmq(&data);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    size_t mem_peak  = malloc_count_peak();
    // Subtract 2.0 bits (Cartesian tree) to measure only inter-block metadata
    double bits      = ((mem_after - mem_before) * 8.0 / n) - 2.0;
    double bits_peak = ((mem_peak  - mem_before) * 8.0 / n) - 2.0;

    write_row(csv, "SDSL_new_rec", B, n, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;

        flush_data_cache(data);
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = rmq(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = rmq(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "SDSL_new_rec", B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

void run_bddst_benchmark(ofstream& csv, int B, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    BDDST_1Bit<int, LinearScan, uint64_t> dst(data, B);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    size_t mem_peak  = malloc_count_peak();
    double bits      = (mem_after - mem_before) * 8.0 / n;
    double bits_peak = (mem_peak  - mem_before) * 8.0 / n;

    write_row(csv, "BDDST_1Bit", B, n, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;

        flush_data_cache(data);
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = dst.query(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = dst.query(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "BDDST_1Bit", B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

void run_st_benchmark(ofstream& csv, int B, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    ST_IntVec<int, LinearScan, uint64_t> st(data, B);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    size_t mem_peak  = malloc_count_peak();
    double bits      = (mem_after - mem_before) * 8.0 / n;
    double bits_peak = (mem_peak  - mem_before) * 8.0 / n;

    write_row(csv, "ST_IntVec", B, n, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;

        flush_data_cache(data);
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = st.query(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = st.query(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "ST_IntVec", B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

void run_rmqrmm_benchmark(ofstream& csv, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    RMQRMM64 rmq(const_cast<int*>(data.data()), data.size());
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    size_t mem_peak  = malloc_count_peak();
    // Subtract 2.0 bits (Cartesian tree) to measure only inter-block metadata
    double bits      = ((mem_after - mem_before) * 8.0 / n) - 2.0;
    double bits_peak = ((mem_peak  - mem_before) * 8.0 / n) - 2.0;

    // RMQRMM64 doesn't have a tunable B parameter, we write 0
    write_row(csv, "RMQRMM64", 512, n, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;

        flush_data_cache(data);
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = rmq.queryRMQ(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = rmq.queryRMQ(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "RMQRMM64", 512, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

void run_bddst_ct_benchmark(ofstream& csv, int B, uint64_t n, const vector<int>& data, Timer& timer) {
    flush_data_cache(data);
    malloc_count_reset_peak();
    size_t mem_before = malloc_count_current();
    timer.start();
    BDDST_CT<int, uint64_t> dst(data, B);
    auto [build_real, build_cpu] = timer.stop();
    size_t mem_after = malloc_count_current();
    size_t mem_peak  = malloc_count_peak();
    // Subtract 2.0 bits (Cartesian tree) to measure only inter-block metadata
    double bits      = ((mem_after - mem_before) * 8.0 / n) - 2.0;
    double bits_peak = ((mem_peak  - mem_before) * 8.0 / n) - 2.0;

    write_row(csv, "BDDST_CT", B, n, "build", 0, build_cpu, bits, bits_peak);

    volatile size_t sink = 0;
    for (uint64_t range_len : query_ranges) {
        auto queries = gen_queries_with_range(n, range_len, NUM_QUERIES);
        if (queries.empty()) continue;

        flush_data_cache(data);
        for (int i = 0; i < NUM_WARMUP; ++i)
            sink = dst(queries[i % NUM_QUERIES].l, queries[i % NUM_QUERIES].r);

        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = dst(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "BDDST_CT", B, n, "query", range_len, query_cpu / NUM_QUERIES, bits);
    }
}

int main() {
    ofstream csv("results.csv");
    write_header(csv);
    Timer timer;

    int rotation = 0;
    for (uint64_t n : full) {
        auto data = gen_random_perm(n);

        auto run_all = [&]() {
            int r = rotation % 5;
            for(int i = 0; i < 5; ++i) {
                int algo = (r + i) % 5;
                if(algo == 0) run_sdsl_benchmark<1024>(csv, n, data, timer);
                if(algo == 1) run_bddst_benchmark(csv, 2048, n, data, timer);
                if(algo == 2) run_rmqrmm_benchmark(csv, n, data, timer);
                if(algo == 3) run_st_benchmark(csv, 2048, n, data, timer);
                if(algo == 4) run_bddst_ct_benchmark(csv, 2048, n, data, timer);
            }
        };

        run_all();
        rotation++;
    }

    csv.close();
    return 0;
}

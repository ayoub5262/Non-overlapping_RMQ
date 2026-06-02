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
#include <unistd.h>

#include <sdsl/rmq_succinct_rec_new.hpp>
#include "BDDST.hpp"
#include "rmq/includes/RMQRMM64.h"


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

vector<QueryPair> gen_random_queries(int n, int k = 10000) {
    vector<QueryPair> queries;
    queries.reserve(k);
    mt19937 rng(456);
    uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < k; ++i) {
        int a = dist(rng), b = dist(rng);
        if (a > b) swap(a, b);
        queries.push_back({a, b});
    }
    return queries;
}

// Generates k queries each with exactly query_range = r-l+1
vector<QueryPair> gen_queries_by_range(int n, int range_len, int k = 10000) {
    vector<QueryPair> queries;
    queries.reserve(k);
    mt19937 rng(789);
    int max_start = n - range_len;
    if (max_start <= 0) {
        for (int i = 0; i < k; ++i) queries.push_back({0, n - 1});
        return queries;
    }
    uniform_int_distribution<int> dist(0, max_start);
    for (int i = 0; i < k; ++i) {
        int l = dist(rng);
        queries.push_back({l, l + range_len - 1});
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

    // Returns {wall_nanoseconds, cpu_nanoseconds}
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
    csv << "algorithm,scenario,size,real_time_ns,cpu_time_ns,bits_per_element\n";
}

void write_row(ofstream& csv,
               const string& algo, const string& scenario,
               long long size, double real_ns, double cpu_ns, double bits)
{
    csv << algo << "," << scenario << "," << size << ","
        << fixed << setprecision(2)
        << real_ns << "," << cpu_ns << "," << bits << "\n";
    csv.flush();

    cout << setw(14) << algo
              << " | " << setw(5) << scenario
              << " | n=" << setw(12) << size
              << " | real=" << setw(16) << fixed << setprecision(2) << real_ns << " ns"
              << " | bits=" << setw(8) << bits << "\n";
}

// Extra write_row for the query-range experiment, includes query_range column
void write_row_ranged(ofstream& csv,
                      const string& algo, long long n,
                      int range_len, double real_ns, double cpu_ns)
{
    csv << algo << ",QueryByRange," << n << ","
        << fixed << setprecision(2)
        << real_ns << "," << cpu_ns << "," << range_len << "\n";
    csv.flush();

    cout << setw(14) << algo
              << " | QueryByRange"
              << " | n=" << setw(10) << n
              << " | range=" << setw(10) << range_len
              << " | real=" << setw(12) << fixed << setprecision(2) << real_ns << " ns\n";
}

// ============================================================
//  Benchmark sizes
//  full:  10 to 10^9  
// ============================================================

const vector<long long> full  = {10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};

const int NUM_QUERIES = 10'000;



int main() {
    ofstream csv("results.csv");
    write_header(csv);

    Timer timer;

    // ────────────────────────────────────────────────────────
    //  3. BD_DST  
    // ────────────────────────────────────────────────────────
    cout << "\n=== BD_DST ===\n";
    for (long long n : full) {
        auto data    = gen_random_perm(n);
        auto queries = gen_random_queries(n, NUM_QUERIES);

        size_t mem_before = malloc_count_current();
        timer.start();
        BDDST<int, uint64_t> dst(data, 256);
        auto [build_real, build_cpu] = timer.stop();
        size_t mem_after = malloc_count_current();
        double bits = (mem_after - mem_before) * 8.0 / n;

        write_row(csv, "BD_DST", "Build", n, build_real, build_cpu, bits);

        volatile size_t sink = 0;
        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = dst.query(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "BD_DST", "Query", n,
                  query_real / NUM_QUERIES, query_cpu / NUM_QUERIES, bits);
    }


    // ────────────────────────────────────────────────────────
    //  4. SDSL RecNew  
    // ────────────────────────────────────────────────────────
    cout << "\n=== SDSL RecNew ===\n";
    for (long long n : full) {
        auto data    = gen_random_perm(n);
        auto queries = gen_random_queries(n, NUM_QUERIES);

        size_t mem_before = malloc_count_current();
        timer.start();
        sdsl::rmq_succinct_rec_new<true, 0, 1024> rmq(&data);
        auto [build_real, build_cpu] = timer.stop();
        size_t mem_after = malloc_count_current();
        double bits = (mem_after - mem_before) * 8.0 / n;

        write_row(csv, "SDSL_RecNew", "Build", n, build_real, build_cpu, bits);

        volatile size_t sink = 0;
        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = rmq(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "SDSL_RecNew", "Query", n,
                  query_real / NUM_QUERIES, query_cpu / NUM_QUERIES, bits);
    }

    // ────────────────────────────────────────────────────────
    //  5. RMQRMM64 
    // ────────────────────────────────────────────────────────
    cout << "\n=== RMQRMM64 ===\n";
    for (long long n : full) {
        auto data    = gen_random_perm(n);
        auto queries = gen_random_queries(n, NUM_QUERIES);

        size_t mem_before = malloc_count_current();
        timer.start();
        RMQRMM64 rmq(data.data(), data.size());
        auto [build_real, build_cpu] = timer.stop();
        size_t mem_after = malloc_count_current();
        double bits = (mem_after - mem_before) * 8.0 / n;

        write_row(csv, "RMQRMM64", "Build", n, build_real, build_cpu, bits);

        volatile size_t sink = 0;
        timer.start();
        for (int i = 0; i < NUM_QUERIES; ++i)
            sink = rmq.queryRMQ(queries[i].l, queries[i].r);
        auto [query_real, query_cpu] = timer.stop();

        write_row(csv, "RMQRMM64", "Query", n,
                  query_real / NUM_QUERIES, query_cpu / NUM_QUERIES, bits);
    }


    // ────────────────────────────────────────────────────────
    //  6. Query-Range Experiment
    //     Fixed n = 10^7, vary query range length from 10 to 10^7
    //     Shows how query time changes with range length
    //     if short queries stay in L1 cache and long queries cause L3/RAM misses
    //
    //     bits_per_element column is repurposed for query_range here.
    // ────────────────────────────────────────────────────────
    cout << "\n=== Query-Range Experiment (n=10^7) ===\n";

    const long long N_RANGE = 10'000'000;
    const vector<int> range_sizes = {10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000};

    // Build all structures once on the same data
    auto range_data = gen_random_perm(N_RANGE);

    BDDST<int, uint64_t> bd(range_data, 256);
    sdsl::rmq_succinct_rec_new<true, 0, 1024> sdsl_r(&range_data);
    RMQRMM64 rmm(range_data.data(), range_data.size());

    for (int range_len : range_sizes) {
        auto q = gen_queries_by_range(N_RANGE, range_len, NUM_QUERIES);

        // BD_DST
        { volatile size_t s = 0; timer.start();
          for (int i = 0; i < NUM_QUERIES; ++i) s = bd.query(q[i].l, q[i].r);
          auto [r, c] = timer.stop();
          write_row_ranged(csv, "BD_DST", N_RANGE, range_len, r/NUM_QUERIES, c/NUM_QUERIES); }

        // SDSL_RecNew
        { volatile size_t s = 0; timer.start();
          for (int i = 0; i < NUM_QUERIES; ++i) s = sdsl_r(q[i].l, q[i].r);
          auto [r, c] = timer.stop();
          write_row_ranged(csv, "SDSL_RecNew", N_RANGE, range_len, r/NUM_QUERIES, c/NUM_QUERIES); }

        // RMQRMM64
        { volatile size_t s = 0; timer.start();
          for (int i = 0; i < NUM_QUERIES; ++i) s = rmm.queryRMQ(q[i].l, q[i].r);
          auto [r, c] = timer.stop();
          write_row_ranged(csv, "RMQRMM64", N_RANGE, range_len, r/NUM_QUERIES, c/NUM_QUERIES); }

    }

    csv.close();
    return 0;
}

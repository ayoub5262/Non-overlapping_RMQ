/**
 * @file benchmark_common.hpp
 * @brief A helper file for custom RMQ Benchmarking.
 *
 * Implements high-precision CPU timing routines, memory peak profiling integration
 * (`malloc_count`), synthetic data generators (RAND, LCP, INC, DEC), and CSV formatting.
 */

#pragma once

#include <chrono>
#include <ostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <memory>

#include "malloc_count/malloc_count.h"
#include "sdsl-lite/include/sdsl/cst_sct3.hpp"
#include "sdsl-lite/include/sdsl/construct.hpp"

using namespace std;

// dead Code Elimination (from FL-RMQ benchmark)
template<class T>
void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

// data Generators 

inline vector<int> gen_random_perm(uint64_t n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
    return v;
}

inline vector<int> gen_lcp_array(uint64_t n) {
    string text(n, 'A');
    ifstream file("../english.1024MB", ios::binary);
    if(file){
        file.read(&text[0], n);
        // For n = 10^9 the program crush at construct_im because the file contains zero symbols
        replace(text.begin(), text.end(),'\0', 'A');
        cout << " english " << flush;
    } else {
        mt19937 rng(42);
        uniform_int_distribution<char> dist('A', 'Z');
        for (uint64_t i {0}; i < n; ++i)
            text[i] = dist(rng);
    }
    sdsl::cst_sct3<> cst;
    sdsl::construct_im(cst, text, 1);
    vector<int> v(n);
    for (uint64_t i {0}; i < n; ++i)
        v[i] = cst.lcp[i];
    return v;
}

// pseudo monotone increasing
inline vector<int> gen_increasing_trend(uint64_t n, int delta = 10000) {
    vector<int> v(n);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(-delta, delta);
    for (uint64_t i {0}; i < n; ++i) 
        v[i] = static_cast<int>(i) + dist(rng);
    
    return v;
}

// Pseudo monotone decreasing
inline vector<int> gen_decreasing_trend(uint64_t n, int delta = 10000) {
    vector<int> v(n);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(-delta, delta);
    for (uint64_t i { 0 }; i < n; ++i) 
        v[i] = static_cast<int>(n - i) + dist(rng);
    
    return v;
}

// query Generator 

struct QueryPair { uint64_t l, r; };

// Generates k queries each of exactly range_len elements [l, l+range_len-1].
inline vector<QueryPair> gen_queries_with_range(uint64_t n, int seed_offset ,uint64_t range_len, int k) {
    if (range_len >= n) return {};
    vector<QueryPair> queries;
    queries.reserve(k);
    mt19937 rng(789 + seed_offset);
    uniform_int_distribution<uint64_t> dist(0, n - range_len);
    for (int i {0}; i < k; ++i) {
        uint64_t l { dist(rng) };
        queries.push_back({l, l + range_len - 1});
    }
    return queries;
}

//HighPrecision Timer 
struct Timer {
    std::chrono::high_resolution_clock::time_point cpu_start;
    void start() { cpu_start = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end {std::chrono::high_resolution_clock::now()};
        return std::chrono::duration<double, std::nano>(end - cpu_start).count();
    }
};

// CSV Output 

inline void write_header(ofstream& csv) {
    csv << "algorithm,block_size,size,data_type,scenario,query_range,"
           "cpu_time_ns,bits_per_element,bits_per_element_peak\n";
}

inline void write_row(ofstream& csv, const string& algo, int block_size,
               uint64_t size, const string& data_type, const string& scenario,
               uint64_t query_range, double cpu_time_ns,
               double bits, double bits_peak = -1.0)
{
    csv << algo << "," << block_size << "," << size << ","
        << data_type << "," << scenario << "," << query_range << ","
        << fixed << setprecision(3)
        << cpu_time_ns << "," << bits << "," << bits_peak << "\n";
    csv.flush();
}

// Constants 

const vector<uint64_t> array_sizes  {100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};
const vector<uint64_t> query_ranges {10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000};
const int NUM_QUERIES { 10'000 };
const int NUM_WARMUP  { 500 };


/* Generic Benchmark Runner 

 BuildFn : () -> std::unique_ptr<RMQType>
   unique_ptr avoids copy/move constraints on non-movable types (RMQ64)
   Example:
     [&]() { return std::make_unique<MyRMQ>(data, B); }

 QueryFn : (RMQType&, uint64_t l, uint64_t r) -> size_t
   Example:
     [](MyRMQ& rmq, uint64_t l, uint64_t r) -> size_t { return rmq.query(l,r); } 
*/

template<typename BuildFn, typename QueryFn>
void run_generic(ofstream& csv, const string& algo_name, int block_size,
                 uint64_t n, const string& data_type,
                 Timer& timer,
                 BuildFn&& build_fn,
                 QueryFn&& query_fn)
{
    cout << "    -> Building " << algo_name;
    if (block_size > 0) cout << " with B=" << block_size;
    cout << " ... " << flush;

    // measure space usage and construction time
    malloc_count_reset_peak();
    size_t mem_before {malloc_count_current()};
    timer.start();
    auto rmq_ptr {build_fn()};                   // unique_ptr<RMQType>
    double build_cpu {timer.stop()};
    size_t mem_after { malloc_count_current()};
    size_t mem_peak {malloc_count_peak()};
    double bits {(mem_after - mem_before) * 8.0 / n};
    double bits_peak  {(mem_peak  - mem_before) * 8.0 / n};

    cout << "Done" << endl;
    write_row(csv, algo_name, block_size, n, data_type, "build", 0,
              build_cpu, bits, bits_peak);

    size_t sink { 0 };
    for (uint64_t range_len : query_ranges) {
        if (range_len >= n) continue;
        // measure query time Best of 5 batches
        double best_query_cpu { 1e18 }; 
        for (int iter {0}; iter < 5; ++iter) {
            // for each batch new query pairs to avoid branch predectability
            auto queries {gen_queries_with_range(n, iter ,range_len, NUM_QUERIES)};
            timer.start();
            for (int i {0}; i < NUM_QUERIES; ++i)
                sink ^= query_fn(*rmq_ptr, queries[i].l, queries[i].r);
            double current_run_cpu {timer.stop()};
            if (current_run_cpu < best_query_cpu) {
                best_query_cpu = current_run_cpu;
            }
        }
        do_not_optimize(sink);

        write_row(csv, algo_name, block_size, n, data_type, "query",
                  range_len, best_query_cpu / NUM_QUERIES, bits);
    }
}

/*
Outer Loop Helper 
Drives the (n, data_type) loop and calls cb(n, data_type, data) for each
combination, generating the appropriate input data automatically.
*/
template<typename Callback>
void for_each_scenario(const vector<uint64_t>& sizes, Callback&& cb) {
    for (uint64_t n : sizes) {
        for (const string& data_type : {"RAND", "LCP", "INC", "DEC"}) {
            cout << "Generating " << data_type << " data for n=" << n << "..." << flush;
            vector<int> data;
            if      (data_type == "RAND") data = gen_random_perm(n);
            else if (data_type == "LCP")  data = gen_lcp_array(n);
            else if (data_type == "INC")  data = gen_increasing_trend(n);
            else                          data = gen_decreasing_trend(n);
            cout << " Done" << endl;
            cout << "Running Scenario: " << data_type << " (n=" << n << ")" << endl;
            cb(n, data_type, data);
        }
    }
}

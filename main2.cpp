#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <bit>

// RMQ Data Structures
#include <sdsl/rmq_succinct_rec_new.hpp>
#include "BD_dst.hpp"
#include "rmq/includes/RMQRMM64.h"

// Memory tracker
#include "malloc_count/malloc_count.h"

// ============================================================
//  DATA GENERATOR
// ============================================================

std::vector<int> gen_random_perm(int n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), std::mt19937(42));
    return v;
}

// ============================================================
//  QUERY GENERATOR — uniformly random [l, r] with l <= r
// ============================================================

struct QueryPair { int l, r; };

std::vector<QueryPair> gen_random_queries(int n, int k = 1000) {
    std::vector<QueryPair> queries;
    queries.reserve(k);
    std::mt19937 rng(456);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < k; ++i) {
        int a = dist(rng), b = dist(rng);
        if (a > b) std::swap(a, b);
        queries.push_back({a, b});
    }
    return queries;
}

// ============================================================
//  BENCHMARK SIZES
//  Full:  10 to 10^8 — for compact structures (BigDataDST, SDSL, RMQRMM64)
//  Heavy: 10 to 10^7 — for O(n log n) space structures (DST Nested/Flat)
//         At n=10^8 they need ~11 GB and get OOM-killed.
// ============================================================

static void FullSizes(benchmark::internal::Benchmark* b) {
    for (long long size = 10; size <= 1000'000'000; size *= 10)
        b->Arg(size);
}



// ============================================================
//  BUILD BENCHMARKS
// ============================================================


static void BM_BD_DST_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        size_t mem_before = malloc_count_current();
        BigDataDST<int, uint32_t> bd_dst(data, 256);
        size_t mem_after = malloc_count_current();
        state.counters["bits_per_element"] = ((mem_after - mem_before) * 8.0) / state.range(0);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_BD_DST_Build)->Apply(FullSizes)->Iterations(1)->Complexity(benchmark::oAuto);

static void BM_SDSL_RecNew_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        size_t mem_before = malloc_count_current();
        sdsl::rmq_succinct_rec_new<true, 0, 1024> rmq(&data);
        size_t mem_after = malloc_count_current();
        state.counters["bits_per_element"] = ((mem_after - mem_before) * 8.0) / state.range(0);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_SDSL_RecNew_Build)->Apply(FullSizes)->Iterations(1)->Complexity(benchmark::oAuto);

static void BM_RMQRMM64_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        size_t mem_before = malloc_count_current();
        RMQRMM64 rmq(data.data(), data.size());
        size_t mem_after = malloc_count_current();
        state.counters["bits_per_element"] = ((mem_after - mem_before) * 8.0) / state.range(0);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_RMQRMM64_Build)->Apply(FullSizes)->Iterations(1)->Complexity(benchmark::oAuto);

// ============================================================
//  QUERY BENCHMARKS
//  Structure is built once (setup), queries are pre-generated.
//  qi cycles through the query pool so the branch predictor
//  sees realistic variation without re-generating queries in
//  the hot loop.
// ============================================================


static void BM_BD_DST_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    size_t mem_before = malloc_count_current();
    BigDataDST<int, uint32_t> bd_dst(data, 256);
    size_t mem_after = malloc_count_current();
    state.counters["bits_per_element"] = ((mem_after - mem_before) * 8.0) / state.range(0);

    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(bd_dst.query(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_BD_DST_Query)->Apply(FullSizes)->Complexity(benchmark::o1);

static void BM_SDSL_RecNew_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    size_t mem_before = malloc_count_current();
    sdsl::rmq_succinct_rec_new<true, 0, 1024> rmq(&data);
    size_t mem_after = malloc_count_current();
    state.counters["bits_per_element"] = ((mem_after - mem_before) * 8.0) / state.range(0);

    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(rmq(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_SDSL_RecNew_Query)->Apply(FullSizes)->Complexity(benchmark::o1);

static void BM_RMQRMM64_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    size_t mem_before = malloc_count_current();
    RMQRMM64 rmq(data.data(), data.size());
    size_t mem_after = malloc_count_current();
    state.counters["bits_per_element"] = ((mem_after - mem_before) * 8.0) / state.range(0);

    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(rmq.queryRMQ(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_RMQRMM64_Query)->Apply(FullSizes)->Complexity(benchmark::o1);

BENCHMARK_MAIN();
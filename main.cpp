#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <bit>

// RMQ Data Structures
#include <sdsl/rmq_succinct_rec_new.hpp>
#include "dst_flattend.hpp"
#include "dst.hpp"
#include "BD_dst.hpp"
#include "rmq/includes/RMQRMM64.h"

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
//  BENCHMARK SIZES: 10^4 to 10^8
// ============================================================

static void CustomSizes(benchmark::internal::Benchmark* b) {
    for (long long size = 10'000; size <= 100'000'000; size *= 10)
        b->Arg(size);
}

// ============================================================
//  BUILD BENCHMARKS
//  Note: heavy structures (DST Nested/Flat) allocate O(n log n)
//  memory, so we cap iterations to avoid OOM at large n.
// ============================================================

static void BM_DST_Nested_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        DisjointSparseTable dst(data);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_DST_Nested_Build)->Apply(CustomSizes)->Iterations(1)->Complexity(benchmark::oNLogN);

static void BM_DST_Flat_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        DisjointSparseTableFlat dst(data);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_DST_Flat_Build)->Apply(CustomSizes)->Iterations(1)->Complexity(benchmark::oNLogN);

static void BM_BD_DST_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        BigDataDST<int> bd_dst(data);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_BD_DST_Build)->Apply(CustomSizes)->Iterations(1)->Complexity(benchmark::oAuto);

static void BM_SDSL_RecNew_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        sdsl::rmq_succinct_rec_new<true, 0, 1024> rmq(&data);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_SDSL_RecNew_Build)->Apply(CustomSizes)->Iterations(1)->Complexity(benchmark::oAuto);

static void BM_RMQRMM64_Build(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    for (auto _ : state) {
        RMQRMM64 rmq(data.data(), data.size());
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_RMQRMM64_Build)->Apply(CustomSizes)->Iterations(1)->Complexity(benchmark::oAuto);

// ============================================================
//  QUERY BENCHMARKS
//  Structure is built once (setup), queries are pre-generated.
//  qi cycles through the query pool so the branch predictor
//  sees realistic variation without re-generating queries in
//  the hot loop.
// ============================================================

static void BM_DST_Nested_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    DisjointSparseTable dst(data);
    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(dst.query(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_DST_Nested_Query)->Apply(CustomSizes)->Complexity(benchmark::o1);

static void BM_DST_Flat_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    DisjointSparseTableFlat dst(data);
    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(dst.query(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_DST_Flat_Query)->Apply(CustomSizes)->Complexity(benchmark::o1);

static void BM_BD_DST_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    BigDataDST<int> bd_dst(data);
    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(bd_dst.query(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_BD_DST_Query)->Apply(CustomSizes)->Complexity(benchmark::o1);

static void BM_SDSL_RecNew_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    sdsl::rmq_succinct_rec_new<true, 0, 1024> rmq(&data);
    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(rmq(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_SDSL_RecNew_Query)->Apply(CustomSizes)->Complexity(benchmark::o1);

static void BM_RMQRMM64_Query(benchmark::State& state) {
    auto data = gen_random_perm(state.range(0));
    RMQRMM64 rmq(data.data(), data.size());
    auto queries = gen_random_queries(state.range(0));
    size_t qi = 0;
    for (auto _ : state) {
        const auto& q = queries[qi++ % queries.size()];
        benchmark::DoNotOptimize(rmq.queryRMQ(q.l, q.r));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_RMQRMM64_Query)->Apply(CustomSizes)->Complexity(benchmark::o1);

BENCHMARK_MAIN();
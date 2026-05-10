#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <fstream>
#include "malloc_count/malloc_count.h"

#include <bit>
#include <cstdint>
#include <cstddef>

namespace dst {
    #include "dst.hpp"
}

namespace dst_flat {
    #include "dst_flattend.hpp"
}

#include "BD_dst.hpp"

#include <sdsl/rmq_succinct_rec_new.hpp>
#include "rmq/includes/RMQRMM64.h"

std::vector<int> gen_random_perm(size_t n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), std::mt19937(42));
    return v;
}

// Generate some random queries
struct Query {
    size_t l, r;
};

std::vector<Query> gen_queries(size_t n, size_t num_queries) {
    std::vector<Query> queries(num_queries);
    std::mt19937 gen(1337);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    for (size_t i = 0; i < num_queries; ++i) {
        size_t l = dist(gen);
        size_t r = dist(gen);
        if (l > r) std::swap(l, r);
        queries[i] = {l, r};
    }
    return queries;
}

template<typename Func>
double measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    size_t num_queries = 100000;
    
    std::ofstream csv_file("benchmark_results.csv");
    csv_file << std::fixed << std::setprecision(4);
    csv_file << "N,Structure,Build Time (s),Build Space (MB),Query Time (s),Queries/sec\n";

    for (size_t n = 10000; n <= 100000000; n *= 10) {
        std::vector<int> data = gen_random_perm(n);
        std::vector<Query> queries = gen_queries(n, num_queries);

        std::cout << "Testing N = " << n << "..." << std::endl;

        // 1. dst::DisjointSparseTable
        {
            size_t mem_before = malloc_count_current();
            dst::DisjointSparseTable* rmq_dst = nullptr;
            double build_time = measure_time([&]() {
                rmq_dst = new dst::DisjointSparseTable(data);
            });
            size_t mem_after = malloc_count_current();
            double space_mb = (mem_after - mem_before) / (1024.0 * 1024.0);

            volatile int dummy = 0;
            double query_time = measure_time([&]() {
                for (const auto& q : queries) {
                    dummy += rmq_dst->query(q.l, q.r);
                }
            });
            
            csv_file << n << ",DST(vector<vector>)," << build_time << "," << space_mb << "," << query_time << "," << num_queries / query_time << "\n";
            csv_file.flush();
            delete rmq_dst;
        }

        // 2. dst_flat::DisjointSparseTable
        {
            size_t mem_before = malloc_count_current();
            dst_flat::DisjointSparseTable* rmq_dst_flat = nullptr;
            double build_time = measure_time([&]() {
                rmq_dst_flat = new dst_flat::DisjointSparseTable(data);
            });
            size_t mem_after = malloc_count_current();
            double space_mb = (mem_after - mem_before) / (1024.0 * 1024.0);

            volatile int dummy = 0;
            double query_time = measure_time([&]() {
                for (const auto& q : queries) {
                    dummy += rmq_dst_flat->query(q.l, q.r);
                }
            });
            
            csv_file << n << ",DST(flattened)," << build_time << "," << space_mb << "," << query_time << "," << num_queries / query_time << "\n";
            csv_file.flush();
            delete rmq_dst_flat;
        }

        // 3. BigDataDST
        {
            size_t mem_before = malloc_count_current();
            BigDataDST<int>* rmq_bd_dst = nullptr;
            double build_time = measure_time([&]() {
                rmq_bd_dst = new BigDataDST<int>(data);
            });
            size_t mem_after = malloc_count_current();
            double space_mb = (mem_after - mem_before) / (1024.0 * 1024.0);

            volatile int dummy = 0;
            double query_time = measure_time([&]() {
                for (const auto& q : queries) {
                    dummy += rmq_bd_dst->query(q.l, q.r);
                }
            });
            
            csv_file << n << ",BigDataDST," << build_time << "," << space_mb << "," << query_time << "," << num_queries / query_time << "\n";
            csv_file.flush();
            delete rmq_bd_dst;
        }

        // 4. SDSL
        {
            size_t mem_before = malloc_count_current();
            sdsl::rmq_succinct_rec_new<true, 0, 1024>* rmq_sdsl = nullptr;
            double build_time = measure_time([&]() {
                rmq_sdsl = new sdsl::rmq_succinct_rec_new<true, 0, 1024>(&data);
            });
            size_t mem_after = malloc_count_current();
            double space_mb = (mem_after - mem_before) / (1024.0 * 1024.0);

            volatile size_t dummy = 0;
            double query_time = measure_time([&]() {
                for (const auto& q : queries) {
                    dummy += (*rmq_sdsl)(q.l, q.r);
                }
            });
            
            csv_file << n << ",SDSL (succinct_rec)," << build_time << "," << space_mb << "," << query_time << "," << num_queries / query_time << "\n";
            csv_file.flush();
            delete rmq_sdsl;
        }

        // 5. RMQRMM64
        {
            size_t mem_before = malloc_count_current();
            RMQRMM64* rmq_ferrada = nullptr;
            double build_time = measure_time([&]() {
                rmq_ferrada = new RMQRMM64(data.data(), data.size());
            });
            size_t mem_after = malloc_count_current();
            double space_mb = (mem_after - mem_before) / (1024.0 * 1024.0);

            volatile size_t dummy = 0;
            double query_time = measure_time([&]() {
                for (const auto& q : queries) {
                    dummy += rmq_ferrada->queryRMQ(q.l, q.r);
                }
            });
            
            csv_file << n << ",RMQRMM64 (Ferrada)," << build_time << "," << space_mb << "," << query_time << "," << num_queries / query_time << "\n";
            csv_file.flush();
            delete rmq_ferrada;
        }
    }

    csv_file.close();
    std::cout << "Done! Results saved in benchmark_results.csv\n";
    return 0;
}
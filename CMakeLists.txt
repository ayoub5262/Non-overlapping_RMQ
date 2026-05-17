cmake_minimum_required(VERSION 3.14)
project(RMQ_Benchmark CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Maximum performance flags for meaningful benchmarks
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")

# --- 1. Ferrada RMQRMM64 library (compiled from source) ---
add_library(rmqrmm64 STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/rmq/RMQRMM64.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmq/includes/Basic_rmq.cpp
)
target_include_directories(rmqrmm64 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/rmq/includes
)
target_compile_options(rmqrmm64 PRIVATE -O3 -march=native -DNDEBUG)

# --- 2. malloc_count (compiled from source) ---
add_library(malloc_count_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/malloc_count/malloc_count.c
)
target_include_directories(malloc_count_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/malloc_count
)

# --- 3. Chrono Benchmark Executable ---
add_executable(rmq_bench main.cpp)

target_include_directories(rmq_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}               # BD_dst.hpp
    /home/ayoub/include                       # sdsl headers (system-installed)
    ${CMAKE_CURRENT_SOURCE_DIR}/rmq/includes  # RMQRMM64.h
)
target_link_directories(rmq_bench PRIVATE /home/ayoub/lib)

target_link_libraries(rmq_bench
    PRIVATE
    rmqrmm64
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

# --- 3.5. Chrono Benchmark 3 Executable ---
add_executable(rmq_bench3 main3.cpp)

target_include_directories(rmq_bench3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}               # BD_dst.hpp
    /home/ayoub/include                       # sdsl headers (system-installed)
    ${CMAKE_CURRENT_SOURCE_DIR}/rmq/includes  # RMQRMM64.h
)
target_link_directories(rmq_bench3 PRIVATE /home/ayoub/lib)

target_link_libraries(rmq_bench3
    PRIVATE
    rmqrmm64
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

# --- 4. Google Benchmark (for main2.cpp) ---
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# --- 5. Google Benchmark Executable ---
add_executable(main_bench main2.cpp)

target_include_directories(main_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}               # BD_dst.hpp
    /home/ayoub/include                       # sdsl headers (system-installed)
    ${CMAKE_CURRENT_SOURCE_DIR}/rmq/includes  # RMQRMM64.h
)
target_link_directories(main_bench PRIVATE /home/ayoub/lib)

target_link_libraries(main_bench
    PRIVATE
    rmqrmm64
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
    benchmark::benchmark
)

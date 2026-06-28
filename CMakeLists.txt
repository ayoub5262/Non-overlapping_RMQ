cmake_minimum_required(VERSION 3.14)
project(RMQ_Benchmark CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Maximum performance flags for meaningful benchmarks
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")

# malloc_count (compiled from source) 
add_library(malloc_count_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/malloc_count/malloc_count.c
)
target_include_directories(malloc_count_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/malloc_count
)

#  Benchmark Executable (benchmark.cpp) 
add_executable(run_benchmark benchmark.cpp)

target_include_directories(run_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/ayoub/include
    ${CMAKE_CURRENT_SOURCE_DIR}/FL-RMQ/include
)
target_link_directories(run_benchmark PRIVATE /home/ayoub/lib)

target_link_libraries(run_benchmark
    PRIVATE
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

# ST vs BDDST Executable (ST_vs_BDST.cpp) 
add_executable(st_vs_bddst ST_vs_BDST.cpp)

target_include_directories(st_vs_bddst PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/ayoub/include
)
target_link_directories(st_vs_bddst PRIVATE /home/ayoub/lib)

target_link_libraries(st_vs_bddst
    PRIVATE
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

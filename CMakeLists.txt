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

#  Benchmark Executable (benchmark1.cpp) 
add_executable(run_benchmark1 benchmark1.cpp BbST/bbst.cpp)
target_compile_definitions(run_benchmark1 PRIVATE MINI_BLOCKS)

target_include_directories(run_benchmark1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/ayoub/include
    ${CMAKE_CURRENT_SOURCE_DIR}/FL-RMQ/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sdsl-lite/include
)
target_link_directories(run_benchmark1 PRIVATE /home/ayoub/lib)

target_link_libraries(run_benchmark1
    PRIVATE
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

#  Benchmark Executable (benchmark2.cpp) 
add_executable(run_benchmark2 benchmark2.cpp BbST/bbst.cpp)
target_compile_definitions(run_benchmark2 PRIVATE MINI_BLOCKS)

target_include_directories(run_benchmark2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/ayoub/include
    ${CMAKE_CURRENT_SOURCE_DIR}/FL-RMQ/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sdsl-lite/include
)
target_link_directories(run_benchmark2 PRIVATE /home/ayoub/lib)

target_link_libraries(run_benchmark2
    PRIVATE
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

#  Benchmark Executable (benchmark3.cpp) 
add_executable(run_benchmark3 benchmark3.cpp BbST/bbst.cpp)
target_compile_definitions(run_benchmark3 PRIVATE MINI_BLOCKS)

target_include_directories(run_benchmark3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/ayoub/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sdsl-lite/include
)
target_link_directories(run_benchmark3 PRIVATE /home/ayoub/lib)

target_link_libraries(run_benchmark3
    PRIVATE
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

# ST vs BDDST Executable (ST_vs_BDST.cpp) 
add_executable(st_vs_bdst ST_vs_BDST.cpp)

target_include_directories(st_vs_bdst PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/ayoub/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sdsl-lite/include
)
target_link_directories(st_vs_bdst PRIVATE /home/ayoub/lib)

target_link_libraries(st_vs_bdst
    PRIVATE
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)

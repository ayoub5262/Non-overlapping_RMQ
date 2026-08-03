cmake_minimum_required(VERSION 3.14)
project(RMQ_Benchmark CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(BDST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# malloc_count (compiled from source) 
add_library(malloc_count_lib STATIC
    ${ROOT_DIR}/malloc_count/malloc_count.c
)
target_include_directories(malloc_count_lib PUBLIC
    ${ROOT_DIR}/malloc_count
)

# Test Executable (test.cpp)
add_executable(run_test test.cpp)
target_include_directories(run_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BDST_INCLUDE_DIR}
    ${ROOT_DIR}
    /home/ayoub/include
    ${ROOT_DIR}/sdsl-lite/include
)
target_link_directories(run_test PRIVATE /home/ayoub/lib)
target_link_libraries(run_test
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
    ${BDST_INCLUDE_DIR}
    ${ROOT_DIR}
    /home/ayoub/include
    ${ROOT_DIR}/sdsl-lite/include
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


#  Benchmark Executable (benchmark1.cpp) 
add_executable(run_benchmark1 benchmark1.cpp )
target_include_directories(run_benchmark1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BDST_INCLUDE_DIR}
    ${ROOT_DIR}
    /home/ayoub/include
    ${ROOT_DIR}/sdsl-lite/include
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
add_executable(run_benchmark2 benchmark2.cpp 
    ${ROOT_DIR}/rmq/RMQRMM64.cpp
    ${ROOT_DIR}/rmq/includes/Basic_rmq.cpp
)
target_include_directories(run_benchmark2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BDST_INCLUDE_DIR}
    ${ROOT_DIR}
    /home/ayoub/include
    ${ROOT_DIR}/sdsl-lite/include
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
add_executable(run_benchmark3 benchmark3.cpp)

target_include_directories(run_benchmark3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BDST_INCLUDE_DIR}
    ${ROOT_DIR}
    /home/ayoub/include
    ${ROOT_DIR}/FL-RMQ/include
    ${ROOT_DIR}/sdsl-lite/include
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

#  Benchmark Executable (benchmark4.cpp) 
add_executable(run_benchmark4 benchmark4.cpp ${ROOT_DIR}/BbST/bbst.cpp)
target_compile_definitions(run_benchmark4 PRIVATE MINI_BLOCKS)

target_include_directories(run_benchmark4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BDST_INCLUDE_DIR}
    ${ROOT_DIR}
    /home/ayoub/include
    ${ROOT_DIR}/sdsl-lite/include
    
)
target_link_directories(run_benchmark4 PRIVATE /home/ayoub/lib)

target_link_libraries(run_benchmark4
    PRIVATE
    sdsl
    divsufsort
    divsufsort64
    malloc_count_lib
    dl
)






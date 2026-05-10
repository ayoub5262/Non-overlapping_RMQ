CXX = g++
CXXFLAGS = -O3 -std=c++20 -pedantic -Wall -Wextra -I/home/ayoub/include -Irmq/includes
LDFLAGS = -L/home/ayoub/lib
LIBS = rmq/rmqrmmBP.a -lsdsl -ldivsufsort -ldivsufsort64 -ldl

all: second_bench

second_bench: second_bench.cpp malloc_count/malloc_count.c
	$(CXX) $(CXXFLAGS) second_bench.cpp malloc_count/malloc_count.c $(LDFLAGS) $(LIBS) -o second_bench

clean:
	rm -f second_bench

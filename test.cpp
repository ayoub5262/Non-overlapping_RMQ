#include <iostream>
#include <random>
#include <vector>
#include <numeric>


#include "BDDST.hpp"
#include "BDDST3_1Bit.hpp"
#include "BDDST0.hpp"

using namespace std;


//g++ -std=c++20 -O3 -I./sdsl-lite/include -L./sdsl-lite/build/lib -L./sdsl-lite/lib test.cpp -o 1 -lsdsl -ldivsufsort -ldivsufsort64 && ./1



vector<int> gen_random_perm(uint64_t n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
    return v;
}

int main(){
    vector<int> v = gen_random_perm(10000);
    BDDST b1(v, 512);
    BDDST3_1Bit<> b2(v, 512);
    BDDST0<> b3(v, 512);

    cout << (b1.query(10, 50) == b2.query(10, 50)) << endl;
    cout << (b1.query(10, 50) == b3.query(10, 50)) << endl;

}
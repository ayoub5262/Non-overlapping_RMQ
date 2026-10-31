#include <iostream>
#include "BD_dst1.hpp"
#include "BD_dst.hpp"
#include <random>
#include <numeric>
#include <vector>


using namespace std;

vector<int> gen_random_perm(int n) {
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), mt19937(42));
    return v;
}

int main(){
    vector<int> v = gen_random_perm(1000);
    BDDST<int> b1(v,64);
    BigDataDST<int> b2(v,64);
    if (b1.query(10, 20) == b2.query(10, 20)) 
        cout << "true" << endl;
}
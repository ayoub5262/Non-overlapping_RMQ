#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

class SqrtDecomposition{
    std::vector<int> blocks;
    std::vector<int> data;
    int block_size;
    int n;
public:
    SqrtDecomposition(const std::vector<int> &arr, const int b_s): data(arr), block_size(b_s) {
        n = data.size();
        blocks.resize(std::ceil(static_cast<float>(n)/block_size));

        for (int i{0}; i < n; i += block_size){
            int b_idx = i / block_size;
            blocks[b_idx] = i;

            for(int j{i+1}; (j-i) < block_size && j < n; ++j)
                blocks[b_idx] = data[blocks[b_idx]] > data[j] ? j : blocks[b_idx];
        }
    }

    int query(int l, int r){
        int b_l {l / block_size};
        int b_r {r / block_size};
        int res {l};
        
        // case 1: query is one block 
        if (b_l == b_r){
            for (int i{l}; i <= r; ++i)
                res = data[res] > data [i] ? i : res;
            return res;
        }

        // case 2: query span multiple blocks

        // read the left intra block
        int end_l {(b_l + 1) * block_size};
        for (int i{l}; i < end_l; ++i)
            res = data[res] > data[i] ? i : res;

        // read the right intra block
        int start_r {(b_r) * block_size};
        for (int i{start_r}; i <= r; ++i)
            res = data[res] > data[i] ? i : res;

        // read the inter blocks
        int q_l {b_l + 1};
        int q_r {b_r - 1};

        for (int i{q_l}; i <= q_r; ++i) {
            res = data[res] > data[blocks[i]] ? blocks[i] : res;
        }

        return res;

    }
};
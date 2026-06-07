#include <vector>
#include <bit>
#include <iostream>

using namespace std;

class DST {
  vector<vector<int>> table;
  vector<int> data;
  int n, block_size, blocks, levels;

public:
  DST(vector<int> &arr, int B) : data(arr), block_size(B) {
    n = data.size();
    blocks = (n + block_size - 1) / block_size;
    levels = blocks <= 1 ? 1 : bit_width(static_cast<unsigned>(blocks - 1));
    table.resize(levels, vector<int>(blocks));

    // Calculate the block minima
    for (int i{0}; i < blocks; ++i) {
      int min_idx {i * block_size};
      int end {min(n, (i + 1) * block_size)};
      for (int j {i * block_size + 1}; j < end; ++j) {
        if (data[j] < data[min_idx]) min_idx = j;
      }
      table[0][i] = min_idx;
    }

    // Construct the Disjoint Sparse Table
    for (int h {2}; h <= levels; ++h) {
      int range {1 << h};
      int half {range / 2};
      int h_idx {h - 1};

      // Compute the left and right minima for each range starting from center
      for (int center {half}; center < blocks; center += range) {

        // compute the left minima
        table[h_idx][center - 1] = table[0][center - 1];
        int l_start {center - half};

        for (int i {center - 2}; i >= l_start; --i)
          table[h_idx][i] = data[table[0][i]] < data[table[h_idx][i + 1]] ? table[0][i] : table[h_idx][i + 1];

        // compute the right minima
        table[h_idx][center] = table[0][center];
        int r_end{min(blocks, center + half)};

        for (int i {center + 1}; i < r_end; ++i)
          table[h_idx][i] = data[table[0][i]] < data[table[h_idx][i - 1]] ? table[0][i] : table[h_idx][i - 1];
      }
    }
  }

  int query(int l, int r) {
    if (l < 0) l = 0;
    if (r >= n) r = n - 1;
    if (l > r) return -1;
    
    int l_block { l / block_size};
    int r_block { r / block_size};
    int res { l } ;

    // Case 1: the query is in one block
    if (l_block == r_block){
        for(int i{l}; i <= r; ++i)
            res = data[i] < data[res] ? i : res;
        return res;
    }

    // Case 2: the query span multiple blocks
    // compute the left intra block minima
    int l_end {(l_block + 1) * block_size};
    for(int i{l}; i < l_end; ++i)
        res = data[i] < data [res] ? i : res;

    // compute the right intra block minima
    int r_start {r_block * block_size};
    for(int i{r_start}; i <= r; ++i)
        res = data[i] < data [res] ? i : res;

    // compute the intra blocks minima
    int ql {l_block + 1};
    int qr {r_block - 1};
    if(ql <= qr){
        int bw {bit_width(static_cast<unsigned>(ql^qr))};
        int h_idx {bw == 0 ? 0 : bw - 1};
        if(data[table[h_idx][ql]] < data[res]) res = table[h_idx][ql]; 
        if(data[table[h_idx][qr]] < data[res]) res = table[h_idx][qr]; 
    }

    return res;
  }
};

int main(){
    vector<int> v {1,3,4,5,6,72,1,3,4,5,6,72,0};
    DST table(v, 2);
    cout << table.query(0, 12) << endl;
}
#include <vector>
#include <algorithm>


class LookupTable{
    std::vector<std::vector <int>> table;
    int n;

public:
    LookupTable(const std::vector<int>& arr){
        n = arr.size();
        table.assign(n, std::vector<int>(n));

        for (int i {0}; i < n; ++i){
            table[i][i] = arr[i];

            for (int j {i + 1}; j < n; ++j){
                table[i][j] = std::min(table[i][j-1], arr[j]);
            }
        }
    }

    int query(int l, int r){
        return table[l][r];
    }


};

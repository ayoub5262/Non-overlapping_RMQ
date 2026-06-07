#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>

// ============================================================
//  Linear-Space RMQ — based on:
//  "A Simple Linear-Space Data Structure for Constant-Time
//   Range Minimum Query" — Stephane Durocher (2013)
//
//  Key idea: For a query [i,j], pick level l = floor(log(log(j-i+1))).
//  Blocks at level l have size s(l) = 2^{2^l} = 2, 4, 16, 256, ...
//
//  Precompute for each level l:
//    C [l][i] = index of min of A[i .. end of i's s(l)-block]   (suffix)
//    Cp[l][i] = index of min of A[start of i's s(l)-block .. i]  (prefix)
//    D [l]    = sparse table over s(l)-block representatives
//               D[l][k][b] = index of min in blocks b..b+2^k-1
//
//  Query(i,j):
//    If i and j are in the same s(l)-block  → recurse one level down
//    Otherwise                              → min(suffix_i, D_middle, prefix_j)
// ============================================================

class LinearRMQ {
private:
    const std::vector<int>& A;
    int n;
    int L;  // number of levels precomputed

    std::vector<int> log_table;      // log_table[i]     = floor(log2(i)),        i >= 1
    std::vector<int> log_log_table;  // log_log_table[i] = floor(log2(log2(i))),  i >= 2

    // C [l][i] = index of min of A[i .. end of i's s(l)-block]
    // Cp[l][i] = index of min of A[start of i's s(l)-block .. i]
    std::vector<std::vector<int>> C, Cp;

    // D[l][k][b] = index of min over s(l)-blocks b..b+2^k-1
    // (sparse table over block representatives at each level)
    std::vector<std::vector<std::vector<int>>> D;

    // s(x) = 2^{2^x}  → 2, 4, 16, 256, ...
    int s(int x) const {
        return 1 << (1 << x);
    }

    // Query within a single s(l)-block: A[i..j], guaranteed i/s(l) == j/s(l)
    int query_in_block(int l, int i, int j) const {
        if (i == j) return i;

        // Base case: block size 2 — just compare directly
        if (l == 0) {
            return A[i] <= A[j] ? i : j;
        }

        // Subdivide into s(l-1)-sub-blocks
        int sl1 = s(l - 1);
        int bi = i / sl1;
        int bj = j / sl1;

        if (bi == bj) {
            // i and j in the same sub-block — recurse
            return query_in_block(l - 1, i, j);
        }

        // i and j in different sub-blocks
        // suffix of i's sub-block vs prefix of j's sub-block
        int best = (A[C[l-1][i]] <= A[Cp[l-1][j]]) ? C[l-1][i] : Cp[l-1][j];

        // Middle complete sub-blocks [bi+1 .. bj-1] via sparse table D[l-1]
        if (bj > bi + 1) {
            int span = bj - bi - 1;                        // number of middle blocks
            int k    = log_table[span];                    // largest power of 2 fitting in span
            int m1   = D[l-1][k][bi + 1];
            int m2   = D[l-1][k][bj - 1 - (1 << k) + 1];
            int mid  = (A[m1] <= A[m2]) ? m1 : m2;
            if (A[mid] < A[best]) best = mid;
        }

        return best;
    }

    // Build the sparse table (D[l]) over block-representative indices
    void build_sparse_table(int l, int block_size) {
        int nb = (n + block_size - 1) / block_size;  // number of blocks at this level
        if (nb == 0) return;

        int rows = log_table[nb] + 1;
        D[l].assign(rows, std::vector<int>(nb));

        // k=0: representative of each block = C[l][block_start] (= min of whole block)
        for (int b = 0; b < nb; ++b)
            D[l][0][b] = C[l][b * block_size];

        // k>0: standard sparse table doubling
        for (int k = 1; (1 << k) <= nb; ++k)
            for (int b = 0; b + (1 << k) <= nb; ++b) {
                int m1 = D[l][k-1][b];
                int m2 = D[l][k-1][b + (1 << (k-1))];
                D[l][k][b] = (A[m1] <= A[m2]) ? m1 : m2;
            }
    }

public:
    // Build O(n) time, O(n) space
    explicit LinearRMQ(const std::vector<int>& input)
        : A(input), n((int)input.size())
    {
        if (n == 0) return;

        // ── 1. Precompute log tables ──────────────────────────────────────
        log_table.assign(n + 1, 0);
        for (int i = 2; i <= n; ++i)
            log_table[i] = log_table[i >> 1] + 1;

        log_log_table.assign(n + 1, 0);
        for (int i = 2; i <= n; ++i)
            log_log_table[i] = (log_table[i] >= 1) ? log_table[log_table[i]] : 0;

        // ── 2. How many levels do we need? ───────────────────────────────
        //  Stop when one block already covers the whole array: s(L) >= n
        L = 1;
        while (L < 6 && s(L) < n) ++L;

        C.assign(L, std::vector<int>(n, 0));
        Cp.assign(L, std::vector<int>(n, 0));
        D.resize(L);

        // ── 3. Level 0 — block size s(0) = 2 ────────────────────────────
        for (int b = 0; b * 2 < n; ++b) {
            int lo = b * 2;
            int hi = std::min(n - 1, lo + 1);

            // Suffix C[0]: right to left
            C[0][hi] = hi;
            for (int i = hi - 1; i >= lo; --i)
                C[0][i] = (A[i] <= A[C[0][i + 1]]) ? i : C[0][i + 1];

            // Prefix Cp[0]: left to right
            Cp[0][lo] = lo;
            for (int i = lo + 1; i <= hi; ++i)
                Cp[0][i] = (A[i] < A[Cp[0][i - 1]]) ? i : Cp[0][i - 1];
        }
        build_sparse_table(0, s(0));

        // ── 4. Higher levels: build from the level below ─────────────────
        for (int l = 1; l < L; ++l) {
            int sl  = s(l);       // current block size
            int sl1 = s(l - 1);   // sub-block size

            for (int b = 0; b * sl < n; ++b) {
                int lo = b * sl;
                int hi = std::min(n - 1, lo + sl - 1);

        // ── Suffix C[l]: scan sub-blocks right to left
                // Rightmost sub-block inside this s(l)-block
                int sb_lo = (hi / sl1) * sl1;  // start of rightmost sub-block
                for (int i = sb_lo; i <= hi; ++i)
                    C[l][i] = C[l-1][i];

                // Running best = min of all sub-blocks to the right of current one
                // Initially it is the min of the rightmost sub-block
                int right_best = C[l-1][sb_lo];   // C[l-1][sub-block start] = min of whole sub-block

                for (int sb = sb_lo - sl1; sb >= lo; sb -= sl1) {
                    int sb_hi = sb + sl1 - 1;
                    // right_best already holds min of everything right of 'sb'
                    for (int i = sb; i <= sb_hi; ++i) {
                        int suf = C[l-1][i];   // min from i to end of i's sub-block
                        C[l][i] = (A[suf] <= A[right_best]) ? suf : right_best;
                    }
                    // Update right_best: include the min of sub-block 'sb'
                    int sb_min = C[l-1][sb];   // min of the whole sub-block sb
                    if (A[sb_min] < A[right_best]) right_best = sb_min;
                }

                // ── Prefix Cp[l]: scan sub-blocks left to right
                // Leftmost sub-block: directly from Cp[l-1]
                int sb_hi_first = std::min(hi, lo + sl1 - 1);
                for (int i = lo; i <= sb_hi_first; ++i)
                    Cp[l][i] = Cp[l-1][i];

                int left_best = Cp[l-1][sb_hi_first];  // min of leftmost sub-block

                for (int sb = lo + sl1; sb <= hi; sb += sl1) {
                    int sb_end = std::min(hi, sb + sl1 - 1);
                    for (int i = sb; i <= sb_end; ++i) {
                        int pre = Cp[l-1][i];
                        Cp[l][i] = (A[pre] <= A[left_best]) ? pre : left_best;
                    }
                    // Extend left_best to include this sub-block
                    int sb_min = Cp[l-1][sb_end];
                    if (A[sb_min] < A[left_best]) left_best = sb_min;
                }
            }

            build_sparse_table(l, sl);
        }
    }

    // O(1) Query — returns the INDEX of the minimum in A[i..j]
    int query(int i, int j) const {
        if (i == j) return i;
        if (i > j)  std::swap(i, j);

        // Pick the right level: s(l) ≤ (j-i+1) < s(l+1)
        int len = j - i;   // length - 1; we use j-i as in the skeleton
        int l   = (len < 2) ? 0 : std::min(log_log_table[len], L - 1);

        int sl = s(l);
        int bi = i / sl;
        int bj = j / sl;

        // ── Same block: recurse inside ────────────────────────────────────
        if (bi == bj) {
            return query_in_block(l, i, j);
        }

        // ── Different blocks ──────────────────────────────────────────────
        // Suffix of i's block + prefix of j's block
        int best = (A[C[l][i]] <= A[Cp[l][j]]) ? C[l][i] : Cp[l][j];

        // Middle complete blocks [bi+1 .. bj-1] via sparse table
        if (bj > bi + 1) {
            int span = bj - bi - 1;
            int k    = log_table[span];
            int m1   = D[l][k][bi + 1];
            int m2   = D[l][k][bj - 1 - (1 << k) + 1];
            int mid  = (A[m1] <= A[m2]) ? m1 : m2;
            if (A[mid] < A[best]) best = mid;
        }

        return best;
    }

    // Convenience: returns the minimum VALUE in A[i..j]
    int query_val(int i, int j) const {
        return A[query(i, j)];
    }
};
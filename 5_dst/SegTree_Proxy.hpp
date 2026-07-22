#pragma once

#include <vector>
#include <bit>
#include <cstdint>

template <typename T, typename Tidx, typename Proxy>
class SegTree_Proxy {
    const Proxy& proxy;
    Tidx n;
    Tidx n_pow2;
    std::vector<Tidx> seg_tree;

public:
    // We accept block_size to match BDST_1Bit_Proxy signature, but ignore it
    // because Segment Tree doesn't group into blocks, it just builds over all elements.
    SegTree_Proxy(const Proxy& proxy, Tidx block_size) : proxy(proxy) {
        n = proxy.size();
        n_pow2 = std::bit_ceil(static_cast<unsigned>(n));
        if (n_pow2 > 0) {
            seg_tree.assign(2 * n_pow2, n);
            for (Tidx i = 0; i < n; ++i) {
                seg_tree[n_pow2 + i] = i;
            }
            for (Tidx i = n_pow2 - 1; i > 0; --i) {
                Tidx l = seg_tree[2 * i];
                Tidx r = seg_tree[2 * i + 1];
                if (l == n) seg_tree[i] = r;
                else if (r == n) seg_tree[i] = l;
                else seg_tree[i] = (proxy[r] <= proxy[l]) ? r : l; // rightmost tie-break
            }
        }
    }

    Tidx query(Tidx l, Tidx r) const {
        if (l == r) return l;
        Tidx l_idx = l + n_pow2;
        Tidx r_idx = r + n_pow2;
        Tidx best = n;

        while (l_idx <= r_idx) {
            if (l_idx & 1) {
                Tidx b = seg_tree[l_idx++];
                if (best == n) best = b;
                else if (b != n && proxy[b] <= proxy[best]) best = b;
            }
            if (!(r_idx & 1)) {
                Tidx b = seg_tree[r_idx--];
                if (best == n) best = b;
                else if (b != n && proxy[b] <= proxy[best]) best = b;
            }
            l_idx >>= 1;
            r_idx >>= 1;
        }
        return best;
    }

    size_t table_bits() const {
        return seg_tree.size() * sizeof(Tidx) * 8;
    }
};

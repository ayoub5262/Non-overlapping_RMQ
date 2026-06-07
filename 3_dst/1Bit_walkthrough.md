# 1-Bit Running Minimum Encoding in Disjoint Sparse Tables

The `BDDST3_1Bit` architecture achieves the theoretical lower bounds of succinct data structures by compressing the inter-block Disjoint Sparse Table index down to strictly **$\approx 1.5$ bits per element per level**, while strictly maintaining $\mathcal{O}(1)$ query times. 

Here is a step-by-step breakdown of how the 1-Bit Encoding and Rank/Select dictionaries accomplish this.

---

## 1. The Storage Strategy: 1:1 Natural Mapping
In a standard Disjoint Sparse Table, we typically store the absolute block index (which takes $\log M$ bits) or a relative intra-branch offset (which takes $\log(dist)$ bits).

In `BDDST3_1Bit`, we completely abandon multi-bit offsets. 
At any level $h$, the array is perfectly partitioned by centers into left and right branches. Every block belongs to exactly one branch. Therefore, we allocate an `sdsl::bit_vector` of exactly `blocks` size. 

**The mapping is 1:1.** Block $i$ maps directly to bit index $i$. 
This completely eliminates complex offset math (like `range_base` or `actual_dist`).

---

## 2. The Encoding Process (Building the Tree)
A Disjoint Sparse Table computes a **Running Prefix Minimum** outwards from the center. Because it is a linear prefix sequence, we can encode it entirely as a sequence of boolean events: "Did the minimum change?"

As we scan outwards from the center:
1. **New Minimum (`1`)**: If the current block is strictly smaller than the running minimum, it becomes the new minimum. We record a `1` at its bit index.
   - *Note: The very first block adjacent to the center is ALWAYS considered a new minimum, so it acts as an uncrossable `1` barrier.*
2. **Larger or Tie (`0`)**: If the current block is $\ge$ the running minimum, the minimum does not change. We record a `0`.

**Example (Left Branch going from $C-1$ down to $0$):**
| Block Index | 6 (C-1) | 5 | 4 | 3 | 2 | 1 | 0 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Block Value | 5 | 8 | 1 | 4 | 1 | 6 | 0 |
| Running Min | 5 | 5 | 1 | 1 | 1 | 1 | 0 |
| **Bit Vector** | **`1`** | **`0`** | **`1`** | **`0`** | **`0`** | **`0`** | **`1`** |

---

## 3. The Decoding Process (Rank and Select)
If we only store $1$s and $0$s, how do we instantly know the absolute block index of the minimum without sequentially scanning the bits? We use **Rank and Select Dictionaries**.

- `rank(i)`: Returns the total number of `1`s strictly before index `i`.
- `select(k)`: Returns the absolute index of the $k$-th `1`.

These operations are performed in strict $\mathcal{O}(1)$ mathematical time by querying pre-computed succinct metadata headers in the `sdsl-lite` library (using hardware instructions like `popcnt`).

### Querying the Left Branch
Suppose we query the left branch `[L, center - 1]`. The minimum for this range is the **last new minimum encountered before we reached $L$** from the center. 
Because the left branch is built right-to-left, we are looking for the **FIRST `1` at or to the right of $L$**.

**The $\mathcal{O}(1)$ Math:**
```cpp
// 1. Count how many 1s exist strictly before L
Tidx num_ones_before = level_rank[level](L);

// 2. Add 1 to target the VERY FIRST '1' that exists inside our range
Tidx target_rank = num_ones_before + 1;

// 3. Instantly fetch its absolute block index!
Tidx actual_block_l = level_select[level](target_rank);
```

### Querying the Right Branch
Suppose we query the right branch `[center, R]`. The minimum for this range is the **last new minimum encountered before we reached $R$** from the center.
Because the right branch is built left-to-right, we are looking for the **LAST `1` at or to the left of $R$**.

**The $\mathcal{O}(1)$ Math:**
```cpp
// 1. Count how many 1s exist up to and including R
// (Rank is exclusive, so we query R + 1)
Tidx total_ones_in_range = level_rank[level](R + 1);

// 2. The rank itself IS the target! We want the last one!
Tidx target_rank = total_ones_in_range;

// 3. Instantly fetch its absolute block index!
Tidx actual_block_r = level_select[level](target_rank);
```

---

## 4. Theoretical Guarantees

### Barrier Safety
Because we hardcode a `1` at `center - 1` (for the left branch) and `center` (for the right branch), it acts as a perfect structural barrier. `select` is mathematically guaranteed to find a `1` before it crosses the center into the other branch. There is zero risk of cross-contamination.

### Space Complexity
* **Bit Vector:** Exactly $1.0$ bits per element per level.
* **Rank/Select Metadata Overhead:** $\approx 0.5$ bits per element per level.
* **Total Space:** $\approx 1.5$ bits per element per level.

By abandoning standard pointers and embracing the linear prefix topology of the Disjoint Sparse Table, `BDDST3_1Bit` achieves extreme theoretical succinctness while remaining branchless and strictly $\mathcal{O}(1)$.

# BDDST0: Deep Dive Architecture

BDDST0 is an ultra-optimized, mathematically pure implementation of a **Disjoint Sparse Table (DST)** that uses **Position-Dependent Bit-Widths** to achieve extreme memory compression without sacrificing query speed.

This document breaks down the implementation step-by-step. We assume the reader is already familiar with the standard Range Minimum Query (RMQ) problem and the classical Disjoint Sparse Table (DST) structure.

---

## 1. The Core Variables

```cpp
sdsl::int_vector<0> base_row;
std::vector<sdsl::int_vector<0>> level_buf;
```

BDDST0 maps a complex tree structure into completely flat 1D bit-arrays:
- **`base_row` (Level 0):** Stores the absolute minimum of each individual block, compressed into exactly `log_block` bits.
- **`level_buf` (Levels 1+):** A 2D structure where each index `level` contains a massive, continuous 1D bit-canvas storing all the trees for that level.

---

## 2. Level 0: The Base Blocks

Before building the tree, we divide the raw array into **Blocks** of size `block_size` (let's call it $B$).

```cpp
base_row = sdsl::int_vector<0>(blocks, 0, log_block);
std::vector<Tidx> base_abs(blocks);

for (Tidx i { 0 }; i < blocks; ++i) {
    Tidx min_idx {i << log_block};
    Tidx end {std::min(n, (i + 1) << log_block)};
    min_idx = QueryStrategy::scan_block(data, min_idx, end - 1);
    base_row[i] = min_idx - (i << log_block);
    base_abs[i] = min_idx;
}
```

- We scan each block to find its absolute minimum element.
- **`base_row[i]`**: Stores the *relative* intra-block offset. Because the block size is $B$, the offset fits perfectly in $\log_2(B)$ bits (`log_block`).
- **`base_abs[i]`**: An uncompressed temporary array storing the absolute minimum index. We use this strictly to speed up the tree-building loops.

---

## 3. Building the Ranges (The Trees)

A traditional DST builds ranges for $h=1, 2, 3...$
- $h=1$ covers 2 blocks
- $h=2$ covers 4 blocks
- $h=3$ covers 8 blocks

### 3.1 Skipping "Level 1" (Size 2 Ranges)
Notice that our loop starts at `level = 1` but calculates `half` as `1 << 1 = 2`.
```cpp
for (Tidx level {1}; level < levels; ++level) {
    const Tidx half {Tidx(1) << level}; // If level = 1, half = 2!
```
This means our first tree loop builds ranges of size $2 \times 2 = 4$ blocks. **We completely skip ranges of size 2.** 
Why? Because a range of size 2 is perfectly split down the middle into 1 block on the left and 1 on the right. The "prefix minimum" of 1 block is just the block itself. If we built this level, it would literally just be an identical memory clone of `base_row`.

### 3.2 The Bit-Canvas Allocation
```cpp
const Tidx half_bits {prefix_bits(half, log_block)};
const Tidx bits_per_range {2 * half_bits};
level_buf[level] = sdsl::int_vector<0>(num_centers * bits_per_range, 0, 1);
```
For a given level, we calculate exactly how many bits a single range requires (`bits_per_range`). Then, we allocate one massive, continuous 1-bit array to hold all the ranges packed back-to-back.

### 3.3 The Distance 0 Duplication Trick
```cpp
// Left side
level_buf[level].set_int(range_base, base_row[center - 1], log_block);
...
// Right side
level_buf[level].set_int(right_base, base_row[center], log_block);
```
Before growing the tree branches outward, the very first element written for the left branch (`center - 1`) and the right branch (`center`) is an exact copy of `base_row`. 
We purposefully duplicate this memory so that later, when a query requests a distance of `0`, the mathematical bit-offset simply resolves to the start of the buffer and cleanly reads `log_block` bits. This eliminates unpredictable `if` branches in the query!

### 3.4 The Mathematics of Centers
```cpp
const Tidx num_centers {(blocks - 1 - half) / (half * 2) + 1};
for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
    const Tidx center {range_idx * (half * 2) + half};
    const Tidx center_data {center << log_block};
```
For every level, the ranges are identically sized. We mathematically calculate exactly how many ranges (`num_centers`) fit into the total blocks. 
The `center` is the block index acting as the root of the range. For example, if `half = 2` (ranges of size 4), the first range spans blocks `[0, 3]`. The center is perfectly calculated as `0 * 4 + 2 = 2`.

### 3.5 The Asymmetric Relative Offsets (The Compression Secret)
When building the left and right branches, the relative offset math is deliberately asymmetric to guarantee the value fits in the shrinking bit-width!

**Left Branch:**
```cpp
level_buf[level].set_int(bit_idx, left_min - (i << log_block), width);
```
The left minimum `left_min` is the absolute index of the minimum in the range `[i, center - 1]`. This index is physically located somewhere between the current block `i` and the center. Therefore, it is guaranteed to be $\ge i \times B$. By subtracting `i * B`, the value becomes a small relative offset bounded strictly by the distance to the center.

**Right Branch:**
```cpp
level_buf[level].set_int(bit_idx, right_min - center_data, width);
```
The right minimum `right_min` is in the range `[center, i]`. This index is guaranteed to be $\ge center \times B$. By subtracting `center * B`, the value again becomes a small relative offset bounded strictly by the distance from the center.

### 3.6 Position-Dependent Bit-Widths (The Magic Shrinking Bits)
```cpp
const uint8_t width {bits_for_dist(dist, log_block)};
```
Because of the clever asymmetric offsets above, the maximum possible value we are storing at a given distance `dist` from the center is strictly bounded by $dist \times B$. 

If the maximum value is $dist \times B$, how many bits do we need to store it?
$\text{Bits} = \log_2(dist \times B) = \log_2(B) + \log_2(dist)$

Therefore, the number of bits required to store the answer dynamically shrinks as we get closer to the center! An element at distance `dist` mathematically only requires exactly `log_block + log2(dist)` bits. As we move away from the center, the bit-widths smoothly grow, perfectly matching the mathematical worst-case bounds. This is what allows BDDST0 to crush standard DST memory overheads.

---

## 4. The Query Phase

When a query requests the range `[l, r]`:

```cpp
const Tidx b_l {l >> log_block};
const Tidx b_r {r >> log_block};
```
We immediately map the raw indices to their absolute boundary blocks.

### 4.1 The Inner Blocks
The tree is only responsible for querying the "fully complete" inner blocks. We isolate them:
```cpp
const Tidx inner_b_l {b_l + 1};
const Tidx inner_b_r {b_r - 1};
```

### 4.2 The Magic XOR (Finding the lowest common tree in O(1))
```cpp
if (inner_b_l <= inner_b_r) {
    const int bw = std::bit_width(static_cast<unsigned>(inner_b_l ^ inner_b_r));
```
How do we know which level of the tree to query?
By simply bitwise XORing the two block indices (`inner_b_l ^ inner_b_r`), the highest differing bit instantly tells us the mathematical height of the tree where both blocks share the exact same `center` (root). This calculates the Lowest Common Ancestor (LCA) in a single CPU instruction!

### 4.3 Adjacent Blocks (The Level 0 Skip)
```cpp
    if (bw <= 1) {
        const Tidx inner_b_l_abs {(inner_b_l << log_block) + base_row[inner_b_l]};
```
If `bw <= 1`, the blocks are either the exact same block or immediately adjacent. We don't even query the tree! We directly read their values from `base_row` (Level 0).

### 4.4 Tree Lookup (O(1) Branchless Math)
```cpp
    } else {
        const Tidx dist_inner_b_l {center - 1 - inner_b_l};
        const uint8_t width_inner_b_l {bits_for_dist(dist_inner_b_l, log_block)};
        const Tidx bit_inner_b_l {range_base + prefix_bits(dist_inner_b_l, log_block)};
        const Tidx inner_b_l_abs {(inner_b_l << log_block) + level_buf[level].get_int(bit_inner_b_l, width_inner_b_l)};
```
If the blocks are far apart, they meet at `center`.
Because of our flat bit-canvas structure, there are no pointers to follow.
1. We calculate the mathematical distance `dist` from the center.
2. We use $O(1)$ prefix-sum math `prefix_bits` to jump directly to the exact bit offset in the 1D canvas.
3. We read `width` bits and shift it back into an absolute index.

**Notice:** There are **zero `if` branches** in this logic! Because we duplicated the $dist = 0$ elements (Section 3.3), even if `dist_inner_b_l == 0`, `prefix_bits` safely returns `0` and reads exactly `log_block` bits. This keeps CPU pipelines completely unblocked.

### 4.5 Boundary Scans (Linear Skip Scan)
```cpp
const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
if (data[res] >= data[bl_min]) {
    // manual scan...
}
```
Finally, we handle the partial boundary blocks `b_l` and `b_r`. 
Before doing a slow linear memory scan of the partial block, we use a **Linear Skip Scan** optimization. We check `base_row` to see if the absolute minimum of the *entire* boundary block is even capable of beating our current `res`. If the absolute best element in that block is worse than what we found in the tree, we skip scanning the block entirely!

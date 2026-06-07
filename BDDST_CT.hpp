#pragma once

#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>

#include "sdsl-lite/include/sdsl/rmq_support.hpp"
#include "sdsl-lite/include/sdsl/int_vector.hpp"
#include "sdsl-lite/include/sdsl/rank_support.hpp"
#include "sdsl-lite/include/sdsl/select_support.hpp"
#include "sdsl-lite/include/sdsl/util.hpp"

template <typename T = int, typename Tidx = std::uint64_t>
class BDDST_CT {
private:
    sdsl::int_vector<0> base_row;
    
    sdsl::bit_vector tree_bits;
    sdsl::rank_support_v5<> tree_rank;
    sdsl::select_support_mcl<1> tree_select;
    
    sdsl::int_vector<> level_ones_before;
    
    const std::vector<T>* data;
    sdsl::rmq_succinct_sct<true> intra_rmq;
    
    Tidx block_size, log_block, blocks, levels;

    void copy(const BDDST_CT& rm) {
        base_row = rm.base_row;
        tree_bits = rm.tree_bits;
        tree_rank = rm.tree_rank;
        tree_rank.set_vector(&tree_bits);
        tree_select = rm.tree_select;
        tree_select.set_vector(&tree_bits);
        level_ones_before = rm.level_ones_before;
        data = rm.data;
        intra_rmq = rm.intra_rmq;
        block_size = rm.block_size;
        log_block = rm.log_block;
        blocks = rm.blocks;
        levels = rm.levels;
    }

public:
    typedef Tidx size_type;
    typedef Tidx value_type;

    BDDST_CT() : data(nullptr) {}

    BDDST_CT(const std::vector<T>& arr, const Tidx B_S) : data(&arr), block_size(B_S), intra_rmq(&arr) {
        log_block = std::bit_width(static_cast<uint64_t>(block_size)) - 1;
        Tidx n {static_cast<Tidx>(data->size())};
        blocks = (n + block_size - 1) / block_size;
        levels = blocks <= 1 ? 1 : std::bit_width(static_cast<uint64_t>(blocks - 1));

        // Level 0 Per block absolute minimums as intra-block offsets 
        base_row = sdsl::int_vector<0>(blocks, 0, log_block);
        for (Tidx i { 0 }; i < blocks; ++i) {
            Tidx min_idx {i << log_block};
            Tidx end {std::min(n, (i + 1) << log_block)};
            min_idx = intra_rmq(min_idx, end - 1);
            base_row[i] = min_idx - (i << log_block);
        }

        if (levels <= 1) return;
        
        level_ones_before = sdsl::int_vector<>(levels, 0);
        
        // Build temporary level bit vectors
        std::vector<sdsl::bit_vector> tmp_levels(levels);
        Tidx total_ones = 0;

        for (Tidx level {1}; level < levels; ++level) {
            const Tidx half {Tidx(1) << level};
            if (blocks <= half) {
                tmp_levels[level] = sdsl::bit_vector(blocks, 0);
                level_ones_before[level] = total_ones;
                continue;
            }
            
            const Tidx num_centers {(blocks - 1 - half) / (half * 2) + 1};
            
            tmp_levels[level] = sdsl::bit_vector(blocks, 0);
            Tidx ones_in_level = 0;

            for (Tidx range_idx {0}; range_idx < num_centers; ++range_idx) {
                const Tidx center {range_idx * (half * 2) + half};

                // Build Left
                Tidx left_min_block {center - 1};
                tmp_levels[level][center - 1] = 1; // Start of running min is always a new minimum
                ones_in_level++;
                
                for (Tidx dist {1}; dist < half && center - 1 >= dist; ++dist) {
                    const Tidx i {center - 1 - dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx left_min_idx {(left_min_block << log_block) + base_row[left_min_block]};
                    
                    if ((*data)[cur_min_idx] < (*data)[left_min_idx]) {
                        left_min_block = i;
                        tmp_levels[level][i] = 1;
                        ones_in_level++;
                    }
                }

                // Build Right
                Tidx right_min_block {center};
                tmp_levels[level][center] = 1; // Start of running min is always a new minimum
                ones_in_level++;
                
                for (Tidx dist {1}; dist < half && center + dist < blocks; ++dist) {
                    const Tidx i {center + dist};
                    
                    const Tidx cur_min_idx {(i << log_block) + base_row[i]};
                    const Tidx right_min_idx {(right_min_block << log_block) + base_row[right_min_block]};
                    
                    if ((*data)[cur_min_idx] < (*data)[right_min_idx]) {
                        right_min_block = i;
                        tmp_levels[level][i] = 1;
                        ones_in_level++;
                    }
                }
            }
            level_ones_before[level] = total_ones;
            total_ones += ones_in_level;
        }

        // Concatenate all levels into monolithic tree_bits
        tree_bits = sdsl::bit_vector(levels * blocks, 0);
        for (Tidx level {1}; level < levels; ++level) {
            for (Tidx i {0}; i < blocks; ++i) {
                tree_bits[level * blocks + i] = tmp_levels[level][i];
            }
        }

        // Initialize O(1) Rank and Select supports
        sdsl::util::init_support(tree_rank, &tree_bits);
        sdsl::util::init_support(tree_select, &tree_bits);
    }

    BDDST_CT(const BDDST_CT& rm) {
        copy(rm);
    }

    BDDST_CT(BDDST_CT&& rm) {
        *this = std::move(rm);
    }

    BDDST_CT& operator=(const BDDST_CT& rm) {
        if (this != &rm) {
            copy(rm);
        }
        return *this;
    }

    BDDST_CT& operator=(BDDST_CT&& rm) {
        if (this != &rm) {
            base_row = std::move(rm.base_row);
            tree_bits = std::move(rm.tree_bits);
            tree_rank = std::move(rm.tree_rank);
            tree_rank.set_vector(&tree_bits);
            tree_select = std::move(rm.tree_select);
            tree_select.set_vector(&tree_bits);
            level_ones_before = std::move(rm.level_ones_before);
            data = rm.data;
            intra_rmq = std::move(rm.intra_rmq);
            block_size = rm.block_size;
            log_block = rm.log_block;
            blocks = rm.blocks;
            levels = rm.levels;
        }
        return *this;
    }

    void swap(BDDST_CT& rm) {
        base_row.swap(rm.base_row);
        tree_bits.swap(rm.tree_bits);
        sdsl::util::swap_support(tree_rank, rm.tree_rank, &tree_bits, &(rm.tree_bits));
        sdsl::util::swap_support(tree_select, rm.tree_select, &tree_bits, &(rm.tree_bits));
        level_ones_before.swap(rm.level_ones_before);
        std::swap(data, rm.data);
        intra_rmq.swap(rm.intra_rmq);
        std::swap(block_size, rm.block_size);
        std::swap(log_block, rm.log_block);
        std::swap(blocks, rm.blocks);
        std::swap(levels, rm.levels);
    }

    void set_vector(const std::vector<T>* v) {
        data = v;
    }

    Tidx operator()(Tidx l, Tidx r) const {
        const Tidx b_l {l >> log_block};
        const Tidx b_r {r >> log_block};
        Tidx res {l};

        if (b_l == b_r) return intra_rmq(l, r);

        const Tidx inner_b_l {b_l + 1};
        const Tidx inner_b_r {b_r - 1};

        if (inner_b_l <= inner_b_r) {
            const int bw = std::bit_width(static_cast<uint64_t>(inner_b_l ^ inner_b_r));

            if (bw <= 1) {
                const Tidx inner_b_l_abs {(inner_b_l << log_block) + base_row[inner_b_l]};
                if ((*data)[inner_b_l_abs] < (*data)[res]) res = inner_b_l_abs;
                
                if (inner_b_l != inner_b_r) {
                    const Tidx inner_b_r_abs {(inner_b_r << log_block) + base_row[inner_b_r]};
                    if ((*data)[inner_b_r_abs] < (*data)[res]) res = inner_b_r_abs;
                }
            } else {
                const Tidx level {static_cast<Tidx>(bw - 1)};

                // Left lookup using Rank/Select in O(1)
                Tidx ones_before = level_ones_before[level];
                Tidx offset = level * blocks;
                Tidx local_rank_l = tree_rank(offset + inner_b_l) - ones_before;
                Tidx actual_block_l = tree_select(ones_before + local_rank_l + 1) - offset;
                
                const Tidx inner_b_l_abs {(actual_block_l << log_block) + base_row[actual_block_l]};

                // Right lookup using Rank/Select in O(1)
                Tidx local_rank_r = tree_rank(offset + inner_b_r + 1) - ones_before;
                Tidx actual_block_r = tree_select(ones_before + local_rank_r) - offset;
                
                const Tidx inner_b_r_abs {(actual_block_r << log_block) + base_row[actual_block_r]};

                if ((*data)[inner_b_l_abs] < (*data)[res]) res = inner_b_l_abs;
                if ((*data)[inner_b_r_abs] < (*data)[res]) res = inner_b_r_abs;
            }
        }

        // Left Boundary Scan
        const Tidx bl_min {(b_l << log_block) + base_row[b_l]};
        if ((*data)[res] >= (*data)[bl_min]) {
            const Tidx end_l {(b_l + 1) << log_block};
            if (l + 1 < end_l) {
                const Tidx left_min {intra_rmq(l + 1, end_l - 1)};
                if ((*data)[res] > (*data)[left_min]) res = left_min;
            }
        }

        // Right Boundary Scan
        const Tidx br_min {(b_r << log_block) + base_row[b_r]};
        if ((*data)[res] >= (*data)[br_min]) {
            const Tidx start_r {b_r << log_block};
            if (start_r <= r) {
                const Tidx right_min {intra_rmq(start_r, r)};
                if ((*data)[res] > (*data)[right_min]) res = right_min;
            }
        }

        return res;
    }
    
    size_type size() const {
        return data ? data->size() : 0;
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v=nullptr, std::string name="") const {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        
        written_bytes += base_row.serialize(out, child, "base_row");
        written_bytes += tree_bits.serialize(out, child, "tree_bits");
        written_bytes += tree_rank.serialize(out, child, "tree_rank");
        written_bytes += tree_select.serialize(out, child, "tree_select");
        written_bytes += level_ones_before.serialize(out, child, "level_ones_before");
        written_bytes += intra_rmq.serialize(out, child, "intra_rmq");
        
        written_bytes += sdsl::write_member(block_size, out, child, "block_size");
        written_bytes += sdsl::write_member(log_block, out, child, "log_block");
        written_bytes += sdsl::write_member(blocks, out, child, "blocks");
        written_bytes += sdsl::write_member(levels, out, child, "levels");
        
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in) {
        base_row.load(in);
        tree_bits.load(in);
        tree_rank.load(in, &tree_bits);
        tree_select.load(in, &tree_bits);
        level_ones_before.load(in);
        intra_rmq.load(in);
        
        sdsl::read_member(block_size, in);
        sdsl::read_member(log_block, in);
        sdsl::read_member(blocks, in);
        sdsl::read_member(levels, in);
    }
    
    size_t table_bits() const {
        size_t total {sdsl::size_in_bytes(base_row) * 8};
        total += sdsl::size_in_bytes(tree_bits) * 8;
        total += sdsl::size_in_bytes(tree_rank) * 8;
        total += sdsl::size_in_bytes(tree_select) * 8;
        total += sdsl::size_in_bytes(level_ones_before) * 8;
        return total;
    }
};

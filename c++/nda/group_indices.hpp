// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions used in nda::group_indices_view.
 */

#pragma once

#include "./layout/idx_map.hpp"
#include "./layout/permutation.hpp"
#include "./stdutil/array.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace nda {

  namespace detail {

    // Check if given integer arrays correspond to a partition of the set `J = {0, 1, ..., R-1}`.
    template <size_t R, size_t... Rs>
    constexpr bool is_partition_of_indices(std::array<int, Rs> const &...grps) {
      // check that the total number of array elements is equal to R
      constexpr auto Rstot = (Rs + ...);
      static_assert(Rstot == R, "Error in nda::detail::is_partition_of_indices: Number of indices has to be equal to the rank");

      // check that every element of J is in one and only one group
      auto count = stdutil::make_initialized_array<R>(0); // FIXME : not necessary in C++20
      auto l     = [&](auto &&grp) mutable -> void {
        for (int u = 0; u < grp.size(); ++u) {
          if (grp[u] < 0) throw std::runtime_error("Error in nda::detail::is_partition_of_indices: Negative indices are not allowed");
          if (grp[u] >= R) throw std::runtime_error("Error in nda::detail::is_partition_of_indices: Indices larger than the rank are not allowed");
          count[grp[u]]++;
        }
      };
      (l(grps), ...);

      return (count == stdutil::make_initialized_array<R>(1));
    }

    // Given an original stride order and a partition of the indices, determine the new stride order of the grouped
    // nda::idx_map.
    template <size_t R, size_t... Rs>
    constexpr std::array<int, sizeof...(Rs)> stride_order_of_grouped_idx_map(std::array<int, R> const &stride_order,
                                                                             std::array<int, Rs> const &...grps) {
      // inverse permutation of the stride order
      auto mem_pos = permutations::inverse(stride_order);

      // check if the indices/dimensions in a group are consecutive in memory and find the slowest varying index within
      // the group
      auto min_mem_pos = [&mem_pos](auto &&grp) {
        int min = R, max = 0;
        for (int idx : grp) {
          int v = mem_pos[idx];
          if (v > max) max = v;
          if (v < min) min = v;
        }
        bool idx_are_consecutive_in_memory = (max - min + 1 == grp.size());
        if (!idx_are_consecutive_in_memory)
          throw std::runtime_error("Error in nda::detail::stride_order_of_grouped_idx_map: Indices are not consecutive in memory");
        return min;
      };

      // number of groups in the partition
      constexpr int Ngrps = sizeof...(Rs);

      // array containing the slowest varying index within each group
      std::array<int, Ngrps> min_mem_positions{min_mem_pos(grps)...};

      // rank each group by its slowest varying index, i.e. the slowest group gets rank 0 and the fastest rank Ngrps-1
      std::array<int, Ngrps> mem_pos_out = stdutil::make_initialized_array<Ngrps>(0); // FIXME : not necessary in C++20
      for (int u = 0; u < Ngrps; ++u) {
        for (int i = 0; i < Ngrps; ++i) {
          if (min_mem_positions[i] < min_mem_positions[u]) ++mem_pos_out[u];
        }
      }

      // the new stride order is the inverse of mem_pos_out
      return permutations::inverse(mem_pos_out);
    }

  } // namespace detail

  /**
   * @addtogroup layout_utils
   * @{
   */

  /**
   * @brief A group of indices.
   * @tparam Is Indices.
   */
  template <int... Is>
  struct idx_group_t {
    /// std::array containing the indices.
    static constexpr std::array<int, sizeof...(Is)> as_std_array{Is...};
  };

  /**
   * @brief Variable template for an nda::idx_group_t.
   * @tparam Is Indices.
   */
  template <int... Is>
  inline idx_group_t<Is...> idx_group = {}; // NOLINT (global variable template is what we want here)

  /**
   * @brief Given an nda::idx_map and a partition of its indices, return a new nda::idx_map with the grouped indices.
   *
   * @details It is used to create an nda::group_indices_view: `group_indices_view(A, idx_group<i,j,...>,
   * idx_group<k,l,...>, ...)`. The resulting index map is of rank `sizeof...(IdxGrps)` with merged indices, one for
   * each group.
   *
   * Preconditions:
   * - The given index groups correspond to a partition of the set `{0, 1, ..., R - 1}` where `R` is the rank.
   * - In each group the indices are consecutive in memory .
   *
   * For example, let's look at the following strider order: `S = (2, 1, 0)`. That means that dimension 0 is the fastest
   * varying index, then dimension 1 and dimension 2 is slowest. We say that dimensions 0 and 1 are consecutive in
   * memory. The same goes for dimensions 2 and 1. However, dimensions 0 and 2 are not consecutive in memory.
   *
   * To check if a group of indices are consecutive in memory, we look at the inverse permutation of the stride order.
   * The stride order `S` is a permutation which maps an integer j to a dimension/index i of the array such that
   * `S[j] = i` is the (j+1)-th slowest varying index, i.e. `S[0]` is the slowest index and `S[R - 1]` is the fastest.
   * The inverse permutation, `M = S^{-1}`, therefore tells us how fast a given index varies, e.g. `M[i] = j` means that
   * index i varies the (j+1)-th slowest.
   *
   * @tparam Rank Rank of the original nda::idx_map.
   * @tparam StaticExtents Static extents of the original nda::idx_map.
   * @tparam StrideOrder Stride order of the original nda::idx_map.
   * @tparam LayoutProp Layout property of the original nda::idx_map.
   * @tparam IdxGrps Groups of indices.
   *
   * @param idxm Original nda::idx_map.
   * @return New nda::idx_map with the grouped indices.
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp, typename... IdxGrps>
  auto group_indices_layout(idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> const &idxm, IdxGrps...) {
    // compile-time checks
    static_assert(LayoutProp == layout_prop_e::contiguous, "Error in nda::group_indices_layout: Only contiguous layouts are supported");
    static_assert(detail::is_partition_of_indices<Rank>(IdxGrps::as_std_array...),
                  "Error in nda::group_indices_layout: Given index groups are not a valid partition");

    // decoded original stride order
    static constexpr auto stride_order = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>::stride_order;

    // number of elements in a group
    auto total_len_of_a_grp = [&idxm](auto &&grp) {
      return std::accumulate(grp.begin(), grp.end(), 1L, [&idxm](long l, long u) { return l * idxm.lengths()[u]; });
    };

    // minimum stride of a group
    auto min_stride_of_a_grp = [&idxm](auto &&grp) {
      return std::accumulate(grp.begin(), grp.end(), idxm.size(), [&idxm](long s, long u) { return std::min(s, idxm.strides()[u]); });
    };

    // new rank, shape, strides and stride order
    static constexpr int new_rank = sizeof...(IdxGrps);
    std::array<long, new_rank> new_extents{total_len_of_a_grp(IdxGrps::as_std_array)...};
    std::array<long, new_rank> new_strides{min_stride_of_a_grp(IdxGrps::as_std_array)...};
    constexpr auto new_stride_order = encode(detail::stride_order_of_grouped_idx_map(stride_order, IdxGrps::as_std_array...));

    return idx_map<new_rank, 0, new_stride_order, layout_prop_e::contiguous>{new_extents, new_strides};
  }

  /** @} */

} // namespace nda

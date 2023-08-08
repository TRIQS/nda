// Copyright (c) 2019-2022 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "basic_array.hpp"
#include <algorithm>

namespace nda {

  namespace details {

    // 2 compile time helpers functioms

    // -- is_partition_of_indices --
    //
    // checks if the groups are a partition of the indices [0,R[
    // where R is the dimension of the array being regrouped
    // it will throw (compiler stop) if index is out of bounds
    template <size_t R, size_t... Rs>
    constexpr bool is_partition_of_indices(std::array<int, Rs> const &...grps) {

      // check we have exactly R indices ...
      // FIXME in C++20, we can have constexpr string, hence have more informative error message.
      auto Rstot = (Rs + ...);
      if (Rstot < R) throw "Too few indices in the group. Should be the rank";
      if (Rstot > R) throw "Too many indices in the group. Should be the rank";

      // we go over each indices, in each group and count them
      std::array<int, R> count          = stdutil::make_initialized_array<R>(0); // FIXME : not necessary in C++20
      std::array<int, R> correct_answer = stdutil::make_initialized_array<R>(1); // FIXME : not necessary in C++20

      auto l = [&](auto &&grp) mutable -> void {
        for (int u = 0; u < grp.size(); ++u) {
          if (grp[u] < 0) throw "Negative index !";
          if (grp[u] >= R) throw "Index larger than the rank !!";
          count[grp[u]]++;
        }
      };
      (l(grps), ...); // execute all lambdas
      return (count == correct_answer);
    }

    //---------------------------

    // --- stride_order_of_grouped_idx_map ---
    //
    // Given a stride_order and some groups, returns the stride_order of the grouped idx_map
    // compile time only
    template <size_t R, size_t... Rs>
    constexpr std::array<int, sizeof...(Rs)> stride_order_of_grouped_idx_map(std::array<int, R> const &stride_order,
                                                                             std::array<int, Rs> const &...grps) {
      // stride_order is permutation which by definition is such that
      // stride_order[0] is the slowest index, stride_order[1] the next, stride_order[Rank-1] the fastest.

      // We need to work with the inverse permutation mem_pos.
      // For each index k, mem_pos[k] gives the position of index k in the memory layout, 0 for slowest to R-1 for fastest.
      // e.g. so by definition   mem_pos[ stride_order[0]] = 0,  mem_pos[ stride_order[1]] = 1, etc...

      // For each group, we look at the mem_pos of its indices.
      // They must be consecutive to be regrouped.
      // We select their min, it will give us the mem_pos of the regrouped view, which we will then invert to get the stride_order.

      auto mem_pos = permutations::inverse(stride_order);

      // Find the minimum memory position of the indices in this group
      // Throw (compile time, i.e. stop compilation) if the indices are not consecutive in memory in this group
      auto min_mem_pos = [&mem_pos](auto &&grp) {
        // m = minimum, M = maximum
        int m = R, M = 0;
        for (int idx : grp) {
          int v = mem_pos[idx];
          if (v > M) M = v;
          if (v < m) m = v;
        }
        bool idx_are_consecutive_in_memory = (M - m + 1 == grp.size());
        if (!idx_are_consecutive_in_memory)
          throw "Indices are not consecutive in memory"; // FIXME : can I have a better error message at compile time ??
        return m;
      };

      // The Number of groups <-> Dimension of the returned idx_map.
      constexpr int Ngrps = sizeof...(Rs);

      // An array containing the minimal memory position for each group
      std::array<int, Ngrps> min_mem_positions{min_mem_pos(grps)...};

      // The problem is that they are not consecutive numbers, they run from 0 to R -1, not Ngrps -1
      // We compress them back to [0, Ngrps[ by counting how many there are before each of them
      std::array<int, Ngrps> mem_pos_out = stdutil::make_initialized_array<Ngrps>(0); // FIXME : not necessary in C++20
      for (int u = 0; u < Ngrps; ++u) {
        for (int i = 0; i < Ngrps; ++i) {
          if (min_mem_positions[i] < min_mem_positions[u]) ++mem_pos_out[u];
        }
      }

      // The new stride_order is the inverse of mem_pos_out
      return permutations::inverse(mem_pos_out);
    }
  } // namespace details

  //---------------------------

  /// A group of indices to be merged together
  template <int... Is>
  struct idx_group_t {
    static constexpr std::array<int, sizeof...(Is)> as_std_array{Is...};
  };

  // idx_group<0,1> etc.. Helper variable template.
  // The user call will be (for the idxmap, then propagated to the array in group_indices_view
  //
  //  group_indices_layout(idxm, idx_group<i, j>, idx_group<k,l>)
  // will return an new regrouped idx_map
  //
  template <int... Is>
  inline idx_group_t<Is...> idx_group = {};

  //---------------------------

  /**
  * \param idxm an idx_map
  * \param IdxGrps some idx_group<i,j>  
  * Usage : group_indices_view(A, idx_group<i,j>, idx_group<k,l>, ...)
  * Precondition :
  *   - the groups indices [ {i,j}, {k,l}, .... ] define a partition of the indices of idxm
  *   - In each group, the indices are consecutive in memory, as indicated by StrideOrder.
  *
  * \return a new idxmap of rank sizeof (IdxGrps...) seeing the same data, with merged indices, one index for each group of indices.
  */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp, typename... IdxGrps>
  auto group_indices_layout(idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> const &idxm, IdxGrps...) {

    static_assert(LayoutProp == layout_prop_e::contiguous, "Not implemented for non contiguous arrays");

    // decoded StrideOrder
    static constexpr auto stride_order = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>::stride_order;

    static_assert(details::is_partition_of_indices<Rank>(IdxGrps::as_std_array...),
                  "The indices provided in the groups are not a partition of the indices of the array, i.e. [0, Rank[");

    /// new lengths and strides.
    auto total_len_of_a_grp = [&idxm](auto &&grp) {
      auto ll = std::accumulate(grp.begin(), grp.end(), 1L, [&idxm](long l, long u) { return l * idxm.lengths()[u]; });
      return ll;
    };

    auto min_stride_of_a_grp = [&idxm](auto &&grp) {
      return std::accumulate(grp.begin(), grp.end(), idxm.size(), [&idxm](long s, long u) { return std::min(s, idxm.strides()[u]); });
    };

    static constexpr int new_rank = sizeof...(IdxGrps);
    std::array<long, new_rank> new_extents{total_len_of_a_grp(IdxGrps::as_std_array)...};
    std::array<long, new_rank> new_strides{min_stride_of_a_grp(IdxGrps::as_std_array)...};

    // new layout using the new stride_order
    using new_layout_t =
       idx_map<new_rank, 0, encode(details::stride_order_of_grouped_idx_map(stride_order, IdxGrps::as_std_array...)), layout_prop_e::contiguous>;

    return new_layout_t{new_extents, new_strides};
  }
} // namespace nda

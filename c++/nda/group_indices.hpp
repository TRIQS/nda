#pragma once
#include "basic_array.hpp"
#include <algorithm>

namespace nda {

  template <size_t R, size_t... Rs>
  constexpr bool check_grouping(std::array<int, R> const &stride_order, std::array<int, Rs> const &... grps) {
    // Now check that all indices are present
    auto l = [](auto &&grp) {
      uint64_t r = 0;
      for (int u = 0; u < grp.size(); ++u) r |= (1ull << grp[u]);
      return r;
    };
    uint64_t bit_pattern = (l(grps) | ...);
    // all bits 0> R-1 should be 1
    if (bit_pattern != ((1ull << R) - 1)) return false;

    // check that all groups are contiguous in memory

    auto check_one_group = [&stride_order](auto &&grp) {
      // find min max
      int m = R, M = 0;
      for (int u = 0; u < grp.size(); ++u) {
        int v = stride_order[grp[u]];
        if (v > M) M = v;
        if (v < m) m = v;
      }
      return (M - m + 1 == grp.size());
    };

    return (check_one_group(grps) and ...);
  }

  //---------------------------

  template <int... Is>
  struct idx_group_t {
    static constexpr std::array<int, sizeof...(Is)> as_std_array{Is...};
  };

  template <int... Is>
  inline idx_group_t<Is...> idx_group = {};

  //---------------------------

  /**
  * Regroup indices for a C array
  * Usage : group_indices_view(A, std::index_{0,1}, {2,3})
  * Precondition :
  *   - every indices is listed in the {...} exactly once.
  *   - the indices in one group are consecutive in memory.
  */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp, typename... IntSequences>
  auto group_indices_layout(idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> const &idxm, IntSequences...) {
    using Idx_t = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;

    static_assert(StaticExtents == 0, "Not yet implemented for static extents");
    static_assert(Idx_t::is_stride_order_C(), "Not yet implemented for non C orders");
    static_assert(LayoutProp == layout_prop_e::contiguous, "Not yet implemented for non contiguous arrays");

    static_assert(check_grouping(Idx_t::stride_order, IntSequences::as_std_array...), "Improper indices in group indices");

    /// Now compute the new lengths and strides.
    auto total_len_of_a_grp = [&idxm](auto &&grp) {
      auto l = std::accumulate(grp.begin(), grp.end(), 1, [&idxm](long l, long u) { return l * idxm.lengths()[u]; });
      return l;
    };

    auto min_stride_of_a_grp = [&idxm](auto &&grp) {
      return std::accumulate(grp.begin(), grp.end(), idxm.size(), [&idxm](long s, long u) { return std::min(s, idxm.strides()[u]); });
    };

    static constexpr int new_rank = sizeof...(IntSequences);
    std::array<long, new_rank> new_extents{total_len_of_a_grp(IntSequences::as_std_array)...};
    std::array<long, new_rank> new_strides{min_stride_of_a_grp(IntSequences::as_std_array)...};

    return idx_map<new_rank, 0, encode(permutations::identity<new_rank>()), layout_prop_e::contiguous>{new_extents, new_strides};
  }
} // namespace nda

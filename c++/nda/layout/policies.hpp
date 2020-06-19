#pragma once
#include "idx_map.hpp"

namespace nda {

  struct C_stride_layout;
  struct F_stride_layout;

  struct C_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>;

    using with_lowest_guarantee_t = C_stride_layout;
    using contiguous_t            = C_layout;
  };

  struct F_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>;

    using with_lowest_guarantee_t = F_stride_layout;
    using contiguous_t            = F_layout;
  };

  struct C_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>;

    using with_lowest_guarantee_t = C_stride_layout;
    using contiguous_t            = C_layout;
  };

  struct F_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>;

    using with_lowest_guarantee_t = F_stride_layout;
    using contiguous_t            = F_layout;
  };

  template <uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  struct basic_layout {
    // FIXME C++20 : StrideOrder will be a std::array<int, Rank> WITH SAME rank
    template <int Rank>
    using mapping = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;

    using with_lowest_guarantee_t = basic_layout<StaticExtents, StrideOrder, layout_prop_e::none>;
    using contiguous_t            = basic_layout<StaticExtents, StrideOrder, layout_prop_e::contiguous>;
  };

  template <uint64_t StrideOrder>
  using contiguous_layout_with_stride_order = basic_layout<0, StrideOrder, layout_prop_e::contiguous>;

} // namespace nda

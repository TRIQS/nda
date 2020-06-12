#pragma once
#include "idx_map.hpp"

namespace nda {

  struct C_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>;
  };

  struct F_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>;
  };

  struct C_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>;
  };

  struct F_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>;
  };

  template <uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  struct basic_layout {
    // FIXME C++20 : StrideOrder will be a std::array<int, Rank> WITH SAME rank
    template <int Rank>
    using mapping = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;
  };

} // namespace nda

#pragma once
#include "idx_map.hpp"

namespace nda {

  struct C_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, layout_prop_e::contiguous>;
  };

  struct F_layout {
    template <int Rank>
    using mapping = idx_map<Rank, Fortran_stride_order<Rank>, layout_prop_e::contiguous>;
  };

  struct C_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, layout_prop_e::none>;
  };

  struct F_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, Fortran_stride_order<Rank>, layout_prop_e::none>;
  };

  template <uint64_t StrideOrder, layout_prop_e LayoutProp>
  struct layout {
    template <int Rank>
    using mapping = idx_map<Rank, StrideOrder, LayoutProp>;
  };

} // namespace nda

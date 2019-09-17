#pragma once
#include "idx_map.hpp"

namespace nda {

  struct C_contiguous_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, layout_info_e::contiguous>;
  };

  struct F_contiguous_layout {
    template <int Rank>
    using mapping = idx_map<Rank, Fortran_stride_order<Rank>, layout_info_e::contiguous>;
  };

  struct C_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, layout_info_e::none>;
  };

  struct F_layout {
    template <int Rank>
    using mapping = idx_map<Rank, Fortran_stride_order<Rank>, layout_info_e::none>;
  };

  template <uint64_t StrideOrder, layout_info_e LayoutInfo>
  struct layout {
    template <int Rank>
    using mapping = idx_map<Rank, StrideOrder, LayoutInfo>;
  };

} // namespace nda

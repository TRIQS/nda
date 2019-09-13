#pragma once
#include "idx_map.hpp"

namespace nda {

  struct C_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, layout_info_e::contiguous>;
  };

  struct C_strided_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, layout_info_e::none>;
  };

  struct C_strided_1d_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, layout_info_e::strided_1d>;
  };

} // namespace nda

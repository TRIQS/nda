// Copyright (c) 2019 Simons Foundation
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

#include "h5.hpp"

namespace nda::h5_details {

  // implementation of the write
  void write(h5::group g, std::string const &name, h5::datatype ty, void *start, int rank, bool is_complex, long const *lens, long const *strides,
             long total_size) {

    h5::array_interface::h5_array_view v{ty, start, rank, is_complex};

    auto [L_tot, strides_h5] = h5::array_interface::get_L_tot_and_strides_h5(strides, rank, total_size);

    for (int u = 0; u < rank; ++u) { // size of lhs may be size of hte rhs vector + 1 if complex. Can not simply use =
      v.slab.count[u]  = lens[u];
      v.slab.stride[u] = strides_h5[u];
      v.L_tot[u]       = L_tot[u];
    }

    h5::array_interface::write(g, name, v, true);
  }
} // namespace nda::h5_details

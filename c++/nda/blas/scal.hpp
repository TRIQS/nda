// Copyright (c) 2019-2021 Simons Foundation
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
#include "../concepts.hpp"
#include "../mem/address_space.hpp"
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

namespace nda::blas {

  // --------
  template <typename X>
  requires(MemoryVector<X> or is_conj_array_expr<X>)
  void scal(get_value_t<X> alpha, X &&x) {
    static_assert(is_blas_lapack_v<get_value_t<X>>, "Vector hold value_type incompatible with blas");

    if constexpr (mem::on_host<X>) {
      f77::scal(x.size(), alpha, x.data(), x.indexmap().strides()[0]);
    } else {
#if defined(NDA_HAVE_DEVICE)
      device::scal(x.size(), alpha, x.data(), x.indexmap().strides()[0]);
#else
      static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
#endif
    }
  }

} // namespace nda::blas

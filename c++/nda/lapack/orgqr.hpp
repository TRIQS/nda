// Copyright (c) 2020-2022 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell, Jason Kaye

#pragma once

#include "../lapack.hpp"
#include "nda/concepts.hpp"

namespace nda::lapack {

  /**
   * Generates an M-by-N real matrix Q with orthonormal columns, which is
   * defined as the first N columns of a product of K elementary reflectors of
   * order M:
   *
   * Q  =  H(1) H(2) . . . H(k)
   *
   * as returned by GEQRF with real M-by-N matrix A.
   *
   * [in,out]  a is real array, dimension (LDA,N)
   *           On entry, the i-th column must contain the vector which
   *           defines the elementary reflector H(i), for i = 1,2,...,k, as
   *           returned by GEQRF or GEQP3 in the first k columns of its array
   *           argument A.
   *           On exit, the M-by-N matrix Q.
   *
   * [in]      tau is a real array, dimension (K)
   *           TAU(i) must contain the scalar factor of the elementary
   *           reflector H(i), as returned by GEQRF or GEQP3.
   *
   * [return]  info is INTEGER
   *           = 0: successful exit. 
   *           < 0: if INFO = -i, the i-th argument had an illegal value.
   */
  template <MemoryMatrix A, MemoryVector TAU>
    requires(mem::on_host<A> and std::is_same_v<double, get_value_t<A>> and have_same_value_type_v<A, TAU>
             and mem::have_compatible_addr_space<A, TAU>)
  int orgqr(A &&a, TAU &&tau) {
    static_assert(has_F_layout<A>, "C order not implemented");
    static_assert(mem::have_host_compatible_addr_space<A, TAU>,
                  "orgqr is only implemented on the CPU, but was provided but was provided non-host compatible array");

    using T     = get_value_t<A>;
    auto [m, n] = a.shape();
    auto k      = tau.size();

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // First call to get the optimal buffersize
    T bufferSize_T{};
    int info = 0;
    lapack::f77::orgqr(m, std::min(m, n), k, a.data(), get_ld(a), tau.data(), &bufferSize_T, -1, info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // Allocate work buffer and perform actual library call
    nda::array<T, 1, C_layout, heap<mem::get_addr_space<A>>> work(bufferSize);
    lapack::f77::orgqr(m, std::min(m, n), k, a.data(), get_ld(a), tau.data(), work.data(), bufferSize, info);

    if (info) NDA_RUNTIME_ERROR << "Error in orgqr : info = " << info;
    return info;
  }

} // namespace nda::lapack

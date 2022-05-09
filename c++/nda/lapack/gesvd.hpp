// Copyright (c) 2020-2021 Simons Foundation
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

#include "../lapack.hpp"

namespace nda::lapack {

  ///
  ///  $$ A = U S {}^t V$$
  /// A is destroyed during the computation
  template <MemoryMatrix A, MemoryMatrix U, MemoryMatrix V>

  requires(have_same_value_type_v<A, U, V> and is_blas_lapack_v<typename A::value_type>)

  int gesvd1(A &a, array_view<double, 1> c, U &u, V &v) {

    static_assert(has_F_layout<A>, "C order not implemented");
    static_assert(has_F_layout<U>, "C order not implemented");
    static_assert(has_F_layout<V>, "C order not implemented");

    int info = 0;

    using T = get_value_t<A>;
    static_assert(is_blas_lapack_v<T>, "Not implemented");

    if constexpr (std::is_same_v<T, double>) {

      // first call to get the optimal lwork
      T work1[1];
      lapack::f77::gesvd('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), c.data(), u.data(), get_ld(u), v.data(), get_ld(v), work1, -1,
                         info);

      int lwork = std::round(work1[0]) + 1;
      array<T, 1> work(lwork);

      lapack::f77::gesvd('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), c.data(), u.data(), get_ld(u), v.data(), get_ld(v), work.data(),
                         lwork, info);

    } else {

      auto rwork = array<double, 1>(5 * std::min(a.extent(0), a.extent(1)));

      // first call to get the optimal lwork
      T work1[1];
      lapack::f77::gesvd('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), c.data(), u.data(), get_ld(u), v.data(), get_ld(v), work1, -1,
                         rwork.data(), info);

      int lwork = std::round(std::real(work1[0])) + 1;
      array<T, 1> work(lwork);

      lapack::f77::gesvd('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), c.data(), u.data(), get_ld(u), v.data(), get_ld(v), work.data(),
                         lwork, rwork.data(), info);
    }

    if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
    return info;
  }

  inline int gesvd(matrix_view<double, F_layout> a, array_view<double, 1> c, matrix_view<double, F_layout> u, matrix_view<double, F_layout> v) {
    return gesvd1(a, c, u, v);
  }

  inline int gesvd(matrix_view<dcomplex, F_layout> a, array_view<double, 1> c, matrix_view<dcomplex, F_layout> u, matrix_view<dcomplex, F_layout> v) {
    return gesvd1(a, c, u, v);
  }

} // namespace nda::lapack

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
#include <type_traits>

namespace nda::lapack {

  ///
  template <MatrixView A, MatrixView B, VectorView C>
  int gelss(A &a, B &b, C &c, double rcond, int &rank) requires(nda::blas::have_same_element_type_and_it_is_blas_type_v<A, B> && std::is_same_v<nda::blas::remove_complex_t<get_value_t<A>>,get_value_t<C>>) {

    int info = 0;

    using T = typename A::value_type;

    // We enforce Fortran order by making a copy if necessary.
    // If both matrix are in C, call itself twice : ok we pass &
    if constexpr (not A::layout_t::is_stride_order_Fortran()) {
      auto af = matrix<T, F_layout>{a};
      info    = gelss(af, b, c, rcond, rank);
      a = af;
      return info;

    } else if constexpr (not B::layout_t::is_stride_order_Fortran()) {

      auto bf = matrix<T, F_layout>{b};
      info    = gelss(a, bf, c, rcond, rank);
      b = bf;
      return info;

    } else { // do not compile useless code !

      // Must be lapack compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(b.indexmap().min_stride() == 1);
      EXPECTS(c.indexmap().min_stride() == 1);

      auto dm = std::min(get_n_rows(a), get_n_cols(a));
      if (c.size() < dm) c.resize(dm);
      int nrhs = get_n_cols(b);

      if constexpr (std::is_same_v<T, double>) {

        // first call to get the optimal lwork
        T work1[1];
        f77::gelss(get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), c.data(), rcond, rank, work1, -1, info);

        //int lwork = r_round(work1[0]);
        int lwork = int(work1[0]);
        array<T, 1> work(lwork);

        f77::gelss(get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), c.data(), rcond, rank, work.data(), lwork, info);

      } else if constexpr (std::is_same_v<T, dcomplex>) {

        auto rwork = array<double, 1>(5 * dm);

        // first call to get the optimal lwork
        T work1[1];
        f77::gelss(get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), c.data(), rcond, rank, work1, -1, rwork.data(),
                   info);

        //int lwork = r_round(work1[0]);
        int lwork = int(real(work1[0]));
        array<T, 1> work(lwork);

        f77::gelss(get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), c.data(), rcond, rank, work.data(), lwork,
                   rwork.data(), info);
      } else
        static_assert(false and always_true<A>, "Internal logic error");

      if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
      return info;
    }
  }

} // namespace nda::lapack

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
  template <MemoryMatrix A, MemoryMatrix B, MemoryMatrix C>
  requires(have_same_value_type_v<A, B, C> and is_blas_lapack_v<get_value_t<A>>) int gelss(A &a, B &b, C &c, double rcond, int &rank) {

    int info = 0;

    using T = typename A::value_type;

    // We enforce Fortran order by making a copy if necessary.
    // If both matrix are in C, call itself twice : ok we pass &
    if constexpr (has_C_layout<A>) {
      auto af = matrix<T, F_layout>{a};
      info    = gelss(af, b, c, rcond, rank);
      return info;

    } else if constexpr (has_C_layout<B>) {
      auto bf = matrix<T, F_layout>{b};
      info    = gelss(a, bf, c, rcond, rank);
      return info;

    } else { // do not compile useless code !

      // Must be lapack compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(b.indexmap().min_stride() == 1);
      EXPECTS(c.indexmap().min_stride() == 1);

      // Copy since it is altered by gelss
      auto a2 = a;

      auto dm = std::min(get_n_rows(a2), get_n_cols(a2));
      if (c.size() < dm) c.resize(dm);
      int nrhs = get_n_cols(b);

      if constexpr (std::is_same_v<T, double>) {

        // first call to get the optimal lwork
        T work1[1];
        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data(), get_ld(a2), b.data(), get_ld(b), c.data(), rcond, rank, work1, -1, info);

        int lwork = r_round(work1[0]);
        array<T, 1> work(lwork);

        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data(), get_ld(a2), b.data(), get_ld(b), c.data(), rcond, rank, work.data(), lwork, info);

      } else if constexpr (std::is_same_v<T, dcomplex>) {

        auto rwork = array<double, 1>(5 * dm);

        // first call to get the optimal lwork
        T work1[1];
        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data(), get_ld(a2), b.data(), get_ld(b), c.data(), rcond, rank, work1, -1, rwork.data(),
                   info);

        int lwork = r_round(work1[0]);
        array<T, 1> work(lwork);

        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data(), get_ld(a2), b.data(), get_ld(b), c.data(), rcond, rank, work.data(), lwork,
                   rwork.data(), info);
      } else
        static_assert(false and always_true<A>, "Internal logic error");

      if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
      return info;
    }
  }

} // namespace nda::lapack

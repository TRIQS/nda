/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <complex>
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

namespace nda::blas {

  ///
  template <CONCEPT(MatrixView) A, CONCEPT(VectorView) C, CONCEPT(MatrixView) U, CONCEPT(MatrixView) V>

  // FIXME : a1 -> a after compile
  REQUIRES(have_same_element_type_and_it_is_blas_type_v<A, C, U, V>) int gesvd(A const &a1, C &c, U &u, V &v) {

    int info = 0;

    using T = typename A::value_type;

    // We enforce Fortran order by making a copy if necessary.
    // If both matrix are in C, call itself twice : ok we pass &
    if constexpr (not A::layout_t::is_stride_order_Fortran()) {
      auto af = matrix<T, F_layout>{a1};
      info    = gesvd(af, c, u, v);
      return info;
    }

    else if constexpr (not U::layout_t::is_stride_order_Fortran()) {

      auto uf = matrix<T, F_layout>{u};
      info    = gesvd(a1, c, uf, v);
      u       = uf;
      return info;

    } else if constexpr (not V::layout_t::is_stride_order_Fortran()) {

      auto vf = matrix<T, F_layout>{v};
      info    = gesvd(a1, c, u, vf);
      u       = uf;
      return info;
    } else { // do not compile useless code !

      auto a2 = a1;

      if constexpr (std::is_same_v<T, double>) {

        // first call to get the optimal lwork
        T work1[1];
        f77::gesvd('A', 'A', get_n_rows(a2), get_n_cols(a2), a2.data_start(), get_ld(a2), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                   get_ld(v), work1, -1, info);

        int lwork = r_round(work1[0]);
        arrays::vector<T> work(lwork);

        f77::gesvd('A', 'A', get_n_rows(a2), get_n_cols(a2), a2.data_start(), get_ld(a2), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                   get_ld(v), work.data_start(), lwork, info);

      } else if constexpr (std::is_same_v<T, dcomplex>) {

        auto rwork = array<double, 1>(5 * std::min(first_dim(a2), second_dim(a2)));

        // first call to get the optimal lwork
        T work1[1];
        f77::gesvd('A', 'A', get_n_rows(a2), get_n_cols(a2), a2.data_start(), get_ld(a2), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                   get_ld(v), work1, -1, rwork.data_start(), info);

        int lwork = r_round(work1[0]);
        arrays::vector<T> work(lwork);

        f77::gesvd('A', 'A', get_n_rows(a2), get_n_cols(a2), a2.data_start(), get_ld(a2), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                   get_ld(v), work.data_start(), lwork, rwork.data_start(), info);

      } else
        static_assert(false and always_true<A>, "Internal logic error");

      if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
      return info;
    }
  }
} // namespace nda::blas

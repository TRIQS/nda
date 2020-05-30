#pragma once
#include <complex>
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

#include <optional>
#include <itertools/itertools.hpp>
#include "f77/cxx_interface.hpp"

namespace nda::blas {

  ///
  template <CONCEPT(MatrixView) A, CONCEPT(MatrixView) B, CONCEPT(MatrixView) C>

  REQUIRES(have_same_element_type_and_it_is_blas_type_v<A, B, C>)

  // FIXME : why A const and B not ??
  // CF Nils ??
  // FIXME change a_in to a after pass test
  int gelss(A const &a_in, B &b, C &c, double rcond, int &rank) {

    int info = 0;

    using T = typename A::value_type;

    // We enforce Fortran order by making a copy if necessary.
    // If both matrix are in C, call itself twice : ok we pass &
    if constexpr (not A::layout_t::is_stride_order_Fortran()) {
      auto af = matrix<T, F_layout>{a_in};
      info    = gelss(af, b, c, rcond, rank);
      return info;

    } else if constexpr (not B::layout_t::is_stride_order_Fortran()) {

      auto bf = matrix<T, F_layout>{b};
      info    = gelss(a_in, bf, c, rcond, rank);
      return info;

    } else { // do not compile useless code !

      // Must be lapack compatible
      EXPECTS(a_in.indexmap().min_stride() == 1);
      EXPECTS(b.indexmap().min_stride() == 1);
      EXPECTS(c.indexmap().min_stride() == 1);

      // Copy since it is altered by gelss
      auto a2 = a_in;

      auto dm = std::min(get_n_rows(a2), get_n_cols(a2));
      if (c.size() < dm) c.resize(dm);
      int nrhs = get_n_cols(b);

      if constexpr (std::is_same_v<T, double>) {

        // first call to get the optimal lwork
        T work1[1];
        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data_start(), get_ld(a2), b.data_start(), get_ld(b), c.data_start(), rcond, rank, work1,
                   -1, info);

        int lwork = r_round(work1[0]);
        array<T, 1> work(lwork);

        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data_start(), get_ld(a2), b.data_start(), get_ld(b), c.data_start(), rcond, rank,
                   work.data_start(), lwork, info);

      } else if constexpr (std::is_same_v<T, dcomplex>) {

        auto rwork = array<double, 1>(5 * dm);

        // first call to get the optimal lwork
        T work1[1];
        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data_start(), get_ld(a2), b.data_start(), get_ld(b), c.data_start(), rcond, rank, work1,
                   -1, rwork.data_start(), info);

        int lwork = r_round(work1[0]);
        array<T, 1> work(lwork);

        f77::gelss(get_n_rows(a2), get_n_cols(a2), nrhs, a2.data_start(), get_ld(a2), b.data_start(), get_ld(b), c.data_start(), rcond, rank,
                   work.data_start(), lwork, rwork.data_start(), info);
      } else
        static_assert(false and always_true<A>, "Internal logic error");

      if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
      return info;
    }
  }

} // namespace nda::blas

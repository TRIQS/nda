// Copyright (c) 2019-2020 Simons Foundation
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

#pragma once
#include <complex>
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

namespace nda::blas {

  /**
   * Compute c <- alpha a*b + beta * c using BLAS dgemm or zgemm 
   * 
   * using a generic version for non lapack types or more complex types
   * largely suboptimal. For testing, or in case of value_type being a complex type.
   * SHOULD not be used with double and dcomplex
   *
   * \private : DO NOT DOCUMENT, testing only ??
   */
  template <MatrixView A, MatrixView B, MatrixView Out>

  void gemm_generic(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &c) {

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    EXPECTS(b.extent(1) == c.extent(1));

    c() = 0;
    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < b.extent(1); ++j) {
        typename A::value_type acc = 0;
        for (int k = 0; k < a.extent(1); ++k) acc += alpha * a(i, k) * b(k, j);
        c(i, j) = acc + beta * c(i, j);
      }
  }

  /**
   * Compute c <- alpha a*b + beta * c using BLAS dgemm or zgemm 
   *
   * @param c Out parameter. Can be a temporary view (hence the &&).
   *         
   * @Precondition : 
   *       * c has the correct dimension given a, b. 
   *         gemm does not resize the object, 
   */
  template <MatrixView A, MatrixView B, MatrixView C>

  REQUIRES(have_same_value_type_v<A, B, C> and is_blas_lapack_v<typename A::value_type>)

  void gemm(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, C &&c) {

    using C_t = std::decay_t<C>;

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    EXPECTS(b.extent(1) == c.extent(1));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    // We need to see if C is in Fortran order or C order
    if constexpr (C_t::is_stride_order_C()) {
      // C order. We compute the transpose of the product in this case
      // since BLAS is in Fortran order
      char trans_a = get_trans(b, true);
      char trans_b = get_trans(a, true);
      int m        = (trans_a == 'N' ? get_n_rows(b) : get_n_cols(b));
      int n        = (trans_b == 'N' ? get_n_cols(a) : get_n_rows(a));
      int k        = (trans_a == 'N' ? get_n_cols(b) : get_n_rows(b));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, b.data(), get_ld(b), a.data(), get_ld(a), beta, c.data(), get_ld(c));
    } else {
      // C is in fortran or, we compute the product.
      char trans_a = get_trans(a, false);
      char trans_b = get_trans(b, false);
      int m        = (trans_a == 'N' ? get_n_rows(a) : get_n_cols(a));
      int n        = (trans_b == 'N' ? get_n_cols(b) : get_n_rows(b));
      int k        = (trans_a == 'N' ? get_n_cols(a) : get_n_rows(a));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, a.data(), get_ld(a), b.data(), get_ld(b), beta, c.data(), get_ld(c));
    }
  }

} // namespace nda::blas

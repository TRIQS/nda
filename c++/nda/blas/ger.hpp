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
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

namespace nda::blas {

  /**
   * Calls ger on a matrix, matrix_view, array, array_view of rank 2
   *  m += alpha * x * ty
   *
   * @tparam X array, array_view of rank 1
   * @tparam Y array, array_view of rank 1
   * @tparam M matrix, matrix_view, array, array_view of rank 2
   * @param alpha
   * @param x 
   * @param y
   * @param m The result. Can be a temporary view. 
   *         
   * @StaticPrecondition : X, Y, M have the same value_type and it is complex<double> or double         
   * @Precondition : 
   *       * m has the correct dimension given a, b. 
   */
  template <typename X, typename Y, typename M>
  void ger(typename X::value_type alpha, X const &x, Y const &y, M &&m) {

    using M_t = std::decay_t<M>;
    static_assert(is_regular_or_view_v<M_t>, "ger: Out must be a matrix or matrix_view");
    static_assert(is_regular_or_view_v<M_t>, "gemm: Out must be a matrix, matrix_view, array or array_view of rank 2");
    static_assert(have_same_element_type_and_it_is_blas_type_v<X, Y, M_t>,
                  "Matrices must have the same element type and it must be double, complex ...");

    static_assert(X::rank == 1, "X must be of rank 1");
    static_assert(Y::rank == 1, "Y must be of rank 1");
    static_assert(M_t::rank == 2, "C must be of rank 2");

    EXPECTS(m.extent(1) == x.extent(0));
    EXPECTS(m.extent(0) == y.extent(0));
    // Must be lapack compatible
    EXPECTS(m.indexmap().min_stride() == 1);

    //for (int i = 0; i< x.extent(0); ++i)
    //for (int j = 0; j< y.extent(0); ++j)
    //m(i,j) += alpha * x(i) * y(j);
    //return ;

    auto idx = m.indexmap(); // FIXME should not need a copy
    // if in C, we need to call fortran with transposed matrix
    if constexpr (idx.is_stride_order_C())
      f77::ger(get_n_rows(m), get_n_cols(m), alpha, y.data(), y.indexmap().strides()[0], x.data(), x.indexmap().strides()[0], m.data(), get_ld(m));
    else
      f77::ger(get_n_rows(m), get_n_cols(m), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
  }

} // namespace nda::blas

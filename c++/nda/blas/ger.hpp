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
#include "tools.hpp"
#include "interface/cxx_interface.hpp"
#include "../mem/address_space.hpp"

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
  template <MemoryVector X, MemoryVector Y, MemoryMatrix M>
  requires(have_same_value_type_v<X, Y, M> and is_blas_lapack_v<get_value_t<X>>)
  void ger(typename X::value_type alpha, X const &x, Y const &y, M &&m) {

    EXPECTS(m.extent(0) == x.extent(0));
    EXPECTS(m.extent(1) == y.extent(0));
    // Must be lapack compatible
    EXPECTS(m.indexmap().min_stride() == 1);

    static constexpr auto X_adr_spc = mem::get_addr_space<X>;
    static constexpr auto Y_adr_spc = mem::get_addr_space<Y>;
    static constexpr auto M_adr_spc = mem::get_addr_space<M>;
    static_assert(X_adr_spc == Y_adr_spc && Y_adr_spc == M_adr_spc);

    // if in C, we need to call fortran with transposed matrix
    if (has_C_layout<M>) {
      ger(alpha, y, x, transpose(m));
      return;
    }

    if constexpr (mem::on_host<X>) {
      f77::ger(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
    } else {
      cuda::ger(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
    }
  }

  /**
   * Calculate the outer product of two (contiguous) arrays a and b
   *
   *  $$ c_{i,j,k,...,u,v,w,...} = a_{i,j,k,...} * b_{u,v,w,...} $$
   *
   * Both A and B can be scalars. Uses blas::ger for efficient implementation.
   *
   * @tparam A array or scalar type
   * @tparam B array or scalar type
   * @param a The first array
   * @param b The second array
   *
   * @return The outer product of a and b as defined above
   */
  template <ArrayOrScalar A, ArrayOrScalar B>
  auto outer_product(A const &a, B const &b) {

    if constexpr (Scalar<A> or Scalar<B>) {
      return a * b;
    } else {
      static_assert(has_contiguous_layout<A> and has_contiguous_layout<B>);
      auto res = zeros<get_value_t<A>, mem::get_addr_space<A>>(stdutil::join(a.shape(), b.shape()));

      auto a_vec = reshaped_view(a, std::array{a.size()});
      auto b_vec = reshaped_view(b, std::array{b.size()});
      auto mat   = reshaped_view(res, std::array{a.size(), b.size()});
      ger(1.0, a_vec, b_vec, mat);

      return res;
    }
  }

} // namespace nda::blas

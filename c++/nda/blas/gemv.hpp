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
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "tools.hpp"
#include "interface/cxx_interface.hpp"
#include "../mem/address_space.hpp"

namespace nda::blas {

  // make the generic version for non lapack types or more complex types
  // largely suboptimal
  template <typename A, typename B, typename Out>
  void gemv_generic(get_value_t<A> alpha, A const &a, B const &b, get_value_t<A> beta, Out &c) {
    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    c() = 0;
    for (int i = 0; i < a.extent(0); ++i) {
      for (int k = 0; k < a.extent(1); ++k) c(i) += alpha * a(i, k) * b(k);
      c(i) += beta * c(i);
    }
  }

  /**
   * Calls gemv on a matrix, matrix_view, array, array_view of rank 2
   * to compute c <- alpha a*b + beta * c
   *
   * @tparam A matrix, matrix_view, array, array_view of rank 2
   * @tparam B matrix, matrix_view, array, array_view of rank 1
   * @tparam C matrix, matrix_view, array, array_view of rank 1
   * @param alpha
   * @param a 
   * @param b
   * @param beta
   * @param c The result. Can be a temporary view. 
   *         
   * @StaticPrecondition : A, B, C have the same value_type and it is complex<double> or double         
   * @Precondition : 
   *       * c has the correct dimension given a, b. 
   *         gemm does not resize the object, 
   *
   *
   */
  template <Matrix X, MemoryVector B, MemoryVector C>
  requires((MemoryMatrix<X> or is_conj_array_expr<X>) and //
           have_same_value_type_v<X, B, C> and            //
           is_blas_lapack_v<get_value_t<X>>)              //
  void gemv(get_value_t<X> alpha, X const &x, B const &b, get_value_t<X> beta, C &&c) {

    auto to_mat = []<Matrix Z>(Z const &z) -> decltype(auto) {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a = to_mat(x);

    static constexpr bool conj_A = is_conj_array_expr<X>;

    using A = decltype(a);
    static_assert(mem::have_same_addr_space_v<A, B, C>);

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    char op_a                    = get_op<conj_A, /*transpose =*/!has_F_layout<A>>;
    auto [m, n] = a.shape();
    if constexpr (has_C_layout<A>) std::swap(m, n);

    if constexpr (mem::on_host<A>) {
      f77::gemv(op_a, m, n, alpha, a.data(), get_ld(a), b.data(), b.indexmap().strides()[0], beta, c.data(), c.indexmap().strides()[0]);
    } else {
#if defined(NDA_HAVE_CUDA)
      device::gemv(op_a, m, n, alpha, a.data(), get_ld(a), b.data(), b.indexmap().strides()[0], beta, c.data(), c.indexmap().strides()[0]);
#else
      static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
      return std::decay_t<X>::value_type{0};
#endif
    }
  }

} // namespace nda::blas

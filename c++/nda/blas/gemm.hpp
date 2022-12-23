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
#include <complex>
#include "tools.hpp"
#include "interface/cxx_interface.hpp"
#include "../declarations.hpp"
#include "../mem/address_space.hpp"

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
  template <Matrix A, Matrix B, MemoryMatrix Out>
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
  template <Matrix X, Matrix Y, MemoryMatrix C>
  requires((MemoryMatrix<X> or is_conj_array_expr<X>) and                        //
           (MemoryMatrix<Y> or is_conj_array_expr<Y>) and                        //
           have_same_value_type_v<X, Y, C> and is_blas_lapack_v<get_value_t<X>>) //
  void gemm(get_value_t<X> alpha, X const &x, Y const &y, get_value_t<X> beta, C &&c) {

    auto to_mat = []<typename Z>(Z const &z) -> auto & {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a = to_mat(x);
    auto &b = to_mat(y);

    static constexpr bool conj_A = is_conj_array_expr<X>;
    static constexpr bool conj_B = is_conj_array_expr<Y>;

    using A = decltype(a);
    using B = decltype(b);
    static_assert(mem::have_same_addr_space_v<A, B, C>, "Matrices must have same memory address space");

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    EXPECTS(b.extent(1) == c.extent(1));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    // c is in C order: compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      gemm(alpha, transpose(y), transpose(x), beta, transpose(std::forward<C>(c)));
      return;
    } else { // c is in Fortran order
      char op_a   = get_op<conj_A, /*transpose =*/has_C_layout<A>>;
      char op_b   = get_op<conj_B, /*transpose =*/has_C_layout<B>>;
      auto [m, k] = a.shape();
      auto n      = b.extent(1);

      if constexpr (mem::on_host<A>) {
        f77::gemm(op_a, op_b, m, n, k, alpha, a.data(), get_ld(a), b.data(), get_ld(b), beta, c.data(), get_ld(c));
      } else { // on device
#if defined(NDA_HAVE_DEVICE)
        device::gemm(op_a, op_b, m, n, k, alpha, a.data(), get_ld(a), b.data(), get_ld(b), beta, c.data(), get_ld(c));
#else
        static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
#endif
      }
    }
  }

} // namespace nda::blas

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

#include "../layout/policies.hpp"
#include "../basic_array.hpp"
#include "../blas/gemm.hpp"
#include "../blas/gemv.hpp"

namespace nda {

  // Helper variable template to check if the three Matrix types can be passed to gemm
  // Only certain combinations of the memory layout (C/Fortran) and conjugation are allowed
  template <Matrix A, Matrix B, MemoryMatrix C, bool conj_A = blas::is_conj_matrix_expr<A>, bool conj_B = blas::is_conj_matrix_expr<B>>
  requires((MemoryMatrix<A> or conj_A) and (MemoryMatrix<B> or conj_B))
  static constexpr bool is_valid_gemm_triple = []() {
    using blas::has_F_layout;
    if constexpr (has_F_layout<C>) {
      return !(conj_A and has_F_layout<A>) and !(conj_B and has_F_layout<B>);
    } else {
      return !(conj_B and !has_F_layout<B>) and !(conj_A and !has_F_layout<A>);
    }
  }();

  /**
   * @tparam L NdArray with algebra 'M' 
   * @tparam R 
   * @param l : lhs
   * @param r : rhs
   * @return the matrix multiplication
   *   Implementation varies 
   */
  template <Matrix L, Matrix R>
  auto matmul(L &&l, R &&r) {
    EXPECTS_WITH_MESSAGE(l.shape()[1] == r.shape()[0], "Matrix product : dimension mismatch in matrix product " << l << " " << r);

    static constexpr auto L_adr_spc = mem::get_addr_space<L>;
    static constexpr auto R_adr_spc = mem::get_addr_space<R>;
    static_assert(L_adr_spc == R_adr_spc, "Error: Matrix Product requires arguments with same Adress space");
    static_assert(L_adr_spc != mem::None);

    using promoted_type = decltype(get_value_t<L>{} * get_value_t<R>{});
    using matrix_t      = basic_array<promoted_type, 2, C_layout /*FIXME*/, 'M', nda::heap<mem::combine<L_adr_spc, R_adr_spc>>>;
    auto result         = matrix_t(l.shape()[0], r.shape()[1]);

    if constexpr (is_blas_lapack_v<promoted_type>) {

      auto as_container = []<typename A>(A &&a) -> decltype(auto) {
        if constexpr (std::is_same_v<get_value_t<A>, promoted_type> and (MemoryMatrix<A> or blas::is_conj_matrix_expr<A>))
          return std::forward<A>(a);
        else
          return matrix_t{std::forward<A>(a)};
      };

      // MSAN has no way to know that we are calling with beta = 0, hence
      // this is not necessaru
      // of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      result = 0;
#endif
#endif

      // Check if we have a valid set of matrices for a gemm call
      // If not, form a new matrix from any conjugate matrix expressions
      if constexpr (!is_valid_gemm_triple<decltype(as_container(l)), decltype(as_container(r)), matrix_t>) {
        blas::gemm(1, make_regular(as_container(l)), make_regular(as_container(r)), 0, result);
      } else {
        blas::gemm(1, as_container(l), as_container(r), 0, result);
      }

    } else {
      blas::gemm_generic(1, l, r, 0, result);
    }
    return result;
  }

  /**
   * @tparam L NdArray with algebra 'M' 
   * @tparam R 
   * @param l : lhs
   * @param r : rhs
   * @return the matrix multiplication
   *   Implementation varies 
   */

  template <Matrix L, Vector R>
  auto matvecmul(L &&l, R &&r) {
    EXPECTS_WITH_MESSAGE(l.shape()[1] == r.shape()[0], "Matrix Vector product : dimension mismatch in matrix product " << l << " " << r);

    static constexpr auto L_adr_spc = mem::get_addr_space<L>;
    static constexpr auto R_adr_spc = mem::get_addr_space<R>;
    static_assert(L_adr_spc == R_adr_spc, "Error: Matrix Product requires arguments with same Adress space");
    static_assert(L_adr_spc != mem::None);

    using promoted_type = decltype(get_value_t<L>{} * get_value_t<R>{});
    vector<promoted_type, heap<L_adr_spc>> result(l.shape()[0]);

    if constexpr (is_blas_lapack_v<promoted_type>) {

      auto as_container = []<typename A>(A &&a) -> decltype(auto) {
        if constexpr (std::is_same_v<get_value_t<A>, promoted_type> and (MemoryMatrix<A> or blas::is_conj_matrix_expr<A>))
          return std::forward<A>(a);
        else
          return basic_array<promoted_type, get_rank<A>, C_layout, 'A', heap<L_adr_spc>>{std::forward<A>(a)};
      };

      // MSAN has no way to know that we are calling with beta = 0, hence
      // this is not necessaru
      // of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      result = 0;
#endif
#endif

      if constexpr (blas::is_conj_matrix_expr<decltype(as_container(l))> and
	            blas::has_F_layout<decltype(as_container(l))>) {
	blas::gemv(1, make_regular(as_container(l)), as_container(r), 0, result);
      } else {
        blas::gemv(1, as_container(l), as_container(r), 0, result);
      }
    } else {
      blas::gemv_generic(1, l, r, 0, result);
    }
    return result;
  }

} // namespace nda

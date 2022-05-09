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
#include <type_traits>

#include "../traits.hpp"
#include "../concepts.hpp"
#include "../mapped_functions.hpp"

namespace nda {
  using dcomplex = std::complex<double>;
}

namespace nda::blas {

  // Check if a type is a conjugate matrix expression
  template <typename M>
  static constexpr bool is_conj_matrix_expr = false;
  template <MemoryMatrix M>
  static constexpr bool is_conj_matrix_expr<expr_call<conj_f, M>> = true;
  template <typename M>
  requires(!std::is_same_v<M, std::remove_cvref_t<M>>) static constexpr bool is_conj_matrix_expr<M> = is_conj_matrix_expr<std::remove_cvref_t<M>>;

  // ==== Layout Checks (Fortran/C) for both MemoryMatrix and conj(MemoryMatrix)

  template <Matrix M>
  requires(MemoryMatrix<M> or is_conj_matrix_expr<M>)
  static constexpr bool has_F_layout = [](){
    if constexpr (blas::is_conj_matrix_expr<M>) return has_F_layout<decltype(std::get<0>(std::declval<M>().a))>;
    else return std::remove_cvref_t<M>::is_stride_order_Fortran();
  }();

  template <Matrix M>
  requires(MemoryMatrix<M> or is_conj_matrix_expr<M>)
  static constexpr bool has_C_layout = [](){
    if constexpr (blas::is_conj_matrix_expr<M>) return has_C_layout<decltype(std::get<0>(std::declval<M>().a))>;
    else return std::remove_cvref_t<M>::is_stride_order_C();
  }();

  // Determine the blas matrix operation tag ('N','T','C') based on the bools for conjugation and transposition
  template <bool conj, bool transpose>
  const char get_op = []() {
    static_assert(!(conj and not transpose), "Cannot use conjugate of a matrix in blas operations. Please perform operation before call");
    if constexpr (conj and transpose)
      return 'C';
    else if constexpr (transpose)
      return 'T';
    else // !conj and !transpose
      return 'N';
  }();

  // LDA in lapack jargon
  template <MemoryMatrix A>
  int get_ld(A const &a) {
    return a.indexmap().strides()[has_F_layout<A> ? 1 : 0];
  }

} // namespace nda::blas

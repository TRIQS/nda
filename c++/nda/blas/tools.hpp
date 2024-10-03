// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides various traits and utilities for the BLAS interface.
 */

#pragma once

#include "../concepts.hpp"
#include "../map.hpp"
#include "../mapped_functions.hpp"

#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @ingroup linalg_blas
   * @brief Alias for `std::complex<double>` type.
   */
  using dcomplex = std::complex<double>;

} // namespace nda

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /// Constexpr variable that is true if the given type is a conjugate lazy expression.
  template <typename A>
  static constexpr bool is_conj_array_expr = false;

  /// Specialization of nda::blas::is_conj_array_expr for the conjugate lazy expressions.
  template <MemoryArray A>
  static constexpr bool is_conj_array_expr<expr_call<detail::conj_f, A>> = true;

  // Specialization of nda::blas::is_conj_array_expr for cvref types.
  template <typename A>
    requires(!std::is_same_v<A, std::remove_cvref_t<A>>)
  static constexpr bool is_conj_array_expr<A> = is_conj_array_expr<std::remove_cvref_t<A>>;

  /// Constexpr variable that is true if the given nda::Array type has a Fortran memory layout.
  template <Array A>
    requires(MemoryArray<A> or is_conj_array_expr<A>)
  static constexpr bool has_F_layout = []() {
    if constexpr (is_conj_array_expr<A>)
      return has_F_layout<decltype(std::get<0>(std::declval<A>().a))>;
    else
      return std::remove_cvref_t<A>::is_stride_order_Fortran();
  }();

  /// Constexpr variable that is true if the given nda::Array type has a C memory layout.
  template <Array A>
    requires(MemoryArray<A> or is_conj_array_expr<A>)
  static constexpr bool has_C_layout = []() {
    if constexpr (is_conj_array_expr<A>)
      return has_C_layout<decltype(std::get<0>(std::declval<A>().a))>;
    else
      return std::remove_cvref_t<A>::is_stride_order_C();
  }();

  /**
   * @brief Variable template that determines the BLAS matrix operation tag ('N','T','C') based on the given boolean
   * flags for conjugation and transposition.
   *
   * @tparam conj Boolean flag for conjugation.
   * @tparam transpose Boolean flag for transposition.
   */
  template <bool conj, bool transpose>
  const char get_op = []() {
    static_assert(!(conj and not transpose), "Error in nda::blas::get_op: Cannot use conjugate operation alone in blas operations");
    if constexpr (conj and transpose)
      return 'C';
    else if constexpr (transpose)
      return 'T';
    else // !conj and !transpose
      return 'N';
  }();

  /**
   * @brief Get the leading dimension in LAPACK jargon of an nda::MemoryMatrix.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a nda::MemoryMatrix object.
   * @return Leading dimension.
   */
  template <MemoryMatrix A>
  int get_ld(A const &a) {
    return a.indexmap().strides()[has_F_layout<A> ? 1 : 0];
  }

  /**
   * @brief Get the number of columns in LAPACK jargon of an nda::MemoryMatrix.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a nda::MemoryMatrix object.
   * @return Number of columns.
   */
  template <MemoryMatrix A>
  int get_ncols(A const &a) {
    return a.shape()[has_F_layout<A> ? 1 : 0];
  }

  /** @} */

} // namespace nda::blas

// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

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
   * @ingroup linalg_blas_utils
   * @brief Alias for `std::complex<float>` type.
   */
  using scomplex = std::complex<float>;

  /**
   * @ingroup linalg_blas_utils
   * @brief Alias for `std::complex<double>` type.
   */
  using dcomplex = std::complex<double>;

} // namespace nda

namespace nda::blas {

  /**
   * @addtogroup linalg_blas_utils
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

  /**
   * @brief Get the underlying array of a conjugate lazy expression or return the array itself in case it is an
   * nda::MemoryArray.
   *
   * @tparam A nda::Array type.
   * @param a Conjugate expression or array/view.
   * @return nda::MemoryArray object.
   */
  template <Array A>
    requires(MemoryArray<A> or is_conj_array_expr<A>)
  MemoryArray decltype(auto) get_array(A &&a) {
    if constexpr (is_conj_array_expr<A>) {
      return std::get<0>(std::forward<A>(a).a);
    } else {
      return std::forward<A>(a);
    }
  }

  /// Constexpr variable that is true if the given nda::Array type has nda::F_layout.
  template <Array A>
    requires(MemoryArray<A> or is_conj_array_expr<A>)
  static constexpr bool has_F_layout = []() {
    if constexpr (is_conj_array_expr<A>)
      return has_F_layout<decltype(std::get<0>(std::declval<A>().a))>;
    else
      return std::remove_cvref_t<A>::is_stride_order_Fortran();
  }();

  /// Constexpr variable that is true if the given nda::Array type has nda::C_layout.
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
    static_assert(!(conj and not transpose), "Error in nda::blas::get_op: Cannot use conjugate operation alone in BLAS operations");
    if constexpr (conj and transpose)
      return 'C';
    else if constexpr (transpose)
      return 'T';
    else // !conj and !transpose
      return 'N';
  }();

  /**
   * @brief Get the leading dimension of an nda::MemoryArray with rank 1 or 2 for BLAS/LAPACK calls.
   *
   * @details The leading dimension is the stride between two consecutive columns (rows) of a matrix in Fortran (C)
   * layout. For 1-dimensional arrays, we simply return the size of the array.
   *
   * @tparam A nda::MemoryArray type.
   * @param a nda::MemoryArray object.
   * @return Leading dimension for BLAS/LAPACK calls.
   */
  template <MemoryArray A>
    requires(get_rank<A> == 1 or get_rank<A> == 2)
  int get_ld(A const &a) {
    if constexpr (get_rank<A> == 1) {
      return a.size();
    } else {
      return a.indexmap().strides()[has_F_layout<A> ? 1 : 0];
    }
  }

  /**
   * @brief Get the number of columns of an nda::MemoryArray with rank 1 or 2 for BLAS/LAPACK calls.
   *
   * @details The number of columns corresponds to the extent of the second (first) dimension of a matrix in Fortran
   * (C) layout. For 1-dimensional arrays, we return 1.
   *
   * @tparam A nda::MemoryArray type.
   * @param a nda::MemoryArray object.
   * @return Number of columns for BLAS/LAPACK calls.
   */
  template <MemoryArray A>
    requires(get_rank<A> == 1 or get_rank<A> == 2)
  int get_ncols(A const &a) {
    if constexpr (get_rank<A> == 1) {
      return 1;
    } else {
      return a.shape()[has_F_layout<A> ? 1 : 0];
    }
  }

  /** @} */

} // namespace nda::blas

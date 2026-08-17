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
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../macros.hpp"
#include "../map.hpp"
#include "../mapped_functions.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <complex>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @ingroup linalg_blas_utils
   * @brief Alias for `std::complex<double>` type.
   */
  using dcomplex = std::complex<double>;

} // namespace nda

namespace nda::blas_lapack {

  /**
   * @addtogroup linalg_blas_utils
   * @{
   */

  /// Constexpr variable that is true if the given type is a conjugate lazy expression.
  template <typename A>
  static constexpr bool is_conj_array_expr = false;

  /// Specialization of nda::blas_lapack::is_conj_array_expr for the conjugate lazy expressions.
  template <MemoryArray A>
  static constexpr bool is_conj_array_expr<expr_call<detail::conj_f, A>> = true;

  // Specialization of nda::blas_lapack::is_conj_array_expr for cvref types.
  template <typename A>
    requires(!std::is_same_v<A, std::remove_cvref_t<A>>)
  static constexpr bool is_conj_array_expr<A> = is_conj_array_expr<std::remove_cvref_t<A>>;

  /**
   * @brief Get the underlying array of a conjugate lazy expression or return the array itself in case it is an
   * nda::MemoryArray.
   *
   * @details The returned array carries the constness of the expression object, i.e. a const conjugate expression can
   * only ever be a read-only BLAS/tensor operand, while a non-const one can also serve as a destination.
   *
   * @tparam A nda::Array type.
   * @param a Conjugate expression or array/view.
   * @return nda::MemoryArray object.
   */
  template <Array A>
    requires(MemoryArray<A> or is_conj_array_expr<A>)
  MemoryArray decltype(auto) get_array(A &&a) {
    if constexpr (is_conj_array_expr<A>) {
      return std::forward<A>(a).operand();
    } else {
      return std::forward<A>(a);
    }
  }

  /// Constexpr variable that is true if all given nda::Array types have nda::F_layout.
  template <Array... As>
    requires((MemoryArray<As> or is_conj_array_expr<As>) and ...)
  static constexpr bool has_F_layout = ([]<typename A>() constexpr {
    if constexpr (is_conj_array_expr<A>)
      return has_F_layout<decltype(std::declval<A>().operand())>;
    else
      return std::remove_cvref_t<A>::is_stride_order_Fortran();
  }.template operator()<As>() and ...);

  /// Constexpr variable that is true if all given nda::Array types have nda::C_layout.
  template <Array... As>
    requires((MemoryArray<As> or is_conj_array_expr<As>) and ...)
  static constexpr bool has_C_layout = ([]<typename A>() constexpr {
    if constexpr (is_conj_array_expr<A>)
      return has_C_layout<decltype(std::declval<A>().operand())>;
    else
      return std::remove_cvref_t<A>::is_stride_order_C();
  }.template operator()<As>() and ...);

  /**
   * @brief Variable template that determines the BLAS matrix operation tag ('N','T','C') based on the given boolean
   * flags for conjugation and transposition.
   *
   * @tparam conj Boolean flag for conjugation.
   * @tparam transpose Boolean flag for transposition.
   */
  template <Array A>
  static constexpr char get_op = []() {
    auto constexpr conj      = is_conj_array_expr<A>;
    auto constexpr transpose = has_C_layout<A>;
    static_assert(!(conj and not transpose), "Error in nda::blas_lapack::get_op: Cannot use conjugate operation alone in BLAS operations");
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

  /**
   * @brief Alias for an nda::vector with the same value type and address space as the given type.
   * @tparam A nda::MemoryArray type.
   */
  template <MemoryArray A>
  using vector_value_t = vector<get_value_t<A>, heap<mem::get_addr_space<A>>>;

  /**
   * @brief Alias for an nda::vector with the same address space as the given type and its value type determined by
   * `nda::get_fp_t<A>`.
   *
   * @tparam A nda::MemoryArray type.
   */
  template <MemoryArray A>
  using vector_fp_t = vector<get_fp_t<A>, heap<mem::get_addr_space<A>>>;

  /**
   * @brief Given a 2- or 3-dimensional array get an array of pointers to each of the submatrices/subvectors indexed by
   * the slowest varying dimension.
   *
   * @tparam A nda::MemoryArray of rank 2 or 3.
   * @param a Input array.
   * @return nda::vector of pointers to each submatrix/subvector.
   */
  template <Array A>
    requires(MemoryArrayOfRank<A, 3> or MemoryArrayOfRank<A, 2>)
  auto batch_ptrs(A &&a) { // NOLINT (temporary views are allowed here)
    using ptr_t           = decltype(a.data());
    auto const idx        = (has_C_layout<A> ? 0 : (get_rank<A> == 3 ? 2 : 1));
    auto const batch_size = a.shape()[idx];
    auto const stride     = a.indexmap().strides()[idx];

    auto ptrs = vector<ptr_t>(batch_size);
    for (int i = 0; auto &ptr : ptrs) ptr = a.data() + i++ * stride;
    return ptrs;
  }

  /**
   * @brief Resize or check the size of a 1D array/view.
   *
   * @details This function is similar to nda::resize_or_check_if_view except that
   * - it only works for 1D arrays/views,
   * - it does not resize or throw an error if the size is too big and
   * - it expects that the memory is contiguous.
   *
   * @tparam A Type of the object.
   * @param a Object to resize or check.
   * @param min_size Minimum size.
   */
  template <typename A>
    requires(is_regular_or_view_v<A> and get_rank<A> == 1)
  void resize_or_check_work_buffer(A &a, long min_size) {
    if (a.size() >= min_size) {
      EXPECTS(a.indexmap().min_stride() == 1);
      return;
    }
    if constexpr (is_regular_v<A>) {
      a.resize(min_size);
    } else {
      NDA_RUNTIME_ERROR << "Error in nda::blas_lapack::resize_or_check_work_buffer: Size mismatch: " << a.size() << " < " << min_size;
    }
  }

  /**
   * @brief BLAS/LAPACK compatible array type.
   * 
   * @tparam A Array type.
   * @tparam R Optional required rank.
   */
  template <typename A, int R = -1>
  concept BlasArray = (R == -1 ? MemoryArray<A> : MemoryArrayOfRank<A, R>) and is_blas_lapack_v<get_value_t<A>>;

  /**
   * @brief BLAS/LAPACK compatible array type with real value type.
   * 
   * @tparam A Array type.
   * @tparam R Optional required rank.
   */
  template <typename A, int R = -1>
  concept BlasArrayReal = BlasArray<A, R> and AnyOf<get_value_t<A>, float, double>;

  /**
   * @brief BLAS/LAPACK compatible array type with complex value type.
   * 
   * @tparam A Array type.
   * @tparam R Optional required rank.
   */
  template <typename A, int R = -1>
  concept BlasArrayCplx = BlasArray<A, R> and AnyOf<get_value_t<A>, std::complex<float>, std::complex<double>>;

  /**
   * @brief BLAS/LAPACK compatible array or conjugate lazy expression type.
   * 
   * @tparam A Array type.
   * @tparam R Optional required rank.
   */
  template <typename A, int R = -1>
  concept BlasArrayOrConj =
     BlasArray<A, R> or ((R == -1 ? Array<A> : ArrayOfRank<A, R>) and is_conj_array_expr<A> and is_blas_lapack_v<get_value_t<A>>);

  /**
   * @brief BLAS/LAPACK compatible array type that has the same value type as the reference array type and a compatible
   * address space.
   *
   * @tparam A Array type.
   * @tparam B Reference array type.
   * @tparam R Optional required rank.
   */
  template <typename A, typename B, int R = -1>
  concept BlasArrayFor = BlasArrayOrConj<B> and BlasArray<A, R> and have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B>;

  /**
   * @brief BLAS/LAPACK compatible array or conjugate lazy expression type that has the same value type as the reference 
   * array type and a compatible address space.
   *
   * @tparam A Array type.
   * @tparam B Reference array type.
   * @tparam R Optional required rank.
   */
  template <typename A, typename B, int R = -1>
  concept BlasArrayOrConjFor =
     BlasArrayOrConj<B> and BlasArrayOrConj<A, R> and have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B>;

  /**
   * @brief BLAS/LAPACK compatible pivot array type that has a compatible address space with the reference array type.
   *
   * @tparam A Array type.
   * @tparam B Reference array type.
   * @tparam R Optional required rank.
   */
  template <typename A, typename B, int R = -1>
  concept PivotArrayFor = BlasArrayOrConj<B> and (R == -1 ? MemoryArray<A> : MemoryArrayOfRank<A, R>)
     and std::is_same_v<get_value_t<A>, int> and mem::have_compatible_addr_space<A, B>;

  /**
   * @brief BLAS/LAPACK compatible array type that has a compatible floating-point value type and address space with the 
   * reference array type.
   *
   * @tparam A Array type.
   * @tparam B Reference array type.
   * @tparam R Optional required rank.
   */
  template <typename A, typename B, int R = -1>
  concept BlasArrayRealFor =
     BlasArrayOrConj<B> and BlasArray<A, R> and std::is_same_v<get_value_t<A>, get_fp_t<B>> and mem::have_compatible_addr_space<A, B>;

  /** @} */

} // namespace nda::blas_lapack

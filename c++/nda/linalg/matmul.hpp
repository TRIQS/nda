// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic matrix-matrix multiplication.
 */

#pragma once

#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../blas/gemm.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../layout/policies.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <type_traits>
#include <utility>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  namespace detail {

    // Generic matrix-matrix multiplication for types not supported by BLAS.
    template <Matrix A, Matrix B, MemoryMatrix C>
      requires(mem::have_host_compatible_addr_space<A, B, C>)
    void gemm_generic(auto alpha, A const &a, B const &b, auto beta, C &&c) { // NOLINT (temporary views are allowed here)
      // check the dimensions of the input/output arrays/views
      auto const [m, k] = a.shape();
      auto const [l, n] = b.shape();
      EXPECTS(k == l);
      EXPECTS(m == c.extent(0));
      EXPECTS(n == c.extent(1));

      // perform the matrix-matrix multiplication
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          c(i, j) = beta * c(i, j);
          for (int r = 0; r < k; ++r) c(i, j) += alpha * a(i, r) * b(r, j);
        }
      }
    }

    // Make compile time checks if blas::gemm can handle the given input matrix. If it can, simply forward the matrix.
    // Otherwise, return a copy with the given value type T, layout policy LP and container policy CP.
    template <typename T, typename LP, typename CP, MemoryMatrix C, Matrix A>
    decltype(auto) get_gemm_matrix(A &&a) {
      using namespace blas_lapack;
      if constexpr (requires { get_array(a); } and std::is_same_v<get_value_t<A>, T>) {
        if constexpr (MemoryMatrix<A>
                      or (is_conj_array_expr<A> and ((has_F_layout<C> and has_C_layout<A>) or (has_C_layout<C> and has_F_layout<A>)))) {
          return std::forward<A>(a);
        } else {
          return matrix<T, LP, CP>{a};
        }
      } else {
        return matrix<T, LP, CP>{a};
      }
    }

    // Make the call to nda::blas::gemm (with copies of the matrices if they are not contiguous).
    template <Matrix A, Matrix B, MemoryMatrix C>
    void make_gemm_call(A const &a, B const &b, C &c) {
      if (blas_lapack::get_array(a).is_contiguous()) {
        if (blas_lapack::get_array(b).is_contiguous()) {
          blas::gemm(1, a, b, 0, c);
        } else {
          blas::gemm(1, a, make_regular(b), 0, c);
        }
      } else {
        if (blas_lapack::get_array(b).is_contiguous()) {
          blas::gemm(1, make_regular(a), b, 0, c);
        } else {
          blas::gemm(1, make_regular(a), make_regular(b), 0, c);
        }
      }
    }

    // Get the layout policy for a given array type.
    template <Array A>
    using get_layout_policy = typename std::remove_cvref_t<decltype(make_regular(std::declval<A>()))>::layout_policy_t;

  } // namespace detail

  /**
   * @brief Compute the matrix-matrix product of two nda::matrix objects.
   *
   * @details This function computes the matrix-matrix product 
   * \f[
   *   \mathbf{C} = \mathbf{A} \mathbf{B} \; ,
   * \f]
   * where \f$ \mathbf{A} \f$, \f$ \mathbf{B} \f$ and \f$ \mathbf{C} \f$ are \f$ m \times k \f$, \f$ k \times n \f$ and
   * \f$ m \times n \f$ matrices, respectively.
   * 
   * The behaviour of this function is similar to nda::blas::gemm, except that it allows
   * - lazy expressions as input,
   * - the value types of the input matrices to be different from each other and
   * - the value types of the input matrices to be different from nda::is_blas_lapack_v.
   *
   * We try to call nda::blas::gemm whenever possible, i.e. when the value type of the result is compatible with
   * nda::is_blas_lapack_v, even if this requires to make copies of the input arrays/views. Otherwise, we perform a very
   * naive and inefficient matrix-matrix multiplication manually.
   *
   * Therefore, if performance is important, users should make sure to pass input arrays/views which are compatible with
   * nda::blas::gemm.
   * 
   * The resulting nda::matrix has
   * - its value type deduced from the multiplication of the value types of the input matrices,
   * - nda::F_layout if both inputs are in F-layout and nda::C_layout otherwise and
   * - its address space set to the nda::mem::common_addr_space of the input matrices.
   * 
   * @note This function might make copies of the input arrays/views. When working on the device memory space, this may 
   * lead to runtime errors if the copying fails.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$ of size \f$ m \times k \f$.
   * @param b Input matrix \f$ \mathbf{B} \f$ of size \f$ k \times n \f$.
   * @return Resulting matrix of the matrix-matrix multiplication of size \f$ m \times n \f$.
   */
  template <Matrix A, Matrix B>
    requires(mem::have_compatible_addr_space<A, B>)
  auto matmul(A &&a, B &&b) { // NOLINT (temporary views are allowed here)
    // get the return type
    using value_t    = decltype(a(0, 0) * b(0, 0));
    using layout_pol = std::conditional_t<get_layout_info<A>.stride_order == get_layout_info<B>.stride_order, detail::get_layout_policy<A>, C_layout>;
    using cont_pol   = heap<mem::common_addr_space<A, B>>;
    using return_t   = matrix<value_t, layout_pol, cont_pol>;

    // result matrix (MSAN complains if it is not initialized)
    auto res = return_t(a.shape()[0], b.shape()[1]);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    res = 0;
#endif
#endif

    // perform matrix-matrix multiplication (if possible we try to call blas::gemm even if this requires making copies)
    if constexpr (is_blas_lapack_v<value_t>) {
      // check at compile time if we need to make a copy of the input matrices
      auto &&a_mat = detail::get_gemm_matrix<value_t, layout_pol, cont_pol, return_t>(a);
      auto &&b_mat = detail::get_gemm_matrix<value_t, layout_pol, cont_pol, return_t>(b);

      // check at runtime if the input matrices are contiguous, make copies if not and call blas::gemm
      detail::make_gemm_call(a_mat, b_mat, res);
    } else {
      detail::gemm_generic(1, a, b, 0, res);
    }
    return res;
  }

  /** @} */

} // namespace nda::linalg

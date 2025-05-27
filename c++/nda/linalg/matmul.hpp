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
#include "../exceptions.hpp"
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

    /// Generic matrix-matrix multiplication for types not supported by BLAS.
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
      if constexpr (requires { blas::get_array(a); } and std::is_same_v<get_value_t<A>, T>) {
        if constexpr (MemoryMatrix<A>
                      or (blas::is_conj_array_expr<A>
                          and ((blas::has_F_layout<C> and blas::has_C_layout<A>) or (blas::has_C_layout<C> and blas::has_F_layout<A>)))) {
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
      auto try_gemm = []<typename A2, typename B2, typename C2>(A2 &&a2, B2 &&b2, C2 &&c2) {
        if constexpr (requires { blas::gemm(1, std::forward<A2>(a2), std::forward<B2>(b2), 0, std::forward<C2>(c2)); }) {
          blas::gemm(1, std::forward<A2>(a2), std::forward<B2>(b2), 0, std::forward<C2>(c2));
        } else {
          NDA_RUNTIME_ERROR << "Error in nda::linalg::matmul: Cannot call blas::gemm with the given input arrys/views.";
        }
      };
      if (blas::get_array(a).is_contiguous()) {
        if (blas::get_array(b).is_contiguous()) {
          try_gemm(a, b, c);
        } else {
          try_gemm(a, nda::make_regular(b), c);
        }
      } else {
        if (blas::get_array(b).is_contiguous()) {
          try_gemm(nda::make_regular(a), b, c);
        } else {
          try_gemm(nda::make_regular(a), nda::make_regular(b), c);
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
   *   \mathrm{op}_A(\mathbf{A}) \mathrm{op}_B(\mathbf{B}) \; ,
   * \f]
   * where \f$ \mathrm{op}_A(\mathbf{A}) \f$ and \f$ \mathrm{op}_B(\mathbf{B}) \f$ are \f$ m \times k \f$ and \f$ k 
   * \times n \f$ matrices, respectively. \f$ \mathrm{op}_i \f$ can be some lazy operation, e.g. nda::conj, nda::sin, 
   * etc.
   *
   * We try to call nda::blas::gemm whenever possible, i.e. when the value type of the result is compatible with
   * nda::is_blas_lapack_v, even if this requires to make copies of the input arrays/views. Otherwise, we perform a very
   * naive and inefficient matrix-matrix multiplication manually.
   *
   * Therefore, if performance is important, users should make sure to pass input arrays/views which are compatible with
   * nda::blas::gemm.
   *
   * @note The layout of the returned matrix depends on the layout of the input matrices. If both input matrices are in
   * nda::F_layout, the returned matrix is also in nda::F_layout. Otherwise, it is in nda::C_layout.
   * 
   * @warning This function might make copies of the input arrays/views. When working on the device memory space, this 
   * may lead to runtime errors if the copying fails.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @param a Input matrix \f$ \mathrm{op}_A(\mathbf{A}) \f$ of size \f$ m \times k \f$.
   * @param b Input matrix \f$ \mathrm{op}_B(\mathbf{B}) \f$ of size \f$ k \times n \f$.
   * @return Resulting matrix of the matrix-matrix multiplication of size \f$ m \times n \f$.
   */
  template <Matrix A, Matrix B>
  auto matmul(A &&a, B &&b) { // NOLINT (temporary views are allowed here)
    // get the return type
    using value_t    = decltype(a(0, 0) * b(0, 0));
    using cont_pol   = heap<mem::common_addr_space<A, B>>;
    using layout_pol = std::conditional_t<get_layout_info<A>.stride_order == get_layout_info<B>.stride_order, detail::get_layout_policy<A>, C_layout>;
    using return_t   = matrix<value_t, layout_pol, cont_pol>;

    // result matrix (MSAN complains if it is not initialized)
    auto res = return_t(a.shape()[0], b.shape()[1]);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    res = 0;
#endif
#endif

    // perform matrix-matrix multiplication (if possible we try to call blas::gemv even if this requires making copies)
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

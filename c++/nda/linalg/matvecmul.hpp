// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic matrix-vector multiplication.
 */

#pragma once

#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../blas/gemv.hpp"
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

    // Generic matrix-vector multiplication for types not supported by BLAS.
    template <Matrix A, Vector X, MemoryVector Y>
      requires(nda::mem::have_host_compatible_addr_space<A, X, Y>)
    void gemv_generic(auto alpha, A const &a, X const &x, auto beta, Y &&y) { // NOLINT (temporary views are allowed here)
      // check the dimensions of the input/output arrays/views
      auto const [m, n] = a.shape();
      EXPECTS(n == x.size());
      EXPECTS(m == y.size());

      // perform the matrix-vector multiplication
      for (int i = 0; i < m; ++i) {
        y(i) = beta * y(i);
        for (int j = 0; j < n; ++j) y(i) += alpha * a(i, j) * x(j);
      }
    }

    // Make compile time checks if blas::gemv can handle the given input vector. If it can, simply forward the vector.
    // Otherwise, return a copy with the given value type T and container policy CP.
    template <typename T, typename CP, Vector X>
    decltype(auto) get_gemv_vector(X &&x) {
      if constexpr (std::is_same_v<get_value_t<X>, T> and MemoryVector<X>) {
        return std::forward<X>(x);
      } else {
        return vector<T, CP>{x};
      }
    }

    // Make compile time checks if blas::gemv can handle the given input matrix. If it can, simply forward the matrix.
    // Otherwise, return a copy with the given value type T and container policy CP.
    template <typename T, typename CP, Matrix A>
    decltype(auto) get_gemv_matrix(A &&a) {
      if constexpr (requires { blas::get_array(a); } and std::is_same_v<get_value_t<A>, T>) {
        if constexpr (MemoryMatrix<A> or (blas::is_conj_array_expr<A> and blas::has_C_layout<A>)) {
          return std::forward<A>(a);
        } else {
          return matrix<T, C_layout, CP>{a};
        }
      } else {
        return matrix<T, C_layout, CP>{a};
      }
    }

    // Make the call to nda::blas::gemv with a copy of the matrix if it is not contiguous.
    template <Matrix A, Vector X, MemoryVector Y>
    void make_gemv_call(A const &a, X const &x, Y &y) {
      auto try_gemv = []<typename A2, typename X2, typename Y2>(A2 &&a2, X2 &&x2, Y2 &&y2) {
        if constexpr (requires { blas::gemv(1, std::forward<A2>(a2), std::forward<X2>(x2), 0, std::forward<Y2>(y2)); }) {
          blas::gemv(1, std::forward<A2>(a2), std::forward<X2>(x2), 0, std::forward<Y2>(y2));
        } else {
          NDA_RUNTIME_ERROR << "Error in nda::linalg::matvecmul: Cannot call blas::gemv with the given input arrys/views.";
        }
      };
      if (blas::get_array(a).is_contiguous()) {
        try_gemv(a, x, y);
      } else {
        try_gemv(nda::make_regular(a), x, y);
      }
    }

  } // namespace detail

  /**
   * @brief Compute the matrix-vector product of an nda::matrix and an nda::vector object.
   *
   * @details This function computes the matrix-vector product 
   * \f[ 
   *   \mathrm{op}_A(\mathbf{A}) \mathrm{op}_x(\mathbf{x}) \; ,
   * \f]
   * where \f$ \mathrm{op}_A(\mathbf{A}) \f$ is an \f$ m \times n \f$ matrix and \f$ \mathrm{op}_x(\mathbf{x}) \f$ is a 
   * vector of size \f$ n \f$. \f$ \mathrm{op}_i \f$ can be some lazy operation, e.g. nda::conj, nda::sin, etc.
   *
   * We try to call nda::blas::gemv whenever possible, i.e. when the value type of the result is compatible with
   * nda::is_blas_lapack_v, even if this requires to make copies of the input arrays/views. Otherwise, we perform a very
   * naive and inefficient matrix-vector multiplication manually.
   *
   * Therefore, if performance is important, users should make sure to pass input arrays/views which are compatible with
   * nda::blas::gemv.
   * 
   * @warning This function might make copies of the input arrays/views. When working on the device memory space, this 
   * may lead to runtime errors if the copying fails.
   *
   * @tparam A nda::Matrix type.
   * @tparam X nda::Vector type.
   * @param a Input matrix \f$ \mathrm{op}_A(\mathbf{A}) \f$ of size \f$ m \times n \f$.
   * @param x Input vector \f$ \mathrm{op}_x(\mathbf{x}) \f$ of size \f$ n \f$.
   * @return Resulting vector of the matrix-vector multiplication of size \f$ m \f$.
   */
  template <Matrix A, Vector X>
    requires(mem::have_compatible_addr_space<A, X>)
  auto matvecmul(A const &a, X const &x) {
    // get the return type
    using value_t  = decltype(a(0, 0) * x(0));
    using cont_pol = heap<mem::common_addr_space<A, X>>;
    using return_t = vector<value_t, cont_pol>;

    // result vector (MSAN complains if it is not initialized)
    auto res = return_t(a.shape()[0]);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    res = 0;
#endif
#endif

    // perform matrix-vector multiplication (if possible we try to call blas::gemv even if this requires making copies)
    if constexpr (is_blas_lapack_v<value_t>) {
      auto &&a_mat = detail::get_gemv_matrix<value_t, cont_pol>(a);
      auto &&x_vec = detail::get_gemv_vector<value_t, cont_pol>(x);

      // check at runtime if the input matrix is contiguous, make copies if not and call blas::gemv
      detail::make_gemv_call(a_mat, x_vec, res);
    } else {
      detail::gemv_generic(1, a, x, 0, res);
    }
    return res;
  }

  /** @} */

} // namespace nda::linalg

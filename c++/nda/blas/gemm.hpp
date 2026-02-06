// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS/cuBLAS `gemm` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../device.hpp"
#include "../layout_transforms.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <utility>

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Interface to the BLAS/cuBLAS `gemm` routine.
   *
   * @details This function performs the matrix-matrix operation
   * \f[
   *   \mathbf{C} \leftarrow \alpha \mathbf{A} \mathbf{B} + \beta \mathbf{C} \;,
   * \f]
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, and \f$ \mathbf{A} \f$, \f$ \mathbf{B} \f$ and \f$ \mathbf{C}
   * \f$ are matrices of size \f$ m \times k \f$, \f$ k \times n \f$ and \f$ m \times n \f$, respectively.
   *
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuBLAS implementation is used.
   * 
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are allowed to be lazy conjugate expressions (see 
   * nda::blas_lapack::is_conj_array_expr). In this case, they are required to have the opposite memory layout of \f$ 
   * \mathbf{C} \f$ (see nda::C_layout vs nda::F_layout).
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj<2> type.
   * @tparam B nda::blas_lapack::BlasArrayOrConjFor<A, 2> type.
   * @tparam C nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input matrix \f$ \mathbf{A} \f$ of size \f$ m \times k \f$.
   * @param b Input matrix \f$ \mathbf{B})\f$ of size \f$ k \times n \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param c Input/Output matrix \f$ \mathbf{C} \f$ of size \f$ m \times n \f$.
   */
  template <BlasArrayOrConj<2> A, BlasArrayOrConjFor<A, 2> B, BlasArrayFor<A, 2> C>
  void gemm(get_value_t<A> alpha, A const &a, B const &b, get_value_t<A> beta, C &&c) {
    // if C is in C-layout, compute the transpose of the product
    if constexpr (has_C_layout<C>) {
      gemm(alpha, transpose(b), transpose(a), beta, transpose(std::forward<C>(c)));
    } else {
      // get underlying matrix in case it is given as a conjugate expression
      auto &mat_a = get_array(a);
      auto &mat_b = get_array(b);

      // check the dimensions of the input/output arrays/views
      auto const [m, k] = mat_a.shape();
      auto const [l, n] = mat_b.shape();
      EXPECTS(k == l);
      EXPECTS(m == c.extent(0));
      EXPECTS(n == c.extent(1));

      // arrays/views must be BLAS compatible
      EXPECTS(mat_a.indexmap().min_stride() == 1);
      EXPECTS(mat_b.indexmap().min_stride() == 1);
      EXPECTS(c.indexmap().min_stride() == 1);

      // perform the actual library call
      if constexpr (mem::have_device_compatible_addr_space<A, B, C>) {
        device::gemm(get_op<A>, get_op<B>, m, n, k, alpha, mat_a.data(), get_ld(mat_a), mat_b.data(), get_ld(mat_b), beta, c.data(), get_ld(c));
      } else {
        f77::gemm(get_op<A>, get_op<B>, m, n, k, alpha, mat_a.data(), get_ld(mat_a), mat_b.data(), get_ld(mat_b), beta, c.data(), get_ld(c));
      }
    }
  }

  /** @} */

} // namespace nda::blas

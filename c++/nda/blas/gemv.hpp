// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS/cuBLAS `gemv` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../device.hpp"
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
   * @brief Interface to the BLAS/cuBLAS `gemv` routine.
   *
   * @details This function performs the matrix-vector operation
   * \f[
   *   \mathbf{y} \leftarrow \alpha \mathbf{A} \mathbf{x} + \beta \mathbf{y} \; ,
   * \f]
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, \f$ \mathbf{A} \f$ is an \f$ m \times n \f$ matrix and \f$
   * \mathbf{x} \f$ and \f$ \mathbf{y} \f$ are vectors of sizes \f$ n \f$ and \f$ m \f$, respectively.
   * 
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuBLAS implementation is used.
   * 
   * @note \f$ \mathbf{A} \f$ is allowed to be a lazy conjugate expression (see nda::blas_lapack::is_conj_array_expr), 
   * in which case it is required to be in nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj<2> type.
   * @tparam X nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam Y nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input matrix \f$ \mathbf{A} \f$ of size \f$ m \times n \f$.
   * @param x Input vector \f$ \mathbf{x} \f$ of size \f$ n \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param y Input/Output vector \f$ \mathbf{y} \f$ of size \f$ m \f$.
   */
  template <BlasArrayOrConj<2> A, BlasArrayFor<A, 1> X, BlasArrayFor<A, 1> Y>
  void gemv(get_value_t<A> alpha, A const &a, X const &x, get_value_t<A> beta, Y &&y) { // NOLINT (temporary views are allowed here)
    // get the underlying matrix in case it is given as a conjugate expression
    auto &mat = get_array(a);

    // check the dimensions of the input/output arrays/views
    auto [m, n] = mat.shape();
    EXPECTS(m == y.size());
    EXPECTS(n == x.size());

    // arrays/views must be BLAS compatible
    EXPECTS(mat.indexmap().min_stride() == 1);

    // swap axis for the transpose case
    if constexpr (has_C_layout<A>) std::swap(m, n);

    // perform actual library call
    if constexpr (mem::have_device_compatible_addr_space<A, X, Y>) {
      device::gemv(get_op<A>, m, n, alpha, mat.data(), get_ld(mat), x.data(), x.indexmap().strides()[0], beta, y.data(), y.indexmap().strides()[0]);
    } else {
      f77::gemv(get_op<A>, m, n, alpha, mat.data(), get_ld(mat), x.data(), x.indexmap().strides()[0], beta, y.data(), y.indexmap().strides()[0]);
    }
  }

  /** @} */

} // namespace nda::blas

// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS/cuBLAS `ger`, `geru` and `gerc` routines.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../device.hpp"
#include "../layout_transforms.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Interface to the BLAS/cuBLAS `ger` and `geru` routine.
   *
   * @details This function performs the rank 1 operation
   * \f[
   *   \mathbf{A} \leftarrow \alpha \mathbf{x} \mathbf{y}^T + \mathbf{A} \; ,
   * \f]
   * where \f$ \alpha \f$ is a scalar, \f$ \mathbf{x} \f$ is an \f$ m \f$ element vector, \f$ \mathbf{y} \f$ is an \f$ n
   * \f$ element vector and \f$ \mathbf{A} \f$ is an \f$ m \times n \f$ matrix.
   * 
   * The vector \f$ \mathbf{y} \f$ is never conjugated. For complex vectors, it calls `geru`. Use nda::blas::gerc to 
   * conjugate \f$ \mathbf{y} \f$.
   * 
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuBLAS implementation is used.
   *
   * @tparam X nda::blas_lapack::BlasArray<1> type.
   * @tparam Y nda::blas_lapack::BlasArrayFor<X, 1> type.
   * @tparam A nda::blas_lapack::BlasArrayFor<X, 2> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param x Input vector \f$ \mathbf{x} \f$ of size \f$ m \f$.
   * @param y Input vector \f$ \mathbf{y} \f$  of size \f$ n \f$.
   * @param a Input/Output matrix \f$ \mathbf{A} \f$  of size \f$ m \times n \f$ to which the outer product is added.
   */
  template <BlasArray<1> X, BlasArrayFor<X, 1> Y, BlasArrayFor<X, 2> A>
  void ger(get_value_t<X> alpha, X const &x, Y const &y, A &&a) { // NOLINT (temporary views are allowed here)
    // for C-layout arrays/views, call ger with the transpose and swap x and y
    if constexpr (has_C_layout<A>) {
      ger(alpha, y, x, transpose(a));
      return;
    }

    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == x.size());
    EXPECTS(n == y.size());

    // arrays/views must be BLAS compatible
    EXPECTS(a.indexmap().min_stride() == 1);

    // perform actual library call
    if constexpr (mem::have_device_compatible_addr_space<X, Y, A>) {
      device::ger(m, n, alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], a.data(), get_ld(a));
    } else {
      f77::ger(m, n, alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], a.data(), get_ld(a));
    }
  }

  /**
   * @brief Interface to the BLAS/cuBLAS `gerc` routine.
   *
   * @details This function performs the rank 1 operation
   * \f[
   *   \mathbf{A} \leftarrow \alpha \mathbf{x} \mathbf{y}^H + \mathbf{A} \; ,
   * \f]
   * where \f$ \alpha \f$ is a scalar, \f$ \mathbf{x} \f$ is an \f$ m \f$ element vector, \f$ \mathbf{y} \f$ is an \f$ n
   * \f$ element vector and \f$ \mathbf{A} \f$ is an \f$ m \times n \f$ matrix.
   *
   * For real vectors/matrices, it calls nda::blas::ger.
   * 
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuBLAS implementation is used.
   *
   * @tparam X nda::blas_lapack::BlasArray<1> type.
   * @tparam Y nda::blas_lapack::BlasArrayFor<X, 1> type.
   * @tparam A nda::blas_lapack::BlasArrayFor<X, 2> type with nda::F_layout.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param x Input vector \f$ \mathbf{x} \f$ of size \f$ m \f$.
   * @param y Input vector \f$ \mathbf{y} \f$  of size \f$ n \f$.
   * @param a Input/Output matrix \f$ \mathbf{A} \f$  of size \f$ m \times n \f$ to which the outer product is added.
   */
  template <BlasArray<1> X, BlasArrayFor<X, 1> Y, BlasArrayFor<X, 2> A>
    requires(has_F_layout<A>)
  void gerc(get_value_t<X> alpha, X const &x, Y const &y, A &&a) { // NOLINT (temporary views are allowed here)
    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == x.size());
    EXPECTS(n == y.size());

    // arrays/views must be BLAS compatible
    EXPECTS(a.indexmap().min_stride() == 1);

    // perform actual library call
    if constexpr (!is_complex_v<get_value_t<X>>) {
      return ger(alpha, x, y, a);
    } else if constexpr (mem::have_device_compatible_addr_space<X, Y, A>) {
      device::gerc(m, n, alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], a.data(), get_ld(a));
    } else {
      f77::gerc(m, n, alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], a.data(), get_ld(a));
    }
  }

  /** @} */

} // namespace nda::blas

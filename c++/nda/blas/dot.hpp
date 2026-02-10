// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS/cuBLAS `dot`, `dotu` and `dotc` routines.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../device.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Interface to the BLAS/cuBLAS `%dot` and `dotu` routines.
   *
   * @details This function forms the dot product of two vectors. It calculates \f$ \mathbf{x}^T \mathbf{y} \f$.
   * 
   * The first argument is never conjugated. For complex vectors, it calls `dotu`. Use nda::blas::dotc to conjugate
   * the first argument.
   * 
   * If the input vectors satisfy nda::mem::have_device_compatible_addr_space, the cuBLAS implementation is used.
   *
   * @tparam X nda::blas_lapack::BlasArray<1> type.
   * @tparam Y nda::blas_lapack::BlasArrayFor<X, 1> type.
   * @param x Input vector \f$ \mathbf{x} \f$.
   * @param y Input vector \f$ \mathbf{y} \f$.
   * @return Result of \f$ \mathbf{x}^T \mathbf{y} \f$.
   */
  template <BlasArray<1> X, BlasArrayFor<X, 1> Y>
  auto dot(X const &x, Y const &y) {
    // check the dimensions of the input/output arrays/views
    EXPECTS(x.size() == y.size());

    // perform actual library call
    if constexpr (mem::have_device_compatible_addr_space<X, Y>) {
      return device::dot(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
    } else {
      return f77::dot(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
    }
  }

  /**
   * @brief Interface to the BLAS/cuBLAS `%dotc` routine.
   *
   * @details This function forms the dot product of two vectors. It calculates \f$ \mathbf{x}^H \mathbf{y} \f$.
   * 
   * For real vectors, it calls nda::blas::dot and returns a real result.
   * 
   * If the input vectors satisfy nda::mem::have_device_compatible_addr_space, the cuBLAS implementation is used.
   *
   * @tparam X nda::blas_lapack::BlasArray<1> type.
   * @tparam Y nda::blas_lapack::BlasArrayFor<X, 1> type.
   * @param x Input vector \f$ \mathbf{x} \f$.
   * @param y Input vector \f$ \mathbf{y} \f$.
   * @return Result of \f$ \mathbf{x}^H \mathbf{y} \f$.
   */
  template <BlasArray<1> X, BlasArrayFor<X, 1> Y>
  auto dotc(X const &x, Y const &y) {
    // check the dimensions of the input/output arrays/views
    EXPECTS(x.size() == y.size());

    // perform actual library call
    if constexpr (!is_complex_v<get_value_t<X>>) {
      return dot(x, y);
    } else if constexpr (mem::have_device_compatible_addr_space<X, Y>) {
      return device::dotc(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
    } else {
      return f77::dotc(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
    }
  }

  /** @} */

} // namespace nda::blas

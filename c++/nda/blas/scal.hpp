// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS/cuBLAS `scal` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../device.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::blas {

  /**
   * @ingroup linalg_blas
   * @brief Interface to the BLAS/cuBLAS `%scal` routine.
   *
   * @details Scales a vector by a constant. This function calculates \f$ \mathbf{x} \leftarrow \alpha \mathbf{x} \f$,
   * where \f$ \alpha \f$ is a scalar constant and \f$ \mathbf{x} \f$ is a vector.
   * 
   * If the input vector satisfies nda::mem::have_device_compatible_addr_space, the cuBLAS implementation is used.
   *
   * @tparam X nda::blas_lapack::BlasArray<1> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param x Input/Output vector \f$ \mathbf{x} \f$ to be scaled.
   */
  template <BlasArray<1> X>
  void scal(get_value_t<X> alpha, X &&x) { // NOLINT (temporary views are allowed here)
    if constexpr (mem::have_device_compatible_addr_space<X>) {
      device::scal(x.size(), alpha, x.data(), x.indexmap().strides()[0]);
    } else {
      f77::scal(x.size(), alpha, x.data(), x.indexmap().strides()[0]);
    }
  }

} // namespace nda::blas

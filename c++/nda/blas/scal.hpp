// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS `scal` routine.
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
   * @brief Interface to the BLAS `scal` routine.
   *
   * @details Scales a vector by a constant. This function calculates
   * \f[
   *   \mathbf{x} \leftarrow \alpha \mathbf{x} ;,
   * \f]
   * where \f$ \alpha \f$ is a scalar constant and \f$ \mathbf{x} \f$ is a vector.
   *
   * @tparam X nda::MemoryVector or a conjugate array expression.
   * @param alpha Input scalar.
   * @param x Input/Output vector to be scaled.
   */
  template <typename X>
    requires(MemoryVector<X> or is_conj_array_expr<X>)
  void scal(get_value_t<X> alpha, X &&x) { // NOLINT (temporary views are allowed here)
    static_assert(is_blas_lapack_v<get_value_t<X>>, "Error in nda::blas::scal: Value type of vector is incompatible with blas");

    if constexpr (mem::on_host<X>) {
      f77::scal(x.size(), alpha, x.data(), x.indexmap().strides()[0]);
    } else {
#if defined(NDA_HAVE_DEVICE)
      device::scal(x.size(), alpha, x.data(), x.indexmap().strides()[0]);
#else
      compile_error_no_gpu();
#endif
    }
  }

} // namespace nda::blas

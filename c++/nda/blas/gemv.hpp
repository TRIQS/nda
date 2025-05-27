// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS `gemv` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

#include <tuple>
#include <utility>

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Interface to the BLAS `gemv` routine.
   *
   * @details This function performs one of the matrix-vector operations
   *
   * - \f$ \mathbf{y} \leftarrow \alpha \mathbf{A} \mathbf{x} + \beta \mathbf{y} \f$,
   * - \f$ \mathbf{y} \leftarrow \alpha \mathbf{A}^* \mathbf{x} + \beta \mathbf{y} \f$ (only if \f$ \mathbf{A} \f$ is
   * in nda::C_layout),
   *
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, \f$ \mathbf{A} \f$ is an \f$ m \times n \f$ matrix and \f$
   * \mathbf{x} \f$ and \f$ \mathbf{y} \f$ are vectors of sizes \f$ n \f$ and \f$ m \f$, respectively.
   *
   * @tparam A nda::Matrix type.
   * @tparam X nda::MemoryVector type.
   * @tparam Y nda::MemoryVector type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input matrix \f$ \mathbf{A} \f$ of size \f$ m \times n \f$.
   * @param x Input vector \f$ \mathbf{x} \f$ of size \f$ n \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param y Input/Output vector \f$ \mathbf{y} \f$ of size \f$ m \f$.
   */
  template <Matrix A, MemoryVector X, MemoryVector Y>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>)
             and have_same_value_type_v<A, X, Y> and mem::have_compatible_addr_space<A, X, Y> and is_blas_lapack_v<get_value_t<A>>)
  void gemv(get_value_t<A> alpha, A const &a, X const &x, get_value_t<A> beta, Y &&y) { // NOLINT (temporary views are allowed here)
    // get the underlying matrix in case it is given as a conjugate expression
    auto &mat = get_array(a);

    // check the dimensions of the input/output arrays/views
    auto [m, n] = mat.shape();
    EXPECTS(m == y.size());
    EXPECTS(n == x.size());

    // arrays/views must be BLAS compatible
    EXPECTS(mat.indexmap().min_stride() == 1);

    // check for conjugate lazy expressions and C-layouts
    char op_a = get_op<is_conj_array_expr<A>, has_C_layout<A>>;
    if constexpr (has_C_layout<A>) std::swap(m, n);

    // perform actual library call
    if constexpr (mem::have_device_compatible_addr_space<A, X, Y>) {
#if defined(NDA_HAVE_DEVICE)
      device::gemv(op_a, m, n, alpha, mat.data(), get_ld(mat), x.data(), x.indexmap().strides()[0], beta, y.data(), y.indexmap().strides()[0]);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::gemv(op_a, m, n, alpha, mat.data(), get_ld(mat), x.data(), x.indexmap().strides()[0], beta, y.data(), y.indexmap().strides()[0]);
    }
  }

  /** @} */

} // namespace nda::blas

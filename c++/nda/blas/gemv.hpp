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
#endif

#include <tuple>

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Generic nda::blas::gemv implementation for types not supported by BLAS/LAPACK.
   *
   * @tparam A Some matrix type.
   * @tparam X Some vector type.
   * @tparam Y Some vector type.
   * @param alpha Input scalar.
   * @param a Input matrix of size m-by-n.
   * @param x Input vector of size n.
   * @param beta Input scalar.
   * @param y Input/Output vector of size m.
   */
  template <typename A, typename X, typename Y>
  void gemv_generic(get_value_t<A> alpha, A const &a, X const &x, get_value_t<A> beta, Y &&y) { // NOLINT (temporary views are allowed here)
    EXPECTS(a.extent(1) == x.extent(0));
    EXPECTS(a.extent(0) == y.extent(0));

    if (beta == get_value_t<A>(0.0)) {
      y = 0 * alpha;
    } else {
      y *= beta;
    }

    for (int i = 0; i < a.extent(0); ++i) {
      for (int k = 0; k < a.extent(1); ++k) y(i) += alpha * a(i, k) * x(k);
    }
  }

  /**
   * @brief Interface to the BLAS `gemv` routine.
   *
   * @details This function performs one of the matrix-vector operations
   *
   * - \f$ \mathbf{y} \leftarrow \alpha \mathbf{A} \mathbf{x} + \beta \mathbf{y} \f$,
   * - \f$ \mathbf{y} \leftarrow \alpha \mathbf{A}^T \mathbf{x} + \beta \mathbf{y} \f$,
   * - \f$ \mathbf{y} \leftarrow \alpha \mathbf{A}^H \mathbf{x} + \beta \mathbf{y} \f$,
   *
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, \f$ \mathbf{x} \f$ and \f$ \mathbf{y} \f$ are vectors and
   * \f$ \mathbf{A} \f$ is an m-by-n matrix.
   *
   * @tparam A nda::Matrix type.
   * @tparam X nda::MemoryVector type.
   * @tparam Y nda::MemoryVector type.
   * @param alpha Input scalar.
   * @param a Input matrix of size m-by-n.
   * @param x Input vector of size n.
   * @param beta Input scalar.
   * @param y Input/Output vector of size m.
   */
  template <Matrix A, MemoryVector X, MemoryVector Y>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>) and have_same_value_type_v<A, X, Y> and is_blas_lapack_v<get_value_t<A>>)
  void gemv(get_value_t<A> alpha, A const &a, X const &x, get_value_t<A> beta, Y &&y) { // NOLINT (temporary views are allowed here)
    // get underlying matrix in case it is given as a lazy expression
    auto to_mat = []<Matrix Z>(Z const &z) -> decltype(auto) {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &mat = to_mat(a);

    // compile-time checks
    using mat_type = decltype(mat);
    static_assert(mem::have_compatible_addr_space<mat_type, X, Y>);

    // runtime checks
    EXPECTS(mat.extent(1) == x.extent(0));
    EXPECTS(mat.extent(0) == y.extent(0));
    EXPECTS(mat.indexmap().min_stride() == 1);
    EXPECTS(x.indexmap().min_stride() == 1);
    EXPECTS(y.indexmap().min_stride() == 1);

    // gather parameters for gemv call
    static constexpr bool conj_A = is_conj_array_expr<A>;
    char op_a                    = get_op<conj_A, /* transpose = */ !has_F_layout<mat_type>>;
    auto [m, n]                  = mat.shape();
    if constexpr (has_C_layout<mat_type>) std::swap(m, n);

    if constexpr (mem::have_device_compatible_addr_space<mat_type, X, Y>) {
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

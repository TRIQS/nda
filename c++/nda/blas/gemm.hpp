// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS `gemm` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../layout_transforms.hpp"
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
   * @brief Interface to the BLAS `gemm` routine.
   *
   * @details This function performs one of the matrix-matrix operations
   * \f[
   *   \mathbf{C} \leftarrow \alpha \mathrm{op}_A(\mathbf{A}) \mathrm{op}_B(\mathbf{B}) + \beta \mathbf{C} \;,
   * \f]
   * where \f$ \mathrm{op}(\mathbf{X}) \f$ is one of
   * - \f$ \mathrm{op}(\mathbf{X}) = \mathbf{X} \f$ or,
   * - \f$ \mathrm{op}(\mathbf{X}) = \mathbf{X}^* \f$ (only if \f$ \mathbf{X} \f$ is in nda::C_layout).
   *
   * Here, \f$ \alpha \f$ and \f$ \beta \f$ are scalars, and \f$ \mathbf{A} \f$, \f$ \mathbf{B} \f$ and \f$ \mathbf{C}
   * \f$ are matrices of size \f$ m \times k \f$, \f$ k \times n \f$ and \f$ m \times n \f$, respectively.
   *
   * @note If matrix \f$ \mathbf{C} \f$ is in nda::C_layout, we transpose both \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ 
   * and swap their order.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @tparam C nda::MemoryMatrix type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input matrix \f$ \mathrm{op}_A(\mathbf{A}) \f$ of size \f$ m \times k \f$.
   * @param b Input matrix \f$ \mathrm{op}_B(\mathbf{B}) \f$ of size \f$ k \times n \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param c Input/Output matrix \f$ \mathbf{C} \f$ of size \f$ m \times n \f$.
   */
  template <Matrix A, Matrix B, MemoryMatrix C>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>) and (MemoryMatrix<B> or is_conj_array_expr<B>)
             and have_same_value_type_v<A, B, C> and mem::have_compatible_addr_space<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
  void gemm(get_value_t<A> alpha, A const &a, B const &b, get_value_t<A> beta, C &&c) {
    // if C is in C-layout, compute the transpose of the product in Fortran order
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

      // check for conjugate lazy expressions and C-layouts
      char op_a = get_op<is_conj_array_expr<A>, has_C_layout<A>>;
      char op_b = get_op<is_conj_array_expr<B>, has_C_layout<B>>;

      // perform the actual library call
      if constexpr (mem::have_device_compatible_addr_space<A, B, C>) {
#if defined(NDA_HAVE_DEVICE)
        device::gemm(op_a, op_b, m, n, k, alpha, mat_a.data(), get_ld(mat_a), mat_b.data(), get_ld(mat_b), beta, c.data(), get_ld(c));
#else
        compile_error_no_gpu();
#endif
      } else {
        f77::gemm(op_a, op_b, m, n, k, alpha, mat_a.data(), get_ld(mat_a), mat_b.data(), get_ld(mat_b), beta, c.data(), get_ld(c));
      }
    }
  }

  /** @} */

} // namespace nda::blas

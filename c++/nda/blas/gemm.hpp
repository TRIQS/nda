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
#endif

#include <tuple>
#include <utility>

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Generic nda::blas::gemm implementation for types not supported by BLAS/LAPACK.
   *
   * @tparam A Some matrix type.
   * @tparam B Some matrix type.
   * @tparam C Some matrix type.
   * @param alpha Input scalar.
   * @param a Input matrix of size m-by-k.
   * @param b Input matrix of size k-by-n.
   * @param beta Input scalar.
   * @param c Input/Output matrix of size m-by-n.
   */
  template <Matrix A, Matrix B, MemoryMatrix C>
  void gemm_generic(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta,
                    C &&c) { // NOLINT (temporary views are allowed here)
    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    EXPECTS(b.extent(1) == c.extent(1));
    for (int i = 0; i < a.extent(0); ++i) {
      for (int j = 0; j < b.extent(1); ++j) {
        c(i, j) = beta * c(i, j);
        for (int k = 0; k < a.extent(1); ++k) c(i, j) += alpha * a(i, k) * b(k, j);
      }
    }
  }

  /**
   * @brief Interface to the BLAS `gemm` routine.
   *
   * @details This function performs one of the matrix-matrix operations
   * \f[
   *   \mathbf{C} \leftarrow \alpha \mathrm{op}(\mathbf{A}) \mathrm{op}(\mathbf{B}) + \beta \mathbf{C} \;,
   * \f]
   * where \f$ \mathrm{op}(\mathbf{X}) \f$ is one of
   *
   * - \f$ \mathrm{op}(\mathbf{X}) = \mathbf{X} \f$,
   * - \f$ \mathrm{op}(\mathbf{X}) = \mathbf{X}^T \f$ or
   * - \f$ \mathrm{op}(\mathbf{X}) = \mathbf{X}^H \f$.
   *
   * Here, \f$ \alpha \f$ and \f$ \beta \f$ are scalars, and \f$ \mathbf{A} \f$, \f$ \mathbf{B} \f$ are matrices with
   * \f$ \mathrm{op}(\mathbf{A}) \f$ is an m-by-k matrix, \f$ \mathrm{op}(\mathbf{B}) \f$ is a k-by-n matrix and
   * \f$ \mathrm{op}(\mathbf{C}) \f$ is an m-by-n matrix.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @tparam C nda::MemoryMatrix type.
   * @param alpha Input scalar.
   * @param a Input matrix of size m-by-k.
   * @param b Input matrix of size k-by-n.
   * @param beta Input scalar.
   * @param c Input/Output matrix of size m-by-n.
   */
  template <Matrix A, Matrix B, MemoryMatrix C>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>) and (MemoryMatrix<B> or is_conj_array_expr<B>)
             and have_same_value_type_v<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
  void gemm(get_value_t<A> alpha, A const &a, B const &b, get_value_t<A> beta, C &&c) {
    // get underlying matrix in case it is given as a lazy expression
    auto to_mat = []<typename Z>(Z const &z) -> auto & {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &mat_a = to_mat(a);
    auto &mat_b = to_mat(b);

    // compile-time checks
    using mat_a_type = decltype(mat_a);
    using mat_b_type = decltype(mat_b);
    static_assert(mem::have_compatible_addr_space<mat_a_type, mat_b_type, C>, "Error in nda::blas::gemm: Incompatible memory address spaces");

    // runtime checks
    EXPECTS(mat_a.extent(1) == mat_b.extent(0));
    EXPECTS(mat_a.extent(0) == c.extent(0));
    EXPECTS(mat_b.extent(1) == c.extent(1));
    EXPECTS(mat_a.indexmap().min_stride() == 1);
    EXPECTS(mat_b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    // c is in C order: compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      gemm(alpha, transpose(b), transpose(a), beta, transpose(std::forward<C>(c)));
    } else { // c is in Fortran order
      static constexpr bool conj_A = is_conj_array_expr<A>;
      static constexpr bool conj_B = is_conj_array_expr<B>;
      char op_a                    = get_op<conj_A, /* transpose = */ has_C_layout<mat_a_type>>;
      char op_b                    = get_op<conj_B, /* transpose = */ has_C_layout<mat_b_type>>;
      auto [m, k]                  = mat_a.shape();
      auto n                       = mat_b.extent(1);

      if constexpr (mem::have_device_compatible_addr_space<mat_a_type, mat_b_type, C>) {
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

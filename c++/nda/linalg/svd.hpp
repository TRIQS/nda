// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to compute the singular value decomposition of a matrix.
 */

#pragma once

#include "../basic_array.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/gesvd.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda::linalg {

  /**
   * @addtogroup linalg_decompositions
   * @{
   */

  /**
   * @brief Compute the singular value decomposition (SVD) of a matrix in place.
   *
   * @details The function computes the SVD of a given \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^H \; ,
   * \f]
   * where \f$ \mathbf{U} \f$ is a unitary \f$ m \times m \f$ matrix, \f$ \mathbf{V} \f$ is a unitary \f$ n \times n \f$
   * matrix and \f$ \mathbf{S} \f$ is an \f$ m \times n \f$ matrix with non-negative real numbers on the diagonal.
   *
   * It first constructs the output vector \f$ \mathbf{s} \f$, which contains the singular values, and the output
   * matrices \f$ \mathbf{U} \f$ and \f$ \mathbf{V}^H \f$. It then calls nda::lapack::gesvd to compute the SVD.
   *
   * The resulting matrices \f$ \mathbf{U} \f$ and \f$ \mathbf{V}^H \f$ have the same memory layout, address space and
   * value type as \f$ \mathbf{A} \f$. The vector \f$ \mathbf{s} \f$ has the same address space as \f$ \mathbf{A} \f$
   * and its value type is given by `nda::get_fp_t<A>`.
   *
   * An exception is thrown, if the LAPACK call returns a non-zero value.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the contents of
   * \f$ \mathbf{A} \f$ are destroyed.
   * @return `std::tuple` containing \f$ \mathbf{U} \f$, \f$ \mathbf{s} \f$ and \f$ \mathbf{V}^H \f$.
   */
  template <blas_lapack::BlasArray<2> A>
  auto svd_in_place(A &&a) { // NOLINT (temporary views are allowed here)
    using layout_policy       = nda::detail::layout_to_policy<typename std::remove_cvref_t<A>::layout_t>::type;
    constexpr auto addr_space = mem::get_addr_space<A>;

    // vector s and matrices U and V^H
    auto const [m, n] = a.shape();
    auto s            = vector<get_fp_t<A>, heap<addr_space>>(std::min(m, n));
    auto U            = matrix<get_value_t<A>, layout_policy, heap<addr_space>>(m, m);
    auto VH           = matrix<get_value_t<A>, layout_policy, heap<addr_space>>(n, n);

    // call lapack gesvd
    int info = lapack::gesvd(a, s, U, VH);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::svd_in_place: gesvd returned a non-zero value: info = " << info;

    return std::make_tuple(U, s, VH);
  }

  /**
   * @brief Compute the singular value decomposition (SVD) of a matrix.
   *
   * @details It calls nda::linalg::svd_in_place with a copy of the input matrix \f$ \mathbf{A} \f$.
   *
   * This function makes copies of the input arrays/views. When working on the device memory space, this may lead to
   * runtime errors if the copying fails.
   *
   * @note \f$ \mathbf{A} \f$ is required to have a value type that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$.
   * @return `std::tuple` containing \f$ \mathbf{U} \f$, \f$ \mathbf{s} \f$ and \f$ \mathbf{V}^H \f$.
   */
  template <Matrix A>
    requires(is_blas_lapack_v<get_value_t<A>>)
  auto svd(A const &a) { // NOLINT (temporary views are allowed here)
    return svd_in_place(basic_array{a});
  }

  /** @} */

} // namespace nda::linalg

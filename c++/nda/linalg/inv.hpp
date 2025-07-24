// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to compute the inverse of a matrix.
 */

#pragma once

#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../exceptions.hpp"
#include "../lapack/getrf.hpp"
#include "../lapack/getri.hpp"
#include "../lapack/getrs.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../matrix_functions.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <utility>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Compute the inverse of a \f$ 1 \times 1 \f$ matrix \f$ \mathbf{M} \f$.
   * 
   * @details It throws an exception if the matrix is not invertible (i.e. if \f$ \det(\mathbf{M}) = 0 \f$).
   *
   * @note The inversion is performed in place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m Input/output matrix. On entry, the matrix \f$ \mathbf{M} \f$. On exit, the matrix \f$ \mathbf{M}^{-1} \f$.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  void inv_in_place_1d(M &&m) { // NOLINT (temporary views are allowed here)
    EXPECTS(is_matrix_square(m) and m.extent(0) == 1);
    if (m(0, 0) == get_value_t<M>{0.0}) NDA_RUNTIME_ERROR << "Error in nda::linalg::inv_in_place_1d: Matrix is not invertible";
    m(0, 0) = 1.0 / m(0, 0);
  }

  /**
   * @brief Compute the inverse of a \f$ 2 \times 2 \f$ matrix \f$ \mathbf{M} \f$.
   * 
   * @details It throws an exception if the matrix is not invertible (i.e. if \f$ \det(\mathbf{M}) = 0 \f$).
   *
   * @note The inversion is performed in place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m Input/output matrix. On entry, the matrix \f$ \mathbf{M} \f$. On exit, the matrix \f$ \mathbf{M}^{-1} \f$.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  void inv_in_place_2d(M &&m) { // NOLINT (temporary views are allowed here)
    EXPECTS(is_matrix_square(m) and m.extent(0) == 2);

    // calculate the determinant of the matrix
    auto const det = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
    if (det == get_value_t<M>{0.0}) NDA_RUNTIME_ERROR << "Error in nda::linalg::inv_in_place_2d: Matrix is not invertible";
    auto const detinv = 1.0 / det;

    // multiply the adjoint by the inverse determinant
    std::swap(m(0, 0), m(1, 1));
    m(0, 0) *= +detinv;
    m(1, 1) *= +detinv;
    m(1, 0) *= -detinv;
    m(0, 1) *= -detinv;
  }

  /**
   * @brief Compute the inverse of a \f$ 3 \times 3 \f$ matrix \f$ \mathbf{M} \f$.
   * 
   * @details It throws an exception if the matrix is not invertible (i.e. if \f$ \det(\mathbf{M}) = 0 \f$).
   *
   * @note The inversion is performed in place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m Input/output matrix. On entry, the matrix \f$ \mathbf{M} \f$. On exit, the matrix \f$ \mathbf{M}^{-1} \f$.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  void inv_in_place_3d(M &&m) { // NOLINT (temporary views are allowed here)
    EXPECTS(is_matrix_square(m) and m.extent(0) == 3);

    // calculate the cofactors of the matrix
    auto const b00 = +m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    auto const b10 = -m(1, 0) * m(2, 2) + m(1, 2) * m(2, 0);
    auto const b20 = +m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    auto const b01 = -m(0, 1) * m(2, 2) + m(0, 2) * m(2, 1);
    auto const b11 = +m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
    auto const b21 = -m(0, 0) * m(2, 1) + m(0, 1) * m(2, 0);
    auto const b02 = +m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
    auto const b12 = -m(0, 0) * m(1, 2) + m(0, 2) * m(1, 0);
    auto const b22 = +m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

    // calculate the determinant of the matrix
    auto const det = m(0, 0) * b00 + m(0, 1) * b10 + m(0, 2) * b20;
    if (det == get_value_t<M>{0.0}) NDA_RUNTIME_ERROR << "Error in nda::linalg::inv_in_place_3d: Matrix is not invertible";
    auto const detinv = 1.0 / det;

    // multiply the cofactors by the inverse determinant
    m(0, 0) = detinv * b00;
    m(0, 1) = detinv * b01;
    m(0, 2) = detinv * b02;
    m(1, 0) = detinv * b10;
    m(1, 1) = detinv * b11;
    m(1, 2) = detinv * b12;
    m(2, 0) = detinv * b20;
    m(2, 1) = detinv * b21;
    m(2, 2) = detinv * b22;
  }

  /**
   * @brief Compute the inverse of an \f$ n \times n \f$ matrix \f$ \mathbf{M} \f$.
   * 
   * @details For small matrices (\f$ 1 \times 1 \f$, \f$ 2 \times 2 \f$ or \f$ 3 \times 3 \f$), it directly computes
   * the matrix inversion using one of the optimized routines nda::linalg::inv_in_place_1d, nda::linalg::inv_in_place_2d
   * or nda::linalg::inv_in_place_3d.
   *
   * For larger matrices, it calls nda::lapack::getrf and nda::lapack::getri.
   * 
   * It throws an exception if the matrix is not invertible, i.e. if \f$ \det(\mathbf{M}) = 0 \f$, or if a call to
   * LAPACK fails.
   *
   * @note The inversion is performed in place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m Input/output matrix. On entry, the matrix \f$ \mathbf{M} \f$. On exit, the matrix \f$ \mathbf{M}^{-1} \f$.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  void inv_in_place(M &&m) { // NOLINT (temporary views are allowed here)
    EXPECTS(is_matrix_square(m));

    // use optimized routines for small matrices, otherwise use LAPACK routines
    auto const dim = m.shape()[0];
    if (dim == 1) {
      inv_in_place_1d(m);
    } else if (dim == 2) {
      inv_in_place_2d(m);
    } else if (dim == 3) {
      inv_in_place_3d(m);
    } else if (dim > 3) {
      // LU factorization with getrf
      auto ipiv = vector<int, nda::heap<nda::mem::get_addr_space<M>>>(dim);
      int info  = nda::lapack::getrf(m, ipiv);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::inv_in_place: getrf routine failed: info = " << info;

      // calculate the inverse with getri
      info = nda::lapack::getri(m, ipiv);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::inv_in_place: getri routine failed: info = " << info;
    }
  }

  /**
   * @brief Compute the inverse of an \f$ n \times n \f$ matrix \f$ \mathbf{M} \f$.
   * 
   * @details The given matrix/view is not modified. It first makes a copy of the matrix/view and then 
   * 
   * - uses nda::lapack::getrf and nda::lapack::getrs to compute the inverse in case the matrix's memory space is 
   * compatible with the device memory space. The resulting inverse matrix is always in nda::F_layout.
   * - calls nda::linalg::inv_in_place if the matrix is stored on the host memory space.
   * 
   * @warning This function makes copies of the input arrays/views. When working on the device memory space, this may
   * lead to runtime errors if the copying fails.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m Input matrix. The matrix \f$ \mathbf{M} \f$. 
   * @return The inverse matrix \f$ \mathbf{M}^{-1} \f$.
   */
  template <Matrix M>
    requires(get_algebra<M> == 'M' and is_blas_lapack_v<get_value_t<M>>)
  auto inv(M const &m) {
    EXPECTS(is_matrix_square(m));
    auto m_copy = make_regular(m);

    // for device compatible address spaces, we use getrf and getrs, otherwise we call inv_in_place
    if constexpr (nda::mem::have_device_compatible_addr_space<M>) {
      // LU factorization with getrf
      auto ipiv = vector<int, heap<nda::mem::get_addr_space<M>>>(m_copy.extent(0));
      int info  = nda::lapack::getrf(m_copy, ipiv);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::inv: getrf routine failed: info = " << info;

      // calculate the inverse with getrs and the identity matrix
      auto B = matrix<get_value_t<M>, F_layout, heap<nda::mem::get_addr_space<M>>>{transpose(eye<get_value_t<M>>(m_copy.extent(0)))};
      info   = nda::lapack::getrs(m_copy, B, ipiv);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::inv: getrs routine failed: info = " << info;
      return B;
    } else {
      inv_in_place(m_copy);
      return m_copy;
    }
  }

  /** @} */

} // namespace nda::linalg

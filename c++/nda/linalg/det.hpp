// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to compute the determinant of a matrix.
 */

#pragma once

#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../exceptions.hpp"
#include "../lapack/getrf.hpp"
#include "../matrix_functions.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Compute the determinant of a \f$ 1 \times 1 \f$ matrix \f$ \mathbf{M} \f$.
   *
   * @tparam M nda::Matrix type.
   * @param m Input matrix. The matrix \f$ \mathbf{M} \f$.
   * @return The determinant \f$ \det(\mathbf{M}) \f$.
   */
  template <Matrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  auto det_1d(M const &m) {
    EXPECTS(is_matrix_square(m) and m.shape()[0] == 1);
    return m(0, 0);
  }

  /**
   * @brief Compute the determinant of a \f$ 2 \times 2 \f$ matrix \f$ \mathbf{M} \f$.
   *
   * @tparam M nda::Matrix type.
   * @param m Input matrix. The matrix \f$ \mathbf{M} \f$.
   * @return The determinant \f$ \det(\mathbf{M}) \f$.
   */
  template <Matrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  auto det_2d(M const &m) {
    EXPECTS(is_matrix_square(m) and m.shape()[0] == 2);
    return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
  }

  /**
   * @brief Compute the inverse of a \f$ 3 \times 3 \f$ matrix \f$ \mathbf{M} \f$.
   *
   * @tparam M nda::Matrix type.
   * @param m Input matrix. The matrix \f$ \mathbf{M} \f$.
   * @return The determinant \f$ \det(\mathbf{M}) \f$.
   */
  template <Matrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  auto det_3d(M const &m) {
    EXPECTS(is_matrix_square(m) and m.shape()[0] == 3);
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
       + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
  }

  /**
   * @brief Compute the determinant of an \f$ n \times n \f$ matrix \f$ \mathbf{M} \f$.
   * 
   * @details For small matrices (\f$ 1 \times 1 \f$, \f$ 2 \times 2 \f$ or \f$ 3 \times 3 \f$), it directly computes
   * the determinant using one of the optimized routines nda::linalg::det_1d, nda::linalg::det_2d and 
   * nda::linalg::det_3d.
   *
   * For larger matrices, it calls nda::lapack::getrf and calculates the determinant from its LU decomposition.
   * 
   * It throws an exception if the call to nda::lapack::getrf fails.
   *
   * @note The matrix \f$ \mathbf{M} \f$ is modified if its number of rows/columns is greater than 3.
   *
   * @tparam M nda::Matrix type.
   * @param m Input/output matrix. On entry, the matrix \f$ \mathbf{M} \f$. On exit, the matrix \f$ \mathbf{M} \f$ or
   * the LU decomposition of \f$ \mathbf{M} \f$ from nda::lapack::getrf.
   * @return The determinant \f$ \det(\mathbf{M}) \f$.
   */
  template <Matrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  auto det_in_place(M &&m) { // NOLINT (temporary views are allowed here)
    EXPECTS(is_matrix_square(m));

    // use optimized routines for small matrices, otherwise use LAPACK routine
    auto const dim = m.shape()[0];
    if (dim == 1) {
      return det_1d(m);
    } else if (dim == 2) {
      return det_2d(m);
    } else if (dim == 3) {
      return det_3d(m);
    } else if (dim > 3) {
      // LU factorization with getrf
      auto ipiv = vector<int, sso<100>>(dim);
      int info  = nda::lapack::getrf(m, ipiv);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::det_in_place: getrf routine failed: info = " << info;

      // calculate the determinant from the LU decomposition
      auto det    = get_value_t<M>{1};
      int n_flips = 0;
      for (int i = 0; i < dim; i++) {
        det *= m(i, i);
        // count the number of column interchanges performed by getrf
        if (ipiv(i) != i + 1) ++n_flips;
      }

      return ((n_flips % 2 == 1) ? -det : det);
    } else {
      // empty matrix
      return get_value_t<M>{1};
    }
  }

  /**
   * @brief Compute the determinant of an \f$ n \times n \f$ matrix \f$ \mathbf{M} \f$.
   * 
   * @details The given matrix/view is not modified. It first makes a copy of the matrix/view and then calls 
   * nda::linalg::det_in_place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m Input matrix. The matrix \f$ \mathbf{M} \f$. 
   * @return The determinant \f$ \det(\mathbf{M}) \f$.
   */
  template <Matrix M>
    requires(get_algebra<M> == 'M' and nda::mem::have_host_compatible_addr_space<M> and is_blas_lapack_v<get_value_t<M>>)
  auto det(M const &m) {
    EXPECTS(is_matrix_square(m));
    auto m_copy = make_regular(m);
    return det_in_place(m_copy);
  }

  /** @} */

} // namespace nda::linalg

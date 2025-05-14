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
#include "../clef/make_lazy.hpp"
#include "../concepts.hpp"
#include "../exceptions.hpp"
#include "../lapack/getrf.hpp"
#include "../lapack/getri.hpp"
#include "../layout/policies.hpp"
#include "../matrix_functions.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../print.hpp"
#include "../traits.hpp"

#include <iostream>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Compute the determinant of a square matrix/view.
   *
   * @details It uses nda::lapack::getrf to compute the LU decomposition of the matrix and then calculates the
   * determinant by multiplying the diagonal elements of the \f$ \mathbf{U} \f$ matrix and taking into account that
   * `getrf` may change the ordering of the rows/columns of the matrix.
   *
   * The given matrix/view is modified in place.
   *
   * @tparam M Type of the matrix/view.
   * @param m Matrix/view object.
   * @return Determinant of the matrix/view.
   */
  template <typename M>
  auto determinant_in_place(M &m)
    requires(is_matrix_or_view_v<M>)
  {
    using value_t = get_value_t<M>;
    static_assert(std::is_convertible_v<value_t, double> or std::is_convertible_v<value_t, std::complex<double>>,
                  "Error in nda::determinant_in_place: Value type needs to be convertible to double or std::complex<double>");
    static_assert(not std::is_const_v<M>, "Error in nda::determinant_in_place: Value type cannot be const");

    // special case for an empty matrix
    if (m.empty()) return value_t{1};

    // check if the matrix is square
    if (m.extent(0) != m.extent(1)) NDA_RUNTIME_ERROR << "Error in nda::determinant_in_place: Matrix is not square: " << m.shape();

    // calculate the LU decomposition using lapack getrf
    const int dim = m.extent(0);
    basic_array<int, 1, C_layout, 'A', sso<100>> ipiv(dim);
    int info = lapack::getrf(m, ipiv); // it is ok to be in C order
    if (info < 0) NDA_RUNTIME_ERROR << "Error in nda::determinant_in_place: info = " << info;

    // calculate the determinant from the LU decomposition
    auto det    = value_t{1};
    int n_flips = 0;
    for (int i = 0; i < dim; i++) {
      det *= m(i, i);
      // count the number of column interchanges performed by getrf
      if (ipiv(i) != i + 1) ++n_flips;
    }

    return ((n_flips % 2 == 1) ? -det : det);
  }

  /**
   * @brief Compute the determinant of a square matrix/view.
   *
   * @details The given matrix/view is not modified. It first makes a copy of the given matrix/view and then calls
   * nda::determinant_in_place with the copy.
   *
   * @tparam M Type of the matrix/view.
   * @param m Matrix/view object.
   * @return Determinant of the matrix/view.
   */
  template <typename M>
  auto determinant(M const &m) {
    auto m_copy = make_regular(m);
    return determinant_in_place(m_copy);
  }

  namespace clef {
    /**
     * @brief Lazy version of nda::determinant.
     */
    CLEF_MAKE_FNT_LAZY(determinant)
  } // namespace clef

  /** @} */

} // namespace nda

// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to compute the determinant and inverse of a matrix.
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

  // For small matrices (2x2 and 3x3), we directly
  // compute the matrix inversion rather than calling the
  // LaPack routine
  // ---------- Small Inverse Benchmarks ---------
  //          Run on (16 X 2400 MHz CPUs) (see benchmarks/small_inv.cpp)
  // ---------------------------------------------
  // Matrix Size      Time (old)        Time (new)
  //     1             502 ns            59.0 ns
  //     2             595 ns            61.7 ns
  //     3             701 ns            67.5 ns

  /**
   * @brief Compute the inverse of a 1-by-1 matrix.
   *
   * @details The inversion is performed in place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m nda::MemoryMatrix object to be inverted.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M' and mem::on_host<M>)
  void inverse1_in_place(M &&m) { // NOLINT (temporary views are allowed here)
    if (m(0, 0) == 0.0) NDA_RUNTIME_ERROR << "Error in nda::inverse1_in_place: Matrix is not invertible";
    m(0, 0) = 1.0 / m(0, 0);
  }

  /**
   * @brief Compute the inverse of a 2-by-2 matrix.
   *
   * @details The inversion is performed in place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m nda::MemoryMatrix object to be inverted.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M' and mem::on_host<M>)
  void inverse2_in_place(M &&m) { // NOLINT (temporary views are allowed here)
    // calculate the adjoint of the matrix
    std::swap(m(0, 0), m(1, 1));

    // calculate the inverse determinant of the matrix
    auto det = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
    if (det == 0.0) NDA_RUNTIME_ERROR << "Error in nda::inverse2_in_place: Matrix is not invertible";
    auto detinv = 1.0 / det;

    // multiply the adjoint by the inverse determinant
    m(0, 0) *= +detinv;
    m(1, 1) *= +detinv;
    m(1, 0) *= -detinv;
    m(0, 1) *= -detinv;
  }

  /**
   * @brief Compute the inverse of a 3-by-3 matrix.
   *
   * @details The inversion is performed in place.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m nda::MemoryMatrix object to be inverted.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M' and mem::on_host<M>)
  void inverse3_in_place(M &&m) { // NOLINT (temporary views are allowed here)
    // calculate the cofactors of the matrix
    auto b00 = +m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    auto b10 = -m(1, 0) * m(2, 2) + m(1, 2) * m(2, 0);
    auto b20 = +m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    auto b01 = -m(0, 1) * m(2, 2) + m(0, 2) * m(2, 1);
    auto b11 = +m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
    auto b21 = -m(0, 0) * m(2, 1) + m(0, 1) * m(2, 0);
    auto b02 = +m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
    auto b12 = -m(0, 0) * m(1, 2) + m(0, 2) * m(1, 0);
    auto b22 = +m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

    // calculate the inverse determinant of the matrix
    auto det = m(0, 0) * b00 + m(0, 1) * b10 + m(0, 2) * b20;
    if (det == 0.0) NDA_RUNTIME_ERROR << "Error in nda::inverse3_in_place: Matrix is not invertible";
    auto detinv = 1.0 / det;

    // fill the matrix by multiplying the cofactors by the inverse determinant
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
   * @brief Compute the inverse of an n-by-n matrix.
   *
   * @details The inversion is performed in place.
   *
   * For small matrices (1-by-1, 2-by-2, 3-by-3), we directly compute the matrix inversion using the optimized routines:
   * nda::inverse1_in_place, nda::inverse2_in_place, nda::inverse3_in_place.
   *
   * For larger matrices, it uses nda::lapack::getrf and nda::lapack::getri.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m nda::MemoryMatrix object to be inverted.
   */
  template <MemoryMatrix M>
    requires(get_algebra<M> == 'M')
  void inverse_in_place(M &&m) { // NOLINT (temporary views are allowed here)
    EXPECTS(is_matrix_square(m, true));

    // nothing to do if the matrix/view is empty
    if (m.empty()) return;

    // use optimized routines for small matrices
    if constexpr (mem::on_host<M>) {
      if (m.shape()[0] == 1) {
        inverse1_in_place(m);
        return;
      }

      if (m.shape()[0] == 2) {
        inverse2_in_place(m);
        return;
      }

      if (m.shape()[0] == 3) {
        inverse3_in_place(m);
        return;
      }
    }

    // use getrf and getri from lapack for larger matrices
    array<int, 1> ipiv(m.extent(0));
    int info = lapack::getrf(m, ipiv); // it is ok to be in C order
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::inverse_in_place: Matrix is not invertible: info = " << info;
    info = lapack::getri(m, ipiv);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::inverse_in_place: Matrix is not invertible: info = " << info;
  }

  /**
   * @brief Compute the inverse of an n-by-n matrix.
   *
   * @details  The given matrix/view is not modified. It first makes copy of the given matrix/view and then calls
   * nda::inverse_in_place with the copy.
   *
   * @tparam M nda::MemoryMatrix type.
   * @param m nda::MemoryMatrix object to be inverted.
   * @return Inverse of the matrix.
   */
  template <Matrix M>
  auto inverse(M const &m)
    requires(get_algebra<M> == 'M')
  {
    EXPECTS(is_matrix_square(m, true));
    auto r = make_regular(m);
    inverse_in_place(r);
    return r;
  }

  /** @} */

} // namespace nda

namespace nda::clef {

  /**
   * @ingroup linalg_tools
   * @brief Lazy version of nda::determinant.
   */
  CLEF_MAKE_FNT_LAZY(determinant)

} // namespace nda::clef

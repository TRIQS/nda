// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to solve eigenvalue problems for general (non-symmetric) matrices.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_array.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/geev.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../matrix_functions.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Get the complex eigenvalues from nda::lapack::geev output for real matrices.
   *
   * @details For real matrices, nda::lapack::geev stores the computed eigenvalues in two real vectors, \f$
   * \mathbf{w}^{(r)} \f$ and \f$ \mathbf{w}^{(i)} \f$.
   *
   * The actual (complex) eigenvalues \f$ \lambda_j \f$ are given by \f$ \lambda_j = w^{(r)}_j + i w^{(i)}_j \f$.
   *
   * Use nda::linalg::get_geev_eigenvectors to get eigenvectors.
   *
   * @tparam WR nda::Vector with double value type.
   * @tparam WI nda::Vector with double value type.
   * @param wr The real parts of the computed eigenvalues.
   * @param wi The imaginary parts of the computed eigenvalues.
   * @return An nda::array containing the complex eigenvalues.
   */
  template <Vector WR, Vector WI>
    requires(mem::have_host_compatible_addr_space<WR, WI> and std::same_as<double, get_value_t<WR>> and have_same_value_type_v<WR, WI>)
  auto get_geev_eigenvalues(const WR &wr, const WI &wi) {
    // check the dimensions of the input arrays/views
    auto const n = wr.size();
    EXPECTS(n == wi.size());

    // generate eigenvalues
    auto lambda = array<std::complex<double>, 1>(n);
    for (long i = 0; i < n; ++i) lambda(i) = std::complex<double>(wr(i), wi(i));
    return lambda;
  }

  /**
   * @brief Get the complex left/right eigenvectors from nda::lapack::geev output for real matrices.
   *
   * @details For real matrices, nda::lapack::geev stores the computed complex eigenvalues in two real vectors, \f$
   * \mathbf{w}^{(r)} \f$ and \f$ \mathbf{w}^{(i)} \f$, and the left/right eigenvectors in packed format in the columns
   * \f$ \mathbf{v}^{(l)}_j \f$/\f$ \mathbf{v}^{(r)}_j \f$ of a real matrix.
   *
   * The complex eigenvalues \f$ \lambda_j \f$ are given by \f$ \lambda_j = w^{(r)}_j + i w^{(i)}_j \f$.
   *
   * The left/right eigenvectors \f$ \mathbf{x}_j \f$ are unpacked as follows:
   * - If the eigenvalue \f$ \lambda_j \f$ is real, i.e. if \f$ w^{(i)}_j = 0 \f$, then the corresponding left/right
   * eigenvector is given by \f$ \mathbf{x}_j = \mathbf{v}^{(\alpha)}_j \f$.
   * - If the eigenvalues \f$ \lambda_j \f$ and \f$ \lambda_{j + 1} \f$ form a complex conjugate pair, i.e. if \f$
   * w^{(i)}_j > 0 \f$, then the two corresponding left/right eigenvectors are given by \f$ \mathbf{x}_j =
   * \mathbf{v}^{(\alpha)}_j + i \mathbf{v}^{(\alpha)}_{j+1} \f$ and \f$ \mathbf{x}_{j+1} = \mathbf{v}^{(\alpha)}_j - i
   * \mathbf{v}^{(\alpha)}_{j+1} \f$.
   *
   * Use nda::linalg::get_geev_eigenvalues to get eigenvalues.
   *
   * @tparam WI nda::Vector with double value type.
   * @tparam VA nda::Matrix with double value type.
   * @param wi The imaginary parts of the computed eigenvalues.
   * @param va The left/right eigenvectors in packed format.
   * @return An nda::matrix containing the complex left/right eigenvectors.
   */
  template <Vector WI, Matrix VA>
    requires(mem::have_host_compatible_addr_space<WI, VA> and std::same_as<double, get_value_t<WI>> and have_same_value_type_v<WI, VA>)
  auto get_geev_eigenvectors(const WI &wi, const VA &va) {
    using namespace std::complex_literals;
    static_assert(nda::blas::has_F_layout<VA>, "Error in nda::linalg::get_geev_eigenvectors: VA must have Fortran layout");

    // check the dimensions of the input arrays/views
    auto const n = wi.size();
    EXPECTS(va.shape() == (std::array<long, 2>{n, n}));

    // unpack eigenvectors
    auto X = matrix<std::complex<double>, F_layout>(n, n);
    long j = 0;
    while (j < n) {
      if (wi(j) > 0.0) {
        // complex conjugate eigenvalue pair --> we need to unpack the eigenvectors
        for (long i = 0; i < n; ++i) {
          X(i, j)     = std::complex<double>{va(i, j), va(i, j + 1)};
          X(i, j + 1) = std::complex<double>{va(i, j), -va(i, j + 1)};
        }
        j += 2;
      } else {
        // real eigenvalue --> eigenvector is purely real
        X(nda::range::all, j) = va(nda::range::all, j);
        ++j;
      }
    }

    return X;
  }

  namespace detail {

    // Implementation for complex matrices - straightforward call to geev.
    template <MemoryMatrix A>
      requires(is_complex_v<get_value_t<A>>)
    auto eig_impl(A &&a, char jobvl, char jobvr) { // NOLINT (temporary views are allowed here)
      using arr_t  = array<get_value_t<A>, 1>;
      using mat_t  = matrix<get_value_t<A>, F_layout>;
      auto const n = a.extent(0);

      // early return if the matrix is empty
      if (a.empty()) return std::make_tuple(arr_t{}, mat_t{}, mat_t{});

      // allocate outputs
      auto lambda = arr_t(n);
      auto U      = mat_t(jobvl == 'V' ? n : 0, jobvl == 'V' ? n : 0);
      auto V      = mat_t(jobvr == 'V' ? n : 0, jobvr == 'V' ? n : 0);

      // make the call to geev
      int info = nda::lapack::geev(a, lambda, U, V, jobvl, jobvr);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eig_impl: geev routine failed: info = " << info;

      return std::make_tuple(std::move(lambda), std::move(U), std::move(V));
    }

    // Implementation for real matrices.
    template <MemoryMatrix A>
      requires(std::same_as<double, get_value_t<A>>)
    auto eig_impl(A &&a, char jobvl, char jobvr) { // NOLINT (temporary views are allowed here)
      using arr_t  = array<std::complex<double>, 1>;
      using mat_t  = matrix<std::complex<double>, F_layout>;
      auto const n = a.extent(0);

      // early return if the matrix is empty
      if (a.empty()) return std::make_tuple(arr_t{}, mat_t{}, mat_t{});

      // allocate outputs
      auto wr = array<double, 1>(n);
      auto wi = array<double, 1>(n);
      auto vl = matrix<double, F_layout>(jobvl == 'V' ? n : 0, jobvl == 'V' ? n : 0);
      auto vr = matrix<double, F_layout>(jobvr == 'V' ? n : 0, jobvr == 'V' ? n : 0);

      // make the call to geev
      int info = nda::lapack::geev(a, wr, wi, vl, vr, jobvl, jobvr);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eig_impl: geev routine failed: info = " << info;

      // get eigenvalues and eigenvectors from geev output
      auto lambda = get_geev_eigenvalues(wr, wi);
      auto U      = (jobvl == 'V') ? get_geev_eigenvectors(wi, vl) : mat_t{};
      auto V      = (jobvr == 'V') ? get_geev_eigenvectors(wi, vr) : mat_t{};

      return std::make_tuple(std::move(lambda), std::move(U), std::move(V));
    }

  } // namespace detail

  /**
   * @brief Compute the eigenvalues and right eigenvectors of a general matrix.
   *
   * @details It computes the right eigenvectors \f$ \mathbf{v}_j \f$ and eigenvalues \f$ \lambda_j \f$ of the matrix
   * \f$ \mathbf{A} \f$ such that
   * \f[
   *  \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{v}_j \; .
   * \f]
   *
   * It calls nda::lapack::geev and, for real matrices, retrieves the complex eigenvalues and eigenvectors using
   * nda::linalg::get_geev_eigenvalues and nda::linalg::get_geev_eigenvectors.
   *
   * It throws an exception if the LAPACK call fails.
   *
   * @note The given matrix/view must have Fortran layout and is modified during the computation.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it is overwritten.
   * @return `std::pair` containing an nda::array with the complex eigenvalues \f$ \lambda_j \f$ and a nda::matrix with
   * the complex right eigenvectors \f$ \mathbf{v}_j \f$ as columns.
   */
  template <MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>> and nda::blas::has_F_layout<A>)
  auto eig_in_place(A &&a) {
    auto [lambda, U, V] = detail::eig_impl(std::forward<A>(a), 'N', 'V');
    return std::make_pair(std::move(lambda), std::move(V));
  }

  /**
   * @brief Compute the eigenvalues of a general matrix.
   *
   * @details It computes the eigenvalues \f$ \lambda_j \f$ of the matrix \f$ \mathbf{A} \f$, where
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{v}_j \; .
   * \f]
   *
   * It calls nda::lapack::geev and, for real matrices, retrieves the complex eigenvalues using
   * nda::linalg::get_geev_eigenvalues.
   *
   * It throws an exception if the LAPACK call fails.
   *
   * @note The given matrix/view must have Fortran layout and is modified during the computation.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it is overwritten.
   * @return An nda::array with the complex eigenvalues \f$ \lambda_j \f$.
   */
  template <MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>> and nda::blas::has_F_layout<A>)
  auto eigvals_in_place(A &&a) {
    auto [lambda, U, V] = detail::eig_impl(std::forward<A>(a), 'N', 'N');
    return lambda;
  }

  /**
   * @brief Compute the eigenvalues and right eigenvectors of a general matrix.
   *
   * @details Same as nda::linalg::eig_in_place but makes a copy of the input matrix, leaving the original unchanged.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @return `std::pair` containing an nda::array with the complex eigenvalues \f$ \lambda_j \f$ and an nda::matrix with
   * the complex right eigenvectors \f$ \mathbf{v}_j \f$ as columns.
   */
  template <MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto eig(A const &a) {
    auto m_copy = matrix<get_value_t<A>, F_layout>(a);
    return eig_in_place(m_copy);
  }

  /**
   * @brief Compute the eigenvalues of a general matrix.
   *
   * @details Same as nda::linalg::eigvals_in_place but makes a copy of the input matrix, leaving the original
   * unchanged.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @return An nda::array with the complex eigenvalues \f$ \lambda_j \f$.
   */
  template <MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto eigvals(A const &a) {
    auto m_copy = matrix<get_value_t<A>, F_layout>(a);
    return eigvals_in_place(m_copy);
  }

  /** @} */

} // namespace nda::linalg

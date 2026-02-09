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
   * Use nda::linalg::get_geev_eigenvectors to get corresponding eigenvectors.
   * 
   * @note \f$ \mathbf{w}^{(r)} \f$ and \f$ \mathbf{w}^{(i)} \f$ are required to satisfy 
   * nda::mem::have_host_compatible_addr_space and to have the same real value type.
   *
   * @tparam WR nda::Vector type.
   * @tparam WI nda::Vector type.
   * @param wr Input vector. \f$ \mathbf{w}^{(r)} \f$ containing the real parts of the computed eigenvalues, i.e. \f$ 
   * \mathrm{Re}(\lambda_j) \f$.
   * @param wi Input vector. \f$ \mathbf{w}^{(r)} \f$ containing the imaginary parts of the computed eigenvalues, i.e. 
   * \f$ \mathrm{Im}(\lambda_j) \f$.
   * @return An nda::array containing the complex eigenvalues.
   */
  template <Vector WR, Vector WI>
    requires(mem::have_host_compatible_addr_space<WR, WI> and AnyOf<get_value_t<WR>, float, double> and have_same_value_type_v<WR, WI>)
  auto get_geev_eigenvalues(const WR &wr, const WI &wi) {
    // check the dimensions of the input arrays/views
    auto const n = wr.size();
    EXPECTS(n == wi.size());

    // generate eigenvalues
    using fp_t  = get_fp_t<WR>;
    auto lambda = array<std::complex<fp_t>, 1>(n);
    for (long i = 0; i < n; ++i) lambda(i) = std::complex<fp_t>(wr(i), wi(i));
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
   * The left/right eigenvectors \f$ \mathbf{x}_j \f$ are unpacked as follows (\f$ \alpha \in \{ l, r \} \f$):
   * - If the eigenvalue \f$ \lambda_j \f$ is real, i.e. if \f$ w^{(i)}_j = 0 \f$, then the corresponding left/right
   * eigenvector is given by \f$ \mathbf{x}_j = \mathbf{v}^{(\alpha)}_j \f$.
   * - If the eigenvalues \f$ \lambda_j \f$ and \f$ \lambda_{j + 1} \f$ form a complex conjugate pair, i.e. if \f$
   * w^{(i)}_j > 0 \f$, then the two corresponding left/right eigenvectors are given by \f$ \mathbf{x}_j =
   * \mathbf{v}^{(\alpha)}_j + i \mathbf{v}^{(\alpha)}_{j+1} \f$ and \f$ \mathbf{x}_{j+1} = \mathbf{v}^{(\alpha)}_j - i
   * \mathbf{v}^{(\alpha)}_{j+1} \f$.
   *
   * Use nda::linalg::get_geev_eigenvalues to get eigenvalues.
   * 
   * The resulting matrix is always returned in nda::F_layout.
   * 
   * @note \f$ \mathbf{V}_{\alpha} \f$ and \f$ \mathbf{w}^{(i)} \f$ are required to satisfy 
   * nda::mem::have_host_compatible_addr_space and to have the same real value type.
   *
   * @tparam WI nda::Vector type.
   * @tparam VA nda::Matrix type.
   * @param wi Input vector. \f$ \mathbf{w}^{(r)} \f$ containing the imaginary parts of the computed eigenvalues, i.e. 
   * \f$ \mathrm{Im}(\lambda_j) \f$.
   * @param va Input matrix. \f$ \mathbf{V}_{\alpha} \f$ containing the left/right eigenvectors in packed format.
   * @return An nda::matrix containing the complex left/right eigenvectors.
   */
  template <Vector WI, Matrix VA>
    requires(mem::have_host_compatible_addr_space<WI, VA> and AnyOf<get_value_t<WI>, float, double> and have_same_value_type_v<WI, VA>)
  auto get_geev_eigenvectors(const WI &wi, const VA &va) {
    using namespace std::complex_literals;

    // check the dimensions of the input arrays/views
    auto const n = wi.size();
    EXPECTS(va.shape() == (std::array<long, 2>{n, n}));

    // unpack eigenvectors
    using fp_t = get_fp_t<WI>;
    auto X     = matrix<std::complex<fp_t>, F_layout>(n, n);
    long j     = 0;
    while (j < n) {
      if (wi(j) > 0.0) {
        // complex conjugate eigenvalue pair --> we need to unpack the eigenvectors
        for (long i = 0; i < n; ++i) {
          X(i, j)     = std::complex<fp_t>{va(i, j), va(i, j + 1)};
          X(i, j + 1) = std::complex<fp_t>{va(i, j), -va(i, j + 1)};
        }
        j += 2;
      } else {
        // real eigenvalue --> eigenvector is purely real
        X(range::all, j) = va(range::all, j);
        ++j;
      }
    }

    return X;
  }

  namespace detail {

    // Implementation for complex matrices - straightforward call to geev.
    template <blas_lapack::BlasArrayCplx<2> A>
    auto eig_impl(A &&a, char jobvl, char jobvr) { // NOLINT (temporary views are allowed here)
      using arr_t  = array<get_value_t<A>, 1>;
      using mat_t  = matrix<get_value_t<A>, F_layout>;
      auto const n = a.extent(0);

      // early return if the matrix is empty
      if (a.empty()) return std::make_tuple(arr_t{}, mat_t{}, mat_t{});

      // allocate outputs
      auto lambda = arr_t(n);
      auto U      = (jobvl == 'V') ? mat_t(n, n) : mat_t();
      auto V      = (jobvr == 'V') ? mat_t(n, n) : mat_t();

      // make the call to geev
      int info = lapack::geev(a, lambda, U, V, jobvl, jobvr);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eig_impl: geev routine failed: info = " << info;

      return std::make_tuple(std::move(lambda), std::move(U), std::move(V));
    }

    // Implementation for real matrices.
    template <blas_lapack::BlasArrayReal<2> A>
    auto eig_impl(A &&a, char jobvl, char jobvr) { // NOLINT (temporary views are allowed here)
      using fp_t   = get_fp_t<A>;
      using arr_t  = array<std::complex<fp_t>, 1>;
      using mat_t  = matrix<std::complex<fp_t>, F_layout>;
      auto const n = a.extent(0);

      // early return if the matrix is empty
      if (a.empty()) return std::make_tuple(arr_t{}, mat_t{}, mat_t{});

      // allocate outputs
      auto wr = array<fp_t, 1>(n);
      auto wi = array<fp_t, 1>(n);
      auto vl = (jobvl == 'V') ? matrix<fp_t, F_layout>(n, n) : matrix<fp_t, F_layout>{};
      auto vr = (jobvr == 'V') ? matrix<fp_t, F_layout>(n, n) : matrix<fp_t, F_layout>{};

      // make the call to geev
      int info = lapack::geev(a, wr, wi, vl, vr, jobvl, jobvr);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eig_impl: geev routine failed: info = " << info;

      // get eigenvalues and eigenvectors from geev output
      auto lambda = get_geev_eigenvalues(wr, wi);
      auto U      = (jobvl == 'V') ? get_geev_eigenvectors(wi, vl) : mat_t{};
      auto V      = (jobvr == 'V') ? get_geev_eigenvectors(wi, vr) : mat_t{};

      return std::make_tuple(std::move(lambda), std::move(U), std::move(V));
    }

  } // namespace detail

  /**
   * @brief Compute the eigenvalues and right eigenvectors of a general matrix in place.
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
   * The resulting matrix \f$ \mathbf{V} \f$ containing the eigenvectors is always returned in nda::F_layout.
   *
   * An exception is thrown, if the LAPACK call fails.
   *
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have 
   * nda::F_layout. See nda::linalg::eig for a version that handles nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it is overwritten.
   * @return `std::pair` containing an nda::array with the complex eigenvalues \f$ \lambda_j \f$ and an nda::matrix with
   * the complex right eigenvectors \f$ \mathbf{v}_j \f$ in its columns.
   */
  template <blas_lapack::BlasArray<2> A>
    requires(mem::have_host_compatible_addr_space<A> and blas_lapack::has_F_layout<A>)
  auto eig_in_place(A &&a) {
    auto [lambda, U, V] = detail::eig_impl(std::forward<A>(a), 'N', 'V');
    return std::make_pair(std::move(lambda), std::move(V));
  }

  /**
   * @brief Compute the eigenvalues of a general matrix in place.
   *
   * @details It computes the eigenvalues \f$ \lambda_j \f$ of the matrix \f$ \mathbf{A} \f$ such that
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{v}_j \; .
   * \f]
   *
   * It calls nda::lapack::geev and, for real matrices, retrieves the complex eigenvalues using
   * nda::linalg::get_geev_eigenvalues.
   *
   * An exception is thrown, if the LAPACK call fails.
   *
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have 
   * nda::F_layout. See nda::linalg::eigvals for a version that handles nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it is overwritten.
   * @return An nda::array with the complex eigenvalues \f$ \lambda_j \f$.
   */
  template <blas_lapack::BlasArray<2> A>
    requires(mem::have_host_compatible_addr_space<A> and blas_lapack::has_F_layout<A>)
  auto eigvals_in_place(A &&a) {
    auto [lambda, U, V] = detail::eig_impl(std::forward<A>(a), 'N', 'N');
    return lambda;
  }

  /**
   * @brief Compute the eigenvalues and right eigenvectors of a general matrix.
   *
   * @details It makes a copy of the given matrix/view and calls nda::linalg::eig_in_place with the copy.
   * 
   * The resulting matrix \f$ \mathbf{V} \f$ containing the eigenvectors is always returned in nda::F_layout.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have a value type
   * that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @return `std::pair` containing an nda::array with the complex eigenvalues \f$ \lambda_j \f$ and an nda::matrix with
   * the complex right eigenvectors \f$ \mathbf{v}_j \f$ in its columns.
   */
  template <Matrix A>
    requires(mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto eig(A const &a) {
    auto m_copy = matrix<get_value_t<A>, F_layout>(a);
    return eig_in_place(m_copy);
  }

  /**
   * @brief Compute the eigenvalues of a general matrix.
   *
   * @details It makes a copy of the given matrix/view and calls nda::linalg::eigvals_in_place with the copy. 
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have a value type
   * that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @return An nda::array with the complex eigenvalues \f$ \lambda_j \f$.
   */
  template <Matrix A>
    requires(mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto eigvals(A const &a) {
    auto m_copy = matrix<get_value_t<A>, F_layout>(a);
    return eigvals_in_place(m_copy);
  }

  /** @} */

} // namespace nda::linalg

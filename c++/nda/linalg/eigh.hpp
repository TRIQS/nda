// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to solve (generalized) eigenvalue problems with a symmetric/hermitian matrices.
 */

#pragma once

#include "../basic_array.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/syev.hpp"
#include "../lapack/sygv.hpp"
#include "../lapack/heev.hpp"
#include "../lapack/hegv.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../matrix_functions.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <complex>
#include <type_traits>
#include <utility>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  namespace detail {

    // Perform the call to the LAPACK routines syev/heev for eigh_in_place and eigvalsh_in_place.
    template <typename A>
    auto eigh_impl(A &&a, char jobz) { // NOLINT (temporary views are allowed here)
      // early return if the matrix is empty
      using fp_t = get_fp_t<A>;
      if (a.empty()) return array<fp_t, 1>{};

      // make the call to syev/heev
      auto lambda = array<fp_t, 1>(a.extent(0));
      int info    = 0;
      if constexpr (is_complex_v<get_value_t<A>>) {
        info = nda::lapack::heev(a, lambda, jobz);
      } else {
        info = nda::lapack::syev(a, lambda, jobz);
      }
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eigh_impl: syev/heev routine failed: info = " << info;

      return lambda;
    }

    // Perform the call to the LAPACK routines sygv/hegv for eigh_in_place and eigvalsh_in_place.
    template <typename A, typename B>
    auto eigh_impl(A &&a, B &&b, char jobz, int itype) { // NOLINT (temporary views are allowed here)
      // early return if the matrix is empty
      using fp_t = get_fp_t<A>;
      if (a.empty()) return array<fp_t, 1>{};

      // make the call to sygv/hegv
      auto lambda = array<fp_t, 1>(a.extent(0));
      int info    = 0;
      if constexpr (is_complex_v<get_value_t<A>>) {
        info = nda::lapack::hegv(a, b, lambda, jobz, itype);
      } else {
        info = nda::lapack::sygv(a, b, lambda, jobz, itype);
      }
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eigh_impl: sygv/hegv routine failed: info = " << info;

      return lambda;
    }

  } // namespace detail

  /**
   * @brief Compute the eigenvalues and eigenvectors of a real symmetric or complex hermitian matrix.
   *
   * @details It computes the eigenvectors \f$ \mathbf{v}_i \f$ and eigenvalues \f$ \lambda_i \f$ of the matrix \f$ 
   * \mathbf{A} \f$ such that
   * \f[
   *  \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; .
   * \f]
   * 
   * If the elements of \f$ \mathbf{A} \f$ are real, it calls nda::lapack::syev. If the elements of \f$ \mathbf{A} \f$ 
   * are complex, it calls nda::lapack::heev.
   * 
   * It throws an exception if the call to LAPACK fails.
   *
   * @note The given matrix/view is modified and contains the eigenvectors in its columns after the call.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it contains the orthonormal 
   * eigenvectors \f$ \mathbf{v}_i \f$ in its columns.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>> and nda::blas::has_F_layout<A>)
  auto eigh_in_place(A &&a) {
    return detail::eigh_impl(std::forward<A>(a), 'V');
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of a generalized real symmetric-definite or complex 
   * hermitian-definite eigenvalue problem.
   *
   * @details It computes the eigenvectors \f$ \mathbf{v}_i \f$ and eigenvalues \f$ \lambda_i \f$ of one of the 
   * following eigenvalue problems:
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   * 
   * Here \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are assumed to be real symmetric or complex hermitian. In addition,
   * \f$ \mathbf{B} \f$ is assumed to be positive definite.
   * 
   * If \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are real, it calls nda::lapack::sygv. Otherwise, it calls 
   * nda::lapack::hegv.
   * 
   * It throws an exception if the call to LAPACK fails.
   *
   * @note The given matrices/views are modified.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it contains the normalized 
   * eigenvectors \f$ \mathbf{v}_i \f$ in its columns (see nda::lapack::sygv or nda::lapack::hegv for details).
   * @param b Input/output matrix. On entry, the matrix \f$ \mathbf{B} \f$. On exit, it is overwritten (see 
   * nda::lapack::sygv or nda::lapack::hegv for details).
   * @param itype Specifies the problem to be solved.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <MemoryMatrix A, MemoryMatrix B>
    requires(nda::mem::have_host_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, B>
             and nda::blas::has_F_layout<A> and nda::blas::has_F_layout<B>)
  auto eigh_in_place(A &&a, B &&b, int itype = 1) {
    return detail::eigh_impl(std::forward<A>(a), std::forward<B>(b), 'V', itype);
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of a real symmetric or complex hermitian matrix.
   *
   * @details It computes the eigenvectors \f$ \mathbf{v}_i \f$ and eigenvalues \f$ \lambda_i \f$ of the matrix \f$ 
   * \mathbf{A} \f$ such that
   * \f[
   *  \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; .
   * \f]
   * 
   * It makes a copy of the given matrix/view and calls nda::linalg::eigh_in_place with the copy.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @return `std::pair` containing an nda::array with the real eigenvalues \f$ \lambda_i \f$ in ascending order and an 
   * nda::matrix in nda::F_layout containing the eigenvectors \f$ \mathbf{v}_i \f$ in its columns.
   */
  template <Matrix A>
    requires(Scalar<get_value_t<A>>)
  auto eigh(A const &a) {
    using value_t = get_value_t<A>;
    auto a_copy   = matrix<value_t, F_layout>{a};
    auto lambda   = eigh_in_place(a_copy);
    return std::make_pair(lambda, a_copy);
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of a generalized real symmetric-definite or complex 
   * hermitian-definite eigenvalue problem.
   *
   * @details It computes the eigenvectors \f$ \mathbf{v}_i \f$ and eigenvalues \f$ \lambda_i \f$ of one of the 
   * following eigenvalue problems:
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   * 
   * It makes a copy of the given matrices/views and calls nda::linalg::eigh_in_place(A &&, B&&, int) with the copies. 
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @param b Input matrix. The matrix \f$ \mathbf{B} \f$. 
   * @param itype Specifies the problem to be solved.
   * @return `std::pair` containing an nda::array with the real eigenvalues \f$ \lambda_i \f$ in ascending order and an 
   * nda::matrix in nda::F_layout containing the eigenvectors \f$ \mathbf{v}_i \f$ in its columns.
   */
  template <Matrix A, Matrix B>
    requires(Scalar<get_value_t<A>> and Scalar<get_value_t<B>> and std::is_same_v<get_fp_t<A>, get_fp_t<B>>)
  auto eigh(A const &a, B const &b, int itype = 1) {
    using value_t = std::conditional_t<is_complex_v<get_value_t<A>> or is_complex_v<get_value_t<B>>, std::complex<get_fp_t<A>>, get_fp_t<A>>;
    auto a_copy   = matrix<value_t, F_layout>{a};
    auto b_copy   = matrix<value_t, F_layout>{b};
    auto lambda   = eigh_in_place(a_copy, b_copy, itype);
    return std::make_pair(lambda, a_copy);
  }

  /**
   * @brief Compute the eigenvalues of a real symmetric or complex hermitian matrix.
   *
   * @details It computes the eigenvalues \f$ \lambda_i \f$ of the matrix \f$ \mathbf{A} \f$ such that
   * \f[
   *  \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; .
   * \f]
   * 
   * If the elements of \f$ \mathbf{A} \f$ are real, it calls nda::lapack::syev. If the elements of \f$ \mathbf{A} \f$ 
   * are complex, it calls nda::lapack::heev.
   * 
   * It throws an exception if the call to LAPACK fails.
   *
   * @note The given matrix/view is modified.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, the contents of \f$ \mathbf{A} \f$ 
   * are destroyed.
   * @return An nda::array containing the real eigenvalues in ascending order.
   */
  template <MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>> and nda::blas::has_F_layout<A>)
  auto eigvalsh_in_place(A &&a) {
    return detail::eigh_impl(std::forward<A>(a), 'N');
  }

  /**
   * @brief Compute the eigenvalues of a generalized real symmetric-definite or complex hermitian-definite eigenvalue 
   * problem.
   *
   * @details It computes the eigenvalues \f$ \lambda_i \f$ of one of the following eigenvalue problems:
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   * 
   * Here \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are assumed to be real symmetric or complex hermitian. In addition,
   * \f$ \mathbf{B} \f$ is assumed to be positive definite.
   * 
   * If \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are real, it calls nda::lapack::sygv. Otherwise, it calls 
   * nda::lapack::hegv.
   * 
   * It throws an exception if the call to LAPACK fails.
   *
   * @note The given matrices/views are modified.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, the contents of \f$ \mathbf{A} \f$ 
   * are destroyed.
   * @param b Input/output matrix. On entry, the matrix \f$ \mathbf{B} \f$. On exit, it is overwritten (see 
   * nda::lapack::sygv or nda::lapack::hegv for details).
   * @param itype Specifies the problem to be solved.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <MemoryMatrix A, MemoryMatrix B>
    requires(nda::mem::have_host_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, B>
             and nda::blas::has_F_layout<A> and nda::blas::has_F_layout<B>)
  auto eigvalsh_in_place(A &&a, B &&b, int itype = 1) {
    return detail::eigh_impl(std::forward<A>(a), std::forward<B>(b), 'N', itype);
  }

  /**
   * @brief Compute the eigenvalues of a real symmetric or complex hermitian matrix.
   *
   * @details It computes the eigenvalues \f$ \lambda_i \f$ of the matrix \f$ \mathbf{A} \f$ such that
   * \f[
   *  \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; .
   * \f]
   * 
   * It makes a copy of the given matrix/view and calls nda::linalg::eigvalsh_in_place with the copy.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @return An nda::array containing the real eigenvalues in ascending order.
   */
  template <Matrix A>
    requires(Scalar<get_value_t<A>>)
  auto eigvalsh(A const &a) {
    using value_t = get_value_t<A>;
    auto a_copy   = matrix<value_t, F_layout>{a};
    return eigvalsh_in_place(a_copy);
  }

  /**
   * @brief Compute the eigenvalues of a generalized real symmetric-definite or complex hermitian-definite eigenvalue 
   * problem.
   *
   * @details It computes the eigenvalues \f$ \lambda_i \f$ of one of the following eigenvalue problems:
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   * 
   * It makes a copy of the given matrices/views and calls nda::linalg::eigvalsh_in_place(A &&, B&&, int) with the 
   * copies.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @param a Input matrix. The matrix \f$ \mathbf{A} \f$.
   * @param b Input matrix. The matrix \f$ \mathbf{B} \f$. 
   * @param itype Specifies the problem to be solved.
   * @return An nda::array containing the real eigenvalues in ascending order.
   */
  template <Matrix A, Matrix B>
    requires(Scalar<get_value_t<A>> and Scalar<get_value_t<B>> and std::is_same_v<get_fp_t<A>, get_fp_t<B>>)
  auto eigvalsh(A const &a, B const &b, int itype = 1) {
    using value_t = std::conditional_t<is_complex_v<get_value_t<A>> or is_complex_v<get_value_t<B>>, std::complex<get_fp_t<A>>, get_fp_t<A>>;
    auto a_copy   = matrix<value_t, F_layout>{a};
    auto b_copy   = matrix<value_t, F_layout>{b};
    return eigvalsh_in_place(a_copy, b_copy, itype);
  }

  /** @} */

} // namespace nda::linalg

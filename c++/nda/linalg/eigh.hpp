// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to solve (generalized) eigenvalue problems with symmetric/hermitian matrices.
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
   * @addtogroup linalg_eig
   * @{
   */

  namespace detail {

    // Perform the call to the LAPACK routines syev/heev for eigh_in_place and eigvalsh_in_place.
    template <typename A>
    auto eigh_impl(A &&a, char jobz) { // NOLINT (temporary views are allowed here)
      using fp_t = get_fp_t<A>;

      // early return if the matrix is empty
      if (a.empty()) return array<fp_t, 1>{};

      // make the call to syev/heev
      auto lambda = array<fp_t, 1>(a.extent(0));
      int info    = 0;
      if constexpr (is_complex_v<get_value_t<A>>) {
        info = lapack::heev(a, lambda, jobz);
      } else {
        info = lapack::syev(a, lambda, jobz);
      }
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eigh_impl: syev/heev routine failed: info = " << info;

      return lambda;
    }

    // Perform the call to the LAPACK routines sygv/hegv for eigh_in_place and eigvalsh_in_place.
    template <typename A, typename B>
    auto eigh_impl(A &&a, B &&b, char jobz, int itype) { // NOLINT (temporary views are allowed here)
      using fp_t = get_fp_t<A>;

      // early return if the matrix is empty
      if (a.empty()) return array<fp_t, 1>{};

      // make the call to sygv/hegv
      auto lambda = array<fp_t, 1>(a.extent(0));
      int info    = 0;
      if constexpr (is_complex_v<get_value_t<A>>) {
        info = lapack::hegv(a, b, lambda, jobz, itype);
      } else {
        info = lapack::sygv(a, b, lambda, jobz, itype);
      }
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::eigh_impl: sygv/hegv routine failed: info = " << info;

      return lambda;
    }

  } // namespace detail

  /**
   * @brief Compute the eigenvalues and eigenvectors of a real symmetric or complex hermitian matrix in place.
   *
   * @details It computes the eigenvectors \f$ \mathbf{v}_i \f$ and eigenvalues \f$ \lambda_i \f$ of the matrix \f$ 
   * \mathbf{A} \f$ such that
   * \f[
   *  \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; .
   * \f]
   * 
   * If \f$ \mathbf{A} \f$ is real, it calls nda::lapack::syev. If \f$ \mathbf{A} \f$ is complex, it calls 
   * nda::lapack::heev.
   * 
   * An exception is thrown, if the LAPACK call fails.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have 
   * nda::F_layout. See nda::linalg::eigh for a version that handles nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it contains the orthonormal 
   * eigenvectors \f$ \mathbf{v}_i \f$ in its columns.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <blas_lapack::BlasArray<2> A>
    requires(mem::have_host_compatible_addr_space<A> and blas_lapack::has_F_layout<A>)
  auto eigh_in_place(A &&a) {
    return detail::eigh_impl(std::forward<A>(a), 'V');
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of a generalized real symmetric-definite or complex 
   * hermitian-definite eigenvalue problem in place.
   *
   * @details It computes the eigenvectors \f$ \mathbf{v}_i \f$ and eigenvalues \f$ \lambda_i \f$ of one of the
   * following eigenvalue problems:
   * 
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   * 
   * Here \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are assumed to be real symmetric or complex hermitian. In addition,
   * \f$ \mathbf{B} \f$ is assumed to be positive definite.
   * 
   * If \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are real, it calls nda::lapack::sygv. If \f$ \mathbf{A} \f$ and \f$ 
   * \mathbf{B} \f$ are complex, it calls nda::lapack::hegv.
   * 
   * An exception is thrown, if the LAPACK call fails.
   *
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are required to satisfy nda::mem::have_host_compatible_addr_space 
   * and to have nda::F_layout. See nda::linalg::eigh for a version that handles nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, it contains the normalized 
   * eigenvectors \f$ \mathbf{v}_i \f$ in its columns (see nda::lapack::sygv or nda::lapack::hegv for details).
   * @param b Input/output matrix. On entry, the matrix \f$ \mathbf{B} \f$. On exit, it is overwritten (see 
   * nda::lapack::sygv or nda::lapack::hegv for details).
   * @param itype Specifies the problem to be solved.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <blas_lapack::BlasArray<2> A, blas_lapack::BlasArrayFor<A, 2> B>
    requires(mem::have_host_compatible_addr_space<A, B> and blas_lapack::has_F_layout<A, B>)
  auto eigh_in_place(A &&a, B &&b, int itype = 1) {
    return detail::eigh_impl(std::forward<A>(a), std::forward<B>(b), 'V', itype);
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of a real symmetric or complex hermitian matrix.
   *
   * @details It makes a copy of the input matrix \f$ \mathbf{A} \f$ and calls nda::linalg::eigh_in_place with the copy.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have a value type
   * that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$.
   * @return `std::pair` containing an nda::array with the real eigenvalues \f$ \lambda_i \f$ in ascending order and an 
   * nda::matrix \f$ \mathbf{V} \f$ in nda::F_layout containing the eigenvectors \f$ \mathbf{v}_i \f$ in its columns.
   */
  template <Matrix A>
    requires(mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto eigh(A const &a) {
    auto a_copy = matrix<get_value_t<A>, F_layout>{a};
    auto lambda = eigh_in_place(a_copy);
    return std::make_pair(lambda, a_copy);
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of a generalized real symmetric-definite or complex 
   * hermitian-definite eigenvalue problem.
   *
   * @details It makes copies of the input matrices \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ and calls 
   * nda::linalg::eigh_in_place(A &&, B&&, int) with the copies.
   * 
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are required to satisfy nda::mem::have_host_compatible_addr_space 
   * and to have the same value type that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$.
   * @param b Input matrix \f$ \mathbf{B} \f$. 
   * @param itype Specifies the problem to be solved.
   * @return `std::pair` containing an nda::array with the real eigenvalues \f$ \lambda_i \f$ in ascending order and an 
   * nda::matrix \f$ \mathbf{V} \f$ in nda::F_layout containing the eigenvectors \f$ \mathbf{v}_i \f$ in its columns.
   */
  template <Matrix A, Matrix B>
    requires(mem::have_host_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, B>)
  auto eigh(A const &a, B const &b, int itype = 1) {
    auto a_copy = matrix<get_value_t<A>, F_layout>{a};
    auto b_copy = matrix<get_value_t<B>, F_layout>{b};
    auto lambda = eigh_in_place(a_copy, b_copy, itype);
    return std::make_pair(lambda, a_copy);
  }

  /**
   * @brief Compute the eigenvalues of a real symmetric or complex hermitian matrix in place.
   *
   * @details It computes the eigenvalues \f$ \lambda_i \f$ of the matrix \f$ \mathbf{A} \f$ such that
   * \f[
   *  \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; .
   * \f]
   * 
   * If \f$ \mathbf{A} \f$ is real, it calls nda::lapack::syev. If \f$ \mathbf{A} \f$ is complex, it calls 
   * nda::lapack::heev.
   * 
   * An exception is thrown, if the LAPACK call fails.
   *
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have 
   * nda::F_layout. See nda::linalg::eigvalsh for a version that handles nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, the contents of \f$ \mathbf{A} \f$ 
   * are destroyed.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <blas_lapack::BlasArray<2> A>
    requires(mem::have_host_compatible_addr_space<A> and blas_lapack::has_F_layout<A>)
  auto eigvalsh_in_place(A &&a) {
    return detail::eigh_impl(std::forward<A>(a), 'N');
  }

  /**
   * @brief Compute the eigenvalues of a generalized real symmetric-definite or complex hermitian-definite eigenvalue 
   * problem in place.
   *
   * @details It computes the eigenvalues \f$ \lambda_i \f$ of one of the following eigenvalue problems:
   * 
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   * 
   * Here \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are assumed to be real symmetric or complex hermitian. In addition,
   * \f$ \mathbf{B} \f$ is assumed to be positive definite.
   * 
   * If \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are real, it calls nda::lapack::sygv. If \f$ \mathbf{A} \f$ and \f$ 
   * \mathbf{B} \f$ are complex, it calls nda::lapack::hegv.
   * 
   * An exception is thrown, if the LAPACK call fails.
   *
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are required to satisfy nda::mem::have_host_compatible_addr_space 
   * and to have nda::F_layout. See nda::linalg::eigvalsh for a version that handles nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, the contents of \f$ \mathbf{A} \f$ 
   * are destroyed.
   * @param b Input/output matrix. On entry, the matrix \f$ \mathbf{B} \f$. On exit, it is overwritten (see 
   * nda::lapack::sygv or nda::lapack::hegv for details).
   * @param itype Specifies the problem to be solved.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <blas_lapack::BlasArray<2> A, blas_lapack::BlasArrayFor<A, 2> B>
    requires(mem::have_host_compatible_addr_space<A, B> and blas_lapack::has_F_layout<A, B>)
  auto eigvalsh_in_place(A &&a, B &&b, int itype = 1) {
    return detail::eigh_impl(std::forward<A>(a), std::forward<B>(b), 'N', itype);
  }

  /**
   * @brief Compute the eigenvalues of a real symmetric or complex hermitian matrix.
   *
   * @details It makes a copy of the input matrix \f$ \mathbf{A} \f$ and calls nda::linalg::eigvalsh_in_place with the 
   * copy.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have a value type
   * that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <Matrix A>
    requires(mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto eigvalsh(A const &a) {
    auto a_copy = matrix<get_value_t<A>, F_layout>{a};
    return eigvalsh_in_place(a_copy);
  }

  /**
   * @brief Compute the eigenvalues of a generalized real symmetric-definite or complex hermitian-definite eigenvalue 
   * problem.
   *
   * @details It makes copies of the input matrices \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ and calls 
   * nda::linalg::eigvalsh_in_place(A &&, B&&, int) with the copies.
   * 
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are required to satisfy nda::mem::have_host_compatible_addr_space 
   * and to have the same value type that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$.
   * @param b Input matrix \f$ \mathbf{B} \f$. 
   * @param itype Specifies the problem to be solved.
   * @return An nda::array containing the real eigenvalues \f$ \lambda_i \f$ in ascending order.
   */
  template <Matrix A, Matrix B>
    requires(mem::have_host_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, B>)
  auto eigvalsh(A const &a, B const &b, int itype = 1) {
    auto a_copy = matrix<get_value_t<A>, F_layout>{a};
    auto b_copy = matrix<get_value_t<B>, F_layout>{b};
    return eigvalsh_in_place(a_copy, b_copy, itype);
  }

  /** @} */

} // namespace nda::linalg

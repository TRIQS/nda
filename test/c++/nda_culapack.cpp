// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <complex>
#include <concepts>
#include <limits>
#include <tuple>
#include <type_traits>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;
using nda::mem::Host, nda::mem::Device, nda::mem::Unified;

// Test the CULAPACK gesvd function.
template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_gesvd() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;

  auto A = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  if constexpr (std::same_as<Layout, C_layout>) {
    // CUDA cannot handle when m < n
    A = matrix_t(nda::transpose(A));
  }
  auto [m, n] = A.shape();

  // expected condition number and spectral norm of A from numpy
  constexpr fp_t cond_A = 6.784414066333698;
  constexpr fp_t norm_A = 12.316822252443167;

  // compute SVD
  auto A_d  = to_addr_space<AS>(A);
  auto U_d  = to_addr_space<AS>(matrix_t(m, m));
  auto VH_d = to_addr_space<AS>(matrix_t(n, n));
  auto s_d  = to_addr_space<AS>(nda::vector<fp_t>(std::min(m, n)));
  nda::lapack::gesvd(A_d, s_d, U_d, VH_d);

  // construct diagonal singular value matrix
  auto s     = nda::to_host(s_d);
  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : nda::range(std::min(m, n))) Sigma(i, i) = s(i);

  // check condition number and spectral norm
  EXPECT_NEAR(s(0) / s(s.size() - 1), cond_A, fp_tol<T>);
  EXPECT_NEAR(s(0), norm_A, fp_tol<T>);

  // check backward error
  EXPECT_ARRAY_NEAR(A, nda::to_host(U_d) * Sigma * nda::to_host(VH_d), fp_tol<T>);
}

template <typename T, typename Layout>
void test_gesvd_address_spaces() {
  test_gesvd<T, Layout, Device>();
  test_gesvd<T, Layout, Unified>();
}

template <typename T>
void test_gesvd_layouts() {
  test_gesvd_address_spaces<T, C_layout>();
  test_gesvd_address_spaces<T, F_layout>();
}

TEST(NDA, CULAPACKGesvd) {
  test_gesvd_layouts<float>();
  test_gesvd_layouts<std::complex<float>>();
  test_gesvd_layouts<double>();
  test_gesvd_layouts<std::complex<double>>();
}

// Test CULAPACK geqrf, orgqr and ungqr functions.
template <typename T, nda::mem::AddressSpace AS, bool wide_matrix = false>
void test_geqrf_orgqr_ungqr() {
  using matrix_t = nda::matrix<T, F_layout>;

  auto A = matrix_t{{{1, 1, 1}, {3, 2, 4}, {5, 3, 2}, {2, 4, 5}, {4, 5, 3}}};
  if constexpr (wide_matrix) A = matrix_t{transpose(A)};
  auto [m, n] = A.shape();

  // compute QR factorization , i.e. A = Q * R
  auto Q_d   = to_addr_space<AS>(A);
  auto tau_d = to_addr_space<AS>(nda::vector<T>(std::min(m, n)));
  nda::lapack::geqrf(Q_d, tau_d);

  // extract upper triangular matrix R
  auto Q = nda::to_host(Q_d);
  auto R = nda::matrix<T, F_layout>::zeros(std::min(m, n), n);
  for (int i = 0; i < std::min(m, n); ++i) {
    for (int j = i; j < n; ++j) { R(i, j) = Q(i, j); }
  }

  // extract matrix Q with orthonormal columns
  if constexpr (std::floating_point<T>) {
    nda::lapack::orgqr(Q_d(nda::range::all, nda::range(std::min(m, n))), tau_d);
  } else {
    nda::lapack::ungqr(Q_d(nda::range::all, nda::range(std::min(m, n))), tau_d);
  }
  Q = nda::to_host(Q_d);

  EXPECT_ARRAY_NEAR(A, Q(nda::range::all, nda::range(std::min(m, n))) * R, fp_tol<T>);
}

template <typename T, bool wide_matrix = false>
void test_geqrf_orgqr_ungqr_address_spaces() {
  test_geqrf_orgqr_ungqr<T, Device, wide_matrix>();
  test_geqrf_orgqr_ungqr<T, Unified, wide_matrix>();
}

TEST(NDA, CULAPACKGeqrfUngqrAndOrgqr) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqrf_orgqr_ungqr_address_spaces<float>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<float>>();
  test_geqrf_orgqr_ungqr_address_spaces<double>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqrf_orgqr_ungqr_address_spaces<float, true>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<float>, true>();
  test_geqrf_orgqr_ungqr_address_spaces<double, true>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<double>, true>();
}

// Test CULAPACK geqrf_batch function.
template <typename T, nda::mem::AddressSpace AS, bool wide_matrix = false>
void test_geqrf_batch() {
  using matrix_t = nda::matrix<T, F_layout>;

  auto A = matrix_t{{{1, 1, 1}, {3, 2, 4}, {5, 3, 2}, {2, 4, 5}, {4, 5, 3}}};
  if constexpr (wide_matrix) A = matrix_t{transpose(A)};
  auto [m, n] = A.shape();

  // create batched arrays by stacking copies of A
  constexpr int batch_size = 3;
  auto A_batch             = nda::array<T, 3, F_layout>(m, n, batch_size);
  for (int i = 0; i < batch_size; ++i) A_batch(nda::range::all, nda::range::all, i) = A;

  // compute batched QR factorization on device
  auto A_batch_d = to_addr_space<AS>(A_batch);
  auto tau_d     = to_addr_space<AS>(nda::matrix<T, F_layout>(std::min(m, n), batch_size));
  nda::lapack::geqrf_batch(A_batch_d, tau_d);

  // bring results back to host for verification
  auto A_batch_h = nda::to_host(A_batch_d);
  auto tau_h     = nda::to_host(tau_d);

  // verify each matrix in the batch
  for (int i = 0; i < batch_size; ++i) {
    auto Q_i   = matrix_t{A_batch_h(nda::range::all, nda::range::all, i)};
    auto tau_i = nda::vector<T>{tau_h(nda::range::all, i)};

    // extract upper triangular matrix R
    auto R_i = matrix_t::zeros(std::min(m, n), n);
    for (int k = 0; k < std::min(m, n); ++k) {
      for (int l = k; l < n; ++l) R_i(k, l) = Q_i(k, l);
    }

    // extract matrix Q with orthonormal columns (use CPU lapack for orgqr/ungqr)
    if constexpr (std::floating_point<T>) {
      nda::lapack::orgqr(Q_i(nda::range::all, nda::range(std::min(m, n))), tau_i);
    } else {
      nda::lapack::ungqr(Q_i(nda::range::all, nda::range(std::min(m, n))), tau_i);
    }

    EXPECT_ARRAY_NEAR(A, Q_i(nda::range::all, nda::range(std::min(m, n))) * R_i, fp_tol<T>);
  }
}

template <typename T, bool wide_matrix = false>
void test_geqrf_batch_address_spaces() {
  test_geqrf_batch<T, Device, wide_matrix>();
  test_geqrf_batch<T, Unified, wide_matrix>();
}

TEST(NDA, CULAPACKGeqrfBatch) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqrf_batch_address_spaces<float>();
  test_geqrf_batch_address_spaces<std::complex<float>>();
  test_geqrf_batch_address_spaces<double>();
  test_geqrf_batch_address_spaces<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqrf_batch_address_spaces<float, true>();
  test_geqrf_batch_address_spaces<std::complex<float>, true>();
  test_geqrf_batch_address_spaces<double, true>();
  test_geqrf_batch_address_spaces<std::complex<double>, true>();
}

// Test the CULAPACK getrs and getrf functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_getrs_getrf() {
  using matrix_t   = nda::matrix<T, Layout>;
  using f_matrix_t = nda::matrix<T, F_layout>;
  using fp_t       = nda::get_fp_t<T>;

  auto A    = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1i};
    Ainv /= T{1i};
  }
  auto B = f_matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // tolerance based on condition number: cond(A) ~ 332, ||Ainv||_max = 24
  // error ~ cond(A) * ||Ainv||_max * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  // solve A * X = B using getrf and getrs
  auto A_d    = to_addr_space<AS1>(A);
  auto B_d    = to_addr_space<AS2>(B);
  auto ipiv_d = to_addr_space<AS1>(nda::array<int, 1>(3));
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(A_d, B_d, ipiv_d);
  auto X = matrix_t{nda::to_host(B_d)};
  EXPECT_ARRAY_NEAR(A * X, B, tol);
  EXPECT_ARRAY_NEAR(Ainv * B, X, tol);

  // solve A^T * X = B using getrf and getrs
  A_d = A;
  B_d = B;
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(nda::transpose(A_d), B_d, ipiv_d);
  X = matrix_t{nda::to_host(B_d)};
  EXPECT_ARRAY_NEAR(nda::transpose(A) * X, B, tol);
  EXPECT_ARRAY_NEAR(nda::transpose(Ainv) * B, X, tol);

  // solve A^H * X = B using getrf and getrs
  if constexpr (nda::blas_lapack::has_F_layout<matrix_t>) {
    A_d = A;
    B_d = B;
    nda::lapack::getrf(A_d, ipiv_d);
    nda::lapack::getrs(nda::dagger(A_d), B_d, ipiv_d);
    X = matrix_t{nda::to_host(B_d)};
    EXPECT_ARRAY_NEAR(nda::dagger(A) * X, B, tol);
    EXPECT_ARRAY_NEAR(nda::dagger(Ainv) * B, X, tol);
  }

  // solve A * x = b using getrf and getrs
  A_d      = A;
  auto b   = B(nda::range::all, 0);
  auto b_d = to_addr_space<AS2>(nda::vector<T>{b});
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(A_d, b_d, ipiv_d);
  auto x = nda::to_host(b_d);
  EXPECT_ARRAY_NEAR(A * x, b, tol);
  EXPECT_ARRAY_NEAR(Ainv * b, x, tol);
}

template <typename T, typename Layout>
void test_getrs_getrf_address_spaces() {
  test_getrs_getrf<T, Layout, Device, Device>();
  test_getrs_getrf<T, Layout, Device, Unified>();
  test_getrs_getrf<T, Layout, Unified, Device>();
  test_getrs_getrf<T, Layout, Unified, Unified>();
  test_getrs_getrf<T, Layout, Unified, Host>();
  test_getrs_getrf<T, Layout, Host, Unified>();
}

template <typename T>
void test_getrs_getrf_layouts() {
  test_getrs_getrf_address_spaces<T, C_layout>();
  test_getrs_getrf_address_spaces<T, F_layout>();
}

TEST(NDA, CULAPACKGetrsAndGetrf) {
  test_getrs_getrf_layouts<float>();
  test_getrs_getrf_layouts<std::complex<float>>();
  test_getrs_getrf_layouts<double>();
  test_getrs_getrf_layouts<std::complex<double>>();
}

template <typename T, nda::mem::AddressSpace AS>
void test_rectangular_getrf() {
  using namespace nda::blas_lapack;
  auto A      = nda::matrix<T, F_layout>{{1, 5}, {4, 5}, {3, 6}};
  auto AT     = nda::matrix<T, F_layout>(nda::transpose(A));
  auto A_c    = nda::matrix<T, C_layout>{A};
  auto AT_c   = nda::matrix<T, C_layout>{AT};
  auto ipiv_d = nda::array<int, 1, C_layout, nda::heap<AS>>(2);

  // get the matrices P, L, U from getrf output
  auto get_plu = [](auto const &M, auto const &ipiv, int m, int n) {
    using layout_t   = std::conditional_t<has_C_layout<decltype(M)>, C_layout, F_layout>;
    auto P           = nda::matrix<T, layout_t>::zeros(m, m);
    auto L           = nda::matrix<T, layout_t>::zeros(m, m);
    auto U           = nda::matrix<T, layout_t>::zeros(m, n);
    nda::diagonal(P) = 1;
    nda::diagonal(L) = 1;
    for (int i = 0; i < ipiv.size(); ++i) deep_swap(P(i, nda::range::all), P(ipiv(i) - 1, nda::range::all));
    for (int i = 0; i < m; ++i) {
      L(i, nda::range(i))    = (has_C_layout<decltype(M)> ? M(nda::range(i), i) : M(i, nda::range(i)));
      U(i, nda::range(i, n)) = (has_C_layout<decltype(M)> ? M(nda::range(i, n), i) : M(i, nda::range(i, n)));
    }
    return std::make_tuple(P, L, U);
  };

  // LU decomposition for 3x2 Fortran layout matrix
  auto LU_f_32 = to_addr_space<AS>(A);
  nda::lapack::getrf(LU_f_32, ipiv_d);
  auto [P_f_32, L_f_32, U_f_32] = get_plu(nda::to_host(LU_f_32), nda::to_host(ipiv_d), 3, 2);
  EXPECT_ARRAY_NEAR(P_f_32 * A, L_f_32 * U_f_32, fp_tol<T>);

  // LU decomposition for 2x3 Fortran layout matrix
  auto LU_f_23 = to_addr_space<AS>(AT);
  nda::lapack::getrf(LU_f_23, ipiv_d);
  auto [P_f_23, L_f_23, U_f_23] = get_plu(nda::to_host(LU_f_23), nda::to_host(ipiv_d), 2, 3);
  EXPECT_ARRAY_NEAR(P_f_23 * AT, L_f_23 * U_f_23, fp_tol<T>);

  // LU decomposition for 3x2 C layout matrix
  auto LU_c_32 = to_addr_space<AS>(A_c);
  nda::lapack::getrf(LU_c_32, ipiv_d);
  auto [P_c_32, L_c_32, U_c_32] = get_plu(nda::to_host(LU_c_32), nda::to_host(ipiv_d), 2, 3);
  EXPECT_ARRAY_NEAR(P_c_32 * nda::transpose(A_c), L_c_32 * U_c_32, fp_tol<T>);

  // LU decomposition for 2x3 C layout matrix
  auto LU_c_23 = to_addr_space<AS>(AT_c);
  nda::lapack::getrf(LU_c_23, ipiv_d);
  auto [P_c_23, L_c_23, U_c_23] = get_plu(nda::to_host(LU_c_23), nda::to_host(ipiv_d), 3, 2);
  EXPECT_ARRAY_NEAR(P_c_23 * nda::transpose(AT_c), L_c_23 * U_c_23, fp_tol<T>);
}

template <typename T>
void test_rectangular_getrf_address_spaces() {
  test_rectangular_getrf<T, Device>();
  test_rectangular_getrf<T, Unified>();
}

TEST(NDA, CULAPACKGetrfWithRectangularMatrix) {
  test_rectangular_getrf_address_spaces<float>();
  test_rectangular_getrf_address_spaces<std::complex<float>>();
  test_rectangular_getrf_address_spaces<double>();
  test_rectangular_getrf_address_spaces<std::complex<double>>();
}

// Test CULAPACK getrf_batch, getrs_batch and getri_batch functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_getrs_getrf_getri_batch() {
  using matrix_t = nda::matrix<T, F_layout>;
  using fp_t     = nda::get_fp_t<T>;

  auto A    = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1i};
    Ainv /= T{1i};
  }
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // tolerance based on condition number: cond(A) ~ 332, ||Ainv||_max = 24
  // error ~ cond(A) * ||Ainv||_max * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  // helper to get the i-th matrix view depending on layout
  auto get_mat = [](auto &arr, int i) {
    if constexpr (nda::blas_lapack::has_F_layout<decltype(arr)>) {
      return nda::matrix_view<T, F_layout>(arr(nda::range::all, nda::range::all, i));
    } else {
      return nda::matrix_view<T, C_layout>(arr(i, nda::range::all, nda::range::all));
    }
  };

  // create batched arrays by stacking copies of the matrices
  constexpr int batch_size = 3;
  auto const [m, n]        = A.shape();
  auto const [k, nrhs]     = B.shape();
  auto a_shape             = std::array<long, 3>{m, n, batch_size};
  auto ipiv_shape          = std::array<long, 2>{m, batch_size};
  if constexpr (std::is_same_v<Layout, C_layout>) {
    a_shape    = {batch_size, m, n};
    ipiv_shape = {batch_size, m};
  }
  auto A_batch = nda::array<T, 3, Layout>(a_shape);
  auto B_batch = nda::array<T, 3, F_layout>(m, nrhs, batch_size);
  for (int i = 0; i < batch_size; ++i) {
    get_mat(A_batch, i) = A;
    get_mat(B_batch, i) = B;
  }

  // solve A * X = B using getrf_batch and getrs_batch
  auto Acopy_d = to_addr_space<AS1>(A_batch);
  auto Bcopy_d = to_addr_space<AS2>(B_batch);
  auto ipiv_d  = to_addr_space<AS1>(nda::matrix<int, Layout>(ipiv_shape));
  nda::lapack::getrf_batch(Acopy_d, ipiv_d);
  nda::lapack::getrs_batch(Acopy_d, Bcopy_d, ipiv_d);
  auto Bcopy = nda::to_host(Bcopy_d);
  for (int i = 0; i < batch_size; ++i) {
    auto X = get_mat(Bcopy, i);
    EXPECT_ARRAY_NEAR(A * X, B, tol);
    EXPECT_ARRAY_NEAR(Ainv * B, X, tol);
  }

  // solve A^T * X = B using getrf_batch and getrs_batch
  Acopy_d = A_batch;
  Bcopy_d = B_batch;
  nda::lapack::getrf_batch(Acopy_d, ipiv_d);
  nda::lapack::getrs_batch(nda::transpose(Acopy_d), Bcopy_d, ipiv_d);
  Bcopy = nda::to_host(Bcopy_d);
  for (int i = 0; i < batch_size; ++i) {
    auto X = get_mat(Bcopy, i);
    EXPECT_ARRAY_NEAR(nda::transpose(A) * X, B, tol);
    EXPECT_ARRAY_NEAR(nda::transpose(Ainv) * B, X, tol);
  }

  // solve A^H * X = B using getrf_batch and getrs_batch (F-layout only)
  if constexpr (std::is_same_v<Layout, F_layout>) {
    Acopy_d = A_batch;
    Bcopy_d = B_batch;
    nda::lapack::getrf_batch(Acopy_d, ipiv_d);
    nda::lapack::getrs_batch(nda::conj(nda::transpose(Acopy_d)), Bcopy_d, ipiv_d);
    Bcopy = nda::to_host(Bcopy_d);
    for (int i = 0; i < batch_size; ++i) {
      auto X = get_mat(Bcopy, i);
      EXPECT_ARRAY_NEAR(nda::conj(nda::transpose(A)) * X, B, tol);
      EXPECT_ARRAY_NEAR(nda::conj(nda::transpose(Ainv)) * B, X, tol);
    }
  }

  // compute the inverse of A using getrf_batch and getri_batch
  Acopy_d = A_batch;
  nda::lapack::getrf_batch(Acopy_d, ipiv_d);
  nda::lapack::getri_batch(Acopy_d, ipiv_d);
  auto Acopy = nda::to_host(Acopy_d);
  for (int i = 0; i < batch_size; ++i) EXPECT_ARRAY_NEAR(Ainv, get_mat(Acopy, i), tol);
}

template <typename T, typename Layout>
void test_getrs_getrf_getri_batch_address_spaces() {
  test_getrs_getrf_getri_batch<T, Layout, Device, Device>();
  test_getrs_getrf_getri_batch<T, Layout, Device, Unified>();
  test_getrs_getrf_getri_batch<T, Layout, Unified, Device>();
  test_getrs_getrf_getri_batch<T, Layout, Unified, Unified>();
  test_getrs_getrf_getri_batch<T, Layout, Unified, Host>();
  test_getrs_getrf_getri_batch<T, Layout, Host, Unified>();
}

template <typename T>
void test_getrs_getrf_getri_batch_layouts() {
  test_getrs_getrf_getri_batch_address_spaces<T, C_layout>();
  test_getrs_getrf_getri_batch_address_spaces<T, F_layout>();
}

TEST(NDA, CULAPACKGetrfGetrsGetriBatch) {
  test_getrs_getrf_getri_batch_layouts<float>();
  test_getrs_getrf_getri_batch_layouts<std::complex<float>>();
  test_getrs_getrf_getri_batch_layouts<double>();
  test_getrs_getrf_getri_batch_layouts<std::complex<double>>();
}

// Test CULAPACK getrf_batch function with rectangular matrices.
// This exercises the fallback loop path in getrf_batch_impl when m != n.
template <typename T, typename Layout, nda::mem::AddressSpace AS, bool wide_matrix = false>
void test_rectangular_getrf_batch() {
  auto A = nda::matrix<T, F_layout>{{1, 5}, {4, 5}, {3, 6}};
  if constexpr (wide_matrix) A = nda::matrix<T, F_layout>(nda::transpose(A));
  auto [m, n] = A.shape();

  // get the matrices P, L, U from getrf output
  auto get_plu = [](auto const &M, auto const &ipiv, int rows, int cols) {
    auto P           = nda::matrix<T, F_layout>::zeros(rows, rows);
    auto L           = nda::matrix<T, F_layout>::zeros(rows, rows);
    auto U           = nda::matrix<T, F_layout>::zeros(rows, cols);
    nda::diagonal(P) = 1;
    nda::diagonal(L) = 1;
    for (int i = 0; i < static_cast<int>(ipiv.size()); ++i) deep_swap(P(i, nda::range::all), P(ipiv(i) - 1, nda::range::all));
    for (int i = 0; i < rows; ++i) {
      L(i, nda::range(i))       = M(i, nda::range(i));
      U(i, nda::range(i, cols)) = M(i, nda::range(i, cols));
    }
    return std::make_tuple(P, L, U);
  };

  // create batched arrays by stacking copies of A
  constexpr int batch_size = 3;
  auto a_shape             = std::array<long, 3>{m, n, batch_size};
  auto ipiv_shape          = std::array<long, 2>{std::min(m, n), batch_size};
  if constexpr (std::is_same_v<Layout, C_layout>) {
    a_shape    = {batch_size, m, n};
    ipiv_shape = {batch_size, std::min(m, n)};
  }
  auto A_batch = nda::array<T, 3, Layout>(a_shape);
  for (int i = 0; i < batch_size; ++i) {
    if constexpr (std::is_same_v<Layout, F_layout>) {
      A_batch(nda::range::all, nda::range::all, i) = A;
    } else {
      A_batch(i, nda::range::all, nda::range::all) = nda::matrix<T, C_layout>{A};
    }
  }

  // compute batched LU factorization on device
  auto A_batch_d = to_addr_space<AS>(A_batch);
  auto ipiv_d    = to_addr_space<AS>(nda::matrix<int, Layout>(ipiv_shape));
  nda::lapack::getrf_batch(A_batch_d, ipiv_d);

  // bring results back to host for verification
  auto A_batch_h = nda::to_host(A_batch_d);
  auto ipiv_h    = nda::to_host(ipiv_d);

  // verify each matrix in the batch
  for (int i = 0; i < batch_size; ++i) {
    nda::matrix<T, F_layout> LU_i;
    nda::vector<int> ipiv_i;
    if constexpr (std::is_same_v<Layout, F_layout>) {
      LU_i   = A_batch_h(nda::range::all, nda::range::all, i);
      ipiv_i = ipiv_h(nda::range::all, i);
    } else {
      LU_i   = nda::matrix<T, F_layout>{nda::transpose(nda::matrix<T, C_layout>{A_batch_h(i, nda::range::all, nda::range::all)})};
      ipiv_i = ipiv_h(i, nda::range::all);
    }

    // for C-layout, the factorization is done on the transposed matrix
    auto A_ref          = A;
    auto [m_ref, n_ref] = A_ref.shape();
    if constexpr (std::is_same_v<Layout, C_layout>) {
      A_ref = nda::matrix<T, F_layout>(nda::transpose(A));
      m_ref = A_ref.extent(0);
      n_ref = A_ref.extent(1);
    }

    auto [P_i, L_i, U_i] = get_plu(LU_i, ipiv_i, m_ref, n_ref);
    EXPECT_ARRAY_NEAR(P_i * A_ref, L_i * U_i, fp_tol<T>);
  }
}

template <typename T, typename Layout, bool wide_matrix = false>
void test_rectangular_getrf_batch_address_spaces() {
  test_rectangular_getrf_batch<T, Layout, Device, wide_matrix>();
  test_rectangular_getrf_batch<T, Layout, Unified, wide_matrix>();
}

template <typename T, bool wide_matrix = false>
void test_rectangular_getrf_batch_layouts() {
  test_rectangular_getrf_batch_address_spaces<T, C_layout, wide_matrix>();
  test_rectangular_getrf_batch_address_spaces<T, F_layout, wide_matrix>();
}

TEST(NDA, CULAPACKGetrfBatchWithRectangularMatrix) {
  // tall matrix, i.e. n_rows > n_cols
  test_rectangular_getrf_batch_layouts<float>();
  test_rectangular_getrf_batch_layouts<double>();

  // wide matrix, i.e. n_rows < n_cols
  test_rectangular_getrf_batch_layouts<float, true>();
  test_rectangular_getrf_batch_layouts<double, true>();
}

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
#include <tuple>
#include <type_traits>

// Test the CULAPACK gesvd function.
template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_gesvd() {
  auto A = nda::matrix<T, Layout>{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  if constexpr (std::same_as<Layout, nda::C_layout>) {
    // CUDA cannot handle when m < n
    A = nda::matrix<T, Layout>(nda::transpose(A));
  }
  auto [m, n] = A.shape();

  auto A_d  = to_addr_space<AS>(A);
  auto U_d  = to_addr_space<AS>(nda::matrix<T, Layout>(m, m));
  auto VT_d = to_addr_space<AS>(nda::matrix<T, Layout>(n, n));
  auto S_d  = to_addr_space<AS>(nda::vector<double>(std::min(m, n)));
  nda::lapack::gesvd(A_d, S_d, U_d, VT_d);

  auto S     = nda::to_host(S_d);
  auto Sigma = nda::matrix<double, Layout>::zeros(A.shape());
  for (auto i : nda::range(std::min(m, n))) Sigma(i, i) = S(i);
  EXPECT_ARRAY_NEAR(A, nda::to_host(U_d) * Sigma * nda::to_host(VT_d), 1e-14);
}

TEST(NDA, CULAPACKGesvd) {
  test_gesvd<double, nda::C_layout, nda::mem::Device>();
  test_gesvd<double, nda::F_layout, nda::mem::Device>();
  test_gesvd<std::complex<double>, nda::C_layout, nda::mem::Device>();
  test_gesvd<std::complex<double>, nda::F_layout, nda::mem::Device>();

  test_gesvd<double, nda::C_layout, nda::mem::Unified>();
  test_gesvd<double, nda::F_layout, nda::mem::Unified>();
  test_gesvd<std::complex<double>, nda::C_layout, nda::mem::Unified>();
  test_gesvd<std::complex<double>, nda::F_layout, nda::mem::Unified>();
}

// Test the CULAPACK getrs and getrf functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_getrs_getrf() {
  T fac = 1.0;
  if constexpr (nda::is_complex_v<T>) fac = 1.0i;

  auto A = nda::matrix<T, Layout>{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  A *= fac;
  auto Ainv = nda::matrix<T, Layout>{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  Ainv /= fac;
  auto B = nda::matrix<T, nda::F_layout>{{1, 5}, {4, 5}, {3, 6}};

  // solve A * X = B using getrf and getrs
  auto A_d    = to_addr_space<AS1>(A);
  auto B_d    = to_addr_space<AS2>(B);
  auto ipiv_d = to_addr_space<AS1>(nda::array<int, 1>(3));
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(A_d, B_d, ipiv_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(B_d), B);
  EXPECT_ARRAY_NEAR(Ainv * B, nda::to_host(B_d));

  // solve A^T * X = B using getrf and getrs
  A_d = A;
  B_d = B;
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(nda::transpose(A_d), B_d, ipiv_d);
  EXPECT_ARRAY_NEAR(nda::transpose(A) * nda::to_host(B_d), B);
  EXPECT_ARRAY_NEAR(nda::transpose(Ainv) * B, nda::to_host(B_d));

  // solve A^H * X = B using getrf and getrs
  if constexpr (std::same_as<Layout, nda::F_layout>) {
    A_d = A;
    B_d = B;
    nda::lapack::getrf(A_d, ipiv_d);
    nda::lapack::getrs(nda::conj(nda::transpose(A_d)), B_d, ipiv_d);
    EXPECT_ARRAY_NEAR(nda::conj(nda::transpose(A)) * nda::to_host(B_d), B);
    EXPECT_ARRAY_NEAR(nda::conj(nda::transpose(Ainv)) * B, nda::to_host(B_d));
  }

  // solve A * x = b using getrf and getrs
  A_d      = A;
  auto b_d = to_addr_space<AS2>(nda::vector<T>{B(nda::range::all, 0)});
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(A_d, b_d, ipiv_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(b_d), B(nda::range::all, 0));
  EXPECT_ARRAY_NEAR((Ainv * B)(nda::range::all, 0), nda::to_host(b_d));
}

TEST(NDA, CULAPACKGetrsAndGetrf) {
  test_getrs_getrf<double, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_getrs_getrf<double, nda::F_layout, nda::mem::Device, nda::mem::Unified>();
  test_getrs_getrf<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Device>();
  test_getrs_getrf<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Unified>();

  test_getrs_getrf<double, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_getrs_getrf<double, nda::F_layout, nda::mem::Host, nda::mem::Unified>();
  test_getrs_getrf<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Host>();
  test_getrs_getrf<std::complex<double>, nda::F_layout, nda::mem::Device, nda::mem::Device>();
}

TEST(NDA, CULAPACKGetrfWithRectangularMatrix) {
  auto A      = nda::matrix<double, nda::F_layout>{{1, 5}, {4, 5}, {3, 6}};
  auto AT     = nda::matrix<double, nda::F_layout>(nda::transpose(A));
  auto A_c    = nda::matrix<double, nda::C_layout>{A};
  auto AT_c   = nda::matrix<double, nda::C_layout>{AT};
  auto ipiv_d = nda::cuarray<int, 1>(2);

  // get the matrices P, L, U from getrf output
  auto get_plu = [](auto const &M, auto const &ipiv, int m, int n) {
    using layout_t   = std::conditional_t<nda::blas::has_C_layout<decltype(M)>, nda::C_layout, nda::F_layout>;
    auto P           = nda::matrix<double, layout_t>::zeros(m, m);
    auto L           = nda::matrix<double, layout_t>::zeros(m, m);
    auto U           = nda::matrix<double, layout_t>::zeros(m, n);
    nda::diagonal(P) = 1;
    nda::diagonal(L) = 1;
    for (int i = 0; i < ipiv.size(); ++i) deep_swap(P(i, nda::range::all), P(ipiv(i) - 1, nda::range::all));
    for (int i = 0; i < m; ++i) {
      L(i, nda::range(i))    = (nda::blas::has_C_layout<decltype(M)> ? M(nda::range(i), i) : M(i, nda::range(i)));
      U(i, nda::range(i, n)) = (nda::blas::has_C_layout<decltype(M)> ? M(nda::range(i, n), i) : M(i, nda::range(i, n)));
    }
    return std::make_tuple(P, L, U);
  };

  // LU decomposition for 3x2 Fortran layout matrix
  auto LU_f_32 = nda::to_device(A);
  nda::lapack::getrf(LU_f_32, ipiv_d);
  auto [P_f_32, L_f_32, U_f_32] = get_plu(nda::to_host(LU_f_32), nda::to_host(ipiv_d), 3, 2);
  EXPECT_ARRAY_NEAR(P_f_32 * A, L_f_32 * U_f_32);

  // LU decomposition for 2x3 Fortran layout matrix
  auto LU_f_23 = nda::to_device(AT);
  nda::lapack::getrf(LU_f_23, ipiv_d);
  auto [P_f_23, L_f_23, U_f_23] = get_plu(nda::to_host(LU_f_23), nda::to_host(ipiv_d), 2, 3);
  EXPECT_ARRAY_NEAR(P_f_23 * AT, L_f_23 * U_f_23);

  // LU decomposition for 3x2 C layout matrix
  auto LU_c_32 = nda::to_device(A_c);
  nda::lapack::getrf(LU_c_32, ipiv_d);
  auto [P_c_32, L_c_32, U_c_32] = get_plu(nda::to_host(LU_c_32), nda::to_host(ipiv_d), 2, 3);
  EXPECT_ARRAY_NEAR(P_c_32 * nda::transpose(A_c), L_c_32 * U_c_32);

  // LU decomposition for 2x3 C layout matrix
  auto LU_c_23 = nda::to_device(AT_c);
  nda::lapack::getrf(LU_c_23, ipiv_d);
  auto [P_c_23, L_c_23, U_c_23] = get_plu(nda::to_host(LU_c_23), nda::to_host(ipiv_d), 3, 2);
  EXPECT_ARRAY_NEAR(P_c_23 * nda::transpose(AT_c), L_c_23 * U_c_23);
}

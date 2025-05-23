// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <concepts>
#include <utility>
#include <vector>

// Test the CUBLAS gemm function.
template <typename T, typename Layout1, typename Layout2, typename Layout3, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2,
          nda::mem::AddressSpace AS3>
void test_gemm() {
  constexpr auto a_is_f_layout = std::same_as<Layout1, nda::F_layout>;
  constexpr auto b_is_f_layout = std::same_as<Layout2, nda::F_layout>;
  constexpr auto c_is_f_layout = std::same_as<Layout3, nda::F_layout>;
  auto A                       = nda::matrix<T, Layout1>{{1, 2, 3}, {4, 5, 6}};
  auto B                       = nda::matrix<T, Layout2>{{1, 2}, {3, 4}, {5, 6}};
  auto exp_C                   = nda::matrix<T, Layout3>{{22, 28}, {49, 64}};
  if constexpr (nda::is_complex_v<T>) {
    A *= 1 - 1i;
    B *= 2 - 1i;
    exp_C *= (1 - 1i) * (2 - 1i);
  }
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);

  // C = A * B
  auto C_d = to_addr_space<AS3>(nda::matrix<T, Layout3>(2, 2));
  nda::blas::gemm(1.0, A_d, B_d, 0.0, C_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C_d), exp_C);

  // C = 3 * A * B + 2 * C
  nda::blas::gemm(3, A_d, B_d, 2, C_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C_d), 5 * exp_C);

  // C_t = B^T * A^T
  auto C_t_d = to_addr_space<AS3>(nda::matrix<T, Layout3>(2, 2));
  nda::blas::gemm(1.0, nda::transpose(B_d), nda::transpose(A_d), 0.0, C_t_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C_t_d), nda::transpose(exp_C));

  // C_h = B^H * A^H
  if constexpr ((a_is_f_layout and b_is_f_layout and c_is_f_layout) or (!a_is_f_layout and !b_is_f_layout and !c_is_f_layout)) {
    auto C_h_d = to_addr_space<AS3>(nda::matrix<T, Layout3>(2, 2));
    nda::blas::gemm(1.0, nda::dagger(B_d), nda::dagger(A_d), 0.0, C_h_d);
    EXPECT_ARRAY_NEAR(nda::to_host(C_h_d), nda::dagger(exp_C));
  }

  // contiguous matrix views
  if constexpr (a_is_f_layout and !b_is_f_layout and !c_is_f_layout) {
    auto exp_C_v = nda::matrix<T, Layout3>{{13, 16}, {37, 46}};
    if constexpr (nda::is_complex_v<T>) exp_C_v *= (1 - 1i) * (2 - 1i);
    auto C_v_d = to_addr_space<AS3>(nda::matrix<T, Layout3>(5, 2));
    nda::blas::gemm(1.0, A_d(nda::range::all, nda::range(0, 2)), B_d(nda::range(1, 3), nda::range::all), 0.0,
                    C_v_d(nda::range(2, 4), nda::range::all));
    EXPECT_ARRAY_NEAR(nda::to_host(C_v_d)(nda::range(2, 4), nda::range::all), exp_C_v);
  }
}

TEST(NDA, CUBLASGemm) {
  // double, C-layout
  test_gemm<double, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemm<double, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Device, nda::mem::Unified, nda::mem::Device>();
  test_gemm<double, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<double, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Host, nda::mem::Unified, nda::mem::Unified>();

  // double, F-layout
  test_gemm<double, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemm<double, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Unified>();
  test_gemm<double, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<double, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Unified>();

  // double, mixed layout
  test_gemm<double, nda::C_layout, nda::F_layout, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemm<double, nda::F_layout, nda::C_layout, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<double, nda::C_layout, nda::F_layout, nda::F_layout, nda::mem::Host, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<double, nda::F_layout, nda::C_layout, nda::C_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Unified>();

  // complex, C-layout
  test_gemm<std::complex<double>, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemm<std::complex<double>, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Device, nda::mem::Unified, nda::mem::Device>();
  test_gemm<std::complex<double>, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<std::complex<double>, nda::C_layout, nda::C_layout, nda::C_layout, nda::mem::Host, nda::mem::Unified, nda::mem::Unified>();

  // complex, F-layout
  test_gemm<std::complex<double>, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemm<std::complex<double>, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Unified>();
  test_gemm<std::complex<double>, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<std::complex<double>, nda::F_layout, nda::F_layout, nda::F_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Unified>();

  // complex, mixed layout
  test_gemm<std::complex<double>, nda::C_layout, nda::F_layout, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemm<std::complex<double>, nda::F_layout, nda::C_layout, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<std::complex<double>, nda::C_layout, nda::F_layout, nda::F_layout, nda::mem::Host, nda::mem::Unified, nda::mem::Unified>();
  test_gemm<std::complex<double>, nda::F_layout, nda::C_layout, nda::C_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Unified>();
}

// Test the CUBLAS/Magma gemm_batch, gemm_vbatch and gemm_batch_strided functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS, bool is_vbatch>
void test_gemm_batch() {
  int const batch_count = 4;
  long size             = 2;
  long fac              = 2;
  if constexpr (!is_vbatch) {
    size = 16;
    fac  = 1;
  }

  // create vector of matrices
  std::vector<nda::matrix<T, Layout, nda::heap<AS>>> vec_A, vec_B, vec_C;
  std::vector<nda::matrix<T, Layout>> exp_C;
  for ([[maybe_unused]] auto i : nda::range(batch_count)) {
    auto A = nda::matrix<T, Layout>::rand({size, size});
    auto B = nda::matrix<T, Layout>::rand({size, size});
    auto C = nda::matrix<T, Layout>::zeros({size, size});
    vec_A.push_back(A);
    vec_B.push_back(B);
    vec_C.push_back(C);
    nda::blas::gemm(1.0, A, B, 0.0, C);
    exp_C.push_back(std::move(C));
    size *= fac;
  }

  // test batched gemm routines
  if constexpr (is_vbatch) {
    nda::blas::gemm_vbatch(1.0, vec_A, vec_B, 0.0, vec_C);
  } else {
    nda::blas::gemm_batch(1.0, vec_A, vec_B, 0.0, vec_C);
  }
  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(nda::to_host(vec_C[i]), exp_C[i]);
}

TEST(NDA, CUBLASGemmBatch) {
  test_gemm_batch<double, nda::C_layout, nda::mem::Device, false>();
  test_gemm_batch<double, nda::F_layout, nda::mem::Device, false>();
  test_gemm_batch<std::complex<double>, nda::C_layout, nda::mem::Device, false>();
  test_gemm_batch<std::complex<double>, nda::F_layout, nda::mem::Device, false>();

  test_gemm_batch<double, nda::C_layout, nda::mem::Unified, false>();
  test_gemm_batch<double, nda::F_layout, nda::mem::Unified, false>();
  test_gemm_batch<std::complex<double>, nda::C_layout, nda::mem::Unified, false>();
  test_gemm_batch<std::complex<double>, nda::F_layout, nda::mem::Unified, false>();
}

#ifdef NDA_HAVE_MAGMA
TEST(NDA, MAGMAGemmVbatch) {
  test_gemm_batch<double, nda::C_layout, nda::mem::Device, true>();
  test_gemm_batch<double, nda::F_layout, nda::mem::Device, true>();
  test_gemm_batch<std::complex<double>, nda::C_layout, nda::mem::Device, true>();
  test_gemm_batch<std::complex<double>, nda::F_layout, nda::mem::Device, true>();

  test_gemm_batch<double, nda::C_layout, nda::mem::Unified, true>();
  test_gemm_batch<double, nda::F_layout, nda::mem::Unified, true>();
  test_gemm_batch<std::complex<double>, nda::C_layout, nda::mem::Unified, true>();
  test_gemm_batch<std::complex<double>, nda::F_layout, nda::mem::Unified, true>();
}
#endif // NDA_HAVE_MAGMA

template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_gemm_batch_strided() {
  int const batch_count = 10;
  long const size       = 16;

  // create arrays
  auto arr_A   = nda::array<T, 3, Layout>::rand({batch_count, size, size});
  auto arr_B   = nda::array<T, 3, Layout>::rand({batch_count, size, size});
  auto arr_C   = nda::array<T, 3, Layout>::zeros({batch_count, size, size});
  auto arr_A_d = nda::array<T, 3, Layout, nda::heap<AS>>{arr_A};
  auto arr_B_d = nda::array<T, 3, Layout, nda::heap<AS>>{arr_B};
  auto arr_C_d = nda::array<T, 3, Layout, nda::heap<AS>>{arr_C};

  // test strided, batched gemm routine
  nda::blas::gemm_batch_strided(1.0, arr_A_d, arr_B_d, 0.0, arr_C_d);
  nda::blas::gemm_batch_strided(1.0, arr_A, arr_B, 0.0, arr_C);
  for (auto i : nda::range(batch_count)) {
    EXPECT_ARRAY_NEAR(nda::to_host(arr_C_d(i, nda::range::all, nda::range::all)), arr_C(i, nda::range::all, nda::range::all));
  }
}

TEST(NDA, BLASGemmBatchStrided) {
  test_gemm_batch_strided<double, nda::C_layout, nda::mem::Device>();
  test_gemm_batch_strided<double, nda::C_layout, nda::mem::Unified>();
  test_gemm_batch_strided<std::complex<double>, nda::C_layout, nda::mem::Device>();
  test_gemm_batch_strided<std::complex<double>, nda::C_layout, nda::mem::Unified>();
}

// Test the CUBLAS gemv function.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2, nda::mem::AddressSpace AS3>
void test_gemv() {
  auto x       = nda::vector<T>{1, 2, 3};
  auto x_t     = nda::vector<T>{1, 2, 3, 4};
  auto exp_y   = nda::vector<T>{14, 32, 50, 68};
  auto exp_y_t = nda::vector<T>{70, 80, 90};
  auto A       = nda::matrix<T, Layout>(4, 3);
  nda::for_each(A.shape(), [&A](auto i, auto j) { A(i, j) = i * 3 + j + 1; });
  if constexpr (nda::is_complex_v<T>) {
    A *= 1 - 1i;
    x *= 2 - 1i;
    x_t *= 2 - 1i;
    exp_y *= (1 - 1i) * (2 - 1i);
    exp_y_t *= (1 - 1i) * (2 - 1i);
  }
  auto A_d   = to_addr_space<AS1>(A);
  auto x_d   = to_addr_space<AS2>(x);
  auto x_t_d = to_addr_space<AS2>(x_t);

  // y = A * x
  auto y_d = to_addr_space<AS3>(nda::vector<T>(4));
  nda::blas::gemv(1.0, A_d, x_d, 0.0, y_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_d), exp_y);

  // y = 3 * A * x + 2y
  nda::blas::gemv(3, A_d, x_d, 2, y_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_d), 5 * exp_y);

  // y_t = A^T * x_t
  auto y_t_d = to_addr_space<AS3>(nda::vector<T>(3));
  nda::blas::gemv(1.0, nda::transpose(A_d), x_t_d, 0.0, y_t_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_t_d), exp_y_t);

  if constexpr (std::same_as<Layout, nda::F_layout>) {
    // y_h = A^H * x_t
    auto exp_y_h = exp_y_t;
    if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{210 + 70i, 240 + 80i, 270 + 90i};
    auto y_h_d = to_addr_space<AS3>(nda::vector<T>(3));
    nda::blas::gemv(1.0, nda::dagger(A_d), x_t_d, 0.0, y_h_d);
    EXPECT_ARRAY_NEAR(nda::to_host(y_h_d), exp_y_h);
  } else {
    // contiguous matrix view * strided vector view
    auto x_v                 = nda::vector<T>(6);
    x_v(nda::range(0, 6, 2)) = x;
    auto x_v_d               = to_addr_space<AS2>(x_v);
    auto y_v_d               = to_addr_space<AS3>(nda::vector<T>(4));
    nda::blas::gemv(1, A_d(nda::range(2), nda::range::all), x_v_d(nda::range(0, 6, 2)), 0, y_v_d(nda::range(0, 4, 2)));
    EXPECT_ARRAY_NEAR(nda::to_host(y_v_d)(nda::range(0, 4, 2)), exp_y(nda::range(2)));
  }
}

TEST(NDA, CUBLASGemv) {
  test_gemv<double, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemv<double, nda::C_layout, nda::mem::Device, nda::mem::Unified, nda::mem::Device>();
  test_gemv<double, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemv<double, nda::C_layout, nda::mem::Host, nda::mem::Unified, nda::mem::Unified>();

  test_gemv<double, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemv<double, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Unified>();
  test_gemv<double, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemv<double, nda::F_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Unified>();

  test_gemv<std::complex<double>, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemv<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Device>();
  test_gemv<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemv<std::complex<double>, nda::C_layout, nda::mem::Host, nda::mem::Host, nda::mem::Unified>();

  test_gemv<std::complex<double>, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device>();
  test_gemv<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Device>();
  test_gemv<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified>();
  test_gemv<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Host>();
}

// Test the CUBLAS ger/gerc function.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2, nda::mem::AddressSpace AS3, bool star>
void test_ger(auto ger) {
  using matrix_t = nda::matrix<T, Layout>;
  T fac          = 1.0;
  if constexpr (nda::is_complex_v<T>) fac = 1.0i;

  // resulting 2 x 2 matrix
  auto exp_M1 = nda::matrix<T>{{1, 2}, {2, 4}};
  if constexpr (nda::is_complex_v<T> and not star) exp_M1 *= -1;
  auto M1_d = to_addr_space<AS1>(matrix_t::zeros(2, 2));
  nda::vector<T> v{1, 2};
  v *= fac;
  auto v_d = to_addr_space<AS2>(v);
  ger(1.0, v_d, v_d, M1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M1_d), exp_M1);
  ger(1.0, v_d, v_d, M1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M1_d), exp_M1 * 2);

  // resulting 2 x 3 matrix
  auto exp_M2 = nda::matrix<T>{{3, 4, 5}, {6, 8, 10}};
  if constexpr (nda::is_complex_v<T>) exp_M2 *= fac;
  auto M2_d = to_addr_space<AS1>(matrix_t::zeros(2, 3));
  auto w_d  = to_addr_space<AS3>(nda::vector<T>{3, 4, 5});
  ger(1.0, v_d, w_d, M2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M2_d), exp_M2);
  ger(1.0, v_d, w_d, M2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M2_d), exp_M2 * 2);

  // resulting 3 x 2 matrix
  auto exp_M3 = nda::matrix<T>{{3, 6}, {4, 8}, {5, 10}};
  if constexpr (nda::is_complex_v<T> and not star) exp_M3 *= fac;
  if constexpr (nda::is_complex_v<T> and star) exp_M3 *= -fac;
  auto M3_d = to_addr_space<AS1>(matrix_t::zeros(3, 2));
  ger(1.0, w_d, v_d, M3_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M3_d), exp_M3);
  ger(1.0, w_d, v_d, M3_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M3_d), exp_M3 * 2);

  // outer product of strided views
  auto exp_M4 = nda::matrix<T>{{6, 8, 10}, {12, 16, 20}};
  if constexpr (nda::is_complex_v<T> and not star) exp_M4 *= -1.0;
  auto M4_d      = to_addr_space<AS1>(matrix_t::zeros(2, 3));
  auto v_strided = nda::vector<T>{0, 1, 0, 2, 0};
  v_strided *= fac;
  auto v_strided_d = to_addr_space<AS2>(v_strided);
  auto w_strided   = nda::vector<T>{3, 0, 0, 4, 0, 0, 5};
  w_strided *= fac;
  auto w_strided_d = to_addr_space<AS3>(w_strided);
  ger(2.0, v_strided_d(nda::range(1, 5, 2)), w_strided_d(nda::range(0, 7, 3)), M4_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M4_d), exp_M4);
}

TEST(NDA, CUBLASGer) {
  auto ger = [](auto alpha, auto &&x, auto &&y, auto &&m) { return nda::blas::ger(alpha, x, y, m); };
  test_ger<double, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device, false>(ger);
  test_ger<double, nda::C_layout, nda::mem::Device, nda::mem::Unified, nda::mem::Unified, false>(ger);
  test_ger<double, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified, false>(ger);
  test_ger<double, nda::C_layout, nda::mem::Host, nda::mem::Host, nda::mem::Unified, false>(ger);

  test_ger<double, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device, false>(ger);
  test_ger<double, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Unified, false>(ger);
  test_ger<double, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified, false>(ger);
  test_ger<double, nda::F_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Unified, false>(ger);

  test_ger<std::complex<double>, nda::C_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device, false>(ger);
  test_ger<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Device, false>(ger);
  test_ger<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified, false>(ger);
  test_ger<std::complex<double>, nda::C_layout, nda::mem::Host, nda::mem::Unified, nda::mem::Unified, false>(ger);

  test_ger<std::complex<double>, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device, false>(ger);
  test_ger<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Device, nda::mem::Device, false>(ger);
  test_ger<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified, false>(ger);
  test_ger<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Host, nda::mem::Host, false>(ger);
}

TEST(NDA, CUBLASGerc) {
  auto gerc = [](auto alpha, auto &&x, auto &&y, auto &&m) { return nda::blas::gerc(alpha, x, y, m); };
  test_ger<double, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device, true>(gerc);
  test_ger<double, nda::F_layout, nda::mem::Device, nda::mem::Unified, nda::mem::Device, true>(gerc);
  test_ger<double, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified, true>(gerc);
  test_ger<double, nda::F_layout, nda::mem::Host, nda::mem::Host, nda::mem::Unified, true>(gerc);

  test_ger<std::complex<double>, nda::F_layout, nda::mem::Device, nda::mem::Device, nda::mem::Device, true>(gerc);
  test_ger<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Device, nda::mem::Unified, true>(gerc);
  test_ger<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Unified, nda::mem::Unified, true>(gerc);
  test_ger<std::complex<double>, nda::F_layout, nda::mem::Host, nda::mem::Unified, nda::mem::Unified, true>(gerc);
}

// Test the CUBLAS dot/dotc function.
template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2, bool star>
void test_dot(auto dot) {
  auto exp_dot = [](auto const &a, auto const &b) {
    T res = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      if constexpr (star and nda::is_complex_v<T>) {
        res += std::conj(a(i)) * b(i);
      } else {
        res += a(i) * b(i);
      }
    }
    return res;
  };
  nda::vector<T> a{1, 2, 3, 4, 5};
  nda::vector<T> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<T>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }
  auto a_d = to_addr_space<AS1>(a);
  auto b_d = to_addr_space<AS2>(b);

  // vector dot vector
  EXPECT_COMPLEX_NEAR(dot(a_d, b_d), exp_dot(a, b), 1.e-14);

  // size 0 vectors
  EXPECT_EQ(dot(to_addr_space<AS1>(nda::vector<T>{}), to_addr_space<AS2>(nda::vector<T>{})), T(0));

  // strided vector dot strided vector
  EXPECT_COMPLEX_NEAR(dot(a_d(nda::range(0, 5, 2)), b_d(nda::range(0, 5, 2))), exp_dot(a(nda::range(0, 5, 2)), b(nda::range(0, 5, 2))), 1.e-14);
}

TEST(NDA, CUBLASDot) {
  auto dot = []<typename A, typename B>(A &&a, B &&b) { return nda::blas::dot(std::forward<A>(a), std::forward<B>(b)); };
  test_dot<double, nda::mem::Device, nda::mem::Device, false>(dot);
  test_dot<double, nda::mem::Device, nda::mem::Unified, false>(dot);
  test_dot<double, nda::mem::Unified, nda::mem::Unified, false>(dot);
  test_dot<double, nda::mem::Unified, nda::mem::Host, false>(dot);
  test_dot<std::complex<double>, nda::mem::Device, nda::mem::Device, false>(dot);
  test_dot<std::complex<double>, nda::mem::Unified, nda::mem::Device, false>(dot);
  test_dot<std::complex<double>, nda::mem::Unified, nda::mem::Unified, false>(dot);
  test_dot<std::complex<double>, nda::mem::Host, nda::mem::Unified, false>(dot);
}

TEST(NDA, CUBLASDotc) {
  auto dotc = []<typename A, typename B>(A &&a, B &&b) { return nda::blas::dotc(std::forward<A>(a), std::forward<B>(b)); };
  test_dot<double, nda::mem::Device, nda::mem::Device, true>(dotc);
  test_dot<double, nda::mem::Device, nda::mem::Unified, true>(dotc);
  test_dot<double, nda::mem::Unified, nda::mem::Unified, true>(dotc);
  test_dot<double, nda::mem::Unified, nda::mem::Host, true>(dotc);
  test_dot<std::complex<double>, nda::mem::Device, nda::mem::Device, true>(dotc);
  test_dot<std::complex<double>, nda::mem::Unified, nda::mem::Device, true>(dotc);
  test_dot<std::complex<double>, nda::mem::Unified, nda::mem::Unified, true>(dotc);
  test_dot<std::complex<double>, nda::mem::Host, nda::mem::Unified, true>(dotc);
}

// Test the CUBLAS scal function.
template <nda::mem::AddressSpace AS>
void test_scal() {
  // empty vector
  nda::vector<double> v;
  auto v_d = to_addr_space<AS>(v);
  nda::blas::scal(3.0, v_d);
  EXPECT_TRUE(v_d.empty());

  // scale a double vector by a double
  v   = {1, 2, 3, 4, 5};
  v_d = to_addr_space<AS>(v);
  nda::blas::scal(3.0, v_d);
  EXPECT_ARRAY_NEAR(nda::to_host(v_d), 3.0 * v);

  // scale a double vector by an integer
  v_d = to_addr_space<AS>(v);
  nda::blas::scal(3, v_d);
  EXPECT_ARRAY_NEAR(nda::to_host(v_d), 3 * v);

  // scale a complex double vector by a double
  auto vc = nda::vector<std::complex<double>>{1, 2, 3, 4, 5};
  vc *= 1 - 1i;
  auto vc_d = to_addr_space<AS>(vc);
  nda::blas::scal(3.0, vc_d);
  EXPECT_ARRAY_NEAR(nda::to_host(vc_d), 3.0 * vc);

  // scale by a complex double
  vc_d = to_addr_space<AS>(vc);
  nda::blas::scal(3.0 + 2.0i, vc_d);
  EXPECT_ARRAY_NEAR(nda::to_host(vc_d), (3.0 + 2.0i) * vc);
}

TEST(NDA, CUBLASScal) {
  test_scal<nda::mem::Device>();
  test_scal<nda::mem::Unified>();
}

// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <vector>
#include <utility>

using namespace std::complex_literals;

// Test the BLAS gemm function.
template <typename T, typename Layout1, typename Layout2, typename Layout3>
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

  // C = A * B
  auto C = nda::matrix<T, Layout3>(2, 2);
  nda::blas::gemm(1.0, A, B, 0.0, C);
  EXPECT_ARRAY_NEAR(C, exp_C);

  // C = 3 * A * B + 2 * C
  nda::blas::gemm(3, A, B, 2, C);
  EXPECT_ARRAY_NEAR(C, 5 * exp_C);

  // C_t = B^T * A^T
  auto C_t = nda::matrix<T, Layout3>(2, 2);
  nda::blas::gemm(1.0, nda::transpose(B), nda::transpose(A), 0.0, C_t);
  EXPECT_ARRAY_NEAR(C_t, nda::transpose(exp_C));

  // C_h = B^H * A^H
  if constexpr ((a_is_f_layout and b_is_f_layout and c_is_f_layout) or (!a_is_f_layout and !b_is_f_layout and !c_is_f_layout)) {
    auto C_h = nda::matrix<T, Layout3>(2, 2);
    nda::blas::gemm(1.0, nda::dagger(B), nda::dagger(A), 0.0, C_h);
    EXPECT_ARRAY_NEAR(C_h, nda::dagger(exp_C));
  }

  // contiguous matrix views
  if constexpr (a_is_f_layout and !b_is_f_layout and !c_is_f_layout) {
    auto exp_C_v = nda::matrix<T, Layout3>{{13, 16}, {37, 46}};
    if constexpr (nda::is_complex_v<T>) exp_C_v *= (1 - 1i) * (2 - 1i);
    auto C_v = nda::matrix<T, Layout3>(5, 2);
    nda::blas::gemm(1.0, A(nda::range::all, nda::range(0, 2)), B(nda::range(1, 3), nda::range::all), 0.0, C_v(nda::range(2, 4), nda::range::all));
    EXPECT_ARRAY_NEAR(C_v(nda::range(2, 4), nda::range::all), exp_C_v);
  }
}

template <typename T>
constexpr auto test_gemm_layouts = []() {
  test_gemm<T, nda::C_layout, nda::C_layout, nda::C_layout>();
  test_gemm<T, nda::C_layout, nda::C_layout, nda::F_layout>();
  test_gemm<T, nda::C_layout, nda::F_layout, nda::C_layout>();
  test_gemm<T, nda::C_layout, nda::F_layout, nda::F_layout>();
  test_gemm<T, nda::F_layout, nda::C_layout, nda::C_layout>();
  test_gemm<T, nda::F_layout, nda::C_layout, nda::F_layout>();
  test_gemm<T, nda::F_layout, nda::F_layout, nda::C_layout>();
  test_gemm<T, nda::F_layout, nda::F_layout, nda::F_layout>();
};

TEST(NDA, BLASGemm) {
  test_gemm_layouts<float>();
  test_gemm_layouts<std::complex<float>>();
  test_gemm_layouts<double>();
  test_gemm_layouts<std::complex<double>>();
}

// Test the BLAS gemm_batch, gemm_vbatch and gemm_batch_strided functions.
template <typename T, typename Layout, bool is_vbatch>
void test_gemm_batch() {
  int const batch_count = 4;
  long size             = 2;
  long fac              = 2;
  if constexpr (!is_vbatch) {
    size = 16;
    fac  = 1;
  }

  // create vector of matrices
  std::vector<nda::matrix<T, Layout>> vec_A, vec_B, vec_C, exp_C;
  for ([[maybe_unused]] auto i : nda::range(batch_count)) {
    vec_A.push_back(nda::matrix<T, Layout>::rand({size, size}));
    vec_B.push_back(nda::matrix<T, Layout>::rand({size, size}));
    vec_C.push_back(nda::matrix<T, Layout>::zeros({size, size}));
    auto tmp = nda::matrix<T, Layout>::zeros({size, size});
    nda::blas::gemm(1.0, vec_A.back(), vec_B.back(), 0.0, tmp);
    exp_C.push_back(std::move(tmp));
    size *= fac;
  }

  // test batched gemm routines
  if constexpr (is_vbatch) {
    nda::blas::gemm_vbatch(1.0, vec_A, vec_B, 0.0, vec_C);
  } else {
    nda::blas::gemm_batch(1.0, vec_A, vec_B, 0.0, vec_C);
  }
  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(vec_C[i], exp_C[i]);
}

TEST(NDA, BLASGemmBatch) {
  test_gemm_batch<double, nda::C_layout, false>();
  test_gemm_batch<double, nda::F_layout, false>();
  test_gemm_batch<std::complex<double>, nda::C_layout, false>();
  test_gemm_batch<std::complex<double>, nda::F_layout, false>();
}

TEST(NDA, BLASGemmVbatch) {
  test_gemm_batch<double, nda::C_layout, true>();
  test_gemm_batch<double, nda::F_layout, true>();
  test_gemm_batch<std::complex<double>, nda::C_layout, true>();
  test_gemm_batch<std::complex<double>, nda::F_layout, true>();
}

template <typename T, typename Layout>
void test_gemm_batch_strided() {
  int const batch_count = 10;
  long const size       = 16;

  // create arrays
  auto arr_A = nda::array<T, 3, Layout>::rand({batch_count, size, size});
  auto arr_B = nda::array<T, 3, Layout>::rand({batch_count, size, size});
  auto arr_C = nda::array<T, 3, Layout>::zeros({batch_count, size, size});

  // test strided, batched gemm routine
  nda::blas::gemm_batch_strided(1.0, arr_A, arr_B, 0.0, arr_C);
  for (auto i : nda::range(batch_count)) {
    auto tmp = nda::matrix<T, Layout>::zeros({size, size});
    nda::blas::gemm(1.0, arr_A(i, nda::range::all, nda::range::all), arr_B(i, nda::range::all, nda::range::all), 0.0, tmp);
    EXPECT_ARRAY_NEAR(arr_C(i, nda::range::all, nda::range::all), tmp);
  }
}

TEST(NDA, BLASGemmBatchStrided) {
  test_gemm_batch_strided<double, nda::C_layout>();
  test_gemm_batch_strided<std::complex<double>, nda::C_layout>();
}

// Test the BLAS gemv function.
template <typename T, typename Layout>
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

  // y = A * x
  auto y = nda::vector<T>(4);
  nda::blas::gemv(1.0, A, x, 0.0, y);
  EXPECT_ARRAY_NEAR(y, exp_y);

  // y = 3 * A * x + 2y
  nda::blas::gemv(3, A, x, 2, y);
  EXPECT_ARRAY_NEAR(y, 5 * exp_y);

  // y_t = A^T * x_t
  auto y_t = nda::vector<T>(3);
  nda::blas::gemv(1.0, nda::transpose(A), x_t, 0.0, y_t);
  EXPECT_ARRAY_NEAR(y_t, exp_y_t);

  if constexpr (std::same_as<Layout, nda::F_layout>) {
    // y_h = A^H * x_t
    auto exp_y_h = exp_y_t;
    if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{T{210 + 70i}, T{240 + 80i}, T{270 + 90i}};
    auto y_h = nda::vector<T>(3);
    nda::blas::gemv(1.0, nda::dagger(A), x_t, 0.0, y_h);
    EXPECT_ARRAY_NEAR(y_h, exp_y_h);
  } else {
    // contiguous matrix view * strided vector view
    auto x_v                 = nda::vector<T>(6);
    x_v(nda::range(0, 6, 2)) = x;
    auto y_v                 = nda::vector<T>(4);
    nda::blas::gemv(1, A(nda::range(2), nda::range::all), x_v(nda::range(0, 6, 2)), 0, y(nda::range(0, 4, 2)));
    EXPECT_ARRAY_NEAR(y(nda::range(0, 4, 2)), exp_y(nda::range(2)));
  }
}

TEST(NDA, BLASGemv) {
  test_gemv<float, nda::C_layout>();
  test_gemv<float, nda::F_layout>();
  test_gemv<std::complex<float>, nda::C_layout>();
  test_gemv<std::complex<float>, nda::F_layout>();
  test_gemv<double, nda::C_layout>();
  test_gemv<double, nda::F_layout>();
  test_gemv<std::complex<double>, nda::C_layout>();
  test_gemv<std::complex<double>, nda::F_layout>();
}

// Test the BLAS ger/gerc function.
template <typename T, typename Layout, bool star>
void test_ger(auto ger) {
  T fac = 1.0;
  if constexpr (nda::is_complex_v<T>) fac = 1.0i;

  // resulting 2 x 2 matrix
  auto exp_M1 = nda::matrix<T>{{1, 2}, {2, 4}};
  if constexpr (nda::is_complex_v<T> and not star) exp_M1 *= -1;
  auto M1 = nda::matrix<T, Layout>::zeros(2, 2);
  nda::vector<T> v{1, 2};
  v *= fac;
  ger(1.0, v, v, M1);
  EXPECT_ARRAY_NEAR(M1, exp_M1);
  ger(1.0, v, v, M1);
  EXPECT_ARRAY_NEAR(M1, exp_M1 * 2);

  // resulting 2 x 3 matrix
  auto exp_M2 = nda::matrix<T>{{3, 4, 5}, {6, 8, 10}};
  if constexpr (nda::is_complex_v<T>) exp_M2 *= fac;
  auto M2 = nda::matrix<T, Layout>::zeros(2, 3);
  nda::vector<T> w{3, 4, 5};
  ger(1.0, v, w, M2);
  EXPECT_ARRAY_NEAR(M2, exp_M2);
  ger(1.0, v, w, M2);
  EXPECT_ARRAY_NEAR(M2, exp_M2 * 2);

  // resulting 3 x 2 matrix
  auto exp_M3 = nda::matrix<T>{{3, 6}, {4, 8}, {5, 10}};
  if constexpr (nda::is_complex_v<T> and not star) exp_M3 *= fac;
  if constexpr (nda::is_complex_v<T> and star) exp_M3 *= -fac;
  auto M3 = nda::matrix<T, Layout>::zeros(3, 2);
  ger(1.0, w, v, M3);
  EXPECT_ARRAY_NEAR(M3, exp_M3);
  ger(1.0, w, v, M3);
  EXPECT_ARRAY_NEAR(M3, exp_M3 * 2);

  // outer product of strided views
  auto exp_M4 = nda::matrix<T>{{6, 8, 10}, {12, 16, 20}};
  if constexpr (nda::is_complex_v<T> and not star) exp_M4 *= -1.0;
  auto M4        = nda::matrix<T, Layout>::zeros(2, 3);
  auto v_strided = nda::vector<T>{0, 1, 0, 2, 0};
  v_strided *= fac;
  auto w_strided = nda::vector<T>{3, 0, 0, 4, 0, 0, 5};
  w_strided *= fac;
  ger(2.0, v_strided(nda::range(1, 5, 2)), w_strided(nda::range(0, 7, 3)), M4);
  EXPECT_ARRAY_NEAR(M4, exp_M4);
}

TEST(NDA, BLASGer) {
  auto ger = [](auto alpha, auto &&x, auto &&y, auto &&m) { return nda::blas::ger(alpha, x, y, m); };
  test_ger<float, nda::C_layout, false>(ger);
  test_ger<float, nda::F_layout, false>(ger);
  test_ger<std::complex<float>, nda::C_layout, false>(ger);
  test_ger<std::complex<float>, nda::F_layout, false>(ger);
  test_ger<double, nda::C_layout, false>(ger);
  test_ger<double, nda::F_layout, false>(ger);
  test_ger<std::complex<double>, nda::C_layout, false>(ger);
  test_ger<std::complex<double>, nda::F_layout, false>(ger);
}

TEST(NDA, BLASGerc) {
  auto gerc = [](auto alpha, auto &&x, auto &&y, auto &&m) { return nda::blas::gerc(alpha, x, y, m); };
  test_ger<float, nda::F_layout, true>(gerc);
  test_ger<std::complex<float>, nda::F_layout, true>(gerc);
  test_ger<double, nda::F_layout, true>(gerc);
  test_ger<std::complex<double>, nda::F_layout, true>(gerc);
}

// Test the BLAS dot/dotc function.
template <typename T, bool star>
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

  // vector dot vector
  EXPECT_COMPLEX_NEAR(dot(a, b), exp_dot(a, b), 1.e-14);

  // size 0 vectors
  EXPECT_EQ(dot(nda::vector<T>{}, nda::vector<T>{}), T(0));

  // strided vector dot strided vector
  auto a_v = a(nda::range(0, 5, 2));
  auto b_v = b(nda::range(0, 5, 2));
  EXPECT_COMPLEX_NEAR(dot(a_v, b_v), exp_dot(a_v, b_v), 1.e-14);
}

TEST(NDA, BLASDot) {
  auto dot = []<typename A, typename B>(A &&a, B &&b) { return nda::blas::dot(std::forward<A>(a), std::forward<B>(b)); };
  test_dot<float, false>(dot);
  test_dot<std::complex<float>, false>(dot);
  test_dot<double, false>(dot);
  test_dot<std::complex<double>, false>(dot);
}

TEST(NDA, BLASDotc) {
  auto dotc = []<typename A, typename B>(A &&a, B &&b) { return nda::blas::dotc(std::forward<A>(a), std::forward<B>(b)); };
  test_dot<float, true>(dotc);
  test_dot<std::complex<float>, true>(dotc);
  test_dot<double, true>(dotc);
  test_dot<std::complex<double>, true>(dotc);
}

// Test the BLAS scal function.
TEST(NDA, BLASScalEmptyVector) {
  nda::vector<double> v;
  nda::blas::scal(3.0, v);
  EXPECT_TRUE(v.empty());
}

TEST(NDA, BLASScalFloat) {
  nda::vector<float> v{1, 2, 3, 4, 5};

  // scale by a float
  auto v1 = v;
  auto xd = 3.0f;
  nda::blas::scal(xd, v1);
  EXPECT_ARRAY_NEAR(v1, xd * v);

  // scale by an integer
  auto v2 = v;
  // int32/auto is fine, but int16 avoids type narrowing
  int16_t xi = 3;
  nda::blas::scal(xi, v2);
  EXPECT_ARRAY_NEAR(v2, xi * v);
}

TEST(NDA, BLASScalSComplex) {
  nda::vector<std::complex<float>> v{1, 2, 3, 4, 5};
  v *= 1 - 1i;

  // scale by a float
  auto v1 = v;
  auto xd = 3.0f;
  nda::blas::scal(xd, v1);
  EXPECT_ARRAY_NEAR(v1, xd * v);

  // scale by a complex float
  auto v2 = v;
  auto xc = 3.0f + 2.0if;
  nda::blas::scal(xc, v2);
  EXPECT_ARRAY_NEAR(v2, xc * v);
}

TEST(NDA, BLASScalDouble) {
  nda::vector<double> v{1, 2, 3, 4, 5};

  // scale by a double
  auto v1 = v;
  auto xd = 3.0;
  nda::blas::scal(xd, v1);
  EXPECT_ARRAY_NEAR(v1, xd * v);

  // scale by an integer
  auto v2 = v;
  auto xi = 3;
  nda::blas::scal(xi, v2);
  EXPECT_ARRAY_NEAR(v2, xi * v);
}

TEST(NDA, BLASScalDComplex) {
  nda::vector<std::complex<double>> v{1, 2, 3, 4, 5};
  v *= 1 - 1i;

  // scale by a double
  auto v1 = v;
  auto xd = 3.0;
  nda::blas::scal(xd, v1);
  EXPECT_ARRAY_NEAR(v1, xd * v);

  // scale by a complex double
  auto v2 = v;
  auto xc = 3.0 + 2.0i;
  nda::blas::scal(xc, v2);
  EXPECT_ARRAY_NEAR(v2, xc * v);
}

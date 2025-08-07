// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <vector>

// Test the BLAS gemm function and its generic implementation.
template <typename value_t, typename Layout>
void test_gemm() {
  nda::matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3{{1, 0}, {0, 1}}, M3_gen;
  M3_gen = M3;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);
  EXPECT_ARRAY_NEAR(M1, nda::matrix<value_t>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<value_t>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});

  nda::blas::gemm_generic(1.0, M1, M2, 1.0, M3_gen);
  EXPECT_ARRAY_NEAR(M1, nda::matrix<value_t>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<value_t>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3_gen, nda::matrix<value_t>{{2, 1}, {3, 4}});
}

TEST(NDA, BLASGemm) {
  test_gemm<double, nda::C_layout>();
  test_gemm<double, nda::F_layout>();
  test_gemm<std::complex<double>, nda::C_layout>();
  test_gemm<std::complex<double>, nda::F_layout>();
}

// Test the BLAS gemm_batch function.
template <typename value_t, typename Layout>
void test_gemm_batch() {
  int batch_count = 10;
  long size       = 64;

  auto vec_A = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_B = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_C = std::vector(batch_count, nda::matrix<value_t, Layout>::zeros({size, size}));
  nda::blas::gemm_batch(1.0, vec_A, vec_B, 0.0, vec_C);

  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(make_regular(vec_A[i] * vec_B[i]), vec_C[i]);
}

TEST(NDA, BLASGemmBatch) {
  test_gemm_batch<double, nda::C_layout>();
  test_gemm_batch<double, nda::F_layout>();
  test_gemm_batch<std::complex<double>, nda::C_layout>();
  test_gemm_batch<std::complex<double>, nda::F_layout>();
}

// Test the BLAS gemm_vbatch function.
template <typename value_t, typename Layout>
void test_gemm_vbatch() {
  int batch_count = 10;
  long size       = 64;

  auto vec_A = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_B = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_C = std::vector(batch_count, nda::matrix<value_t, Layout>::zeros({size, size}));
  nda::blas::gemm_vbatch(1.0, vec_A, vec_B, 0.0, vec_C);

  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(make_regular(vec_A[i] * vec_B[i]), vec_C[i]);
}

TEST(NDA, BLASGemmVbatch) {
  test_gemm_vbatch<double, nda::C_layout>();
  test_gemm_vbatch<double, nda::F_layout>();
  test_gemm_vbatch<std::complex<double>, nda::C_layout>();
  test_gemm_vbatch<std::complex<double>, nda::F_layout>();
}

// Test the BLAS gemv function and its generic implementation.
template <typename value_t, typename Layout>
void test_gemv() {
  using namespace nda::clef::literals;

  nda::matrix<value_t, Layout> A(5, 5);
  A(i_, j_) << i_ + 2 * j_ + 1;

  nda::vector<value_t> v(5), w(5);
  v() = 1;
  w() = 0;

  nda::range rg(1, 3);
  nda::blas::gemv(1, A(rg, rg), v(rg), 0, w(rg));
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{0, 10, 12, 0, 0});

  nda::vector<value_t> w_gen(5);
  w_gen() = 0;
  nda::blas::gemv_generic(1, A(rg, rg), v(rg), 0, w_gen(rg));
  EXPECT_ARRAY_NEAR(w_gen, nda::vector<value_t>{0, 10, 12, 0, 0});

  auto AT = nda::make_regular(transpose(A));
  nda::blas::gemv(1, AT(rg, rg), v(rg), 0, w(rg));
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{0, 9, 13, 0, 0});

  nda::blas::gemv_generic(1, AT(rg, rg), v(rg), 0, w_gen(rg));
  EXPECT_ARRAY_NEAR(w_gen, nda::vector<value_t>{0, 9, 13, 0, 0});

  // test operator*
  w()   = -8;
  w(rg) = AT(rg, rg) * v(rg);
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{-8, 9, 13, -8, -8});
}

TEST(NDA, BLASGemv) {
  test_gemv<double, nda::C_layout>();
  test_gemv<double, nda::F_layout>();
  test_gemv<std::complex<double>, nda::C_layout>();
  test_gemv<std::complex<double>, nda::F_layout>();
}

// Test the BLAS ger function.
template <typename T, typename Layout>
void test_ger() {
  // resulting 2 x 2 matrix
  auto M1 = nda::matrix<T, Layout>::zeros(2, 2);
  nda::vector<T> v{1, 2};
  nda::blas::ger(1.0, v, v, M1);
  EXPECT_ARRAY_NEAR(M1, nda::matrix<T>{{1, 2}, {2, 4}});
  nda::blas::ger(1.0, v, v, M1);
  EXPECT_ARRAY_NEAR(M1, nda::matrix<T>{{2, 4}, {4, 8}});

  // resulting 2 x 3 matrix
  auto M2 = nda::matrix<T, Layout>::zeros(2, 3);
  nda::vector<T> w{3, 4, 5};
  nda::blas::ger(1.0, v, w, M2);
  EXPECT_ARRAY_NEAR(M2, nda::matrix<T>{{3, 4, 5}, {6, 8, 10}});
  nda::blas::ger(1.0, v, w, M2);
  EXPECT_ARRAY_NEAR(M2, nda::matrix<T>{{6, 8, 10}, {12, 16, 20}});

  // resulting 3 x 2 matrix
  auto M3 = nda::matrix<T, Layout>::zeros(3, 2);
  nda::blas::ger(1.0, w, v, M3);
  EXPECT_ARRAY_NEAR(M3, nda::matrix<T>{{3, 6}, {4, 8}, {5, 10}});
  nda::blas::ger(1.0, w, v, M3);
  EXPECT_ARRAY_NEAR(M3, nda::matrix<T>{{6, 12}, {8, 16}, {10, 20}});

  // outer product of strided views
  M2             = 0;
  auto v_strided = nda::vector<T>{0, 1, 0, 2, 0};
  auto w_strided = nda::vector<T>{3, 0, 0, 4, 0, 0, 5};
  nda::blas::ger(2.0, v_strided(nda::range(1, 5, 2)), w_strided(nda::range(0, 7, 3)), M2);
  EXPECT_ARRAY_NEAR(M2, nda::matrix<T>{{6, 8, 10}, {12, 16, 20}});
}

TEST(NDA, BLASGer) {
  test_ger<double, nda::C_layout>();
  test_ger<double, nda::F_layout>();
  test_ger<std::complex<double>, nda::C_layout>();
  test_ger<std::complex<double>, nda::C_layout>();
}

// Test the BLAS dot/dotc function.
template <typename T, bool star>
void test_dot() {
  auto dot = [](auto &&a, auto &&b) {
    if constexpr (star) {
      return nda::blas::dotc(a, b);
    } else {
      return nda::blas::dot(a, b);
    }
  };
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
  test_dot<double, false>();
  test_dot<std::complex<double>, false>();
}

TEST(NDA, BLASDotc) {
  test_dot<double, true>();
  test_dot<std::complex<double>, true>();
}

// Test the BLAS scal function.
TEST(NDA, BLASScalEmptyVector) {
  nda::vector<double> v;
  nda::blas::scal(3.0, v);
  EXPECT_TRUE(v.empty());
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

TEST(NDA, BLASScalComplex) {
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

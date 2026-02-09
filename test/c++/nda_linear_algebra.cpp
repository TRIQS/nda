// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <vector>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;

// Test the generic dot/dotc function.
auto exp_dot(auto const &a, auto const &b) {
  auto res = a(0) * b(0);
  for (size_t i = 1; i < a.size(); ++i) res += a(i) * b(i);
  return res;
}

auto exp_dotc(auto const &a, auto const &b) {
  auto res = std::conj(a(0)) * b(0);
  for (size_t i = 1; i < a.size(); ++i) res += std::conj(a(i)) * b(i);
  return res;
}

TEST(NDA, LinearAlgebraDotProduct) {
  // scalars
  std::complex<double> u{1, 2};
  std::complex<double> v{3, -4};
  EXPECT_EQ(nda::linalg::dot(1, 2), 2);
  EXPECT_EQ(nda::linalg::dotc(1, 2), 2);
  EXPECT_DOUBLE_EQ(nda::linalg::dot(2, -5.0), -10.0);
  EXPECT_DOUBLE_EQ(nda::linalg::dotc(2, -5.0), -10.0);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(u, v), u * v, fp_tol<double>);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(u, v), std::conj(u) * v, fp_tol<double>);

  // BLAS compatible vectors
  nda::vector<double> a{1, 2, 3, 4, 5};
  nda::vector<double> b{10, 20, 30, 40, 50};
  EXPECT_DOUBLE_EQ(nda::linalg::dot(a, b), nda::blas::dot(a, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a, b), nda::blas::dotc(a, b), fp_tol<double>);

  nda::vector<float> af{1, 2, 3, 4, 5};
  nda::vector<float> bf{10, 20, 30, 40, 50};
  EXPECT_FLOAT_EQ(nda::linalg::dot(af, bf), nda::blas::dot(af, bf));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(af, bf), nda::blas::dotc(af, bf), fp_tol<float>);

  nda::vector<std::complex<double>> c = a * (1.1 - 2.1i);
  nda::vector<std::complex<double>> d = b * (3 + 4i);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c, d), nda::blas::dot(c, d), fp_tol<double>);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c, d), nda::blas::dotc(c, d), fp_tol<double>);

  nda::vector<std::complex<float>> cf = a * (1.1f - 2.1if);
  nda::vector<std::complex<float>> df = b * (3.0f + 4if);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(cf, df), nda::blas::dot(cf, df), fp_tol<float>);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(cf, df), nda::blas::dotc(cf, df), fp_tol<float>);

  // vectors with different value types
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(a, c), exp_dot(a, c), fp_tol<double>);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a, c), exp_dotc(a, c), fp_tol<double>);

  nda::vector<int> e{1, 2, 3, 4, 5};
  EXPECT_EQ(nda::linalg::dot(e, e), exp_dot(e, e));
  EXPECT_DOUBLE_EQ(nda::linalg::dot(e, b), exp_dot(e, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(e, b), exp_dotc(e, b), fp_tol<double>);

  // lazy expressions
  auto sin_a = nda::make_regular(nda::sin(a));
  EXPECT_DOUBLE_EQ(nda::linalg::dot(nda::sin(a), b), nda::blas::dot(sin_a, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(nda::sin(a), b), nda::blas::dotc(sin_a, b), fp_tol<double>);

  // (strided) vector views
  auto c_v = c(nda::range(0, 5, 2));
  auto d_v = d(nda::range(1, 4));
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c_v, d_v), exp_dot(c_v, d_v), fp_tol<double> * 10); // x10 needed for MKL
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c_v, d_v), exp_dotc(c_v, d_v), fp_tol<double>);
}

// Test the generic matvecmul function.
template <typename T, typename Layout>
void test_matvecmul() {
  auto x       = nda::vector<T>{1, 2, 3};
  auto x_t     = nda::vector<T>{1, 2, 3, 4};
  auto exp_y   = nda::vector<T>{14, 32, 50, 68};
  auto exp_y_t = nda::vector<T>{70, 80, 90};
  auto A       = nda::matrix<T, Layout>(4, 3);
  nda::for_each(A.shape(), [&A](auto i, auto j) { A(i, j) = i * 3 + j + 1; });
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1 - 1i};
    x *= T{2 - 1i};
    x_t *= T{2 - 1i};
    exp_y *= T{(1 - 1i) * (2 - 1i)};
    exp_y_t *= T{(1 - 1i) * (2 - 1i)};
  }

  // y = A * x
  auto y = nda::linalg::matvecmul(A, x);
  EXPECT_ARRAY_NEAR(y, exp_y, fp_tol<T>);

  // y_t = A^T * x_t
  auto y_t = nda::linalg::matvecmul(nda::transpose(A), x_t);
  EXPECT_ARRAY_NEAR(y_t, exp_y_t, fp_tol<T>);

  // y_h = A^H * x_t
  auto exp_y_h = exp_y_t;
  if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{T{210 + 70i}, T{240 + 80i}, T{270 + 90i}};
  auto y_h = nda::linalg::matvecmul(nda::dagger(A), x_t);
  EXPECT_ARRAY_NEAR(y_h, exp_y_h, fp_tol<T>);

  // strided matrix and vector views
  auto y_v = nda::linalg::matvecmul(A(nda::range(0, 4, 2), nda::range(0, 3, 2)), x(nda::range(0, 3, 2)));
  if constexpr (nda::is_complex_v<T>) {
    EXPECT_ARRAY_NEAR(y_v, (nda::vector<T>{T{10 - 30i}, T{34 - 102i}}), fp_tol<T>);
  } else {
    EXPECT_ARRAY_NEAR(y_v, (nda::vector<T>{10, 34}), fp_tol<T>);
  }
}

template <typename T>
void test_matvecmul_layouts() {
  test_matvecmul<T, C_layout>();
  test_matvecmul<T, F_layout>();
}

TEST(NDA, LinearAlgebraMatvecmulGenericGemvBranch) { test_matvecmul_layouts<long>(); }

TEST(NDA, LinearAlgebraMatvecmulBLASBranch) {
  test_matvecmul_layouts<float>();
  test_matvecmul_layouts<std::complex<float>>();
  test_matvecmul_layouts<double>();
  test_matvecmul_layouts<std::complex<double>>();
}

template <typename T>
void test_matvecmul_promotion() {
  auto A_i  = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_fp = nda::matrix<T>{{1, 2}, {3, 4}};
  auto w_i  = nda::vector<int>{1, 1};
  auto w_fp = nda::vector<T>{1, 1};

  auto v_fp1 = nda::linalg::matvecmul(A_fp, w_i);
  static_assert(std::same_as<nda::get_value_t<decltype(v_fp1)>, T>);
  EXPECT_ARRAY_NEAR(v_fp1, (nda::vector<T>{3, 7}), fp_tol<T>);

  auto v_fp2 = nda::linalg::matvecmul(A_i, w_fp);
  static_assert(std::same_as<nda::get_value_t<decltype(v_fp2)>, T>);
  EXPECT_ARRAY_NEAR(v_fp2, (nda::vector<T>{3, 7}), fp_tol<T>);

  auto v_i = nda::linalg::matvecmul(A_i, w_i);
  static_assert(std::same_as<nda::get_value_t<decltype(v_i)>, int>);
  EXPECT_ARRAY_EQ(v_i, (nda::vector<int>{3, 7}));
}

TEST(NDA, LinearAlgebraMatvecmulPromotion) {
  test_matvecmul_promotion<float>();
  test_matvecmul_promotion<std::complex<float>>();
  test_matvecmul_promotion<double>();
  test_matvecmul_promotion<std::complex<double>>();
}

template <typename T>
void test_matvecmul_lazy_expressions() {
  auto A     = nda::array<T, 2>{{1, 2}, {3, 4}};
  auto A_sin = nda::array<T, 2>{nda::sin(A)};
  auto w     = nda::vector<T>{1, 1};
  auto w_sin = nda::vector<T>{nda::sin(w)};
  EXPECT_ARRAY_NEAR(nda::linalg::matvecmul(nda::sin(A), nda::sin(w)), nda::linalg::matvecmul(A_sin, w_sin), fp_tol<T>);
}

TEST(NDA, LinearAlgebraMatvecmulWithLazyExpressions) {
  test_matvecmul_lazy_expressions<float>();
  test_matvecmul_lazy_expressions<std::complex<float>>();
  test_matvecmul_lazy_expressions<double>();
  test_matvecmul_lazy_expressions<std::complex<double>>();
}

// Test the generic matmul function.
template <typename T, typename Layout1, typename Layout2>
void test_matmul() {
  auto A     = nda::matrix<T, Layout1>{{1, 2, 3}, {4, 5, 6}};
  auto B     = nda::matrix<T, Layout2>{{1, 2}, {3, 4}, {5, 6}};
  auto exp_C = nda::matrix<T>{{22, 28}, {49, 64}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1 - 1i};
    B *= T{2 - 1i};
    exp_C *= T{(1 - 1i) * (2 - 1i)};
  }

  // C = A * B
  auto C = nda::linalg::matmul(A, B);
  EXPECT_ARRAY_NEAR(C, exp_C, fp_tol<T>);

  // C_t = B^T * A^T
  auto C_t = nda::linalg::matmul(nda::transpose(B), nda::transpose(A));
  EXPECT_ARRAY_NEAR(C_t, nda::transpose(exp_C), fp_tol<T>);

  // C_h = B^H * A^H
  auto C_h = nda::linalg::matmul(nda::dagger(B), nda::dagger(A));
  EXPECT_ARRAY_NEAR(C_h, nda::dagger(exp_C), fp_tol<T>);

  // strided matrix views
  using nda::range;
  auto exp_C_v = nda::matrix<T>{{16, 20}, {34, 44}};
  if constexpr (nda::is_complex_v<T>) exp_C_v *= T{(1 - 1i) * (2 - 1i)};
  auto C_v                            = nda::matrix<T>(4, 4);
  C_v(range(0, 4, 2), range(0, 4, 2)) = nda::linalg::matmul(A(range::all, range(0, 3, 2)), B(range(0, 3, 2), range::all));
  EXPECT_ARRAY_NEAR(C_v(range(0, 4, 2), range(0, 4, 2)), exp_C_v, fp_tol<T>);
}

template <typename T>
void test_matmul_layouts() {
  test_matmul<T, C_layout, C_layout>();
  test_matmul<T, C_layout, F_layout>();
  test_matmul<T, F_layout, F_layout>();
  test_matmul<T, F_layout, C_layout>();
}

TEST(NDA, LinearAlgebraMatmulGenericGemmBranch) { test_matmul_layouts<long>(); }

TEST(NDA, LinearAlgebraMatmulBLASBranch) {
  test_matmul_layouts<float>();
  test_matmul_layouts<std::complex<float>>();
  test_matmul_layouts<double>();
  test_matmul_layouts<std::complex<double>>();
}

template <typename T>
void test_matmul_promotion() {
  auto A_i  = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_fp = nda::matrix<T>{{1, 2}, {3, 4}};

  auto B_fp1 = nda::linalg::matmul(A_fp, A_i);
  static_assert(std::same_as<nda::get_value_t<decltype(B_fp1)>, T>);
  EXPECT_ARRAY_NEAR(B_fp1, (nda::matrix<T>{{7, 10}, {15, 22}}), fp_tol<T>);

  auto B_fp2 = nda::linalg::matmul(A_i, A_fp);
  static_assert(std::same_as<nda::get_value_t<decltype(B_fp2)>, T>);
  EXPECT_ARRAY_NEAR(B_fp2, (nda::matrix<T>{{7, 10}, {15, 22}}), fp_tol<T>);

  auto B_i = nda::linalg::matmul(A_i, A_i);
  static_assert(std::same_as<nda::get_value_t<decltype(B_i)>, int>);
  EXPECT_ARRAY_EQ(B_i, (nda::matrix<int>{{7, 10}, {15, 22}}));
}

TEST(NDA, LinearAlgebraMatmulPromoteValueType) {
  test_matmul_promotion<float>();
  test_matmul_promotion<std::complex<float>>();
  test_matmul_promotion<double>();
  test_matmul_promotion<std::complex<double>>();
}

template <typename T>
void test_matmul_lazy_expressions() {
  auto A     = nda::array<T, 2>{{1, 2}, {3, 4}};
  auto A_sin = nda::array<T, 2>{nda::sin(A)};
  EXPECT_ARRAY_NEAR(nda::linalg::matmul(nda::sin(A), nda::sin(A)), nda::linalg::matmul(A_sin, A_sin), fp_tol<T>);
}

TEST(NDA, LinearAlgebraMatmulWithLazyExpressions) {
  test_matmul_lazy_expressions<float>();
  test_matmul_lazy_expressions<std::complex<float>>();
  test_matmul_lazy_expressions<double>();
  test_matmul_lazy_expressions<std::complex<double>>();
}

// Test general inverse and determinant functions.
template <typename T, typename Layout>
void test_inv_and_det() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;

  // tolerance based on condition number of worst-case matrix C: cond(C) ~ 332, max(|Cinv|) = 24
  // error ~ cond(C) * max_element * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  // lambda that checks inverse functions for small matrices
  auto check_det_inv = [tol](auto M, auto Minv, T detM) {
    if constexpr (nda::is_complex_v<T>) {
      M *= T{1.0i};
      Minv /= T{1.0i};
      detM *= std::pow(T{1.0i}, M.extent(0));
    }

    auto Minv2 = nda::linalg::inv(M);
    EXPECT_ARRAY_NEAR(Minv, Minv2, tol);
    EXPECT_COMPLEX_NEAR(nda::linalg::det(Minv2), 1.0 / detM, tol);
    auto M2 = nda::linalg::inv(Minv2);
    EXPECT_ARRAY_NEAR(M, M2, tol);
    EXPECT_COMPLEX_NEAR(nda::linalg::det(M2), detM, tol);

    auto Minv3 = M;
    nda::linalg::inv_in_place(Minv3);
    EXPECT_ARRAY_NEAR(Minv, Minv3, tol);
    EXPECT_COMPLEX_NEAR(nda::linalg::det_in_place(Minv3), 1.0 / detM, tol);
  };

  // 1x1 matrix
  auto A    = matrix_t{{3}};
  auto Ainv = matrix_t{{1.0 / 3.0}};
  auto detA = 3.0;
  check_det_inv(A, Ainv, detA);

  // 2x2 matrix
  auto B    = matrix_t{{1, 2}, {0, 1}};
  auto Binv = matrix_t{{1, -2}, {0, 1}};
  auto detB = 1.0;
  check_det_inv(B, Binv, detB);

  // 3x3 matrix
  auto C    = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto Cinv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto detC = 1.0;
  check_det_inv(C, Cinv, detC);

  // 4x4 matrix
  auto D    = matrix_t{{2, 2, 2, 2}, {2, 4, 6, 8}, {2, 6, 12, 20}, {2, 8, 20, 40}};
  auto Dinv = matrix_t{{2, -3, 2, -0.5}, {-3, 7, -5.5, 1.5}, {2, -5.5, 5, -1.5}, {-0.5, 1.5, -1.5, 0.5}};
  auto detD = 16.0;
  check_det_inv(D, Dinv, detD);

  // matrix view
  EXPECT_ARRAY_NEAR(nda::linalg::inv(C(nda::range(0, 2), nda::range(0, 2))), Binv, tol);
  EXPECT_COMPLEX_NEAR(nda::linalg::det(C(nda::range(0, 2), nda::range(0, 2))), detB, tol);
}

template <typename T>
void test_inv_and_det_layouts() {
  test_inv_and_det<T, C_layout>();
  test_inv_and_det<T, F_layout>();
}

TEST(NDA, LinearAlgebraInvAndDet) {
  test_inv_and_det_layouts<float>();
  test_inv_and_det_layouts<std::complex<float>>();
  test_inv_and_det_layouts<double>();
  test_inv_and_det_layouts<std::complex<double>>();
}

// Check that the eigenvectors/values are correct.
void check_eigen(auto const &A, auto const &V, auto const &l, bool is_left = false) {
  if (not is_left) {
    EXPECT_ARRAY_NEAR(A * V, V * nda::diag(l));
  } else {
    EXPECT_ARRAY_NEAR(nda::dagger(V) * A, nda::diag(l) * nda::dagger(V));
  }
}

void check_eigen(auto const &A, auto const &B, auto const &V, auto const &l, int itype = 1) {
  if (itype == 1) {
    EXPECT_ARRAY_NEAR(A * V, B * V * nda::diag(l));
  } else if (itype == 2) {
    EXPECT_ARRAY_NEAR(A * B * V, V * nda::diag(l));
  } else {
    EXPECT_ARRAY_NEAR(B * A * V, V * nda::diag(l));
  }
}

// Create a symmetric or hermitian matrix with restricted eigenvalues.
template <typename T>
auto syhe_matrix(int n, double a = 1e-6, double b = 1.0) {
  using matrix_t = nda::matrix<T, nda::F_layout>;

  // orthogonal/unitary matrix Q
  auto jpvt = nda::zeros<int>(n);
  auto tau  = nda::vector<T>(n);
  auto Q    = nda::matrix<T, nda::F_layout>::rand(n, n);
  nda::lapack::geqp3(Q, jpvt, tau);
  if constexpr (nda::is_complex_v<T>) {
    nda::lapack::ungqr(Q, tau);
  } else {
    nda::lapack::orgqr(Q, tau);
  }

  // diagonal matrix containing the eigenvalues
  auto D = nda::eye<double>(n) * a + nda::diag(nda::rand(n)) * (b - a);

  // return Q * D * Q^H (hermitian/symmetric)
  return matrix_t{Q * D * nda::dagger(Q)};
}

// Test the eigh and eigvalsh functions.
template <typename T>
void test_eigh_eigvalsh() {
  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);

    // use eigh to compute eigenvalues and eigenvectors
    auto [w1, V1] = nda::linalg::eigh(A);
    check_eigen(A, V1, w1);

    // use eigh_in_place to compute eigenvalues and eigenvectors
    auto V2 = A;
    auto w2 = nda::linalg::eigh_in_place(V2);
    check_eigen(A, V2, w2);
    EXPECT_ARRAY_NEAR(V1, V2);
    EXPECT_ARRAY_NEAR(w1, w2);

    // use eigvalsh to compute eigenvalues only
    auto w3 = nda::linalg::eigvalsh(A);
    EXPECT_ARRAY_NEAR(w1, w3);

    // use eigvalsh_in_place to compute eigenvalues only
    auto A4 = A;
    auto w4 = nda::linalg::eigvalsh_in_place(A4);
    EXPECT_ARRAY_NEAR(w1, w4);

    // use eigh with a C-layout matrix
    auto A5       = nda::matrix<T, nda::C_layout>{A};
    auto [w5, V5] = nda::linalg::eigh(A5);
    check_eigen(A5, V5, w5);
    EXPECT_ARRAY_NEAR(V1, V5);
    EXPECT_ARRAY_NEAR(w1, w5);

    // use eigvalsh with a C-layout matrix
    auto w6 = nda::linalg::eigvalsh(A5);
    EXPECT_ARRAY_NEAR(w1, w6);
  }
}

TEST(NDA, LinearAlgebraEighAndEigvalsh) {
  test_eigh_eigvalsh<double>();
  test_eigh_eigvalsh<std::complex<double>>();
}

// Test the eigh and eigvalsh functions for generalized eigenvalue problems.
template <typename T>
void test_generalized_eigh_eigvalsh(int itype) {
  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);
    auto B = syhe_matrix<T>(i, 1e-6, 1);

    // use eigh to compute eigenvalues and eigenvectors
    auto [w1, V1] = nda::linalg::eigh(A, B, itype);
    check_eigen(A, B, V1, w1, itype);

    // use eigh_in_place to compute eigenvalues and eigenvectors
    auto V2 = A;
    auto B2 = B;
    auto w2 = nda::linalg::eigh_in_place(V2, B2, itype);
    check_eigen(A, B, V2, w2, itype);
    EXPECT_ARRAY_NEAR(V1, V2);
    EXPECT_ARRAY_NEAR(w1, w2);

    // use eigvalsh to compute eigenvalues only
    auto w3 = nda::linalg::eigvalsh(A, B, itype);
    EXPECT_ARRAY_NEAR(w1, w3);

    // use eigvalsh_in_place to compute eigenvalues only
    auto A4 = A;
    auto B4 = B;
    auto w4 = nda::linalg::eigvalsh_in_place(A4, B4, itype);
    EXPECT_ARRAY_NEAR(w1, w4);

    // use eigh with a C-layout matrices
    auto A5       = nda::matrix<T, nda::C_layout>{A};
    auto B5       = nda::matrix<T, nda::C_layout>{B};
    auto [w5, V5] = nda::linalg::eigh(A5, B5, itype);
    check_eigen(A, B, V5, w5, itype);
    EXPECT_ARRAY_NEAR(V1, V5);
    EXPECT_ARRAY_NEAR(w1, w5);

    // use eigvalsh with a C-layout matrices
    auto w6 = nda::linalg::eigvalsh(A5, B5, itype);
    EXPECT_ARRAY_NEAR(w1, w6);
  }
}

TEST(NDA, LinearAlgebraGeneralizedEighAndEigvalsh) {
  test_generalized_eigh_eigvalsh<double>(1);
  test_generalized_eigh_eigvalsh<double>(2);
  test_generalized_eigh_eigvalsh<double>(3);
  test_generalized_eigh_eigvalsh<std::complex<double>>(1);
  test_generalized_eigh_eigvalsh<std::complex<double>>(2);
  test_generalized_eigh_eigvalsh<std::complex<double>>(3);
}

// Helper to compare complex eigenvalues (they may be in different order).
void expect_eigenvalues_near(auto const &lambda1, auto const &lambda2) {
  ASSERT_EQ(lambda1.size(), lambda2.size());
  std::vector<std::complex<double>> lambda1_sorted(lambda1.begin(), lambda1.end());
  std::vector<std::complex<double>> lambda2_sorted(lambda2.begin(), lambda2.end());
  auto cmp = [](auto a, auto b) {
    if (std::abs(std::real(a) - std::real(b)) > 1e-10) return std::real(a) < std::real(b);
    return std::imag(a) < std::imag(b);
  };
  std::sort(lambda1_sorted.begin(), lambda1_sorted.end(), cmp);
  std::sort(lambda2_sorted.begin(), lambda2_sorted.end(), cmp);
  for (size_t j = 0; j < lambda1_sorted.size(); ++j) { EXPECT_COMPLEX_NEAR(lambda1_sorted[j], lambda2_sorted[j]); }
}

// Test the eig and eigvals functions for general matrices.
template <typename T>
void test_eig_eigvals() {
  for (auto i : nda::range(1, 6)) {
    auto A    = nda::matrix<T, nda::F_layout>::rand(i, i);
    auto Acpx = nda::matrix<std::complex<double>, nda::F_layout>(A);

    // use eig to compute eigenvalues and eigenvectors
    auto [lambda1, V1] = nda::linalg::eig(A);
    check_eigen(Acpx, V1, lambda1);

    // use eig_in_place to compute eigenvalues and eigenvectors
    auto A2            = nda::matrix<T, nda::F_layout>(A);
    auto [lambda2, V2] = nda::linalg::eig_in_place(A2);
    check_eigen(Acpx, V2, lambda2);
    expect_eigenvalues_near(lambda1, lambda2);

    // use eigvals to compute eigenvalues only
    auto lambda3 = nda::linalg::eigvals(A);
    expect_eigenvalues_near(lambda1, lambda3);

    // use eigvals_in_place to compute eigenvalues only
    auto A4      = nda::matrix<T, nda::F_layout>(A);
    auto lambda4 = nda::linalg::eigvals_in_place(A4);
    expect_eigenvalues_near(lambda1, lambda4);

    // use eig with a C-layout matrix
    auto A5            = nda::matrix<T, nda::C_layout>{A};
    auto A5cpx         = nda::matrix<std::complex<double>, nda::C_layout>(A5);
    auto [lambda5, V5] = nda::linalg::eig(A5);
    check_eigen(A5cpx, V5, lambda5);
    expect_eigenvalues_near(lambda1, lambda5);

    // use eigvals with a C-layout matrix
    auto lambda6 = nda::linalg::eigvals(A5);
    expect_eigenvalues_near(lambda1, lambda6);
  }
}

// Test eig with a matrix that has known complex eigenvalues.
TEST(NDA, LinearAlgebraEigComplexEigenvalues) {
  // 2x2 rotation matrix: eigenvalues are i and -i
  auto A        = nda::matrix<double, nda::F_layout>{{0.0, -1.0}, {1.0, 0.0}};
  auto Acpx     = nda::matrix<std::complex<double>, nda::F_layout>(A);
  auto [w, V]   = nda::linalg::eig(A);
  auto expected = std::array{std::complex<double>{0.0, 1.0}, std::complex<double>{0.0, -1.0}};

  // check eigenvalues (order may vary)
  EXPECT_TRUE((std::abs(w(0) - expected[0]) < 1e-10) || (std::abs(w(0) - expected[1]) < 1e-10));
  EXPECT_TRUE((std::abs(w(1) - expected[0]) < 1e-10) || (std::abs(w(1) - expected[1]) < 1e-10));

  // check eigenvector equation
  check_eigen(Acpx, V, w);
}

TEST(NDA, LinearAlgebraEigAndEigvals) {
  test_eig_eigvals<double>();
  test_eig_eigvals<std::complex<double>>();
}

// Test the norm function.
bool check_norm_p(auto &v, double p) { return nda::linalg::norm(v, p) == std::pow(nda::sum(nda::pow(nda::abs(v), p)), 1.0 / p); };

TEST(NDA, LinearAlgebraNormZeros) {
  const int size = 100;
  auto v         = nda::zeros<double>(size);

  EXPECT_EQ(nda::linalg::norm(v), nda::linalg::norm(v, 2.0));
  EXPECT_EQ(nda::linalg::norm(v, 0.0), 0.0);
  EXPECT_EQ(nda::linalg::norm(v, 1.0), 0.0);
  EXPECT_EQ(nda::linalg::norm(v, 2.0), 0.0);
  EXPECT_EQ(nda::linalg::norm(v, std::numeric_limits<double>::infinity()), 0.0);
  EXPECT_EQ(nda::linalg::norm(v, -std::numeric_limits<double>::infinity()), 0.0);
  EXPECT_EQ(nda::linalg::norm(v, 1.5), 0.0);
}

TEST(NDA, LinearAlgebraNormOnes) {
  const int size = 100;
  auto v         = nda::ones<double>(size);

  EXPECT_EQ(nda::linalg::norm(v), nda::linalg::norm(v, 2.0));
  EXPECT_EQ(nda::linalg::norm(v, 0.0), size);
  EXPECT_EQ(nda::linalg::norm(v, 1.0), size);
  EXPECT_EQ(nda::linalg::norm(v, 2.0), std::sqrt(size));
  EXPECT_EQ(nda::linalg::norm(v, std::numeric_limits<double>::infinity()), 1);
  EXPECT_EQ(nda::linalg::norm(v, -std::numeric_limits<double>::infinity()), 1);
  EXPECT_EQ(nda::linalg::norm(v, 1.5), std::pow(double(size), 1.0 / 1.5));
}

TEST(NDA, LinearAlgebraNormRand) {
  const int size = 100;
  auto v         = nda::rand(size);

  EXPECT_EQ(nda::linalg::norm(v), nda::linalg::norm(v, 2.0));
  EXPECT_EQ(nda::linalg::norm(v, 0.0), size);
  EXPECT_EQ(nda::linalg::norm(v, 1.0), nda::sum(abs(v)));
  EXPECT_EQ(nda::linalg::norm(v, 2.0), std::sqrt(std::real(nda::blas::dotc(v, v))));
  EXPECT_EQ(nda::linalg::norm(v, std::numeric_limits<double>::infinity()), nda::max_element(v));
  EXPECT_EQ(nda::linalg::norm(v, -std::numeric_limits<double>::infinity()), nda::min_element(v));

  EXPECT_TRUE((check_norm_p(v, -1.5)));
  EXPECT_TRUE((check_norm_p(v, -1.0)));
  EXPECT_TRUE((check_norm_p(v, 1.5)));
}

TEST(NDA, LinearAlgebraNormExample) {
  auto run_checks = [](auto const &v) {
    EXPECT_EQ(nda::linalg::norm(v), nda::linalg::norm(v, 2.0));
    EXPECT_EQ(nda::linalg::norm(v, 0.0), 3);
    EXPECT_EQ(nda::linalg::norm(v, 1.0), 4);
    EXPECT_NEAR(nda::linalg::norm(v, 2.0), std::sqrt(7.5), 1e-15);

    EXPECT_TRUE((check_norm_p(v, -1.5)));
    EXPECT_TRUE((check_norm_p(v, -1.0)));
    EXPECT_TRUE((check_norm_p(v, 1.5)));
  };

  auto v = nda::array<double, 1>{-0.5, 0.0, 1.0, 2.5};
  run_checks(v);
  run_checks(1i * v);
  run_checks((1 + 1i) / std::sqrt(2) * v);
  EXPECT_EQ(nda::linalg::norm(v, std::numeric_limits<double>::infinity()), 2.5);
  EXPECT_EQ(nda::linalg::norm(v, -std::numeric_limits<double>::infinity()), 0.0);
}

// Test the outer product function.
template <typename T, typename Layout>
void test_outer_product() {
  using namespace nda::blas_lapack;

  // outer product of two arrays
  auto A = nda::array<T, 2, Layout>::rand(2, 3);
  auto B = nda::array<T, 3, Layout>::rand(4, 5, 6);
  auto C = nda::array<T, 5, Layout>(2, 3, 4, 5, 6);
  for (auto [i, j] : A.indices())
    for (auto [k, l, m] : B.indices()) C(i, j, k, l, m) = A(i, j) * B(k, l, m);
  EXPECT_ARRAY_NEAR(C, nda::linalg::outer_product(A, B), fp_tol<T>);

  // outer product of two vectors
  nda::vector<T> v{1, 2};
  nda::vector<T> w{3, 4, 5};
  auto M = nda::linalg::outer_product(v, w);
  static_assert(nda::get_algebra<decltype(M)> == 'M');
  static_assert(has_C_layout<decltype(M)>);
  EXPECT_ARRAY_NEAR(nda::matrix<T>{{3, 4, 5}, {6, 8, 10}}, M, fp_tol<T>);

  // outer product of a vector and an array
  auto D = nda::linalg::outer_product(v, A);
  static_assert(nda::get_algebra<decltype(D)> == 'A');
  static_assert(has_C_layout<decltype(D)> == has_C_layout<decltype(A)>);
  auto D_exp = nda::array<T, 3, Layout>(2, 2, 3);
  for (auto i : nda::range(2))
    for (auto [j, k] : A.indices()) D_exp(i, j, k) = v(i) * A(j, k);
  EXPECT_ARRAY_NEAR(D, D_exp, fp_tol<T>);

  // outer product of an array and a vector
  auto E = nda::linalg::outer_product(A, v);
  static_assert(nda::get_algebra<decltype(E)> == 'A');
  static_assert(has_C_layout<decltype(E)> == has_C_layout<decltype(A)>);
  auto E_exp = nda::array<T, 3, Layout>(2, 3, 2);
  for (auto [i, j] : A.indices())
    for (auto k : nda::range(2)) E_exp(i, j, k) = A(i, j) * v(k);
  EXPECT_ARRAY_NEAR(E, E_exp, fp_tol<T>);
}

template <typename T>
void test_outer_product_layouts() {
  test_outer_product<T, C_layout>();
  test_outer_product<T, F_layout>();
}

TEST(NDA, LinearAlgebraOuterProduct) {
  test_outer_product_layouts<float>();
  test_outer_product_layouts<std::complex<float>>();
  test_outer_product_layouts<double>();
  test_outer_product_layouts<std::complex<double>>();
}

// Test the generic solve and solve_in_place functions.
template <typename T, typename Layout>
void test_solve() {
  using matrix_t = nda::matrix<T, Layout>;
  using vector_t = nda::vector<T>;
  using fp_t     = nda::get_fp_t<T>;

  /// tolerance based on condition number: cond(A) ~ 332, ||Ainv||_max = 24
  // error ~ cond(A) * ||Ainv||_max * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // solve A * X = B using the exact matrix inverse
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X    = matrix_t{Ainv * B};
  EXPECT_ARRAY_NEAR(matrix_t{A * X}, B, tol);

  // solve A * X = B using solve_in_place
  if constexpr (nda::blas_lapack::has_F_layout<matrix_t>) {
    auto Acopy = matrix_t{A};
    auto Bcopy = matrix_t{B};
    nda::linalg::solve_in_place(Acopy, Bcopy);
    EXPECT_ARRAY_NEAR(matrix_t{A * Bcopy}, B, tol);
    EXPECT_ARRAY_NEAR(X, Bcopy, tol);

    // solve A * x = b using solve_in_place
    Acopy  = A;
    auto b = vector_t{B(nda::range::all, 0)};
    nda::linalg::solve_in_place(Acopy, b);
    EXPECT_ARRAY_NEAR(A * b, B(nda::range::all, 0), tol);
    EXPECT_ARRAY_NEAR(X(nda::range::all, 0), b, tol);
  }

  // solve A * X = B using solve
  auto X2 = nda::linalg::solve(A, B);
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B, tol);
  EXPECT_ARRAY_NEAR(X, X2, tol);

  // solve A * x = b using solve
  auto x = nda::linalg::solve(A, B(nda::range::all, 0));
  EXPECT_ARRAY_NEAR(A * x, B(nda::range::all, 0), tol);
  EXPECT_ARRAY_NEAR(X(nda::range::all, 0), x, tol);
}

template <typename T>
void test_solve_layouts() {
  test_solve<T, C_layout>();
  test_solve<T, F_layout>();
}

TEST(NDA, LinearAlgebraSolve) {
  test_solve_layouts<float>();
  test_solve_layouts<std::complex<float>>();
  test_solve_layouts<double>();
  test_solve_layouts<std::complex<double>>();
}

// Test the svd and svd_in_place functions.
template <typename T, typename Layout>
void test_svd() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;

  auto A = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};

  // expected condition number and spectral norm of A from numpy
  constexpr fp_t cond_A = 6.784414066333698;
  constexpr fp_t norm_A = 12.316822252443167;

  // check backward error of SVD and expected condition number and spectral norm from numpy
  auto check_svd = [cond_A, norm_A](auto const &A, auto const &U, auto const &s, auto const &VH) {
    auto S      = matrix_t::zeros(A.shape());
    diagonal(S) = s;
    EXPECT_ARRAY_NEAR(A, U * S * VH, fp_tol<T>);
    EXPECT_NEAR(s(0) / s(s.size() - 1), cond_A, fp_tol<T>);
    EXPECT_NEAR(s(0), norm_A, fp_tol<T>);
  };

  // compute the SVD of A
  auto [U1, s1, VH1] = nda::linalg::svd(A);
  check_svd(A, U1, s1, VH1);

  // compute the SVD of A in place
  auto A_copy        = A;
  auto [U2, s2, VH2] = nda::linalg::svd_in_place(A_copy);
  check_svd(A, U2, s2, VH2);
}

template <typename T>
void test_svd_layouts() {
  test_svd<T, C_layout>();
  test_svd<T, F_layout>();
}

TEST(NDA, LinearAlgebraSVD) {
  test_svd_layouts<float>();
  test_svd_layouts<std::complex<float>>();
  test_svd_layouts<double>();
  test_svd_layouts<std::complex<double>>();
}

// Test the cross product function.
TEST(NDA, LinearAlgebraCrossProduct) {
  nda::vector<double> e1{1, 0, 0};
  nda::vector<double> e2{0, 1, 0};
  nda::vector<double> e3{0, 0, 1};

  EXPECT_ARRAY_NEAR(nda::linalg::cross_product(e1, e2), e3);
  EXPECT_ARRAY_NEAR(nda::linalg::cross_product(e2, e3), e1);
  EXPECT_ARRAY_NEAR(nda::linalg::cross_product(e3, e1), e2);
}

// Test the get_permutation_matrix and get_permutation_vector functions.
TEST(NDA, LinearAlgebraPermutationMatrixAndVector) {
  // test get_permutation_matrix from pivot indices
  nda::vector<int> ipiv{2, 2, 3};
  auto P_from_ipiv = nda::linalg::get_permutation_matrix<double>(ipiv, 3);

  // starting with identity
  // (i) swap row 0 with row 1 (ipiv[0]=2 -> swap with row 1)
  // (ii) swap row 1 with row 1 (ipiv[1]=2 -> swap with row 1)
  // (iii) swap row 2 with row 2 (ipiv[2]=3 -> swap with row 2)
  auto P_expected = nda::matrix<double>{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
  EXPECT_ARRAY_NEAR(P_from_ipiv, P_expected, fp_tol<double>);

  // test get_permutation_vector from pivot indices
  auto sigma = nda::linalg::get_permutation_vector(ipiv, 3);
  EXPECT_ARRAY_EQ(sigma, (nda::vector<int>{1, 0, 2}));

  // test get_permutation_matrix from permutation vector
  auto P_from_sigma = nda::linalg::get_permutation_matrix<double>(sigma);
  EXPECT_ARRAY_NEAR(P_from_sigma, P_expected, fp_tol<double>);

  // verify that P * sigma gives identity permutation applied in order
  auto test_vec = nda::vector<double>{10, 20, 30};
  auto permuted = nda::vector<double>{test_vec(sigma(0)), test_vec(sigma(1)), test_vec(sigma(2))};
  EXPECT_ARRAY_NEAR(P_from_sigma * test_vec, permuted, fp_tol<double>);

  // test with different layout
  auto P_C_layout = nda::linalg::get_permutation_matrix<double, C_layout>(sigma);
  EXPECT_ARRAY_NEAR(P_C_layout, P_expected, fp_tol<double>);

  // test with complex type
  auto P_complex = nda::linalg::get_permutation_matrix<std::complex<double>>(sigma);
  EXPECT_ARRAY_NEAR(P_complex, nda::matrix<std::complex<double>>(P_expected), fp_tol<double>);

  // test larger permutation
  nda::vector<int> ipiv_large{3, 3, 4, 4};
  auto sigma_large        = nda::linalg::get_permutation_vector(ipiv_large, 4);
  auto P_large_from_ipiv  = nda::linalg::get_permutation_matrix<double>(ipiv_large, 4);
  auto P_large_from_sigma = nda::linalg::get_permutation_matrix<double>(sigma_large);
  EXPECT_ARRAY_NEAR(P_large_from_ipiv, P_large_from_sigma, fp_tol<double>);

  // verify permutation properties: P^T * P = I
  EXPECT_ARRAY_NEAR(nda::transpose(P_from_sigma) * P_from_sigma, nda::eye(3), fp_tol<double>);
  EXPECT_ARRAY_NEAR(nda::transpose(P_large_from_sigma) * P_large_from_sigma, nda::eye(4), fp_tol<double>);

  // test identity permutation
  nda::vector<int> sigma_id{0, 1, 2, 3};
  auto P_id = nda::linalg::get_permutation_matrix<double>(sigma_id);
  EXPECT_ARRAY_NEAR(P_id, nda::eye(4), fp_tol<double>);
}

// Verify that L and U have the correct structure after LU decomposition.
void verify_lu_structure(auto const &A, auto const &sigma, auto const &L, auto const &U, bool rank_deficient) {
  using fp_t = nda::get_fp_t<decltype(A)>;

  // verify dimensions
  auto const [m, n] = A.shape();
  auto const k      = std::min(m, n);
  EXPECT_EQ(L.extent(0), m);
  EXPECT_EQ(L.extent(1), k);
  EXPECT_EQ(U.extent(0), k);
  EXPECT_EQ(U.extent(1), n);

  // verify that P * A = L * U
  auto P = nda::linalg::get_permutation_matrix<nda::get_value_t<decltype(A)>>(sigma);
  EXPECT_ARRAY_NEAR(P * A, L * U, fp_tol<fp_t>);

  // verify L is lower triangular/trapezoidal with unit diagonal
  for (int i = 0; i < m; ++i) {
    if (i < k) { EXPECT_COMPLEX_NEAR(L(i, i), 1.0, fp_tol<fp_t>); }
    for (int j = i + 1; j < k; ++j) EXPECT_COMPLEX_NEAR(L(i, j), 0.0, fp_tol<fp_t>);
  }

  // verify U is upper triangular/trapezoidal
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < i; ++j) EXPECT_COMPLEX_NEAR(U(i, j), 0.0, fp_tol<fp_t>);
  }

  // in case of rank deficiency, check that at least one diagonal element of U is close to zero
  if (rank_deficient and m >= 2 and n >= 2) { EXPECT_NEAR(nda::min_element(nda::abs(nda::diagonal(U))), 0.0, fp_tol<fp_t>); }
}

// Test LU decompositions.
template <typename T, typename Layout>
void test_lu(int m, int n, bool rank_deficient = false) {
  using matrix_t = nda::matrix<T, Layout>;
  auto A         = matrix_t::rand(m, n);

  // introduce rank deficiency if requested
  if (rank_deficient and n >= 2) {
    A(nda::range::all, 1) = A(nda::range::all, 0);
  } else if (rank_deficient and m >= 2) {
    A(1, nda::range::all) = A(0, nda::range::all);
  }

  // LU decomposition
  auto [sigma_1, L_1, U_1] = nda::linalg::lu(A, rank_deficient);
  verify_lu_structure(A, sigma_1, L_1, U_1, rank_deficient);

  // LU decomposition in-place
  if constexpr (nda::blas_lapack::has_F_layout<matrix_t>) {
    auto A_copy              = A;
    auto [sigma_2, L_2, U_2] = nda::linalg::lu_in_place(A_copy, rank_deficient);
    verify_lu_structure(A, sigma_2, L_2, U_2, rank_deficient);
  }
}

template <typename T>
void test_lu_layouts(int m, int n) {
  test_lu<T, F_layout>(m, n);
  test_lu<T, F_layout>(m, n, true);
  test_lu<T, C_layout>(m, n);
  test_lu<T, C_layout>(m, n, true);
}

TEST(NDA, LinearAlgebraLUSquare) {
  auto sizes = std::vector<int>{1, 2, 3, 5, 10, 20};
  for (auto n : sizes) {
    test_lu_layouts<float>(n, n);
    test_lu_layouts<std::complex<float>>(n, n);
    test_lu_layouts<double>(n, n);
    test_lu_layouts<std::complex<double>>(n, n);
  }
}

TEST(NDA, LinearAlgebraLURectangularNarrow) {
  auto shapes = std::vector<std::array<int, 2>>{{2, 1}, {5, 1}, {10, 3}, {20, 7}};
  for (auto [m, n] : shapes) {
    test_lu_layouts<float>(m, n);
    test_lu_layouts<std::complex<float>>(m, n);
    test_lu_layouts<double>(m, n);
    test_lu_layouts<std::complex<double>>(m, n);
  }
}

TEST(NDA, LinearAlgebraLURectangularWide) {
  auto shapes = std::vector<std::array<int, 2>>{{1, 2}, {1, 5}, {3, 10}, {7, 20}};
  for (auto [m, n] : shapes) {
    test_lu_layouts<float>(m, n);
    test_lu_layouts<std::complex<float>>(m, n);
    test_lu_layouts<double>(m, n);
    test_lu_layouts<std::complex<double>>(m, n);
  }
}

// Verify the QR decomposition.
void verify_qr(auto const &A, auto const &sigma, auto const &Q, auto const &R, bool complete) {
  // verify dimensions
  auto const [m, n] = A.shape();
  auto const k      = (complete ? m : std::min(m, n));
  EXPECT_EQ(Q.extent(0), m);
  EXPECT_EQ(Q.extent(1), complete ? m : k);
  EXPECT_EQ(R.extent(0), k);
  EXPECT_EQ(R.extent(1), n);

  // verify factorization A * P = Q * R
  auto P = nda::linalg::get_permutation_matrix<nda::get_value_t<decltype(A)>>(sigma, true);
  EXPECT_ARRAY_NEAR(A * P, Q * R);

  // verify columns of Q are orthogonal
  for (int i = 0; i < k; ++i) {
    EXPECT_COMPLEX_NEAR(nda::linalg::dotc(Q(nda::range::all, i), Q(nda::range::all, i)), 1.0);
    for (int j = i + 1; j < k; ++j) { EXPECT_COMPLEX_NEAR(nda::linalg::dotc(Q(nda::range::all, i), Q(nda::range::all, j)), 0.0); }
  }

  // verify R is upper triangular/trapezoidal
  for (int i = 1; i < k; ++i) {
    for (int j = 0; j < std::min(i, static_cast<int>(n)); ++j) EXPECT_COMPLEX_NEAR(R(i, j), 0.0);
  }
}

// Test QR decompositions.
template <typename T, typename Layout>
void test_qr(int m, int n) {
  using matrix_t = nda::matrix<T, Layout>;
  auto A         = matrix_t::rand(m, n);

  // QR decomposition
  for (auto complete : {true, false}) {
    auto [sigma_1, Q_1, R_1] = nda::linalg::qr(A, complete);
    verify_qr(A, sigma_1, Q_1, R_1, complete);
  }

  // in-place QR decompositions
  if constexpr (nda::blas::has_F_layout<matrix_t>) {
    for (auto complete : {true, false}) {
      auto A_copy              = A;
      auto [sigma_2, Q_2, R_2] = nda::linalg::qr_in_place(A_copy, complete);
      verify_qr(A, sigma_2, Q_2, R_2, complete);
    }
  }
}

TEST(NDA, LinearAlgebraQRSquare) {
  auto sizes = std::vector<int>{1, 2, 3, 5, 10, 20};
  for (auto n : sizes) {
    test_qr<double, nda::F_layout>(n, n);
    test_qr<std::complex<double>, nda::F_layout>(n, n);

    test_qr<double, nda::C_layout>(n, n);
    test_qr<std::complex<double>, nda::C_layout>(n, n);
  }
}

TEST(NDA, LinearAlgebraQRRectangularNarrow) {
  auto shapes = std::vector<std::array<int, 2>>{{2, 1}, {5, 1}, {10, 3}, {20, 7}};
  for (auto [m, n] : shapes) {
    test_qr<double, nda::F_layout>(m, n);
    test_qr<std::complex<double>, nda::F_layout>(m, n);

    test_qr<double, nda::C_layout>(m, n);
    test_qr<std::complex<double>, nda::C_layout>(m, n);
  }
}

TEST(NDA, LinearAlgebraQRRectangularWide) {
  auto shapes = std::vector<std::array<int, 2>>{{1, 2}, {1, 5}, {3, 10}, {7, 20}};
  for (auto [m, n] : shapes) {
    test_qr<double, nda::F_layout>(m, n);
    test_qr<std::complex<double>, nda::F_layout>(m, n);

    test_qr<double, nda::C_layout>(m, n);
    test_qr<std::complex<double>, nda::C_layout>(m, n);
  }
}

// Test linear algebra functions with CLEF placeholders.
TEST(NDA, LinearAlgebraWithClefPlaceholders) {
  using namespace nda::clef::literals;

  auto M = nda::matrix<double>{{1, 2}, {0, 1}};

  // test celf::eval for lazy expression evaluation
  auto I        = nda::eye<double>(2);
  auto lazy_inv = nda::linalg::inv(2.0 * i_ * I - M);
  for (int i = 0; i < 4; ++i) {
    auto result   = nda::clef::eval(lazy_inv, i_ = i);
    auto expected = nda::linalg::inv(2.0 * i * I - M);
    EXPECT_ARRAY_NEAR(result, expected);
  }

  // use clef placeholder to fill nda::vector of matrices: B(i) = inv((i+2) * I - M)
  auto B = nda::vector<nda::matrix<double>>(4);
  for (auto &b : B) b.resize(2, 2);
  B(i_) << nda::linalg::inv(2.0 * i_ * I - M);
  for (int i = 0; i < 4; ++i) {
    auto expected = nda::linalg::inv(2.0 * i * I - M);
    EXPECT_ARRAY_NEAR(B(i), expected);
  }
}

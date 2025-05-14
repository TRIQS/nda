// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <limits>
#include <type_traits>

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
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(u, v), u * v);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(u, v), std::conj(u) * v);

  // BLAS compatible vectors
  nda::vector<double> a{1, 2, 3, 4, 5};
  nda::vector<double> b{10, 20, 30, 40, 50};
  EXPECT_DOUBLE_EQ(nda::linalg::dot(a, b), nda::blas::dot(a, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a, b), nda::blas::dotc(a, b));

  nda::vector<std::complex<double>> c = a * (1.1 - 2.1i);
  nda::vector<std::complex<double>> d = b * (3 + 4i);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c, d), nda::blas::dot(c, d));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c, d), nda::blas::dotc(c, d));

  // vectors with different value types
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(a, c), exp_dot(a, c));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a, c), exp_dotc(a, c));

  nda::vector<int> e{1, 2, 3, 4, 5};
  EXPECT_EQ(nda::linalg::dot(e, e), exp_dot(e, e));
  EXPECT_DOUBLE_EQ(nda::linalg::dot(e, b), exp_dot(e, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(e, b), exp_dotc(e, b));

  // lazy expressions
  auto sin_a = nda::make_regular(nda::sin(a));
  EXPECT_DOUBLE_EQ(nda::linalg::dot(nda::sin(a), b), nda::blas::dot(sin_a, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(nda::sin(a), b), nda::blas::dotc(sin_a, b));

  // (strided) vector views
  auto c_v = c(nda::range(0, 5, 2));
  auto d_v = d(nda::range(1, 4));
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c_v, d_v), exp_dot(c_v, d_v));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c_v, d_v), exp_dotc(c_v, d_v));
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
    A *= 1 - 1i;
    x *= 2 - 1i;
    x_t *= 2 - 1i;
    exp_y *= (1 - 1i) * (2 - 1i);
    exp_y_t *= (1 - 1i) * (2 - 1i);
  }

  // y = A * x
  auto y = nda::linalg::matvecmul(A, x);
  EXPECT_ARRAY_NEAR(y, exp_y);

  // y_t = A^T * x_t
  auto y_t = nda::linalg::matvecmul(nda::transpose(A), x_t);
  EXPECT_ARRAY_NEAR(y_t, exp_y_t);

  // y_h = A^H * x_t
  auto exp_y_h = exp_y_t;
  if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{210 + 70i, 240 + 80i, 270 + 90i};
  auto y_h = nda::linalg::matvecmul(nda::conj(nda::transpose(A)), x_t);
  EXPECT_ARRAY_NEAR(y_h, exp_y_h);

  // strided matrix and vector views
  auto y_v = nda::linalg::matvecmul(A(nda::range(0, 4, 2), nda::range(0, 3, 2)), x(nda::range(0, 3, 2)));
  if constexpr (nda::is_complex_v<T>) {
    EXPECT_ARRAY_EQ(y_v, (nda::vector<T>{10 - 30i, 34 - 102i}));
  } else {
    EXPECT_ARRAY_EQ(y_v, (nda::vector<T>{10, 34}));
  }
}

TEST(NDA, LinearAlgebraMatvecmulGenericGemvBranch) {
  test_matvecmul<long, nda::C_layout>();
  test_matvecmul<long, nda::F_layout>();
}

TEST(NDA, LinearAlgebraMatvecmulBLASBranch) {
  test_matvecmul<double, nda::C_layout>();
  test_matvecmul<double, nda::F_layout>();
  test_matvecmul<std::complex<double>, nda::C_layout>();
  test_matvecmul<std::complex<double>, nda::F_layout>();
}

TEST(NDA, LinearAlgebraMatvecmulPromotion) {
  auto A_i = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_d = nda::matrix<double>{{1, 2}, {3, 4}};
  auto w_i = nda::vector<int>{1, 1};
  auto w_d = nda::vector<double>{1, 1};

  auto v_d1 = nda::linalg::matvecmul(A_d, w_i);
  static_assert(std::same_as<nda::get_value_t<decltype(v_d1)>, double>);
  EXPECT_ARRAY_NEAR(v_d1, (nda::vector<double>{3, 7}), 1.e-13);

  auto v_d2 = nda::linalg::matvecmul(A_i, w_d);
  static_assert(std::same_as<nda::get_value_t<decltype(v_d2)>, double>);
  EXPECT_ARRAY_NEAR(v_d2, (nda::vector<double>{3, 7}), 1.e-13);

  auto v_i = nda::linalg::matvecmul(A_i, w_i);
  static_assert(std::same_as<nda::get_value_t<decltype(v_i)>, int>);
  EXPECT_ARRAY_EQ(v_i, (nda::vector<int>{3, 7}));
}

TEST(NDA, LinearAlgebraMatvecmulWithLazyExpressions) {
  auto A     = nda::array<double, 2>{{1, 2}, {3, 4}};
  auto A_sin = nda::array<double, 2>{nda::sin(A)};
  auto w     = nda::vector<double>{1, 1};
  auto w_sin = nda::vector<double>{nda::sin(w)};
  EXPECT_ARRAY_NEAR(nda::linalg::matvecmul(nda::sin(A), nda::sin(w)), nda::linalg::matvecmul(A_sin, w_sin), 1.e-13);
}

// Test the generic matmul function.
template <typename T, typename Layout1, typename Layout2, typename Layout3>
void test_matmul() {
  auto A     = nda::matrix<T, Layout1>{{1, 2, 3}, {4, 5, 6}};
  auto B     = nda::matrix<T, Layout2>{{1, 2}, {3, 4}, {5, 6}};
  auto exp_C = nda::matrix<T, Layout3>{{22, 28}, {49, 64}};
  if constexpr (nda::is_complex_v<T>) {
    A *= 1 - 1i;
    B *= 2 - 1i;
    exp_C *= (1 - 1i) * (2 - 1i);
  }

  // C = A * B
  auto C = nda::linalg::matmul(A, B);
  EXPECT_ARRAY_NEAR(C, exp_C);

  // C_t = B^T * A^T
  auto C_t = nda::linalg::matmul(nda::transpose(B), nda::transpose(A));
  EXPECT_ARRAY_NEAR(C_t, nda::transpose(exp_C));

  // C_h = B^H * A^H
  auto C_h = nda::linalg::matmul(nda::dagger(B), nda::dagger(A));
  EXPECT_ARRAY_NEAR(C_h, nda::dagger(exp_C));

  // strided matrix views
  auto exp_C_v = nda::matrix<T, Layout3>{{16, 20}, {34, 44}};
  if constexpr (nda::is_complex_v<T>) exp_C_v *= (1 - 1i) * (2 - 1i);
  auto C_v = nda::matrix<T, Layout3>(4, 4);
  C_v(nda::range(0, 4, 2), nda::range(0, 4, 2)) =
     nda::linalg::matmul(A(nda::range::all, nda::range(0, 3, 2)), B(nda::range(0, 3, 2), nda::range::all));
  EXPECT_ARRAY_NEAR(C_v(nda::range(0, 4, 2), nda::range(0, 4, 2)), exp_C_v);
}

TEST(NDA, LinearAlgebraMatmulGenericGemmBranch) {
  test_matmul<long, nda::C_layout, nda::C_layout, nda::C_layout>();
  test_matmul<long, nda::C_layout, nda::C_layout, nda::F_layout>();
  test_matmul<long, nda::C_layout, nda::F_layout, nda::F_layout>();
  test_matmul<long, nda::C_layout, nda::F_layout, nda::C_layout>();
  test_matmul<long, nda::F_layout, nda::F_layout, nda::F_layout>();
  test_matmul<long, nda::F_layout, nda::C_layout, nda::F_layout>();
  test_matmul<long, nda::F_layout, nda::F_layout, nda::C_layout>();
  test_matmul<long, nda::F_layout, nda::C_layout, nda::C_layout>();
}

TEST(NDA, LinearAlgebraMatmulBLASBranch) {
  test_matmul<double, nda::C_layout, nda::C_layout, nda::C_layout>();
  test_matmul<double, nda::C_layout, nda::C_layout, nda::F_layout>();
  test_matmul<double, nda::C_layout, nda::F_layout, nda::F_layout>();
  test_matmul<double, nda::C_layout, nda::F_layout, nda::C_layout>();
  test_matmul<double, nda::F_layout, nda::F_layout, nda::F_layout>();
  test_matmul<double, nda::F_layout, nda::C_layout, nda::F_layout>();
  test_matmul<double, nda::F_layout, nda::F_layout, nda::C_layout>();
  test_matmul<double, nda::F_layout, nda::C_layout, nda::C_layout>();
  test_matmul<std::complex<double>, nda::C_layout, nda::C_layout, nda::C_layout>();
  test_matmul<std::complex<double>, nda::C_layout, nda::C_layout, nda::F_layout>();
  test_matmul<std::complex<double>, nda::C_layout, nda::F_layout, nda::F_layout>();
  test_matmul<std::complex<double>, nda::C_layout, nda::F_layout, nda::C_layout>();
  test_matmul<std::complex<double>, nda::F_layout, nda::F_layout, nda::F_layout>();
  test_matmul<std::complex<double>, nda::F_layout, nda::C_layout, nda::F_layout>();
  test_matmul<std::complex<double>, nda::F_layout, nda::F_layout, nda::C_layout>();
  test_matmul<std::complex<double>, nda::F_layout, nda::C_layout, nda::C_layout>();
}

TEST(NDA, LinearAlgebraMatumulPromoteValueType) {
  auto A_i = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_d = nda::matrix<double>{{1, 2}, {3, 4}};

  auto B_d1 = nda::linalg::matmul(A_d, A_i);
  static_assert(std::is_same_v<nda::get_value_t<decltype(B_d1)>, double>);
  EXPECT_ARRAY_NEAR(B_d1, (nda::matrix<double>{{7, 10}, {15, 22}}), 1.e-13);

  auto B_d2 = nda::linalg::matmul(A_d, A_d);
  static_assert(std::is_same_v<nda::get_value_t<decltype(B_d2)>, double>);
  EXPECT_ARRAY_NEAR(B_d2, (nda::matrix<double>{{7, 10}, {15, 22}}), 1.e-13);

  auto B_i = nda::linalg::matmul(A_i, A_i);
  static_assert(std::is_same_v<nda::get_value_t<decltype(B_i)>, int>);
  EXPECT_ARRAY_NEAR(B_i, (nda::matrix<int>{{7, 10}, {15, 22}}), 1.e-13);
}

TEST(NDA, LinearAlgebraMatmulWithLazyExpressions) {
  auto A     = nda::array<double, 2>{{1, 2}, {3, 4}};
  auto A_sin = nda::array<double, 2>{nda::sin(A)};
  EXPECT_ARRAY_NEAR(nda::linalg::matmul(nda::sin(A), nda::sin(A)), nda::linalg::matmul(A_sin, A_sin), 1.e-13);
}

// Test determinant for a specific memory layout.
template <typename L>
void test_determinant() {
  nda::matrix<double, L> W1(1, 1);
  W1(0, 0) = 1.0;
  EXPECT_NEAR(determinant(W1), 1.0, 1.e-12);

  nda::matrix<double, L> W2{{1.0, 2.0}, {3.0, 4.0}};
  EXPECT_NEAR(determinant(W2), -2.0, 1.e-12);

  nda::matrix<double, L> W3(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W3(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);
  EXPECT_NEAR(determinant(W3), -7.8, 1.e-12);
}

TEST(NDA, LinearAlgebraDeterminant) {
  test_determinant<nda::F_layout>();
  test_determinant<nda::C_layout>();
}

// Test inverse for a specific memory layout.
template <typename L>
void test_inverse() {
  using matrix_t = nda::matrix<double, L>;

  matrix_t W(3, 3), Winv(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  Winv = inverse(W);
  EXPECT_NEAR(determinant(Winv), -1 / 7.8, 1.e-12);

  nda::matrix<double, nda::F_layout> id(W * Winv);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(std::abs(id(i, j)), (i == j ? 1 : 0), 1.e-13);

  // calculate the inverse of the inverse by calling the lapack routines directly
  nda::array<int, 1> ipiv(3);
  ipiv     = 0;
  int info = nda::lapack::getrf(Winv, ipiv);
  EXPECT_EQ(info, 0);
  info = nda::lapack::getri(Winv, ipiv);
  EXPECT_EQ(info, 0);
  EXPECT_ARRAY_NEAR(Winv, W, 1.e-12);
}

TEST(NDA, LinearAlgebraInverse) {
  test_inverse<nda::F_layout>();
  test_inverse<nda::C_layout>();
}

TEST(NDA, LinearAlgebraInverseInvolution) {
  using matrix_t = nda::matrix<double, nda::C_layout>;

  matrix_t W(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);
  auto W_copy = W;

  W = inverse(W);
  W = inverse(W);
  EXPECT_ARRAY_NEAR(W, W_copy, 1.e-12);
}

TEST(NDA, LinearAlgebraInverseSlice) {
  using matrix_t = nda::matrix<double, nda::C_layout>;

  matrix_t W(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  auto V        = W(nda::range(0, 3, 2), nda::range(0, 3, 2));
  matrix_t Vinv = inverse(V);
  matrix_t Vinv_ref{{-0.1, 0.5}, {-0.5, 0.0}};
  EXPECT_ARRAY_NEAR(Vinv, Vinv_ref, 1.e-12);

  W = inverse(W);

  auto U        = W(nda::range(0, 3, 2), nda::range(0, 3, 2));
  matrix_t Uinv = inverse(U);
  matrix_t Uinv_ref{{-5.0, 4.0}, {24.5, -27.4}};
  EXPECT_ARRAY_NEAR(Uinv, Uinv_ref, 1.e-12);
}

TEST(NDA, LinearAlgebraInverseSmall) {
  for (auto n : {1, 2, 3}) {

    nda::matrix<double> W(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) W(i, j) = (i > j ? 0.5 + i + 2.5 * j : i * 0.8 - j - 0.5);

    auto Winv = inverse(W);
    EXPECT_NEAR(determinant(Winv), 1.0 / determinant(W), 1.e-12);
    EXPECT_ARRAY_NEAR(W * Winv, nda::eye<double>(n), 1.e-13);

    auto Winv_inv = inverse(Winv);
    EXPECT_ARRAY_NEAR(Winv_inv, W, 1.e-12);
  }
}

// Check that the eigenvectors/values are correct.
template <typename M, typename V1, typename V2>
void check_eig(M const &m, V1 const &vectors, V2 const &values) {
  for (auto i : nda::range(0, m.extent(0))) {
    EXPECT_ARRAY_NEAR(nda::linalg::matvecmul(m, vectors(nda::range::all, i)), values(i) * vectors(nda::range::all, i), 1.e-13);
  }
}

TEST(NDA, LinearAlgebraEigenelements) {
  // calculate eigenvalues and eigenvectors and check that they are correct
  auto test_eigenelements = [](auto &&M) {
    auto [ev1, vecs] = nda::linalg::eigenelements(M);
    check_eig(M, vecs, ev1);
    auto Mcopy = M;
    auto ev2   = nda::linalg::eigenvalues_in_place(Mcopy);
    EXPECT_ARRAY_NEAR(ev1, ev2);
  };

  // double matrix in C layout
  nda::matrix<double> A(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j <= i; ++j) {
      A(i, j) = (i > j ? i + 2 * j : i - j);
      A(j, i) = A(i, j);
    }
  }
  test_eigenelements(A);

  A()     = 0;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(2, 2) = 8;
  A(0, 2) = 2;
  A(2, 0) = 2;
  test_eigenelements(A);

  A()     = 0;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(2, 2) = 8;
  test_eigenelements(A);

  // double matrix in Fortran layout
  nda::matrix<double, nda::F_layout> D{{1.3, 1.2}, {1.2, 2.2}};
  test_eigenelements(D);

  // complex matrix in C layout
  nda::matrix<std::complex<double>> B{{{1.0, 0.0}, {0.0, 1.0}}, {{0.0, -1.0}, {2.0, 0.0}}};
  test_eigenelements(B);

  // complex matrix in Fortran layout
  nda::matrix<std::complex<double>, nda::F_layout> C{{{1.3, 0.0}, {0.0, 1.1}}, {{0.0, -1.1}, {2.4, 0.0}}};
  test_eigenelements(C);
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
  auto v         = nda::rand<double>(size);

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
  run_checks((1 + 1i) / sqrt(2) * v);
  EXPECT_EQ(nda::linalg::norm(v, std::numeric_limits<double>::infinity()), 2.5);
  EXPECT_EQ(nda::linalg::norm(v, -std::numeric_limits<double>::infinity()), 0.0);
}

// Test the outer product function.
template <typename T, typename Layout>
void test_outer_product() {
  // outer product of two arrays
  auto A = nda::array<T, 2, Layout>::rand(2, 3);
  auto B = nda::array<T, 3, Layout>::rand(4, 5, 6);
  auto C = nda::array<T, 5, Layout>(2, 3, 4, 5, 6);
  for (auto [i, j] : A.indices())
    for (auto [k, l, m] : B.indices()) C(i, j, k, l, m) = A(i, j) * B(k, l, m);
  EXPECT_ARRAY_NEAR(C, nda::linalg::outer_product(A, B));

  // outer product of two vectors
  nda::vector<T> v{1, 2};
  nda::vector<T> w{3, 4, 5};
  auto M = nda::linalg::outer_product(v, w);
  static_assert(nda::get_algebra<decltype(M)> == 'M');
  static_assert(nda::blas::has_C_layout<decltype(M)>);
  EXPECT_ARRAY_NEAR(nda::matrix<T>{{3, 4, 5}, {6, 8, 10}}, M);
}

TEST(NDA, LinearAlgebraOuterProduct) {
  test_outer_product<double, nda::C_layout>();
  test_outer_product<double, nda::F_layout>();
  test_outer_product<std::complex<double>, nda::C_layout>();
  test_outer_product<std::complex<double>, nda::F_layout>();
}

// Test the generic solve and solve_in_place functions.
template <typename value_t, typename Layout>
void test_solve() {
  using matrix_t = nda::matrix<value_t, Layout>;
  using vector_t = nda::vector<value_t>;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // solve A * X = B using the exact matrix inverse
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X    = matrix_t{Ainv * B};
  EXPECT_ARRAY_NEAR(matrix_t{A * X}, B);

  // solve A * X = B using solve_in_place
  if constexpr (nda::blas::has_F_layout<matrix_t>) {
    auto Acopy = matrix_t{A};
    auto Bcopy = matrix_t{B};
    nda::linalg::solve_in_place(Acopy, Bcopy);
    EXPECT_ARRAY_NEAR(matrix_t{A * Bcopy}, B);
    EXPECT_ARRAY_NEAR(X, Bcopy);

    // solve A * x = b using solve_in_place
    Acopy  = A;
    auto b = vector_t{B(nda::range::all, 0)};
    nda::linalg::solve_in_place(Acopy, b);
    EXPECT_ARRAY_NEAR(A * b, B(nda::range::all, 0));
    EXPECT_ARRAY_NEAR(X(nda::range::all, 0), b);
  }

  // solve A * X = B using solve
  auto X2 = nda::linalg::solve(A, B);
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B);
  EXPECT_ARRAY_NEAR(X, X2);

  // solve A * x = b using solve
  auto x = nda::linalg::solve(A, B(nda::range::all, 0));
  EXPECT_ARRAY_NEAR(A * x, B(nda::range::all, 0));
  EXPECT_ARRAY_NEAR(X(nda::range::all, 0), x);
}

TEST(NDA, LinearAlgebraSolve) {
  test_solve<double, nda::C_layout>();
  test_solve<double, nda::F_layout>();
  test_solve<std::complex<double>, nda::C_layout>();
  test_solve<std::complex<double>, nda::F_layout>();
}

// Test the svd and svd_in_place functions.
template <typename T, typename Layout>
void test_svd() {
  using matrix_t = nda::matrix<T, Layout>;

  auto A = matrix_t{{2, -2, 1}, {-4, -8, -8}};
  auto s = nda::vector<double>{12, 3};

  // compute the SVD of A
  auto [U_1, s_1, VH_1] = nda::linalg::svd(A);
  auto S_1              = matrix_t::zeros(A.shape());
  diagonal(S_1)         = s_1;
  EXPECT_ARRAY_NEAR(s_1, s, 1e-14);
  EXPECT_ARRAY_NEAR(A, U_1 * S_1 * VH_1, 1e-14);

  // compute the SVD of A in place
  auto A_copy           = A;
  auto [U_2, s_2, VH_2] = nda::linalg::svd_in_place(A_copy);
  auto S_2              = matrix_t::zeros(A.shape());
  diagonal(S_2)         = s_2;
  EXPECT_ARRAY_NEAR(s, s_2, 1e-14);
  EXPECT_ARRAY_NEAR(A, U_2 * S_2 * VH_2, 1e-14);
}

TEST(NDA, LinearAlgebraSVD) {
  test_svd<double, nda::C_layout>();
  test_svd<double, nda::F_layout>();
  test_svd<std::complex<double>, nda::C_layout>();
  test_svd<std::complex<double>, nda::F_layout>();
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

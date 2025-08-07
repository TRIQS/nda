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

// Test matrix-matrix multiplication for specific memory layouts.
template <typename T, typename L1, typename L2, typename L3>
void test_matmul() {
  nda::matrix<T, L1> M1(2, 3);
  nda::matrix<T, L2> M2(3, 4);
  nda::matrix<T, L2> M3(2, 4), M3b(2, 4);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) { M1(i, j) = i + j; }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) { M2(i, j) = 1 + i - j; }

  // BLAS gemm
  M3b = 0;
  if constexpr (nda::is_blas_lapack_v<T>) { nda::blas::gemm(1, M1, M2, 0, M3b); }

  // operator*
  M3 = 0;
  M3 = M1 * M2;

  // brute force
  auto M4 = M3;
  M4      = 0;
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 4; ++j) M4(i, j) += M1(i, k) * M2(k, j);
  EXPECT_ARRAY_NEAR(M4, M3, 1.e-13);
  if constexpr (nda::is_blas_lapack_v<T>) { EXPECT_ARRAY_NEAR(M4, M3b, 1.e-13); }

  // generic gemm implementation
  nda::blas::gemm_generic(1, M1, M2, 0, M4);
  EXPECT_ARRAY_NEAR(M4, M3, 1.e-13);
}

// Call test_matmul for all various memory layouts.
template <typename T>
void all_test_matmul() {
  test_matmul<T, nda::C_layout, nda::C_layout, nda::C_layout>();
  test_matmul<T, nda::C_layout, nda::C_layout, nda::F_layout>();
  test_matmul<T, nda::C_layout, nda::F_layout, nda::F_layout>();
  test_matmul<T, nda::C_layout, nda::F_layout, nda::C_layout>();
  test_matmul<T, nda::F_layout, nda::F_layout, nda::F_layout>();
  test_matmul<T, nda::F_layout, nda::C_layout, nda::F_layout>();
  test_matmul<T, nda::F_layout, nda::F_layout, nda::C_layout>();
  test_matmul<T, nda::F_layout, nda::C_layout, nda::C_layout>();
}

TEST(NDA, LinearAlgebraMatmul) {
  all_test_matmul<double>();
  all_test_matmul<std::complex<double>>();
  all_test_matmul<long>();
}

TEST(NDA, LinearAlgebraMatumulPromoteValueType) {
  nda::matrix<double> A_d = {{1.0, 2.3}, {3.1, 4.3}};
  nda::matrix<int> B_i    = {{1, 2}, {3, 4}};
  nda::matrix<double> B_d = {{1, 2}, {3, 4}};

  auto C = nda::make_regular(A_d * B_i);
  auto D = nda::make_regular(A_d * B_d);
  static_assert(std::is_same_v<nda::get_value_t<decltype(C)>, double>);
  static_assert(std::is_same_v<nda::get_value_t<decltype(D)>, double>);
  EXPECT_ARRAY_NEAR(C, D, 1.e-13);
}

TEST(NDA, LinearAlgebraMatmulCache) {
  // test with view for possible cache issue
  nda::array<std::complex<double>, 3> A(2, 2, 5);
  A() = -1;
  nda::matrix_view<std::complex<double>> A_v(A(nda::range::all, nda::range::all, 2));
  nda::matrix<std::complex<double>> M1(2, 2), Res(2, 2);
  M1()      = 0;
  M1(0, 0)  = 2;
  M1(1, 1)  = 3.2;
  Res()     = 0;
  Res(0, 0) = 8;
  Res(1, 1) = 16.64;
  A_v()     = 0;
  A_v()     = nda::matrix<std::complex<double>>{M1 * (M1 + 2.0)};
  EXPECT_ARRAY_NEAR(A_v(), Res, 1.e-13);

  // not matmul, just recheck diagonal unity
  Res()     = 0;
  Res(0, 0) = 4;
  Res(1, 1) = 5.2;
  A_v()     = 0;
  A_v()     = nda::matrix<std::complex<double>>{(M1 + 2.0)};
  EXPECT_ARRAY_NEAR(A_v(), Res, 1.e-13);
}

TEST(NDA, LinearAlgebraMatmulAlias) {
  nda::array<std::complex<double>, 3> A(10, 2, 2);
  A() = -1;

  A(4, nda::range::all, nda::range::all) = 1;
  A(5, nda::range::all, nda::range::all) = 2;

  nda::matrix_view<std::complex<double>> M1 = A(4, nda::range::all, nda::range::all);
  nda::matrix_view<std::complex<double>> M2 = A(5, nda::range::all, nda::range::all);

  M1 = M1 * M2;
  EXPECT_ARRAY_NEAR(M1, nda::matrix<std::complex<double>>{{4, 4}, {4, 4}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<std::complex<double>>{{2, 2}, {2, 2}});

  nda::matrix<double> B1(2, 2), B2(2, 2);
  B1() = 2;
  B2() = 3;

  B1 = nda::make_regular(B1) * B2;
  EXPECT_ARRAY_NEAR(B1, nda::matrix<double>{{6, 0}, {0, 6}});
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

TEST(NDA, LinearAlgebraMatvecmulPromotion) {
  nda::matrix<int> A_i    = {{1, 2}, {3, 4}};
  nda::matrix<double> A_d = {{1, 2}, {3, 4}};
  nda::array<int, 1> v_i, w_i    = {1, 1};
  nda::array<double, 1> v_d, w_d = {1, 1};

  v_d = matvecmul(A_d, w_i);
  v_i = matvecmul(A_i, w_i);

  EXPECT_ARRAY_NEAR(v_d, v_i, 1.e-13);
}

// Check that the eigenvectors/values are correct.
template <typename M, typename V1, typename V2>
void check_eig(M const &m, V1 const &vectors, V2 const &values) {
  for (auto i : nda::range(0, m.extent(0))) {
    EXPECT_ARRAY_NEAR(matvecmul(m, vectors(nda::range::all, i)), values(i) * vectors(nda::range::all, i), 1.e-13);
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

TEST(NDA, LinearAlgebraNormZeros) {
  const int size = 100;
  auto v         = nda::zeros<double>(size);

  EXPECT_EQ(nda::norm(v), nda::norm(v, 2.0));
  EXPECT_EQ(nda::norm(v, 0.0), 0.0);
  EXPECT_EQ(nda::norm(v, 1.0), 0.0);
  EXPECT_EQ(nda::norm(v, 2.0), 0.0);
  EXPECT_EQ(nda::norm(v, std::numeric_limits<double>::infinity()), 0.0);
  EXPECT_EQ(nda::norm(v, -std::numeric_limits<double>::infinity()), 0.0);
  EXPECT_EQ(nda::norm(v, 1.5), 0.0);
}

TEST(NDA, LinearAlgebraNormOnes) {
  const int size = 100;
  auto v         = nda::ones<double>(size);

  EXPECT_EQ(nda::norm(v), nda::norm(v, 2.0));
  EXPECT_EQ(nda::norm(v, 0.0), size);
  EXPECT_EQ(nda::norm(v, 1.0), size);
  EXPECT_EQ(nda::norm(v, 2.0), std::sqrt(size));
  EXPECT_EQ(nda::norm(v, std::numeric_limits<double>::infinity()), 1);
  EXPECT_EQ(nda::norm(v, -std::numeric_limits<double>::infinity()), 1);
  EXPECT_EQ(nda::norm(v, 1.5), std::pow(double(size), 1.0 / 1.5));
}

// Check that the p-norm is correct by comparing it to its definition.
bool check_norm_p(auto &v, double p) { return norm(v, p) == std::pow(nda::sum(nda::pow(nda::abs(v), p)), 1.0 / p); };

TEST(NDA, LinearAlgebraNormRand) {
  const int size = 100;
  auto v         = nda::rand<double>(size);

  EXPECT_EQ(nda::norm(v), nda::norm(v, 2.0));
  EXPECT_EQ(nda::norm(v, 0.0), size);
  EXPECT_EQ(nda::norm(v, 1.0), nda::sum(abs(v)));
  EXPECT_EQ(nda::norm(v, 2.0), std::sqrt(std::real(nda::blas::dotc(v, v))));
  EXPECT_EQ(nda::norm(v, std::numeric_limits<double>::infinity()), nda::max_element(v));
  EXPECT_EQ(nda::norm(v, -std::numeric_limits<double>::infinity()), nda::min_element(v));

  EXPECT_TRUE((check_norm_p(v, -1.5)));
  EXPECT_TRUE((check_norm_p(v, -1.0)));
  EXPECT_TRUE((check_norm_p(v, 1.5)));
}

TEST(NDA, LinearAlgebraNormExample) {
  // check various p-norms of a vector
  auto run_checks = [](auto const &v) {
    EXPECT_EQ(nda::norm(v), nda::norm(v, 2.0));
    EXPECT_EQ(nda::norm(v, 0.0), 3);
    EXPECT_EQ(nda::norm(v, 1.0), 4);
    EXPECT_NEAR(nda::norm(v, 2.0), std::sqrt(7.5), 1e-15);

    EXPECT_TRUE((check_norm_p(v, -1.5)));
    EXPECT_TRUE((check_norm_p(v, -1.0)));
    EXPECT_TRUE((check_norm_p(v, 1.5)));
  };

  auto v = nda::array<double, 1>{-0.5, 0.0, 1.0, 2.5};
  run_checks(v);
  run_checks(1i * v);
  run_checks((1 + 1i) / sqrt(2) * v);
  EXPECT_EQ(nda::norm(v, std::numeric_limits<double>::infinity()), 2.5);
  EXPECT_EQ(nda::norm(v, -std::numeric_limits<double>::infinity()), 0.0);
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

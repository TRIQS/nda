// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"
#include "nda/traits.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <limits>

using namespace std::complex_literals;

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

template <typename T>
void test_dotproduct() {
  // scalars
  std::complex<T> u{1, 2};
  std::complex<T> v{3, -4};
  EXPECT_EQ(nda::linalg::dot(1, 2), 2);
  EXPECT_EQ(nda::linalg::dotc(1, 2), 2);
  EXPECT_EQ(nda::linalg::dot(2, -5.0), -10.0);
  EXPECT_EQ(nda::linalg::dotc(2, -5.0), -10.0);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(u, v), u * v);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(u, v), std::conj(u) * v);

  // BLAS compatible vectors
  nda::vector<T> a{1, 2, 3, 4, 5};
  nda::vector<T> b{10, 20, 30, 40, 50};
  EXPECT_EQ(nda::linalg::dot(a, b), nda::blas::dot(a, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a, b), nda::blas::dotc(a, b));

  nda::vector<std::complex<T>> c = a * (1.1 - 2.1i);
  nda::vector<std::complex<T>> d = b * (3 + 4i);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c, d), exp_dot(c, d));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c, d), exp_dotc(c, d));

  // vectors with different value types
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(a, c), exp_dot(a, c));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a, c), exp_dotc(a, c));

  nda::vector<int> e{1, 2, 3, 4, 5};
  EXPECT_EQ(nda::linalg::dot(e, e), exp_dot(e, e));
  EXPECT_EQ(nda::linalg::dot(e, b), exp_dot(e, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(e, b), exp_dotc(e, b));

  // lazy expressions
  auto sin_a = nda::make_regular(nda::sin(a));
  EXPECT_EQ(nda::linalg::dot(nda::sin(a), b), exp_dot(sin_a, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(nda::sin(a), b), exp_dotc(sin_a, b));

  // (strided) vector views
  auto c_v = c(nda::range(0, 5, 2));
  auto d_v = d(nda::range(1, 4));
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c_v, d_v), exp_dot(c_v, d_v));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c_v, d_v), exp_dotc(c_v, d_v));
}

TEST(NDA, LinearAlgebraDotProduct) {
  test_dotproduct<float>();
  test_dotproduct<double>();
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
  if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{T{210 + 70i}, T{240 + 80i}, T{270 + 90i}};
  auto y_h = nda::linalg::matvecmul(nda::conj(nda::transpose(A)), x_t);
  EXPECT_ARRAY_NEAR(y_h, exp_y_h);

  // strided matrix and vector views
  auto y_v = nda::linalg::matvecmul(A(nda::range(0, 4, 2), nda::range(0, 3, 2)), x(nda::range(0, 3, 2)));
  if constexpr (nda::is_complex_v<T>) {
    EXPECT_ARRAY_EQ(y_v, (nda::vector<T>{T{10 - 30i}, T{34 - 102i}}));
  } else {
    EXPECT_ARRAY_EQ(y_v, (nda::vector<T>{10, 34}));
  }
}

template <typename T>
constexpr auto test_matvecmul_layouts = []() {
  test_matvecmul<T, nda::C_layout>();
  test_matvecmul<T, nda::F_layout>();
};

TEST(NDA, LinearAlgebraMatvecmulGenericGemvBranch) { test_matvecmul_layouts<long>(); }

TEST(NDA, LinearAlgebraMatvecmulBLASBranch) {
  test_matvecmul_layouts<float>();
  test_matvecmul_layouts<std::complex<float>>();
  test_matvecmul_layouts<double>();
  test_matvecmul_layouts<std::complex<double>>();
}

template <typename T>
void test_matvecmul_promotion() {
  auto A_i = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_d = nda::matrix<T>{{1, 2}, {3, 4}};
  auto w_i = nda::vector<int>{1, 1};
  auto w_d = nda::vector<T>{1, 1};

  auto v_d1 = nda::linalg::matvecmul(A_d, w_i);
  static_assert(std::same_as<nda::get_value_t<decltype(v_d1)>, T>);
  EXPECT_ARRAY_NEAR(v_d1, (nda::vector<T>{3, 7}), std::numeric_limits<T>::epsilon());

  auto v_d2 = nda::linalg::matvecmul(A_i, w_d);
  static_assert(std::same_as<nda::get_value_t<decltype(v_d2)>, T>);
  EXPECT_ARRAY_NEAR(v_d2, (nda::vector<T>{3, 7}), std::numeric_limits<T>::epsilon());

  auto v_i = nda::linalg::matvecmul(A_i, w_i);
  static_assert(std::same_as<nda::get_value_t<decltype(v_i)>, int>);
  EXPECT_ARRAY_EQ(v_i, (nda::vector<int>{3, 7}));
}

TEST(NDA, LinearAlgebraMatvecmulPromotion) {
  test_matvecmul_promotion<float>();
  test_matvecmul_promotion<double>();
}

TEST(NDA, LinearAlgebraMatvecmulWithLazyExpressions) {
  auto A     = nda::array<double, 2>{{1, 2}, {3, 4}};
  auto A_sin = nda::array<double, 2>{nda::sin(A)};
  auto w     = nda::vector<double>{1, 1};
  auto w_sin = nda::vector<double>{nda::sin(w)};
  EXPECT_ARRAY_NEAR(nda::linalg::matvecmul(nda::sin(A), nda::sin(w)), nda::linalg::matvecmul(A_sin, w_sin), 1.e-13);
}

// Test the generic matmul function.
template <typename T, typename Layout1, typename Layout2>
void test_matmul() {
  auto A     = nda::matrix<T, Layout1>{{1, 2, 3}, {4, 5, 6}};
  auto B     = nda::matrix<T, Layout2>{{1, 2}, {3, 4}, {5, 6}};
  auto exp_C = nda::matrix<T>{{22, 28}, {49, 64}};
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
  auto exp_C_v = nda::matrix<T>{{16, 20}, {34, 44}};
  if constexpr (nda::is_complex_v<T>) exp_C_v *= (1 - 1i) * (2 - 1i);
  auto C_v = nda::matrix<T>(4, 4);
  C_v(nda::range(0, 4, 2), nda::range(0, 4, 2)) =
     nda::linalg::matmul(A(nda::range::all, nda::range(0, 3, 2)), B(nda::range(0, 3, 2), nda::range::all));
  EXPECT_ARRAY_NEAR(C_v(nda::range(0, 4, 2), nda::range(0, 4, 2)), exp_C_v);
}

template <typename T>
constexpr auto test_matmul_all_layouts = []() {
  test_matmul<T, nda::C_layout, nda::C_layout>();
  test_matmul<T, nda::C_layout, nda::F_layout>();
  test_matmul<T, nda::F_layout, nda::F_layout>();
  test_matmul<T, nda::F_layout, nda::C_layout>();
};

TEST(NDA, LinearAlgebraMatmulGenericGemmBranch) { test_matmul_all_layouts<long>(); }

TEST(NDA, LinearAlgebraMatmulBLASBranch) {
  test_matmul_all_layouts<float>();
  test_matmul_all_layouts<std::complex<float>>();
  test_matmul_all_layouts<double>();
  test_matmul_all_layouts<std::complex<double>>();
}

template <typename T>
void test_matmul_promotion() {
  auto A_i = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_d = nda::matrix<T>{{1, 2}, {3, 4}};

  auto B_d1 = nda::linalg::matmul(A_d, A_i);
  static_assert(std::same_as<nda::get_value_t<decltype(B_d1)>, T>);
  EXPECT_ARRAY_NEAR(B_d1, (nda::matrix<T>{{7, 10}, {15, 22}}), std::numeric_limits<T>::epsilon());

  auto B_d2 = nda::linalg::matmul(A_d, A_d);
  static_assert(std::same_as<nda::get_value_t<decltype(B_d2)>, T>);
  EXPECT_ARRAY_NEAR(B_d2, (nda::matrix<T>{{7, 10}, {15, 22}}), std::numeric_limits<T>::epsilon());

  auto B_i = nda::linalg::matmul(A_i, A_i);
  static_assert(std::same_as<nda::get_value_t<decltype(B_i)>, int>);
  EXPECT_ARRAY_NEAR(B_i, (nda::matrix<int>{{7, 10}, {15, 22}}), std::numeric_limits<T>::epsilon());
}

TEST(NDA, LinearAlgebraMatmulPromoteValueType) {
  test_matmul_promotion<float>();
  test_matmul_promotion<double>();
}

TEST(NDA, LinearAlgebraMatmulWithLazyExpressions) {
  auto A     = nda::array<double, 2>{{1, 2}, {3, 4}};
  auto A_sin = nda::array<double, 2>{nda::sin(A)};
  EXPECT_ARRAY_NEAR(nda::linalg::matmul(nda::sin(A), nda::sin(A)), nda::linalg::matmul(A_sin, A_sin), 1.e-13);
}

// Test general inverse and determinant functions.
template <typename T, typename Layout>
void test_inv_and_det() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;
  T fac          = 1.0;
  if constexpr (nda::is_complex_v<T>) fac = 1.0i;
  // condition number for matrix is ~275, but magnitudes are ~5 and we use absolute error
  constexpr double eps_close = 5 * 275 * std::numeric_limits<nda::get_fp_t<T>>::epsilon();

  // A is 3x3, B is 2x2, C is 1x1
  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  A *= fac;
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  Ainv /= fac;
  T detA = std::pow(fac, fp_t{3});
  auto B = matrix_t{{1, 2}, {0, 1}};
  B *= fac;
  auto Binv = matrix_t{{1, -2}, {0, 1}};
  Binv /= fac;
  T detB = std::pow(fac, fp_t{2});
  auto C = matrix_t{{3}};
  C *= fac;
  auto Cinv = matrix_t{{1.0 / 3.0}};
  Cinv /= fac;
  T detC = 3 * fac;

  // lambda that checks inverse functions for small matrices
  auto check_small_mat = [](auto const &M, auto const &Minv, auto detM, auto opt_inv, auto opt_det) {
    auto Minv2 = nda::linalg::inv(M);
    EXPECT_ARRAY_NEAR(Minv, Minv2, eps_close);
    EXPECT_COMPLEX_NEAR(nda::linalg::det(Minv2), 1.0 / detM, eps_close);
    Minv2 = nda::linalg::inv(Minv2);
    static_assert(std::is_same_v<nda::get_value_t<decltype(Minv2)>, T>);
    EXPECT_ARRAY_NEAR(M, Minv2, eps_close);
    EXPECT_COMPLEX_NEAR(nda::linalg::det(Minv2), detM, eps_close);

    auto Minv3 = M;
    nda::linalg::inv_in_place(Minv3);
    EXPECT_ARRAY_NEAR(Minv, Minv3, eps_close);
    EXPECT_COMPLEX_NEAR(nda::linalg::det_in_place(Minv3), 1.0 / detM, eps_close);
    nda::linalg::inv_in_place(Minv3);
    EXPECT_ARRAY_NEAR(M, Minv3, eps_close);
    EXPECT_COMPLEX_NEAR(nda::linalg::det_in_place(Minv3), detM, eps_close);

    auto Minv4 = M;
    opt_inv(Minv4);
    EXPECT_ARRAY_NEAR(Minv, Minv4, eps_close);
    EXPECT_COMPLEX_NEAR(opt_det(Minv4), 1.0 / detM, eps_close);
    opt_inv(Minv4);
    EXPECT_ARRAY_NEAR(M, Minv4, eps_close);
    EXPECT_COMPLEX_NEAR(opt_det(Minv4), detM, eps_close);
  };

  check_small_mat(A, Ainv, detA, [](auto &M) { return nda::linalg::inv_in_place_3d(M); }, [](auto &M) { return nda::linalg::det_3d(M); });
  check_small_mat(B, Binv, detB, [](auto &M) { return nda::linalg::inv_in_place_2d(M); }, [](auto &M) { return nda::linalg::det_2d(M); });
  check_small_mat(C, Cinv, detC, [](auto &M) { return nda::linalg::inv_in_place_1d(M); }, [](auto &M) { return nda::linalg::det_1d(M); });

  // matrix view
  EXPECT_ARRAY_NEAR(nda::linalg::inv(A(nda::range(0, 2), nda::range(0, 2))), Binv, eps_close);
  EXPECT_COMPLEX_NEAR(nda::linalg::det(A(nda::range(0, 2), nda::range(0, 2))), detB, eps_close);

  // 4x4 matrix
  auto D = matrix_t{{2, 2, 2, 2}, {2, 4, 6, 8}, {2, 6, 12, 20}, {2, 8, 20, 40}};
  D *= fac;
  auto Dinv = matrix_t{{2, -3, 2, -0.5}, {-3, 7, -5.5, 1.5}, {2, -5.5, 5, -1.5}, {-0.5, 1.5, -1.5, 0.5}};
  Dinv /= fac;
  T detD = 16 * std::pow(fac, fp_t{4});

  auto Dinv2 = nda::linalg::inv(D);
  static_assert(std::is_same_v<nda::get_value_t<decltype(Dinv2)>, T>);
  EXPECT_ARRAY_NEAR(Dinv, Dinv2, eps_close);
  EXPECT_COMPLEX_NEAR(nda::linalg::det(Dinv2), 1.0 / detD, eps_close);
  Dinv2 = nda::linalg::inv(Dinv2);
  EXPECT_ARRAY_NEAR(D, Dinv2, eps_close);
  // This checks absolute error, but 16 is fairly 'large' in fp32, so scale down to relative error
  EXPECT_COMPLEX_NEAR(nda::linalg::det(Dinv2) / fp_t(16), detD / fp_t(16), eps_close);

  auto Dinv3 = D;
  nda::linalg::inv_in_place(Dinv3);
  EXPECT_ARRAY_NEAR(Dinv, Dinv3, eps_close);
  nda::linalg::inv_in_place(Dinv3);
  EXPECT_ARRAY_NEAR(D, Dinv3, eps_close);
  // This checks absolute error, but 16 is fairly 'large' in fp32, so scale down to relative error
  EXPECT_COMPLEX_NEAR(nda::linalg::det_in_place(Dinv3) / fp_t{16}, detD / fp_t{16}, eps_close);
}

TEST(NDA, LinearAlgebraInvAndDet) {
  test_inv_and_det<float, nda::C_layout>();
  test_inv_and_det<float, nda::F_layout>();
  test_inv_and_det<std::complex<float>, nda::C_layout>();
  test_inv_and_det<std::complex<float>, nda::F_layout>();
  test_inv_and_det<double, nda::C_layout>();
  test_inv_and_det<double, nda::F_layout>();
  test_inv_and_det<std::complex<double>, nda::C_layout>();
  test_inv_and_det<std::complex<double>, nda::F_layout>();
}

// Check that the eigenvectors/values are correct.
void check_eigen(auto const &A, auto const &V, auto const &l, double eps_close) {
  for (auto i : nda::range(0, A.extent(0))) { EXPECT_ARRAY_NEAR(A * V(nda::range::all, i), l(i) * V(nda::range::all, i), eps_close); }
}

void check_eigen(auto const &A, auto const &B, auto const &V, auto const &l, int itype, double eps_close) {
  for (auto i : nda::range(0, A.extent(0))) {
    if (itype == 1) {
      EXPECT_ARRAY_NEAR(A * V(nda::range::all, i), l(i) * B * V(nda::range::all, i), eps_close);
    } else if (itype == 2) {
      EXPECT_ARRAY_NEAR(A * B * V(nda::range::all, i), l(i) * V(nda::range::all, i), eps_close);
    } else {
      EXPECT_ARRAY_NEAR(B * A * V(nda::range::all, i), l(i) * V(nda::range::all, i), eps_close);
    }
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
  using fp_type = nda::get_fp_t<T>;
  auto D        = nda::eye<fp_type>(n) * a + nda::diag(nda::vector<fp_type>::rand(n)) * (b - a);

  // return Q * D * Q^H (hermitian/symmetric)
  return matrix_t{Q * D * nda::dagger(Q)};
}

// Test the eigh and eigvalsh functions.
template <typename T>
void test_eigh_eigvalsh() {
  // Max condition number I got for the syhe_matrix is ~300, matrix vals are ~1
  constexpr double eps_close = 300 * std::numeric_limits<nda::get_fp_t<T>>::epsilon();

  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);

    // use eigh to compute eigenvalues and eigenvectors
    auto [w1, V1] = nda::linalg::eigh(A);
    check_eigen(A, V1, w1, eps_close);

    // use eigh_in_place to compute eigenvalues and eigenvectors
    auto V2 = A;
    auto w2 = nda::linalg::eigh_in_place(V2);
    check_eigen(A, V2, w2, eps_close);
    // Eigenvectors are only the same up to a sign, so some columns in V1 are minus that in V2
    // checking the absolute values should be sufficient in any non-trivial case
    EXPECT_ARRAY_NEAR(nda::abs(V1), nda::abs(V2), eps_close);
    EXPECT_ARRAY_NEAR(w1, w2, eps_close);

    // use eigvalsh to compute eigenvalues only
    auto w3 = nda::linalg::eigvalsh(A);
    EXPECT_ARRAY_NEAR(w1, w3, eps_close);

    // use eigvalsh_in_place to compute eigenvalues only
    auto A4 = A;
    auto w4 = nda::linalg::eigvalsh_in_place(A4);
    EXPECT_ARRAY_NEAR(w1, w4, eps_close);

    // use eigh with a C-layout matrix
    auto A5       = nda::matrix<T, nda::C_layout>{A};
    auto [w5, V5] = nda::linalg::eigh(A5);
    check_eigen(A5, V5, w5, eps_close);
    EXPECT_ARRAY_NEAR(V1, V5, eps_close);
    EXPECT_ARRAY_NEAR(w1, w5, eps_close);

    // use eigvalsh with a C-layout matrix
    auto w6 = nda::linalg::eigvalsh(A5);
    EXPECT_ARRAY_NEAR(w1, w6, eps_close);
  }
}

TEST(NDA, LinearAlgebraEighAndEigvalsh) {
  test_eigh_eigvalsh<float>();
  test_eigh_eigvalsh<std::complex<float>>();
  test_eigh_eigvalsh<double>();
  test_eigh_eigvalsh<std::complex<double>>();
}

// Test the eigh and eigvalsh functions for generalized eigenvalue problems.
template <typename T>
void test_generalized_eigh_eigvalsh(int itype) {
  constexpr double eps_close = 100 * std::numeric_limits<nda::get_fp_t<T>>::epsilon();

  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);
    auto B = syhe_matrix<T>(i, 1e-6, 1);

    // use eigh to compute eigenvalues and eigenvectors
    auto [w1, V1] = nda::linalg::eigh(A, B, itype);
    check_eigen(A, B, V1, w1, itype, eps_close);

    // use eigh_in_place to compute eigenvalues and eigenvectors
    auto V2 = A;
    auto B2 = B;
    auto w2 = nda::linalg::eigh_in_place(V2, B2, itype);
    check_eigen(A, B, V2, w2, itype, eps_close);
    EXPECT_ARRAY_NEAR(V1, V2, eps_close);
    EXPECT_ARRAY_NEAR(w1, w2, eps_close);

    // use eigvalsh to compute eigenvalues only
    auto w3 = nda::linalg::eigvalsh(A, B, itype);
    EXPECT_ARRAY_NEAR(w1, w3, eps_close);

    // use eigvalsh_in_place to compute eigenvalues only
    auto A4 = A;
    auto B4 = B;
    auto w4 = nda::linalg::eigvalsh_in_place(A4, B4, itype);
    EXPECT_ARRAY_NEAR(w1, w4, eps_close);

    // use eigh with a C-layout matrices
    auto A5       = nda::matrix<T, nda::C_layout>{A};
    auto B5       = nda::matrix<T, nda::C_layout>{B};
    auto [w5, V5] = nda::linalg::eigh(A5, B5, itype);
    check_eigen(A, B, V5, w5, itype, eps_close);
    EXPECT_ARRAY_NEAR(V1, V5, eps_close);
    EXPECT_ARRAY_NEAR(w1, w5, eps_close);

    // use eigvalsh with a C-layout matrices
    auto w6 = nda::linalg::eigvalsh(A5, B5, itype);
    EXPECT_ARRAY_NEAR(w1, w6, eps_close);
  }
}

TEST(NDA, LinearAlgebraGeneralizedEighAndEigvalsh) {
  test_generalized_eigh_eigvalsh<float>(1);
  test_generalized_eigh_eigvalsh<float>(2);
  test_generalized_eigh_eigvalsh<float>(3);
  test_generalized_eigh_eigvalsh<std::complex<float>>(1);
  test_generalized_eigh_eigvalsh<std::complex<float>>(2);
  test_generalized_eigh_eigvalsh<std::complex<float>>(3);
  test_generalized_eigh_eigvalsh<double>(1);
  test_generalized_eigh_eigvalsh<double>(2);
  test_generalized_eigh_eigvalsh<double>(3);
  test_generalized_eigh_eigvalsh<std::complex<double>>(1);
  test_generalized_eigh_eigvalsh<std::complex<double>>(2);
  test_generalized_eigh_eigvalsh<std::complex<double>>(3);
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
  run_checks((1 + 1i) / std::sqrt(2) * v);
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
  // condition number for matrix is ~275, but magnitudes are ~5 and we use absolute error
  constexpr double eps_close = 5 * 275 * std::numeric_limits<nda::get_fp_t<value_t>>::epsilon();

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // solve A * X = B using the exact matrix inverse
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X    = matrix_t{Ainv * B};
  EXPECT_ARRAY_NEAR(matrix_t{A * X}, B, eps_close);

  // solve A * X = B using solve_in_place
  if constexpr (nda::blas::has_F_layout<matrix_t>) {
    auto Acopy = matrix_t{A};
    auto Bcopy = matrix_t{B};
    nda::linalg::solve_in_place(Acopy, Bcopy);
    EXPECT_ARRAY_NEAR(matrix_t{A * Bcopy}, B, eps_close);
    EXPECT_ARRAY_NEAR(X, Bcopy, eps_close);

    // solve A * x = b using solve_in_place
    Acopy  = A;
    auto b = vector_t{B(nda::range::all, 0)};
    nda::linalg::solve_in_place(Acopy, b);
    EXPECT_ARRAY_NEAR(A * b, B(nda::range::all, 0), eps_close);
    EXPECT_ARRAY_NEAR(X(nda::range::all, 0), b, eps_close);
  }

  // solve A * X = B using solve
  auto X2 = nda::linalg::solve(A, B);
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B, eps_close);
  EXPECT_ARRAY_NEAR(X, X2, eps_close);

  // solve A * x = b using solve
  auto x = nda::linalg::solve(A, B(nda::range::all, 0));
  EXPECT_ARRAY_NEAR(A * x, B(nda::range::all, 0), eps_close);
  EXPECT_ARRAY_NEAR(X(nda::range::all, 0), x, eps_close);
}

TEST(NDA, LinearAlgebraSolve) {
  test_solve<float, nda::C_layout>();
  test_solve<float, nda::F_layout>();
  test_solve<std::complex<float>, nda::C_layout>();
  test_solve<std::complex<float>, nda::F_layout>();
  test_solve<double, nda::C_layout>();
  test_solve<double, nda::F_layout>();
  test_solve<std::complex<double>, nda::C_layout>();
  test_solve<std::complex<double>, nda::F_layout>();
}

// Test the svd and svd_in_place functions.
template <typename T, typename Layout>
void test_svd() {
  using matrix_t = nda::matrix<T, Layout>;
  // condition number for matrix is 4, but max magnitude is 8 and we use absolute error
  constexpr double eps_close = 4 * 8 * std::numeric_limits<nda::get_fp_t<T>>::epsilon();

  auto A = matrix_t{{2, -2, 1}, {-4, -8, -8}};
  auto s = nda::vector<double>{12, 3};

  // compute the SVD of A
  auto [U_1, s_1, VH_1] = nda::linalg::svd(A);
  auto S_1              = matrix_t::zeros(A.shape());
  diagonal(S_1)         = s_1;
  EXPECT_ARRAY_NEAR(s_1, s, eps_close);
  EXPECT_ARRAY_NEAR(A, U_1 * S_1 * VH_1, eps_close);

  // compute the SVD of A in place
  auto A_copy           = A;
  auto [U_2, s_2, VH_2] = nda::linalg::svd_in_place(A_copy);
  auto S_2              = matrix_t::zeros(A.shape());
  diagonal(S_2)         = s_2;
  EXPECT_ARRAY_NEAR(s, s_2, eps_close);
  EXPECT_ARRAY_NEAR(A, U_2 * S_2 * VH_2, eps_close);
}

TEST(NDA, LinearAlgebraSVD) {
  test_svd<float, nda::C_layout>();
  test_svd<float, nda::F_layout>();
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

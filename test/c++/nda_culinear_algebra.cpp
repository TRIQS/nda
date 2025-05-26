// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <concepts>

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

template <nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_dot() {
  // BLAS compatible vectors
  auto a   = nda::vector<double>{1, 2, 3, 4, 5};
  auto b   = nda::vector<double>{10, 20, 30, 40, 50};
  auto a_d = to_addr_space<AS1>(a);
  auto b_d = to_addr_space<AS2>(b);
  EXPECT_DOUBLE_EQ(nda::linalg::dot(a_d, b_d), nda::blas::dot(a, b));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a_d, b_d), nda::blas::dotc(a, b));

  auto c   = nda::vector<std::complex<double>>{a * (1.1 - 2.1i)};
  auto d   = nda::vector<std::complex<double>>{b * (3 + 4i)};
  auto c_d = to_addr_space<AS1>(c);
  auto d_d = to_addr_space<AS2>(d);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c_d, d_d), exp_dot(c, d));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c_d, d_d), exp_dotc(c, d));

  // vectors with different value types
  if constexpr (nda::mem::have_host_compatible_addr_space<decltype(a_d), decltype(b_d)>) {
    EXPECT_COMPLEX_NEAR(nda::linalg::dot(a_d, c_d), exp_dot(a, c));
    EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a_d, c_d), exp_dotc(a, c));

    auto e   = nda::vector<int>{1, 2, 3, 4, 5};
    auto e_d = to_addr_space<AS1>(e);
    EXPECT_EQ(nda::linalg::dot(e_d, e_d), exp_dot(e, e));
    EXPECT_DOUBLE_EQ(nda::linalg::dot(e_d, b_d), exp_dot(e, b));
    EXPECT_COMPLEX_NEAR(nda::linalg::dotc(e_d, b_d), exp_dotc(e, b));

    // lazy expressions
    auto sin_a = nda::make_regular(nda::sin(a));
    EXPECT_DOUBLE_EQ(nda::linalg::dot(nda::sin(a_d), b), exp_dot(sin_a, b));
    EXPECT_COMPLEX_NEAR(nda::linalg::dotc(nda::sin(a_d), b), exp_dotc(sin_a, b));
  }

  // (strided) vector views
  auto rg1 = nda::range(0, 5, 2);
  auto rg2 = nda::range(1, 4);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c_d(rg1), d_d(rg2)), exp_dot(c(rg1), d(rg2)));
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c_d(rg1), d_d(rg2)), exp_dotc(c(rg1), d(rg2)));
}

TEST(NDA, CULinearAlgebraDotProduct) {
  test_dot<nda::mem::Device, nda::mem::Device>();
  test_dot<nda::mem::Device, nda::mem::Unified>();
  test_dot<nda::mem::Unified, nda::mem::Device>();
  test_dot<nda::mem::Unified, nda::mem::Unified>();
  test_dot<nda::mem::Unified, nda::mem::Host>();
  test_dot<nda::mem::Host, nda::mem::Unified>();
}

// Test the generic matvecmul function.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
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
  auto A_d   = to_addr_space<AS1>(A);
  auto x_d   = to_addr_space<AS2>(x);
  auto x_t_d = to_addr_space<AS2>(x_t);

  // y = A * x
  auto y_d = nda::linalg::matvecmul(A_d, x_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_d), exp_y);

  // y_t = A^T * x_t
  auto y_t_d = nda::linalg::matvecmul(nda::transpose(A_d), x_t_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_t_d), exp_y_t);

  // y_h = A^H * x_t
  if constexpr (nda::blas::has_F_layout<decltype(A_d)>) {
    auto exp_y_h = exp_y_t;
    if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{210 + 70i, 240 + 80i, 270 + 90i};
    auto y_h_d = nda::linalg::matvecmul(nda::conj(nda::transpose(A_d)), x_t_d);
    EXPECT_ARRAY_NEAR(nda::to_host(y_h_d), exp_y_h);
  }

  // strided matrix and vector views
  if constexpr (nda::mem::have_host_compatible_addr_space<decltype(A_d), decltype(x_d)>) {
    auto y_v_d = nda::linalg::matvecmul(A_d(nda::range(0, 4, 2), nda::range(0, 3, 2)), x_d(nda::range(0, 3, 2)));
    if constexpr (nda::is_complex_v<T>) {
      EXPECT_ARRAY_EQ(nda::to_host(y_v_d), (nda::vector<T>{10 - 30i, 34 - 102i}));
    } else {
      EXPECT_ARRAY_EQ(nda::to_host(y_v_d), (nda::vector<T>{10, 34}));
    }
  }
}

TEST(NDA, CULinearAlgebraMatvecmulGenericGemvBranch) {
  test_matvecmul<long, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matvecmul<long, nda::C_layout, nda::mem::Unified, nda::mem::Host>();
  test_matvecmul<long, nda::C_layout, nda::mem::Host, nda::mem::Unified>();

  test_matvecmul<long, nda::F_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matvecmul<long, nda::F_layout, nda::mem::Unified, nda::mem::Host>();
  test_matvecmul<long, nda::F_layout, nda::mem::Host, nda::mem::Unified>();
}

TEST(NDA, CULinearAlgebraMatvecmulBLASBranch) {
  test_matvecmul<double, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_matvecmul<double, nda::F_layout, nda::mem::Unified, nda::mem::Device>();
  test_matvecmul<double, nda::C_layout, nda::mem::Device, nda::mem::Unified>();
  test_matvecmul<std::complex<double>, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_matvecmul<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Device>();
  test_matvecmul<std::complex<double>, nda::C_layout, nda::mem::Device, nda::mem::Unified>();

  test_matvecmul<double, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matvecmul<double, nda::F_layout, nda::mem::Unified, nda::mem::Host>();
  test_matvecmul<double, nda::C_layout, nda::mem::Host, nda::mem::Unified>();
  test_matvecmul<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matvecmul<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Host>();
  test_matvecmul<std::complex<double>, nda::C_layout, nda::mem::Host, nda::mem::Unified>();
}

// Test the generic matmul function.
template <typename T, typename Layout1, typename Layout2, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_matmul() {
  auto A     = nda::matrix<T, Layout1>{{1, 2, 3}, {4, 5, 6}};
  auto B     = nda::matrix<T, Layout2>{{1, 2}, {3, 4}, {5, 6}};
  auto exp_C = nda::matrix<T>{{22, 28}, {49, 64}};
  if constexpr (nda::is_complex_v<T>) {
    A *= 1 - 1i;
    B *= 2 - 1i;
    exp_C *= (1 - 1i) * (2 - 1i);
  }
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);

  // C = A * B
  auto C_d = nda::linalg::matmul(A_d, B_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C_d), exp_C);

  // C_t = B^T * A^T
  auto C_t_d = nda::linalg::matmul(nda::transpose(B_d), nda::transpose(A_d));
  EXPECT_ARRAY_NEAR(nda::to_host(C_t_d), nda::transpose(exp_C));

  // C_h = B^H * A^H --> not working right now because of how we determine the layout of C
  // if constexpr (std::same_as<Layout1, Layout2>) {
  //   auto C_h_d = nda::linalg::matmul(nda::dagger(B_d), nda::dagger(A_d));
  //   EXPECT_ARRAY_NEAR(nda::to_host(C_h_d), nda::dagger(exp_C));
  // }

  // strided matrix views
  if constexpr (nda::mem::have_host_compatible_addr_space<decltype(A_d), decltype(B_d)>) {
    auto exp_C_v = nda::matrix<T>{{16, 20}, {34, 44}};
    if constexpr (nda::is_complex_v<T>) exp_C_v *= (1 - 1i) * (2 - 1i);
    auto C_v_d = nda::matrix<T>(4, 4);
    C_v_d(nda::range(0, 4, 2), nda::range(0, 4, 2)) =
       nda::linalg::matmul(A_d(nda::range::all, nda::range(0, 3, 2)), B_d(nda::range(0, 3, 2), nda::range::all));
    EXPECT_ARRAY_NEAR(C_v_d(nda::range(0, 4, 2), nda::range(0, 4, 2)), exp_C_v);
  }
}

TEST(NDA, CULinearAlgebraMatmulGenericGemmBranch) {
  test_matmul<long, nda::C_layout, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matmul<long, nda::C_layout, nda::F_layout, nda::mem::Host, nda::mem::Unified>();
  test_matmul<long, nda::F_layout, nda::F_layout, nda::mem::Unified, nda::mem::Host>();
  test_matmul<long, nda::F_layout, nda::C_layout, nda::mem::Host, nda::mem::Unified>();
}

TEST(NDA, CULinearAlgebraMatmulBLASBranch) {
  test_matmul<double, nda::C_layout, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_matmul<double, nda::C_layout, nda::F_layout, nda::mem::Unified, nda::mem::Device>();
  test_matmul<double, nda::F_layout, nda::F_layout, nda::mem::Device, nda::mem::Unified>();
  test_matmul<double, nda::F_layout, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_matmul<std::complex<double>, nda::C_layout, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_matmul<std::complex<double>, nda::C_layout, nda::F_layout, nda::mem::Unified, nda::mem::Device>();
  test_matmul<std::complex<double>, nda::F_layout, nda::F_layout, nda::mem::Device, nda::mem::Unified>();
  test_matmul<std::complex<double>, nda::F_layout, nda::C_layout, nda::mem::Device, nda::mem::Device>();

  test_matmul<double, nda::C_layout, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matmul<double, nda::C_layout, nda::F_layout, nda::mem::Host, nda::mem::Unified>();
  test_matmul<double, nda::F_layout, nda::F_layout, nda::mem::Unified, nda::mem::Host>();
  test_matmul<double, nda::F_layout, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matmul<std::complex<double>, nda::C_layout, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
  test_matmul<std::complex<double>, nda::C_layout, nda::F_layout, nda::mem::Host, nda::mem::Unified>();
  test_matmul<std::complex<double>, nda::F_layout, nda::F_layout, nda::mem::Unified, nda::mem::Host>();
  test_matmul<std::complex<double>, nda::F_layout, nda::C_layout, nda::mem::Unified, nda::mem::Unified>();
}

// Test general inverse functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_inv() {
  using matrix_t = nda::matrix<T, Layout>;
  T fac          = 1.0;
  if constexpr (nda::is_complex_v<T>) fac = 1.0i;

  // lambda that checks inverse function
  auto check_inv = [](auto const &M, auto const &Minv) {
    auto M_d    = to_addr_space<AS>(M);
    auto Minv_d = nda::linalg::inv(M_d);
    EXPECT_ARRAY_NEAR(Minv, nda::to_host(Minv_d));
    auto M2_d = nda::linalg::inv(Minv_d);
    EXPECT_ARRAY_NEAR(M, nda::to_host(M2_d));
  };

  // 1x1 matrix
  auto C = matrix_t{{3}};
  C *= fac;
  auto Cinv = matrix_t{{1.0 / 3.0}};
  Cinv /= fac;
  check_inv(C, Cinv);

  // 2x2 matrix
  auto B = matrix_t{{1, 2}, {0, 1}};
  B *= fac;
  auto Binv = matrix_t{{1, -2}, {0, 1}};
  Binv /= fac;
  check_inv(B, Binv);

  // 3x3 matrix
  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  A *= fac;
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  Ainv /= fac;
  check_inv(A, Ainv);

  // 4x4 matrix
  auto D = matrix_t{{2, 2, 2, 2}, {2, 4, 6, 8}, {2, 6, 12, 20}, {2, 8, 20, 40}};
  D *= fac;
  auto Dinv = matrix_t{{2, -3, 2, -0.5}, {-3, 7, -5.5, 1.5}, {2, -5.5, 5, -1.5}, {-0.5, 1.5, -1.5, 0.5}};
  Dinv /= fac;
  check_inv(D, Dinv);
}

TEST(NDA, CULinearAlgebraInv) {
  test_inv<double, nda::C_layout, nda::mem::Device>();
  test_inv<double, nda::F_layout, nda::mem::Device>();
  test_inv<std::complex<double>, nda::C_layout, nda::mem::Device>();
  test_inv<std::complex<double>, nda::F_layout, nda::mem::Device>();

  test_inv<double, nda::C_layout, nda::mem::Unified>();
  test_inv<double, nda::F_layout, nda::mem::Unified>();
  test_inv<std::complex<double>, nda::C_layout, nda::mem::Unified>();
  test_inv<std::complex<double>, nda::F_layout, nda::mem::Unified>();
}

// Test the outer product function.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_outer_product() {
  // outer product of two arrays
  auto A = nda::array<T, 2, Layout>::rand(2, 3);
  auto B = nda::array<T, 3, Layout>::rand(4, 5, 6);
  auto C = nda::array<T, 5, Layout>(2, 3, 4, 5, 6);
  for (auto [i, j] : A.indices())
    for (auto [k, l, m] : B.indices()) C(i, j, k, l, m) = A(i, j) * B(k, l, m);
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);
  EXPECT_ARRAY_NEAR(C, nda::to_host(nda::linalg::outer_product(A_d, B_d)));

  // outer product of two vectors
  nda::vector<T> v{1, 2};
  nda::vector<T> w{3, 4, 5};
  auto v_d = to_addr_space<AS1>(v);
  auto w_d = to_addr_space<AS2>(w);
  auto M_d = nda::linalg::outer_product(v_d, w_d);
  static_assert(nda::get_algebra<decltype(M_d)> == 'M');
  static_assert(nda::blas::has_C_layout<decltype(M_d)>);
  EXPECT_ARRAY_NEAR(nda::matrix<T>{{3, 4, 5}, {6, 8, 10}}, nda::to_host(M_d));
}

TEST(NDA, CULinearAlgebraOuterProduct) {
  test_outer_product<double, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_outer_product<double, nda::F_layout, nda::mem::Device, nda::mem::Unified>();
  test_outer_product<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Device>();
  test_outer_product<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Unified>();
  test_outer_product<double, nda::C_layout, nda::mem::Unified, nda::mem::Host>();
  test_outer_product<std::complex<double>, nda::F_layout, nda::mem::Host, nda::mem::Unified>();
}

// Test the generic solve and solve_in_place functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_solve() {
  auto A   = nda::matrix<T, Layout>{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B   = nda::matrix<T, nda::F_layout>{{1, 5}, {4, 5}, {3, 6}};
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);
  auto b_d = to_addr_space<AS2>(nda::make_regular(B(nda::range::all, 0)));

  // solve A * X = B using the exact matrix inverse
  auto Ainv = nda::matrix<T, Layout>{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X    = nda::matrix<T, Layout>{Ainv * B};
  EXPECT_ARRAY_NEAR(A * X, B);

  // solve A * X = B using solve_in_place
  auto A2_d = A_d;
  auto B2_d = B_d;
  nda::linalg::solve_in_place(A2_d, B2_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(B2_d), B);
  EXPECT_ARRAY_NEAR(X, nda::to_host(B2_d));

  // solve A * x = b using solve_in_place
  A2_d      = A;
  auto b2_d = b_d;
  nda::linalg::solve_in_place(A2_d, b2_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(b2_d), B(nda::range::all, 0));
  EXPECT_ARRAY_NEAR(X(nda::range::all, 0), nda::to_host(b2_d));

  // solve A * X = B using solve
  auto X_d = nda::linalg::solve(A_d, B_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(X_d), B);
  EXPECT_ARRAY_NEAR(X, nda::to_host(X_d));

  // solve A * x = b using solve
  auto x_d = nda::linalg::solve(A_d, b_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(x_d), B(nda::range::all, 0));
  EXPECT_ARRAY_NEAR(X(nda::range::all, 0), nda::to_host(x_d));
}

TEST(NDA, CULinearAlgebraSolve) {
  test_solve<double, nda::C_layout, nda::mem::Device, nda::mem::Device>();
  test_solve<double, nda::F_layout, nda::mem::Device, nda::mem::Unified>();
  test_solve<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Device>();
  test_solve<std::complex<double>, nda::F_layout, nda::mem::Unified, nda::mem::Unified>();
  test_solve<double, nda::F_layout, nda::mem::Host, nda::mem::Unified>();
  test_solve<std::complex<double>, nda::C_layout, nda::mem::Unified, nda::mem::Host>();
}

// Test the svd and svd_in_place functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_svd() {
  auto A = nda::matrix<T, Layout>{{2, -2, 1}, {-4, -8, -8}};
  if constexpr (std::same_as<Layout, nda::F_layout>) {
    // CUDA cannot handle when m < n
    A = nda::matrix<T, Layout>(nda::transpose(A));
  }
  auto s = nda::vector<double>{12, 3};

  // compute the SVD of A
  auto A_d              = to_addr_space<AS>(A);
  auto [U_d, s_d, VH_d] = nda::linalg::svd(A_d);
  auto S                = nda::matrix<T, Layout>::zeros(A.shape());
  nda::diagonal(S)      = nda::to_host(s_d);
  EXPECT_ARRAY_NEAR(s, nda::to_host(s_d), 1e-14);
  EXPECT_ARRAY_NEAR(A, nda::to_host(U_d) * S * nda::to_host(VH_d), 1e-14);

  // compute the SVD of A in place
  auto [U_d2, s_d2, VH_d2] = nda::linalg::svd_in_place(A_d);
  auto S2                  = nda::matrix<T, Layout>::zeros(A.shape());
  nda::diagonal(S2)        = nda::to_host(s_d2);
  EXPECT_ARRAY_NEAR(s, nda::to_host(s_d2), 1e-14);
  EXPECT_ARRAY_NEAR(A, nda::to_host(U_d2) * S2 * nda::to_host(VH_d2), 1e-14);
}

TEST(NDA, CULinearAlgebraSVD) {
  test_svd<double, nda::C_layout, nda::mem::Device>();
  test_svd<double, nda::F_layout, nda::mem::Device>();
  test_svd<std::complex<double>, nda::C_layout, nda::mem::Device>();
  test_svd<std::complex<double>, nda::F_layout, nda::mem::Device>();

  test_svd<double, nda::C_layout, nda::mem::Unified>();
  test_svd<double, nda::F_layout, nda::mem::Unified>();
  test_svd<std::complex<double>, nda::C_layout, nda::mem::Unified>();
  test_svd<std::complex<double>, nda::F_layout, nda::mem::Unified>();
}

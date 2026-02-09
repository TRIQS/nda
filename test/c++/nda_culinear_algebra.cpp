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
using nda::C_layout, nda::F_layout;
using nda::mem::Host, nda::mem::Device, nda::mem::Unified;

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

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_dot() {
  using cplx_t = std::complex<T>;

  // error tolerance need to be increased compared to non-CUDA version (see nda_linear_algebra.cpp)
  constexpr auto tol = fp_tol<T> * 20;

  // BLAS compatible vectors
  auto a   = nda::vector<T>{1, 2, 3, 4, 5};
  auto b   = nda::vector<T>{10, 20, 30, 40, 50};
  auto a_d = to_addr_space<AS1>(a);
  auto b_d = to_addr_space<AS2>(b);
  EXPECT_NEAR(nda::linalg::dot(a_d, b_d), nda::blas::dot(a, b), tol);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a_d, b_d), nda::blas::dotc(a, b), tol);

  auto c   = nda::vector<cplx_t>{a * cplx_t{1.1 - 2.1i}};
  auto d   = nda::vector<cplx_t>{b * cplx_t{3 + 4i}};
  auto c_d = to_addr_space<AS1>(c);
  auto d_d = to_addr_space<AS2>(d);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c_d, d_d), exp_dot(c, d), tol);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c_d, d_d), exp_dotc(c, d), tol);

  // vectors with different value types
  if constexpr (nda::mem::have_host_compatible_addr_space<decltype(a_d), decltype(b_d)>) {
    EXPECT_COMPLEX_NEAR(nda::linalg::dot(a_d, c_d), exp_dot(a, c), tol);
    EXPECT_COMPLEX_NEAR(nda::linalg::dotc(a_d, c_d), exp_dotc(a, c), tol);

    auto e   = nda::vector<int>{1, 2, 3, 4, 5};
    auto e_d = to_addr_space<AS1>(e);
    EXPECT_EQ(nda::linalg::dot(e_d, e_d), exp_dot(e, e));
    EXPECT_NEAR(nda::linalg::dot(e_d, b_d), exp_dot(e, b), tol);
    EXPECT_COMPLEX_NEAR(nda::linalg::dotc(e_d, b_d), exp_dotc(e, b), tol);

    // lazy expressions
    auto sin_a = nda::make_regular(nda::sin(a));
    EXPECT_NEAR(nda::linalg::dot(nda::sin(a_d), b), exp_dot(sin_a, b), tol);
    EXPECT_COMPLEX_NEAR(nda::linalg::dotc(nda::sin(a_d), b), exp_dotc(sin_a, b), tol);
  }

  // (strided) vector views
  auto rg1 = nda::range(0, 5, 2);
  auto rg2 = nda::range(1, 4);
  EXPECT_COMPLEX_NEAR(nda::linalg::dot(c_d(rg1), d_d(rg2)), exp_dot(c(rg1), d(rg2)), tol);
  EXPECT_COMPLEX_NEAR(nda::linalg::dotc(c_d(rg1), d_d(rg2)), exp_dotc(c(rg1), d(rg2)), tol);
}

template <typename T>
void test_dot_address_spaces() {
  test_dot<T, Device, Device>();
  test_dot<T, Device, Unified>();
  test_dot<T, Unified, Device>();
  test_dot<T, Unified, Unified>();
  test_dot<T, Unified, Host>();
  test_dot<T, Host, Unified>();
}

TEST(NDA, CULinearAlgebraDotProduct) {
  test_dot_address_spaces<float>();
  test_dot_address_spaces<double>();
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
    A *= T{1 - 1i};
    x *= T{2 - 1i};
    x_t *= T{2 - 1i};
    exp_y *= T{(1 - 1i) * (2 - 1i)};
    exp_y_t *= T{(1 - 1i) * (2 - 1i)};
  }
  auto A_d   = to_addr_space<AS1>(A);
  auto x_d   = to_addr_space<AS2>(x);
  auto x_t_d = to_addr_space<AS2>(x_t);

  // y = A * x
  auto y_d = nda::linalg::matvecmul(A_d, x_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_d), exp_y, fp_tol<T>);

  // y_t = A^T * x_t
  auto y_t_d = nda::linalg::matvecmul(nda::transpose(A_d), x_t_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_t_d), exp_y_t, fp_tol<T>);

  // y_h = A^H * x_t
  if constexpr (nda::blas::has_F_layout<decltype(A_d)> and not nda::mem::on_device<decltype(A_d)>) {
    auto exp_y_h = exp_y_t;
    if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{T{210 + 70i}, T{240 + 80i}, T{270 + 90i}};
    auto y_h_d = nda::linalg::matvecmul(nda::conj(nda::transpose(A_d)), x_t_d);
    EXPECT_ARRAY_NEAR(nda::to_host(y_h_d), exp_y_h, fp_tol<T>);
  }

  // strided matrix and vector views
  if constexpr (nda::mem::have_host_compatible_addr_space<decltype(A_d), decltype(x_d)>) {
    auto y_v_d = nda::linalg::matvecmul(A_d(nda::range(0, 4, 2), nda::range(0, 3, 2)), x_d(nda::range(0, 3, 2)));
    if constexpr (nda::is_complex_v<T>) {
      EXPECT_ARRAY_NEAR(nda::to_host(y_v_d), (nda::vector<T>{T{10 - 30i}, T{34 - 102i}}), fp_tol<T>);
    } else {
      EXPECT_ARRAY_NEAR(nda::to_host(y_v_d), (nda::vector<T>{10, 34}), fp_tol<T>);
    }
  }
}

template <typename T, typename Layout>
void test_matvecmul_address_spaces() {
  test_matvecmul<T, Layout, Device, Device>();
  test_matvecmul<T, Layout, Device, Unified>();
  test_matvecmul<T, Layout, Unified, Device>();
  test_matvecmul<T, Layout, Unified, Unified>();
  test_matvecmul<T, Layout, Unified, Host>();
  test_matvecmul<T, Layout, Host, Unified>();
}

template <typename T>
void test_matvecmul_layouts() {
  test_matvecmul_address_spaces<T, C_layout>();
  test_matvecmul_address_spaces<T, F_layout>();
}

TEST(NDA, CULinearAlgebraMatvecmulGenericGemvBranch) {
  test_matvecmul<long, C_layout, Unified, Unified>();
  test_matvecmul<long, C_layout, Unified, Host>();
  test_matvecmul<long, C_layout, Host, Unified>();

  test_matvecmul<long, F_layout, Unified, Unified>();
  test_matvecmul<long, F_layout, Unified, Host>();
  test_matvecmul<long, F_layout, Host, Unified>();
}

TEST(NDA, CULinearAlgebraMatvecmulBLASBranch) {
  test_matvecmul_layouts<float>();
  test_matvecmul_layouts<std::complex<float>>();
  test_matvecmul_layouts<double>();
  test_matvecmul_layouts<std::complex<double>>();
}

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_matvecmul_promotion() {
  auto A_i    = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_fp   = nda::matrix<T>{{1, 2}, {3, 4}};
  auto w_i    = nda::vector<int>{1, 1};
  auto w_fp   = nda::vector<T>{1, 1};
  auto A_i_d  = to_addr_space<AS1>(A_i);
  auto A_fp_d = to_addr_space<AS1>(A_fp);
  auto w_i_d  = to_addr_space<AS2>(w_i);
  auto w_fp_d = to_addr_space<AS2>(w_fp);

  auto v_fp1_d = nda::linalg::matvecmul(A_fp_d, w_i_d);
  static_assert(std::same_as<nda::get_value_t<decltype(v_fp1_d)>, T>);
  EXPECT_ARRAY_NEAR(nda::to_host(v_fp1_d), (nda::vector<T>{3, 7}), fp_tol<T>);

  auto v_fp2_d = nda::linalg::matvecmul(A_i_d, w_fp_d);
  static_assert(std::same_as<nda::get_value_t<decltype(v_fp2_d)>, T>);
  EXPECT_ARRAY_NEAR(nda::to_host(v_fp2_d), (nda::vector<T>{3, 7}), fp_tol<T>);

  auto v_i_d = nda::linalg::matvecmul(A_i_d, w_i_d);
  static_assert(std::same_as<nda::get_value_t<decltype(v_i_d)>, int>);
  EXPECT_ARRAY_EQ(nda::to_host(v_i_d), (nda::vector<int>{3, 7}));
}

template <typename T>
void test_matvecmul_promotion_address_spaces() {
  test_matvecmul_promotion<T, Unified, Unified>();
  test_matvecmul_promotion<T, Unified, Host>();
  test_matvecmul_promotion<T, Host, Unified>();
}

TEST(NDA, CULinearAlgebraMatvecmulPromotion) {
  test_matvecmul_promotion_address_spaces<float>();
  test_matvecmul_promotion_address_spaces<std::complex<float>>();
  test_matvecmul_promotion_address_spaces<double>();
  test_matvecmul_promotion_address_spaces<std::complex<double>>();
}

// Test the generic matmul function.
template <typename T, typename Layout1, typename Layout2, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_matmul() {
  auto A     = nda::matrix<T, Layout1>{{1, 2, 3}, {4, 5, 6}};
  auto B     = nda::matrix<T, Layout2>{{1, 2}, {3, 4}, {5, 6}};
  auto exp_C = nda::matrix<T>{{22, 28}, {49, 64}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1 - 1i};
    B *= T{2 - 1i};
    exp_C *= T{(1 - 1i) * (2 - 1i)};
  }
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);

  // C = A * B
  auto C_d = nda::linalg::matmul(A_d, B_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C_d), exp_C, fp_tol<T>);

  // C_t = B^T * A^T
  auto C_t_d = nda::linalg::matmul(nda::transpose(B_d), nda::transpose(A_d));
  EXPECT_ARRAY_NEAR(nda::to_host(C_t_d), nda::transpose(exp_C), fp_tol<T>);

  // C_h = B^H * A^H --> not working right now because of how we determine the layout of C
  // if constexpr (std::same_as<Layout1, Layout2>) {
  //   auto C_h_d = nda::linalg::matmul(nda::dagger(B_d), nda::dagger(A_d));
  //   EXPECT_ARRAY_NEAR(nda::to_host(C_h_d), nda::dagger(exp_C));
  // }

  // strided matrix views
  if constexpr (nda::mem::have_host_compatible_addr_space<decltype(A_d), decltype(B_d)>) {
    using nda::range;
    auto exp_C_v = nda::matrix<T>{{16, 20}, {34, 44}};
    if constexpr (nda::is_complex_v<T>) exp_C_v *= T{(1 - 1i) * (2 - 1i)};
    auto C_v_d                            = nda::matrix<T>(4, 4);
    C_v_d(range(0, 4, 2), range(0, 4, 2)) = nda::linalg::matmul(A_d(range::all, range(0, 3, 2)), B_d(range(0, 3, 2), range::all));
    EXPECT_ARRAY_NEAR(C_v_d(range(0, 4, 2), range(0, 4, 2)), exp_C_v, fp_tol<T>);
  }
}

template <typename T, typename Layout1, typename Layout2>
void test_matmul_address_spaces() {
  test_matmul<T, Layout1, Layout2, Device, Device>();
  test_matmul<T, Layout1, Layout2, Device, Unified>();
  test_matmul<T, Layout1, Layout2, Unified, Device>();
  test_matmul<T, Layout1, Layout2, Unified, Unified>();
  test_matmul<T, Layout1, Layout2, Unified, Host>();
  test_matmul<T, Layout1, Layout2, Host, Unified>();
}

template <typename T>
void test_matmul_layouts() {
  test_matmul_address_spaces<T, C_layout, C_layout>();
  test_matmul_address_spaces<T, C_layout, F_layout>();
  test_matmul_address_spaces<T, F_layout, C_layout>();
  test_matmul_address_spaces<T, F_layout, F_layout>();
}

TEST(NDA, CULinearAlgebraMatmulGenericGemmBranch) {
  test_matmul<long, C_layout, C_layout, Unified, Unified>();
  test_matmul<long, C_layout, F_layout, Host, Unified>();
  test_matmul<long, F_layout, F_layout, Unified, Host>();
  test_matmul<long, F_layout, C_layout, Host, Unified>();
}

TEST(NDA, CULinearAlgebraMatmulBLASBranch) {
  test_matmul_layouts<float>();
  test_matmul_layouts<std::complex<float>>();
  test_matmul_layouts<double>();
  test_matmul_layouts<std::complex<double>>();
}

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_matmul_promotion() {
  auto A_i    = nda::matrix<int>{{1, 2}, {3, 4}};
  auto A_fp   = nda::matrix<T>{{1, 2}, {3, 4}};
  auto A_i_d  = to_addr_space<AS1>(A_i);
  auto A_fp_d = to_addr_space<AS2>(A_fp);

  auto B_fp1_d = nda::linalg::matmul(A_fp_d, A_i_d);
  static_assert(std::same_as<nda::get_value_t<decltype(B_fp1_d)>, T>);
  EXPECT_ARRAY_NEAR(nda::to_host(B_fp1_d), (nda::matrix<T>{{7, 10}, {15, 22}}), fp_tol<T>);

  auto B_fp2_d = nda::linalg::matmul(A_i_d, A_fp_d);
  static_assert(std::same_as<nda::get_value_t<decltype(B_fp2_d)>, T>);
  EXPECT_ARRAY_NEAR(nda::to_host(B_fp2_d), (nda::matrix<T>{{7, 10}, {15, 22}}), fp_tol<T>);

  auto B_i_d = nda::linalg::matmul(A_i_d, A_i_d);
  static_assert(std::same_as<nda::get_value_t<decltype(B_i_d)>, int>);
  EXPECT_ARRAY_EQ(nda::to_host(B_i_d), (nda::matrix<int>{{7, 10}, {15, 22}}));
}

template <typename T>
void test_matmul_promotion_address_spaces() {
  test_matmul_promotion<T, Unified, Unified>();
  test_matmul_promotion<T, Unified, Host>();
  test_matmul_promotion<T, Host, Unified>();
}

TEST(NDA, CULinearAlgebraMatmulPromotion) {
  test_matmul_promotion_address_spaces<float>();
  test_matmul_promotion_address_spaces<std::complex<float>>();
  test_matmul_promotion_address_spaces<double>();
  test_matmul_promotion_address_spaces<std::complex<double>>();
}

// Test general inverse functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_inv() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;

  // tolerance based on condition number of worst-case matrix C: cond(C) ~ 332, max(|Cinv|) = 24
  // error ~ cond(C) * max_element * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  // lambda that checks inverse function
  auto check_inv = [tol](auto M, auto Minv) {
    if constexpr (nda::is_complex_v<T>) {
      M *= T{1.0i};
      Minv /= T{1.0i};
    }

    auto M_d    = to_addr_space<AS>(M);
    auto Minv_d = nda::linalg::inv(M_d);
    EXPECT_ARRAY_NEAR(Minv, nda::to_host(Minv_d), tol);
    auto M2_d = nda::linalg::inv(Minv_d);
    EXPECT_ARRAY_NEAR(M, nda::to_host(M2_d), tol);
  };

  // 1x1 matrix
  auto A    = matrix_t{{3}};
  auto Ainv = matrix_t{{1.0 / 3.0}};
  check_inv(A, Ainv);

  // 2x2 matrix
  auto B    = matrix_t{{1, 2}, {0, 1}};
  auto Binv = matrix_t{{1, -2}, {0, 1}};
  check_inv(B, Binv);

  // 3x3 matrix
  auto C    = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto Cinv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  check_inv(C, Cinv);

  // 4x4 matrix
  auto D    = matrix_t{{2, 2, 2, 2}, {2, 4, 6, 8}, {2, 6, 12, 20}, {2, 8, 20, 40}};
  auto Dinv = matrix_t{{2, -3, 2, -0.5}, {-3, 7, -5.5, 1.5}, {2, -5.5, 5, -1.5}, {-0.5, 1.5, -1.5, 0.5}};
  check_inv(D, Dinv);
}

template <typename T, typename Layout>
void test_inv_address_spaces() {
  test_inv<T, Layout, Device>();
  test_inv<T, Layout, Unified>();
}

template <typename T>
void test_inv_layouts() {
  test_inv_address_spaces<T, C_layout>();
  test_inv_address_spaces<T, F_layout>();
}

TEST(NDA, CULinearAlgebraInv) {
  test_inv_layouts<float>();
  test_inv_layouts<std::complex<float>>();
  test_inv_layouts<double>();
  test_inv_layouts<std::complex<double>>();
}

// Test the outer product function.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_outer_product() {
  using namespace nda::blas_lapack;

  // outer product of two arrays
  auto A = nda::array<T, 2, Layout>::rand(2, 3);
  auto B = nda::array<T, 3, Layout>::rand(4, 5, 6);
  auto C = nda::array<T, 5, Layout>(2, 3, 4, 5, 6);
  for (auto [i, j] : A.indices())
    for (auto [k, l, m] : B.indices()) C(i, j, k, l, m) = A(i, j) * B(k, l, m);
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);
  EXPECT_ARRAY_NEAR(C, nda::to_host(nda::linalg::outer_product(A_d, B_d)), fp_tol<T>);

  // outer product of two vectors
  nda::vector<T> v{1, 2};
  nda::vector<T> w{3, 4, 5};
  auto v_d = to_addr_space<AS1>(v);
  auto w_d = to_addr_space<AS2>(w);
  auto M_d = nda::linalg::outer_product(v_d, w_d);
  static_assert(nda::get_algebra<decltype(M_d)> == 'M');
  static_assert(nda::blas_lapack::has_C_layout<decltype(M_d)>);
  EXPECT_ARRAY_NEAR(nda::matrix<T>{{3, 4, 5}, {6, 8, 10}}, nda::to_host(M_d), fp_tol<T>);

  // outer product of a vector and an array
  auto D_d = nda::linalg::outer_product(v_d, A_d);
  static_assert(nda::get_algebra<decltype(D_d)> == 'A');
  static_assert(has_C_layout<decltype(D_d)> == has_C_layout<decltype(A_d)>);
  auto D_exp = nda::array<T, 3, Layout>(2, 2, 3);
  for (auto i : nda::range(2))
    for (auto [j, k] : A.indices()) D_exp(i, j, k) = v(i) * A(j, k);
  EXPECT_ARRAY_NEAR(nda::to_host(D_d), D_exp, fp_tol<T>);

  // outer product of an array and a vector
  auto E_d = nda::linalg::outer_product(A_d, v_d);
  static_assert(nda::get_algebra<decltype(E_d)> == 'A');
  static_assert(has_C_layout<decltype(E_d)> == has_C_layout<decltype(A_d)>);
  auto E_exp = nda::array<T, 3, Layout>(2, 3, 2);
  for (auto [i, j] : A.indices())
    for (auto k : nda::range(2)) E_exp(i, j, k) = A(i, j) * v(k);
  EXPECT_ARRAY_NEAR(nda::to_host(E_d), E_exp, fp_tol<T>);
}

template <typename T, typename Layout>
void test_outer_product_address_spaces() {
  test_outer_product<T, Layout, Device, Device>();
  test_outer_product<T, Layout, Device, Unified>();
  test_outer_product<T, Layout, Unified, Device>();
  test_outer_product<T, Layout, Unified, Unified>();
  test_outer_product<T, Layout, Unified, Host>();
  test_outer_product<T, Layout, Host, Unified>();
}

template <typename T>
void test_outer_product_layouts() {
  test_outer_product_address_spaces<T, C_layout>();
  test_outer_product_address_spaces<T, F_layout>();
}

TEST(NDA, CULinearAlgebraOuterProduct) {
  test_outer_product_layouts<float>();
  test_outer_product_layouts<std::complex<float>>();
  test_outer_product_layouts<double>();
  test_outer_product_layouts<std::complex<double>>();
}

// Test the generic solve and solve_in_place functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_solve() {
  using matrix_t = nda::matrix<T, Layout>;
  using vector_t = nda::vector<T>;
  using fp_t     = nda::get_fp_t<T>;

  /// tolerance based on condition number: cond(A) ~ 332, ||Ainv||_max = 24
  // error ~ cond(A) * ||Ainv||_max * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = nda::matrix<T, F_layout>{{1, 5}, {4, 5}, {3, 6}};
  auto b = vector_t{B(nda::range::all, 0)};

  // solve A * X = B using the exact matrix inverse
  auto Ainv = nda::matrix<T, Layout>{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X    = nda::matrix<T, Layout>{Ainv * B};
  EXPECT_ARRAY_NEAR(A * X, B, tol);

  // solve A * X = B using solve_in_place
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);
  nda::linalg::solve_in_place(A_d, B_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(B_d), B, tol);
  EXPECT_ARRAY_NEAR(X, nda::to_host(B_d), tol);

  // solve A * x = b using solve_in_place
  A_d      = A;
  auto b_d = to_addr_space<AS2>(b);
  nda::linalg::solve_in_place(A_d, b_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(b_d), b, tol);
  EXPECT_ARRAY_NEAR(X(nda::range::all, 0), nda::to_host(b_d), tol);

  // solve A * X = B using solve
  A_d      = to_addr_space<AS1>(A);
  B_d      = to_addr_space<AS2>(B);
  auto X_d = nda::linalg::solve(A_d, B_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(X_d), B, tol);
  EXPECT_ARRAY_NEAR(X, nda::to_host(X_d), tol);

  // solve A * x = b using solve
  b_d      = to_addr_space<AS2>(b);
  auto x_d = nda::linalg::solve(A_d, b_d);
  EXPECT_ARRAY_NEAR(A * nda::to_host(x_d), b, tol);
  EXPECT_ARRAY_NEAR(X(nda::range::all, 0), nda::to_host(x_d), tol);
}

template <typename T, typename Layout>
void test_solve_address_spaces() {
  test_solve<T, Layout, Device, Device>();
  test_solve<T, Layout, Device, Unified>();
  test_solve<T, Layout, Unified, Device>();
  test_solve<T, Layout, Unified, Unified>();
  test_solve<T, Layout, Unified, Host>();
  test_solve<T, Layout, Host, Unified>();
}

template <typename T>
void test_solve_layouts() {
  test_solve_address_spaces<T, C_layout>();
  test_solve_address_spaces<T, F_layout>();
}

TEST(NDA, CULinearAlgebraSolve) {
  test_solve_layouts<float>();
  test_solve_layouts<std::complex<float>>();
  test_solve_layouts<double>();
  test_solve_layouts<std::complex<double>>();
}

// Test the svd and svd_in_place functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_svd() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;

  auto A = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  if constexpr (std::same_as<Layout, C_layout>) {
    // cuSOLVER cannot handle when m < n
    A = matrix_t(nda::transpose(A));
  }

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
  auto [U1_d, s1_d, VH1_d] = nda::linalg::svd(to_addr_space<AS>(A));
  check_svd(A, nda::to_host(U1_d), nda::to_host(s1_d), nda::to_host(VH1_d));

  // compute the SVD of A in place
  auto A_d                 = to_addr_space<AS>(A);
  auto [U2_d, s2_d, VH2_d] = nda::linalg::svd_in_place(A_d);
  check_svd(A, nda::to_host(U2_d), nda::to_host(s2_d), nda::to_host(VH2_d));
}

template <typename T, typename Layout>
void test_svd_address_spaces() {
  test_svd<T, Layout, Device>();
  test_svd<T, Layout, Unified>();
}

template <typename T>
void test_svd_layouts() {
  test_svd_address_spaces<T, C_layout>();
  test_svd_address_spaces<T, F_layout>();
}

TEST(NDA, CULinearAlgebraSVD) {
  test_svd_layouts<float>();
  test_svd_layouts<std::complex<float>>();
  test_svd_layouts<double>();
  test_svd_layouts<std::complex<double>>();
}

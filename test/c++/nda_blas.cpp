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
using nda::C_layout, nda::F_layout;

// Test BLAS/LAPACK helper traits.
TEST(NDA, BLASandLAPACKToolsTraits) {
  using namespace nda::blas_lapack;
  using mat_c_type = nda::matrix<std::complex<double>, C_layout>;
  using mat_f_type = nda::matrix<std::complex<double>, F_layout>;
  using arr_c_type = nda::array<double, 3, C_layout>;
  using arr_f_type = nda::array<double, 3, F_layout>;
  using vec_type   = nda::vector<float>;

  auto m_c = mat_c_type::rand({2, 3});
  auto m_f = mat_f_type::rand({2, 3});
  auto v   = vec_type{1.0, 2.0};
  auto a_c = arr_c_type::rand({2, 3, 4});
  auto a_f = arr_f_type::rand({2, 3, 4});

  static_assert(not is_conj_array_expr<mat_c_type>);
  static_assert(not is_conj_array_expr<arr_f_type>);
  static_assert(not is_conj_array_expr<vec_type>);
  static_assert(is_conj_array_expr<decltype(nda::conj(m_c))>);
  static_assert(is_conj_array_expr<decltype(nda::conj(m_f)) const &>);
  static_assert(is_conj_array_expr<decltype(nda::conj(m_c)) &&>);
  static_assert(not is_conj_array_expr<decltype(nda::conj(a_c))>);
  static_assert(not is_conj_array_expr<decltype(nda::conj(v))>);
  static_assert(is_conj_array_expr<decltype(nda::dagger(m_c))>);

  static_assert(has_C_layout<mat_c_type, arr_c_type, vec_type>);
  static_assert(has_F_layout<mat_f_type, arr_f_type, vec_type>);
  static_assert(not has_C_layout<mat_c_type, arr_f_type, vec_type>);
  static_assert(not has_F_layout<mat_c_type, arr_f_type, vec_type>);
  static_assert(has_C_layout<decltype(nda::dagger(m_f))>);
  static_assert(has_F_layout<decltype(nda::transpose(a_c))>);

  static_assert(get_op<mat_f_type> == 'N');
  static_assert(get_op<arr_f_type> == 'N');
  static_assert(get_op<mat_c_type> == 'T');
  static_assert(get_op<arr_c_type> == 'T');
  static_assert(get_op<decltype(nda::conj(m_c))> == 'C');
  static_assert(get_op<decltype(nda::dagger(m_f))> == 'C');
}

// Test BLAS/LAPACK helper functions.
TEST(NDA, BLASandLAPACKToolsFunctions) {
  using namespace nda::blas_lapack;

  auto v      = nda::vector<double>(10);
  auto m_c    = nda::matrix<double>::rand(3, 4);
  auto m_f    = nda::matrix<double, F_layout>::rand(3, 4);
  auto m_cplx = nda::matrix<std::complex<double>>{{1.0 + 1i, 2.0}, {3.0, 4.0 - 1i}};

  // get_array
  auto &m_ref = get_array(m_c);
  EXPECT_EQ(&m_ref(0, 0), &m_c(0, 0));

  auto conj_expr   = nda::conj(m_cplx);
  auto &m_cplx_ref = get_array(conj_expr);
  EXPECT_EQ(&m_cplx_ref(0, 0), &m_cplx(0, 0));

  // get_ld
  EXPECT_EQ(get_ld(v), 10);
  EXPECT_EQ(get_ld(nda::vector<double>()), 0);
  EXPECT_EQ(get_ld(m_f), 3);
  EXPECT_EQ(get_ld(m_c), 4);
  EXPECT_EQ(get_ld(m_f(nda::range(2), nda::range(3))), 3);
  EXPECT_EQ(get_ld(m_c(nda::range(2), nda::range(2))), 4);

  // get_ncols
  EXPECT_EQ(get_ncols(v), 1);
  EXPECT_EQ(get_ncols(m_f), 4);
  EXPECT_EQ(get_ncols(m_c), 3);
  EXPECT_EQ(get_ncols(m_f(nda::range(2), nda::range(3))), 3);
  EXPECT_EQ(get_ncols(m_c(nda::range(2), nda::range(2))), 2);
}

// Test BLAS/LAPACK helper concepts.
TEST(NDA, BLASandLAPACKToolsConcepts) {
  using namespace nda::blas_lapack;
  using conj_expr_t = decltype(nda::conj(nda::matrix<std::complex<double>>{}));

  static_assert(BlasArray<nda::matrix<double>>);
  static_assert(BlasArray<nda::array<std::complex<float>, 3, F_layout>, 3>);
  static_assert(not BlasArray<nda::matrix<int>>);
  static_assert(not BlasArray<nda::array<std::complex<float>, 3, F_layout>, 2>);
  static_assert(not BlasArray<conj_expr_t>);

  static_assert(BlasArrayReal<nda::matrix_view<double>>);
  static_assert(BlasArrayReal<nda::vector<float>, 1>);
  static_assert(not BlasArrayReal<nda::matrix_view<std::complex<double>>>);
  static_assert(not BlasArrayReal<nda::vector<float>, 2>);

  static_assert(BlasArrayCplx<nda::matrix_view<std::complex<double>>>);
  static_assert(BlasArrayCplx<nda::array<std::complex<float>, 3>, 3>);
  static_assert(not BlasArrayCplx<nda::matrix_view<double>>);
  static_assert(not BlasArrayCplx<nda::array<std::complex<float>, 3>, 1>);

  static_assert(BlasArrayOrConj<nda::matrix<double>>);
  static_assert(BlasArrayOrConj<conj_expr_t, 2>);
  static_assert(not BlasArrayOrConj<nda::matrix<int>>);
  static_assert(not BlasArrayOrConj<conj_expr_t, 3>);

  static_assert(BlasArrayFor<nda::matrix<double>, nda::matrix<double>>);
  static_assert(BlasArrayFor<nda::vector<std::complex<float>>, nda::matrix_view<std::complex<float>>, 1>);
  static_assert(not BlasArrayFor<nda::matrix<float>, nda::matrix<double>>);
  static_assert(not BlasArrayFor<nda::vector<std::complex<float>>, nda::matrix_view<std::complex<float>>, 2>);

  static_assert(BlasArrayOrConjFor<conj_expr_t, nda::vector<std::complex<double>>>);
  static_assert(BlasArrayOrConjFor<nda::vector_view<double>, nda::matrix<double>, 1>);
  static_assert(not BlasArrayOrConjFor<conj_expr_t, nda::vector<std::complex<float>>>);
  static_assert(not BlasArrayOrConjFor<nda::vector_view<double>, nda::matrix<double>, 3>);

  static_assert(PivotArrayFor<nda::vector<int>, nda::matrix_view<double>>);
  static_assert(PivotArrayFor<nda::array<int, 3>, nda::matrix<float>, 3>);
  static_assert(not PivotArrayFor<nda::vector<long>, nda::matrix_view<double>>);
  static_assert(not PivotArrayFor<nda::array<int, 3>, nda::matrix<float>, 2>);

  static_assert(BlasArrayRealFor<nda::vector_view<double>, nda::matrix<double>>);
  static_assert(BlasArrayRealFor<nda::vector<float>, nda::matrix<std::complex<float>>, 1>);
  static_assert(not BlasArrayRealFor<nda::vector_view<double>, nda::matrix<float>>);
  static_assert(not BlasArrayRealFor<nda::vector<float>, nda::matrix<std::complex<float>>, 3>);
}

// Test the BLAS gemm function.
template <typename T, typename Layout1, typename Layout2, typename Layout3>
void test_gemm() {
  constexpr auto a_is_f_layout = std::same_as<Layout1, F_layout>;
  constexpr auto b_is_f_layout = std::same_as<Layout2, F_layout>;
  constexpr auto c_is_f_layout = std::same_as<Layout3, F_layout>;
  auto A                       = nda::matrix<T, Layout1>{{1, 2, 3}, {4, 5, 6}};
  auto B                       = nda::matrix<T, Layout2>{{1, 2}, {3, 4}, {5, 6}};
  auto exp_C                   = nda::matrix<T, Layout3>{{22, 28}, {49, 64}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1 - 1i};
    B *= T{2 - 1i};
    exp_C *= T{(1 - 1i) * (2 - 1i)};
  }

  // C = A * B
  auto C = nda::matrix<T, Layout3>(2, 2);
  nda::blas::gemm(1.0, A, B, 0.0, C);
  EXPECT_ARRAY_NEAR(C, exp_C, fp_tol<T>);

  // C = 3 * A * B + 2 * C
  nda::blas::gemm(3, A, B, 2, C);
  EXPECT_ARRAY_NEAR(C, 5 * exp_C, fp_tol<T>);

  // C_t = B^T * A^T
  auto C_t = nda::matrix<T, Layout3>(2, 2);
  nda::blas::gemm(1.0, nda::transpose(B), nda::transpose(A), 0.0, C_t);
  EXPECT_ARRAY_NEAR(C_t, nda::transpose(exp_C), fp_tol<T>);

  // C_h = B^H * A^H
  if constexpr ((a_is_f_layout and b_is_f_layout and c_is_f_layout) or (!a_is_f_layout and !b_is_f_layout and !c_is_f_layout)) {
    auto C_h = nda::matrix<T, Layout3>(2, 2);
    nda::blas::gemm(1.0, nda::dagger(B), nda::dagger(A), 0.0, C_h);
    EXPECT_ARRAY_NEAR(C_h, nda::dagger(exp_C), fp_tol<T>);
  }

  // contiguous matrix views
  if constexpr (a_is_f_layout and !b_is_f_layout and !c_is_f_layout) {
    using nda::range;
    auto exp_C_v = nda::matrix<T, Layout3>{{13, 16}, {37, 46}};
    if constexpr (nda::is_complex_v<T>) exp_C_v *= T{(1 - 1i) * (2 - 1i)};
    auto C_v = nda::matrix<T, Layout3>(5, 2);
    nda::blas::gemm(1.0, A(range::all, range(0, 2)), B(range(1, 3), range::all), 0.0, C_v(range(2, 4), range::all));
    EXPECT_ARRAY_NEAR(C_v(range(2, 4), range::all), exp_C_v, fp_tol<T>);
  }
}

template <typename T>
void test_gemm_layouts() {
  test_gemm<T, C_layout, C_layout, C_layout>();
  test_gemm<T, C_layout, C_layout, F_layout>();
  test_gemm<T, C_layout, F_layout, C_layout>();
  test_gemm<T, C_layout, F_layout, F_layout>();
  test_gemm<T, F_layout, C_layout, C_layout>();
  test_gemm<T, F_layout, C_layout, F_layout>();
  test_gemm<T, F_layout, F_layout, C_layout>();
  test_gemm<T, F_layout, F_layout, F_layout>();
};

TEST(NDA, BLASGemm) {
  test_gemm_layouts<float>();
  test_gemm_layouts<std::complex<float>>();
  test_gemm_layouts<double>();
  test_gemm_layouts<std::complex<double>>();
}

// Test the BLAS gemm_batch, gemm_vbatch and gemm_batch_strided functions.
template <typename T, typename Layout1, typename Layout2, typename Layout3, bool is_vbatch>
void test_gemm_batch() {
  int const batch_count = 4;
  long m                = is_vbatch ? 2 : 16;
  long k                = is_vbatch ? 3 : 12;
  long n                = is_vbatch ? 4 : 8;

  // create vector of matrices
  std::vector<nda::matrix<T, Layout1>> vec_A;
  std::vector<nda::matrix<T, Layout2>> vec_B;
  std::vector<nda::matrix<T, Layout3>> vec_C, exp_C;
  for ([[maybe_unused]] auto i : nda::range(batch_count)) {
    vec_A.push_back(nda::matrix<T, Layout1>::rand({m, k}));
    vec_B.push_back(nda::matrix<T, Layout2>::rand({k, n}));
    vec_C.push_back(nda::matrix<T, Layout3>::zeros({m, n}));
    auto tmp = nda::matrix<T, Layout3>::zeros({m, n});
    nda::blas::gemm(1.0, vec_A.back(), vec_B.back(), 0.0, tmp);
    exp_C.push_back(std::move(tmp));
    if (is_vbatch) {
      m *= 2;
      k *= 2;
      n *= 2;
    }
  }

  // test batched gemm routines
  if constexpr (is_vbatch) {
    nda::blas::gemm_vbatch(1.0, vec_A, vec_B, 0.0, vec_C);
  } else {
    nda::blas::gemm_batch(1.0, vec_A, vec_B, 0.0, vec_C);
  }
  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(vec_C[i], exp_C[i], fp_tol<T>);
}

template <typename T, bool is_vbatch>
void test_gemm_batch_layouts() {
  test_gemm_batch<T, C_layout, C_layout, C_layout, is_vbatch>();
  test_gemm_batch<T, C_layout, C_layout, F_layout, is_vbatch>();
  test_gemm_batch<T, C_layout, F_layout, C_layout, is_vbatch>();
  test_gemm_batch<T, C_layout, F_layout, F_layout, is_vbatch>();
  test_gemm_batch<T, F_layout, C_layout, C_layout, is_vbatch>();
  test_gemm_batch<T, F_layout, C_layout, F_layout, is_vbatch>();
  test_gemm_batch<T, F_layout, F_layout, C_layout, is_vbatch>();
  test_gemm_batch<T, F_layout, F_layout, F_layout, is_vbatch>();
};

TEST(NDA, BLASGemmBatch) {
  test_gemm_batch_layouts<float, false>();
  test_gemm_batch_layouts<std::complex<float>, false>();
  test_gemm_batch_layouts<double, false>();
  test_gemm_batch_layouts<std::complex<double>, false>();
}

TEST(NDA, BLASGemmVbatch) {
  test_gemm_batch_layouts<float, true>();
  test_gemm_batch_layouts<std::complex<float>, true>();
  test_gemm_batch_layouts<double, true>();
  test_gemm_batch_layouts<std::complex<double>, true>();
}

template <typename T, typename Layout>
auto make_batch(int bc, long d1, long d2, bool zeros) {
  using arr_t = nda::array<T, 3, Layout>;
  if constexpr (std::same_as<Layout, C_layout>) {
    return (zeros ? arr_t::zeros({bc, d1, d2}) : arr_t::rand({bc, d1, d2}));
  } else {
    return (zeros ? arr_t::zeros({d1, d2, bc}) : arr_t::rand({d1, d2, bc}));
  }
}

template <typename T, typename Layout1, typename Layout2, typename Layout3>
void test_gemm_batch_strided() {
  int const batch_count = 10;
  long const m          = 16;
  long const k          = 12;
  long const n          = 8;
  T const alpha         = 2;
  T const beta          = 0.5;

  // get a view to the i-th matrix in the batch
  auto get_mat = [](auto &arr, auto i) {
    if constexpr (nda::blas_lapack::has_C_layout<decltype(arr)>) {
      return arr(i, nda::ellipsis{});
    } else {
      return arr(nda::ellipsis{}, i);
    }
  };

  auto arr_A = make_batch<T, Layout1>(batch_count, m, k, false);
  auto arr_B = make_batch<T, Layout2>(batch_count, k, n, false);
  auto arr_C = make_batch<T, Layout3>(batch_count, m, n, true);

  // C_i = A_i * B_i
  nda::blas::gemm_batch_strided(1.0, arr_A, arr_B, 0.0, arr_C);
  for (auto i : nda::range(batch_count)) {
    auto exp = nda::matrix<T, F_layout>::zeros({m, n});
    nda::blas::gemm(1.0, get_mat(arr_A, i), get_mat(arr_B, i), 0.0, exp);
    EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
  }

  // C_i = alpha * A_i * B_i + beta * C_i
  nda::blas::gemm_batch_strided(alpha, arr_A, arr_B, beta, arr_C);
  for (auto i : nda::range(batch_count)) {
    auto exp = nda::matrix<T, F_layout>::zeros({m, n});
    nda::blas::gemm(1.0, get_mat(arr_A, i), get_mat(arr_B, i), 0.0, exp);
    nda::blas::gemm(alpha, get_mat(arr_A, i), get_mat(arr_B, i), beta, exp);
    EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
  }

  // C_i = A_i^T * B_i
  arr_A = make_batch<T, Layout1>(batch_count, k, m, false);
  nda::blas::gemm_batch_strided(1.0, nda::transpose(arr_A), arr_B, 0.0, arr_C);
  for (auto i : nda::range(batch_count)) {
    auto exp = nda::matrix<T, F_layout>::zeros({m, n});
    nda::blas::gemm(1.0, nda::transpose(get_mat(arr_A, i)), get_mat(arr_B, i), 0.0, exp);
    EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
  }

  // C_i = A_i * B_i^H
  if constexpr (std::same_as<Layout2, F_layout> and std::same_as<Layout3, F_layout>) {
    arr_A = make_batch<T, Layout1>(batch_count, m, k, false);
    arr_B = make_batch<T, Layout2>(batch_count, n, k, false);
    nda::blas::gemm_batch_strided(1.0, arr_A, nda::conj(nda::transpose(arr_B)), 0.0, arr_C);
    for (auto i : nda::range(batch_count)) {
      auto exp = nda::matrix<T, F_layout>::zeros({m, n});
      nda::blas::gemm(1.0, get_mat(arr_A, i), nda::dagger(get_mat(arr_B, i)), 0.0, exp);
      EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
    }
  }
}

template <typename T>
void test_gemm_batch_strided_layouts() {
  test_gemm_batch_strided<T, C_layout, C_layout, C_layout>();
  test_gemm_batch_strided<T, C_layout, C_layout, F_layout>();
  test_gemm_batch_strided<T, C_layout, F_layout, C_layout>();
  test_gemm_batch_strided<T, C_layout, F_layout, F_layout>();
  test_gemm_batch_strided<T, F_layout, C_layout, C_layout>();
  test_gemm_batch_strided<T, F_layout, C_layout, F_layout>();
  test_gemm_batch_strided<T, F_layout, F_layout, C_layout>();
  test_gemm_batch_strided<T, F_layout, F_layout, F_layout>();
}

TEST(NDA, BLASGemmBatchStrided) {
  test_gemm_batch_strided_layouts<float>();
  test_gemm_batch_strided_layouts<std::complex<float>>();
  test_gemm_batch_strided_layouts<double>();
  test_gemm_batch_strided_layouts<std::complex<double>>();
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
    A *= T{1 - 1i};
    x *= T{2 - 1i};
    x_t *= T{2 - 1i};
    exp_y *= T{(1 - 1i) * (2 - 1i)};
    exp_y_t *= T{(1 - 1i) * (2 - 1i)};
  }

  // y = A * x
  auto y = nda::vector<T>(4);
  nda::blas::gemv(1.0, A, x, 0.0, y);
  EXPECT_ARRAY_NEAR(y, exp_y, fp_tol<T>);

  // y = 3 * A * x + 2y
  nda::blas::gemv(3, A, x, 2, y);
  EXPECT_ARRAY_NEAR(y, 5 * exp_y, fp_tol<T>);

  // y_t = A^T * x_t
  auto y_t = nda::vector<T>(3);
  nda::blas::gemv(1.0, nda::transpose(A), x_t, 0.0, y_t);
  EXPECT_ARRAY_NEAR(y_t, exp_y_t, fp_tol<T>);

  if constexpr (std::same_as<Layout, F_layout>) {
    // y_h = A^H * x_t
    auto exp_y_h = exp_y_t;
    if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{T{210 + 70i}, T{240 + 80i}, T{270 + 90i}};
    auto y_h = nda::vector<T>(3);
    nda::blas::gemv(1.0, nda::dagger(A), x_t, 0.0, y_h);
    EXPECT_ARRAY_NEAR(y_h, exp_y_h, fp_tol<T>);
  } else {
    // contiguous matrix view * strided vector view
    auto x_v                 = nda::vector<T>(6);
    x_v(nda::range(0, 6, 2)) = x;
    auto y_v                 = nda::vector<T>(4);
    nda::blas::gemv(1, A(nda::range(2), nda::range::all), x_v(nda::range(0, 6, 2)), 0, y(nda::range(0, 4, 2)));
    EXPECT_ARRAY_NEAR(y(nda::range(0, 4, 2)), exp_y(nda::range(2)), fp_tol<T>);
  }
}

template <typename T>
void test_gemv_layouts() {
  test_gemv<T, C_layout>();
  test_gemv<T, F_layout>();
}

TEST(NDA, BLASGemv) {
  test_gemv_layouts<float>();
  test_gemv_layouts<std::complex<float>>();
  test_gemv_layouts<double>();
  test_gemv_layouts<std::complex<double>>();
}

// Test the BLAS ger/gerc function.
template <typename T, typename Layout, bool star>
void test_ger() {
  // helper wrapper to call ger or gerc
  auto call_ger = [](auto alpha, auto const &x, auto const &y, auto &&m) {
    if constexpr (star) {
      nda::blas::gerc(alpha, x, y, m);
    } else {
      nda::blas::ger(alpha, x, y, m);
    }
  };

  // helper to compute outer product with optional conjugation
  auto outer_product = [](auto const &x, auto const &y, bool conj_y) {
    auto m = nda::matrix<T, Layout>(x.size(), y.size());
    for (int i = 0; i < x.size(); ++i)
      for (int j = 0; j < y.size(); ++j) m(i, j) = x(i) * (conj_y ? nda::conj(y(j)) : y(j));
    return m;
  };

  // initialize vectors: complex or real depending on T
  nda::vector<T> v(2);
  if constexpr (nda::is_complex_v<T>) {
    v = {T{1.0i}, T{2.0i}};
  } else {
    v = {1, 2};
  }

  // test 1: v ⊗ v starting from zero matrix
  auto exp_M1 = outer_product(v, v, star);
  auto M1     = nda::matrix<T, Layout>::zeros(2, 2);
  call_ger(1.0, v, v, M1);
  EXPECT_ARRAY_NEAR(M1, exp_M1, fp_tol<T>);

  // test 2: v ⊗ v starting from non-zero matrix (test accumulation)
  auto M1_init = nda::matrix<T, Layout>{{10, 20}, {30, 40}};
  auto M1b     = M1_init;
  call_ger(1.0, v, v, M1b);
  EXPECT_ARRAY_NEAR(M1b, M1_init + exp_M1, fp_tol<T>);

  // test 3: v ⊗ w (mixed: v complex/real, w real)
  nda::vector<T> w{3, 4, 5};
  auto exp_M2 = outer_product(v, w, star);
  auto M2     = nda::matrix<T, Layout>::zeros(2, 3);
  call_ger(1.0, v, w, M2);
  EXPECT_ARRAY_NEAR(M2, exp_M2, fp_tol<T>);
  call_ger(1.0, v, w, M2);
  EXPECT_ARRAY_NEAR(M2, exp_M2 * 2.0, fp_tol<T>);

  // test 4: w ⊗ v (swapped)
  auto exp_M3 = outer_product(w, v, star);
  auto M3     = nda::matrix<T, Layout>::zeros(3, 2);
  call_ger(1.0, w, v, M3);
  EXPECT_ARRAY_NEAR(M3, exp_M3, fp_tol<T>);
  call_ger(1.0, w, v, M3);
  EXPECT_ARRAY_NEAR(M3, exp_M3 * 2.0, fp_tol<T>);

  // test 5: strided views
  nda::vector<T> v_full(5), w_full(7);
  if constexpr (nda::is_complex_v<T>) {
    v_full = {0, T{1.0i}, 0, T{2.0i}, 0};
    w_full = {T{3.0i}, 0, 0, T{4.0i}, 0, 0, T{5.0i}};
  } else {
    v_full = {0, 1, 0, 2, 0};
    w_full = {3, 0, 0, 4, 0, 0, 5};
  }
  auto v_strided = v_full(nda::range(1, 5, 2));
  auto w_strided = w_full(nda::range(0, 7, 3));
  auto exp_M4    = outer_product(v_strided, w_strided, star) * 2.0;
  auto M4        = nda::matrix<T, Layout>::zeros(2, 3);
  call_ger(2.0, v_strided, w_strided, M4);
  EXPECT_ARRAY_NEAR(M4, exp_M4, fp_tol<T>);
}

template <typename T, bool star>
void test_ger_layouts() {
  test_ger<T, C_layout, star>();
  test_ger<T, F_layout, star>();
}

TEST(NDA, BLASGer) {
  test_ger_layouts<float, false>();
  test_ger_layouts<std::complex<float>, false>();
  test_ger_layouts<double, false>();
  test_ger_layouts<std::complex<double>, false>();
}

TEST(NDA, BLASGerc) {
  test_ger<float, F_layout, true>();
  test_ger<std::complex<float>, F_layout, true>();
  test_ger<double, F_layout, true>();
  test_ger<std::complex<double>, F_layout, true>();
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
    a *= T{1 + 1i};
    b *= T{1 + 2i};
  }

  // vector dot vector
  EXPECT_COMPLEX_NEAR(dot(a, b), exp_dot(a, b), fp_tol<T>);

  // size 0 vectors
  EXPECT_EQ(dot(nda::vector<T>{}, nda::vector<T>{}), T(0));

  // strided vector dot strided vector
  auto a_v = a(nda::range(0, 5, 2));
  auto b_v = b(nda::range(0, 5, 2));
  EXPECT_COMPLEX_NEAR(dot(a_v, b_v), exp_dot(a_v, b_v), fp_tol<T>);
}

TEST(NDA, BLASDot) {
  test_dot<float, false>();
  test_dot<std::complex<float>, false>();
  test_dot<double, false>();
  test_dot<std::complex<double>, false>();
}

TEST(NDA, BLASDotc) {
  test_dot<float, true>();
  test_dot<std::complex<float>, true>();
  test_dot<double, true>();
  test_dot<std::complex<double>, true>();
}

// Test the BLAS scal function.
template <typename T>
void test_scal() {
  using fp_t = nda::get_fp_t<T>;

  // scale an empty vector
  nda::vector<T> v_empty;
  nda::blas::scal(3.0, v_empty);
  EXPECT_TRUE(v_empty.empty());

  // prepare an input vector
  nda::vector<T> v{1, 2, 3, 4, 5};
  if constexpr (nda::is_complex_v<T>) { v *= T{1 - 1i}; }

  // scale by a scalar float
  auto v1  = v;
  fp_t xfp = 3.0;
  nda::blas::scal(xfp, v1);
  EXPECT_ARRAY_NEAR(v1, xfp * v, fp_tol<T>);

  // scale by an integer
  auto v2 = v;
  int xi  = 3;
  nda::blas::scal(xi, v2);
  EXPECT_ARRAY_NEAR(v2, xi * v, fp_tol<T>);

  // scale by a complex scalar if T is complex
  if constexpr (nda::is_complex_v<T>) {
    auto v3  = v;
    auto xcp = T{3.0 + 2.0i};
    nda::blas::scal(xcp, v3);
    EXPECT_ARRAY_NEAR(v3, xcp * v, fp_tol<T>);
  }
}

TEST(NDA, BLASScal) {
  test_scal<float>();
  test_scal<std::complex<float>>();
  test_scal<double>();
  test_scal<std::complex<double>>();
}

// Test the rank-3 overloads for gemm.
TEST(NDA, BLASGemmRank3Overload) {
  using value_t         = double;
  int const batch_count = 5;
  long const m          = 6;
  long const k          = 4;
  long const n          = 3;

  auto arr_A     = make_batch<value_t, F_layout>(batch_count, m, k, false);
  auto arr_B     = make_batch<value_t, F_layout>(batch_count, k, n, false);
  auto arr_C_old = make_batch<value_t, F_layout>(batch_count, m, n, true);
  auto arr_C_new = nda::array<value_t, 3, F_layout>{arr_C_old};

  nda::blas::gemm_batch_strided(2.0, arr_A, arr_B, 0.0, arr_C_old);
  nda::blas::gemm(2.0, arr_A, arr_B, 0.0, arr_C_new);

  for (auto i : nda::range(batch_count)) { EXPECT_ARRAY_NEAR(arr_C_new(nda::ellipsis{}, i), arr_C_old(nda::ellipsis{}, i), fp_tol<value_t>); }
}

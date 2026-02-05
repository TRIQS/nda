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

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;
using nda::mem::Host, nda::mem::Device, nda::mem::Unified;

template <typename A, typename B>
void test_blas_array_concepts() {
  using namespace nda::blas_lapack;
  if constexpr (nda::mem::have_compatible_addr_space<A, B>) {
    static_assert(BlasArrayFor<A, B>);
    static_assert(BlasArrayOrConjFor<A, B>);
    static_assert(BlasArrayRealFor<A, B>);
  } else {
    static_assert(not BlasArrayFor<A, B>);
    static_assert(not BlasArrayOrConjFor<A, B>);
    static_assert(not BlasArrayRealFor<A, B>);
  }
}

template <typename A, typename B>
void test_pivot_array_concept() {
  using namespace nda::blas_lapack;
  if constexpr (nda::mem::have_compatible_addr_space<A, B>) {
    static_assert(PivotArrayFor<A, B>);
  } else {
    static_assert(not PivotArrayFor<A, B>);
  }
}

// Test BLAS/LAPACK helper concepts with different address spaces.
TEST(NDA, CUBLASandCULAPACKToolsConcepts) {
  using namespace nda::blas_lapack;
  using mat_host_t      = nda::matrix<double, F_layout, nda::heap<Host>>;
  using mat_device_t    = nda::matrix<double, F_layout, nda::heap<Device>>;
  using mat_unified_t   = nda::matrix<double, F_layout, nda::heap<Unified>>;
  using pivot_host_t    = nda::vector<int, nda::heap<Host>>;
  using pivot_device_t  = nda::vector<int, nda::heap<Device>>;
  using pivot_unified_t = nda::vector<int, nda::heap<Unified>>;

  test_blas_array_concepts<mat_device_t, mat_device_t>();
  test_blas_array_concepts<mat_unified_t, mat_device_t>();
  test_blas_array_concepts<mat_host_t, mat_device_t>();
  test_blas_array_concepts<mat_device_t, mat_unified_t>();
  test_blas_array_concepts<mat_unified_t, mat_unified_t>();
  test_blas_array_concepts<mat_host_t, mat_unified_t>();
  test_blas_array_concepts<mat_device_t, mat_host_t>();
  test_blas_array_concepts<mat_unified_t, mat_host_t>();
  test_blas_array_concepts<mat_host_t, mat_host_t>();

  test_pivot_array_concept<pivot_device_t, mat_device_t>();
  test_pivot_array_concept<pivot_unified_t, mat_device_t>();
  test_pivot_array_concept<pivot_host_t, mat_device_t>();
  test_pivot_array_concept<pivot_device_t, mat_unified_t>();
  test_pivot_array_concept<pivot_unified_t, mat_unified_t>();
  test_pivot_array_concept<pivot_host_t, mat_unified_t>();
  test_pivot_array_concept<pivot_device_t, mat_host_t>();
  test_pivot_array_concept<pivot_unified_t, mat_host_t>();
  test_pivot_array_concept<pivot_host_t, mat_host_t>();
}

// Test the CUBLAS gemm function.
template <typename T, typename Layout1, typename Layout2, typename Layout3, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2,
          nda::mem::AddressSpace AS3>
void test_gemm() {
  constexpr auto a_is_f_layout = std::same_as<Layout1, F_layout>;
  constexpr auto b_is_f_layout = std::same_as<Layout2, F_layout>;
  constexpr auto c_is_f_layout = std::same_as<Layout3, F_layout>;
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
  test_gemm<double, C_layout, C_layout, C_layout, Device, Device, Device>();
  test_gemm<double, C_layout, C_layout, C_layout, Device, Unified, Device>();
  test_gemm<double, C_layout, C_layout, C_layout, Unified, Unified, Unified>();
  test_gemm<double, C_layout, C_layout, C_layout, Host, Unified, Unified>();

  // double, F-layout
  test_gemm<double, F_layout, F_layout, F_layout, Device, Device, Device>();
  test_gemm<double, F_layout, F_layout, F_layout, Device, Device, Unified>();
  test_gemm<double, F_layout, F_layout, F_layout, Unified, Unified, Unified>();
  test_gemm<double, F_layout, F_layout, F_layout, Unified, Host, Unified>();

  // double, mixed layout
  test_gemm<double, C_layout, F_layout, C_layout, Device, Device, Device>();
  test_gemm<double, F_layout, C_layout, F_layout, Unified, Unified, Unified>();
  test_gemm<double, C_layout, F_layout, F_layout, Host, Unified, Unified>();
  test_gemm<double, F_layout, C_layout, C_layout, Unified, Host, Unified>();

  // complex, C-layout
  test_gemm<std::complex<double>, C_layout, C_layout, C_layout, Device, Device, Device>();
  test_gemm<std::complex<double>, C_layout, C_layout, C_layout, Device, Unified, Device>();
  test_gemm<std::complex<double>, C_layout, C_layout, C_layout, Unified, Unified, Unified>();
  test_gemm<std::complex<double>, C_layout, C_layout, C_layout, Host, Unified, Unified>();

  // complex, F-layout
  test_gemm<std::complex<double>, F_layout, F_layout, F_layout, Device, Device, Device>();
  test_gemm<std::complex<double>, F_layout, F_layout, F_layout, Device, Device, Unified>();
  test_gemm<std::complex<double>, F_layout, F_layout, F_layout, Unified, Unified, Unified>();
  test_gemm<std::complex<double>, F_layout, F_layout, F_layout, Unified, Host, Unified>();

  // complex, mixed layout
  test_gemm<std::complex<double>, C_layout, F_layout, C_layout, Device, Device, Device>();
  test_gemm<std::complex<double>, F_layout, C_layout, F_layout, Unified, Unified, Unified>();
  test_gemm<std::complex<double>, C_layout, F_layout, F_layout, Host, Unified, Unified>();
  test_gemm<std::complex<double>, F_layout, C_layout, C_layout, Unified, Host, Unified>();
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
  test_gemm_batch<double, C_layout, Device, false>();
  test_gemm_batch<double, F_layout, Device, false>();
  test_gemm_batch<std::complex<double>, C_layout, Device, false>();
  test_gemm_batch<std::complex<double>, F_layout, Device, false>();

  test_gemm_batch<double, C_layout, Unified, false>();
  test_gemm_batch<double, F_layout, Unified, false>();
  test_gemm_batch<std::complex<double>, C_layout, Unified, false>();
  test_gemm_batch<std::complex<double>, F_layout, Unified, false>();
}

#ifdef NDA_HAVE_MAGMA
TEST(NDA, MAGMAGemmVbatch) {
  test_gemm_batch<double, C_layout, Device, true>();
  test_gemm_batch<double, F_layout, Device, true>();
  test_gemm_batch<std::complex<double>, C_layout, Device, true>();
  test_gemm_batch<std::complex<double>, F_layout, Device, true>();

  test_gemm_batch<double, C_layout, Unified, true>();
  test_gemm_batch<double, F_layout, Unified, true>();
  test_gemm_batch<std::complex<double>, C_layout, Unified, true>();
  test_gemm_batch<std::complex<double>, F_layout, Unified, true>();
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
  test_gemm_batch_strided<double, C_layout, Device>();
  test_gemm_batch_strided<double, C_layout, Unified>();
  test_gemm_batch_strided<std::complex<double>, C_layout, Device>();
  test_gemm_batch_strided<std::complex<double>, C_layout, Unified>();
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

  if constexpr (std::same_as<Layout, F_layout>) {
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
  test_gemv<double, C_layout, Device, Device, Device>();
  test_gemv<double, C_layout, Device, Unified, Device>();
  test_gemv<double, C_layout, Unified, Unified, Unified>();
  test_gemv<double, C_layout, Host, Unified, Unified>();

  test_gemv<double, F_layout, Device, Device, Device>();
  test_gemv<double, F_layout, Device, Device, Unified>();
  test_gemv<double, F_layout, Unified, Unified, Unified>();
  test_gemv<double, F_layout, Unified, Host, Unified>();

  test_gemv<std::complex<double>, C_layout, Device, Device, Device>();
  test_gemv<std::complex<double>, C_layout, Unified, Unified, Device>();
  test_gemv<std::complex<double>, C_layout, Unified, Unified, Unified>();
  test_gemv<std::complex<double>, C_layout, Host, Host, Unified>();

  test_gemv<std::complex<double>, F_layout, Device, Device, Device>();
  test_gemv<std::complex<double>, F_layout, Unified, Unified, Device>();
  test_gemv<std::complex<double>, F_layout, Unified, Unified, Unified>();
  test_gemv<std::complex<double>, F_layout, Unified, Host, Host>();
}

// Test the CUBLAS ger/gerc function.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2, nda::mem::AddressSpace AS3, bool star>
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
  auto v_d = to_addr_space<AS2>(v);

  // test 1: v ⊗ v starting from zero matrix
  auto exp_M1 = outer_product(v, v, star);
  auto M1_d   = to_addr_space<AS1>(nda::matrix<T, Layout>::zeros(2, 2));
  call_ger(1.0, v_d, v_d, M1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M1_d), exp_M1, fp_tol<T>);

  // test 2: v ⊗ v starting from non-zero matrix (test accumulation)
  auto M1_init = nda::matrix<T, Layout>{{10, 20}, {30, 40}};
  auto M1b_d   = to_addr_space<AS1>(M1_init);
  call_ger(1.0, v_d, v_d, M1b_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M1b_d), M1_init + exp_M1, fp_tol<T>);

  // test 3: v ⊗ w (mixed: v complex/real, w real)
  nda::vector<T> w{3, 4, 5};
  auto exp_M2 = outer_product(v, w, star);
  auto w_d    = to_addr_space<AS3>(w);
  auto M2_d   = to_addr_space<AS1>(nda::matrix<T, Layout>::zeros(2, 3));
  call_ger(1.0, v_d, w_d, M2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M2_d), exp_M2, fp_tol<T>);
  call_ger(1.0, v_d, w_d, M2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M2_d), exp_M2 * 2.0, fp_tol<T>);

  // test 4: w ⊗ v (swapped)
  auto exp_M3 = outer_product(w, v, star);
  auto M3_d   = to_addr_space<AS1>(nda::matrix<T, Layout>::zeros(3, 2));
  call_ger(1.0, w_d, v_d, M3_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M3_d), exp_M3, fp_tol<T>);
  call_ger(1.0, w_d, v_d, M3_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M3_d), exp_M3 * 2.0, fp_tol<T>);

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
  auto v_full_d  = to_addr_space<AS2>(v_full);
  auto w_full_d  = to_addr_space<AS3>(w_full);
  auto M4_d      = to_addr_space<AS1>(nda::matrix<T, Layout>::zeros(2, 3));
  call_ger(2.0, v_full_d(nda::range(1, 5, 2)), w_full_d(nda::range(0, 7, 3)), M4_d);
  EXPECT_ARRAY_NEAR(nda::to_host(M4_d), exp_M4, fp_tol<T>);
}

template <typename T, typename Layout, bool star>
void test_ger_address_spaces() {
  test_ger<T, Layout, Device, Device, Device, star>();
  test_ger<T, Layout, Device, Unified, Device, star>();
  test_ger<T, Layout, Unified, Unified, Unified, star>();
  test_ger<T, Layout, Unified, Host, Unified, star>();
}

template <typename T, bool star>
void test_ger_layouts() {
  test_ger_address_spaces<T, C_layout, star>();
  test_ger_address_spaces<T, F_layout, star>();
}

TEST(NDA, CUBLASGer) {
  test_ger_layouts<float, false>();
  test_ger_layouts<std::complex<float>, false>();
  test_ger_layouts<double, false>();
  test_ger_layouts<std::complex<double>, false>();
}

TEST(NDA, CUBLASGerc) {
  test_ger_address_spaces<float, F_layout, true>();
  test_ger_address_spaces<std::complex<float>, F_layout, true>();
  test_ger_address_spaces<double, F_layout, true>();
  test_ger_address_spaces<std::complex<double>, F_layout, true>();
}

// Test the CUBLAS dot/dotc function.
template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2, bool star>
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
  auto a_d = to_addr_space<AS1>(a);
  auto b_d = to_addr_space<AS2>(b);

  // vector dot vector
  EXPECT_COMPLEX_NEAR(dot(a_d, b_d), exp_dot(a, b), fp_tol<T>);

  // size 0 vectors
  EXPECT_EQ(dot(to_addr_space<AS1>(nda::vector<T>{}), to_addr_space<AS2>(nda::vector<T>{})), T(0));

  // strided vector dot strided vector
  EXPECT_COMPLEX_NEAR(dot(a_d(nda::range(0, 5, 2)), b_d(nda::range(0, 5, 2))), exp_dot(a(nda::range(0, 5, 2)), b(nda::range(0, 5, 2))), fp_tol<T>);
}

template <typename T, bool star>
void test_dot_address_spaces() {
  test_dot<T, Device, Device, star>();
  test_dot<T, Device, Unified, star>();
  test_dot<T, Unified, Device, star>();
  test_dot<T, Unified, Unified, star>();
  test_dot<T, Unified, Host, star>();
  test_dot<T, Host, Unified, star>();
}

TEST(NDA, CUBLASDot) {
  test_dot_address_spaces<float, false>();
  test_dot_address_spaces<std::complex<float>, false>();
  test_dot_address_spaces<double, false>();
  test_dot_address_spaces<std::complex<double>, false>();
}

TEST(NDA, CUBLASDotc) {
  test_dot_address_spaces<float, true>();
  test_dot_address_spaces<std::complex<float>, true>();
  test_dot_address_spaces<double, true>();
  test_dot_address_spaces<std::complex<double>, true>();
}

// Test the CUBLAS scal function.
template <typename T, nda::mem::AddressSpace AS>
void test_scal() {
  using fp_t = nda::get_fp_t<T>;

  // scale an empty vector
  nda::vector<T> v_empty;
  auto v_empty_d = to_addr_space<AS>(v_empty);
  nda::blas::scal(3.0, v_empty_d);
  EXPECT_TRUE(v_empty_d.empty());

  // prepare an input vector
  nda::vector<T> v{1, 2, 3, 4, 5};
  if constexpr (nda::is_complex_v<T>) { v *= T{1 - 1i}; }

  // scale by a scalar float
  auto v1_d = to_addr_space<AS>(v);
  fp_t xfp  = 3.0;
  nda::blas::scal(xfp, v1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(v1_d), xfp * v, fp_tol<T>);

  // scale by an integer
  auto v2_d = to_addr_space<AS>(v);
  int xi    = 3;
  nda::blas::scal(xi, v2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(v2_d), xi * v, fp_tol<T>);

  // scale by a complex scalar if T is complex
  if constexpr (nda::is_complex_v<T>) {
    auto v3_d = to_addr_space<AS>(v);
    auto xcp  = T{3.0 + 2.0i};
    nda::blas::scal(xcp, v3_d);
    EXPECT_ARRAY_NEAR(nda::to_host(v3_d), xcp * v, fp_tol<T>);
  }
}

template <typename T>
void test_scal_address_spaces() {
  test_scal<T, Device>();
  test_scal<T, Unified>();
}

TEST(NDA, CUBLASScal) {
  test_scal_address_spaces<float>();
  test_scal_address_spaces<std::complex<float>>();
  test_scal_address_spaces<double>();
  test_scal_address_spaces<std::complex<double>>();
}

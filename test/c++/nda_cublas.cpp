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
  EXPECT_ARRAY_NEAR(nda::to_host(C_d), exp_C, fp_tol<T>);

  // C = 3 * A * B + 2 * C
  nda::blas::gemm(3, A_d, B_d, 2, C_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C_d), 5 * exp_C, fp_tol<T>);

  // C_t = B^T * A^T
  auto C_t_d = to_addr_space<AS3>(nda::matrix<T, Layout3>(2, 2));
  nda::blas::gemm(1.0, nda::transpose(B_d), nda::transpose(A_d), 0.0, C_t_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C_t_d), nda::transpose(exp_C), fp_tol<T>);

  // C_h = B^H * A^H
  if constexpr ((a_is_f_layout and b_is_f_layout and c_is_f_layout) or (!a_is_f_layout and !b_is_f_layout and !c_is_f_layout)) {
    auto C_h_d = to_addr_space<AS3>(nda::matrix<T, Layout3>(2, 2));
    nda::blas::gemm(1.0, nda::dagger(B_d), nda::dagger(A_d), 0.0, C_h_d);
    EXPECT_ARRAY_NEAR(nda::to_host(C_h_d), nda::dagger(exp_C), fp_tol<T>);
  }

  // contiguous matrix views
  if constexpr (a_is_f_layout and !b_is_f_layout and !c_is_f_layout) {
    using nda::range;
    auto exp_C_v = nda::matrix<T, Layout3>{{13, 16}, {37, 46}};
    if constexpr (nda::is_complex_v<T>) exp_C_v *= (1 - 1i) * (2 - 1i);
    auto C_v_d = to_addr_space<AS3>(nda::matrix<T, Layout3>(5, 2));
    nda::blas::gemm(1.0, A_d(range::all, range(0, 2)), B_d(range(1, 3), range::all), 0.0, C_v_d(range(2, 4), range::all));
    EXPECT_ARRAY_NEAR(nda::to_host(C_v_d)(range(2, 4), range::all), exp_C_v, fp_tol<T>);
  }
}

template <typename T, typename Layout1, typename Layout2, typename Layout3>
void test_gemm_address_spaces() {
  test_gemm<T, Layout1, Layout2, Layout3, Device, Device, Device>();
  test_gemm<T, Layout1, Layout2, Layout3, Device, Unified, Device>();
  test_gemm<T, Layout1, Layout2, Layout3, Unified, Unified, Unified>();
  test_gemm<T, Layout1, Layout2, Layout3, Host, Unified, Unified>();
}

template <typename T>
void test_gemm_layouts() {
  test_gemm_address_spaces<T, C_layout, C_layout, C_layout>();
  test_gemm_address_spaces<T, C_layout, F_layout, C_layout>();
  test_gemm_address_spaces<T, F_layout, C_layout, F_layout>();
  test_gemm_address_spaces<T, F_layout, F_layout, F_layout>();
}

TEST(NDA, CUBLASGemm) {
  test_gemm_layouts<float>();
  test_gemm_layouts<std::complex<float>>();
  test_gemm_layouts<double>();
  test_gemm_layouts<std::complex<double>>();
}

// Test the CUBLAS/Magma gemm_batch, gemm_vbatch and gemm_batch_strided functions.
template <typename T, typename Layout1, typename Layout2, typename Layout3, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2,
          nda::mem::AddressSpace AS3, bool is_vbatch>
void test_gemm_batch() {
  int const batch_count = 4;
  long m                = is_vbatch ? 2 : 16;
  long k                = is_vbatch ? 3 : 12;
  long n                = is_vbatch ? 4 : 8;

  // create vector of matrices
  std::vector<nda::matrix<T, Layout1, nda::heap<AS1>>> vec_A;
  std::vector<nda::matrix<T, Layout2, nda::heap<AS2>>> vec_B;
  std::vector<nda::matrix<T, Layout3, nda::heap<AS3>>> vec_C;
  std::vector<nda::matrix<T, Layout3>> exp_C;
  for ([[maybe_unused]] auto i : nda::range(batch_count)) {
    auto A = nda::matrix<T, Layout1>::rand({m, k});
    auto B = nda::matrix<T, Layout2>::rand({k, n});
    auto C = nda::matrix<T, Layout3>::zeros({m, n});
    vec_A.push_back(A);
    vec_B.push_back(B);
    vec_C.push_back(C);
    nda::blas::gemm(1.0, A, B, 0.0, C);
    exp_C.push_back(std::move(C));
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
  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(nda::to_host(vec_C[i]), exp_C[i], fp_tol<T>);
}

template <typename T, typename Layout1, typename Layout2, typename Layout3, bool is_vbatch>
void test_gemm_batch_address_spaces() {
  test_gemm_batch<T, Layout1, Layout2, Layout3, Device, Device, Device, is_vbatch>();
  test_gemm_batch<T, Layout1, Layout2, Layout3, Device, Unified, Device, is_vbatch>();
  test_gemm_batch<T, Layout1, Layout2, Layout3, Unified, Unified, Unified, is_vbatch>();
  test_gemm_batch<T, Layout1, Layout2, Layout3, Host, Unified, Unified, is_vbatch>();
}

template <typename T, bool is_vbatch>
void test_gemm_batch_layouts() {
  test_gemm_batch_address_spaces<T, C_layout, C_layout, C_layout, is_vbatch>();
  test_gemm_batch_address_spaces<T, C_layout, F_layout, C_layout, is_vbatch>();
  test_gemm_batch_address_spaces<T, F_layout, C_layout, F_layout, is_vbatch>();
  test_gemm_batch_address_spaces<T, F_layout, F_layout, F_layout, is_vbatch>();
}

TEST(NDA, CUBLASGemmBatch) {
  test_gemm_batch_layouts<float, false>();
  test_gemm_batch_layouts<std::complex<float>, false>();
  test_gemm_batch_layouts<double, false>();
  test_gemm_batch_layouts<std::complex<double>, false>();
}

TEST(NDA, CUBLASGemmVBatch) {
  test_gemm_batch_layouts<float, true>();
  test_gemm_batch_layouts<double, true>();
}

#ifdef NDA_HAVE_MAGMA
TEST(NDA, MAGMAGemmVbatch) {
  test_gemm_batch_layouts<std::complex<float>, true>();
  test_gemm_batch_layouts<std::complex<double>, true>();
}
#endif // NDA_HAVE_MAGMA

template <typename T, typename Layout>
auto make_batch(int bc, long d1, long d2, bool zeros) {
  using arr_t = nda::array<T, 3, Layout>;
  if constexpr (std::same_as<Layout, C_layout>) {
    return (zeros ? arr_t::zeros({bc, d1, d2}) : arr_t::rand({bc, d1, d2}));
  } else {
    return (zeros ? arr_t::zeros({d1, d2, bc}) : arr_t::rand({d1, d2, bc}));
  }
}

template <typename T, typename Layout1, typename Layout2, typename Layout3, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2,
          nda::mem::AddressSpace AS3>
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

  auto arr_A   = make_batch<T, Layout1>(batch_count, m, k, false);
  auto arr_B   = make_batch<T, Layout2>(batch_count, k, n, false);
  auto arr_C   = make_batch<T, Layout3>(batch_count, m, n, true);
  auto arr_A_d = nda::to_device(arr_A);
  auto arr_B_d = nda::to_device(arr_B);
  auto arr_C_d = nda::to_device(arr_C);

  // C_i = A_i * B_i
  nda::blas::gemm_batch_strided(1.0, arr_A_d, arr_B_d, 0.0, arr_C_d);
  arr_C = nda::to_host(arr_C_d);
  for (auto i : nda::range(batch_count)) {
    auto exp = nda::matrix<T, F_layout>::zeros({m, n});
    nda::blas::gemm(1.0, get_mat(arr_A, i), get_mat(arr_B, i), 0.0, exp);
    EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
  }

  // C_i = alpha * A_i * B_i + beta * C_i
  nda::blas::gemm_batch_strided(alpha, arr_A_d, arr_B_d, beta, arr_C_d);
  arr_C = nda::to_host(arr_C_d);
  for (auto i : nda::range(batch_count)) {
    auto exp = nda::matrix<T, F_layout>::zeros({m, n});
    nda::blas::gemm(1.0, get_mat(arr_A, i), get_mat(arr_B, i), 0.0, exp);
    nda::blas::gemm(alpha, get_mat(arr_A, i), get_mat(arr_B, i), beta, exp);
    EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
  }

  // C_i = A_i^T * B_i
  arr_A   = make_batch<T, Layout1>(batch_count, k, m, false);
  arr_A_d = nda::to_device(arr_A);
  nda::blas::gemm_batch_strided(1.0, nda::transpose(arr_A_d), arr_B_d, 0.0, arr_C_d);
  arr_C = nda::to_host(arr_C_d);
  for (auto i : nda::range(batch_count)) {
    auto exp = nda::matrix<T, F_layout>::zeros({m, n});
    nda::blas::gemm(1.0, nda::transpose(get_mat(arr_A, i)), get_mat(arr_B, i), 0.0, exp);
    EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
  }

  // C_i = A_i * B_i^H
  if constexpr (std::same_as<Layout2, F_layout> and std::same_as<Layout3, F_layout>) {
    arr_A   = make_batch<T, Layout1>(batch_count, m, k, false);
    arr_B   = make_batch<T, Layout2>(batch_count, n, k, false);
    arr_A_d = nda::to_device(arr_A);
    arr_B_d = nda::to_device(arr_B);
    nda::blas::gemm_batch_strided(1.0, arr_A_d, nda::conj(nda::transpose(arr_B_d)), 0.0, arr_C_d);
    arr_C = nda::to_host(arr_C_d);
    for (auto i : nda::range(batch_count)) {
      auto exp = nda::matrix<T, F_layout>::zeros({m, n});
      nda::blas::gemm(1.0, get_mat(arr_A, i), nda::dagger(get_mat(arr_B, i)), 0.0, exp);
      EXPECT_ARRAY_NEAR(get_mat(arr_C, i), exp, fp_tol<T>);
    }
  }
}

template <typename T, typename Layout1, typename Layout2, typename Layout3>
void test_gemm_batch_strided_address_spaces() {
  test_gemm_batch_strided<T, Layout1, Layout2, Layout3, Device, Device, Device>();
  test_gemm_batch_strided<T, Layout1, Layout2, Layout3, Device, Unified, Device>();
  test_gemm_batch_strided<T, Layout1, Layout2, Layout3, Unified, Unified, Unified>();
  test_gemm_batch_strided<T, Layout1, Layout2, Layout3, Host, Unified, Unified>();
}

template <typename T>
void test_gemm_batch_strided_layouts() {
  test_gemm_batch_strided_address_spaces<T, C_layout, C_layout, C_layout>();
  test_gemm_batch_strided_address_spaces<T, C_layout, F_layout, C_layout>();
  test_gemm_batch_strided_address_spaces<T, F_layout, C_layout, F_layout>();
  test_gemm_batch_strided_address_spaces<T, F_layout, F_layout, F_layout>();
}

TEST(NDA, BLASGemmBatchStrided) {
  test_gemm_batch_strided_layouts<float>();
  test_gemm_batch_strided_layouts<std::complex<float>>();
  test_gemm_batch_strided_layouts<double>();
  test_gemm_batch_strided_layouts<std::complex<double>>();
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
  auto y_d = to_addr_space<AS3>(nda::vector<T>(4));
  nda::blas::gemv(1.0, A_d, x_d, 0.0, y_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_d), exp_y, fp_tol<T>);

  // y = 3 * A * x + 2y
  nda::blas::gemv(3, A_d, x_d, 2, y_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_d), 5 * exp_y, fp_tol<T>);

  // y_t = A^T * x_t
  auto y_t_d = to_addr_space<AS3>(nda::vector<T>(3));
  nda::blas::gemv(1.0, nda::transpose(A_d), x_t_d, 0.0, y_t_d);
  EXPECT_ARRAY_NEAR(nda::to_host(y_t_d), exp_y_t, fp_tol<T>);

  if constexpr (std::same_as<Layout, F_layout>) {
    // y_h = A^H * x_t
    auto exp_y_h = exp_y_t;
    if constexpr (nda::is_complex_v<T>) exp_y_h = nda::vector<T>{T{210 + 70i}, T{240 + 80i}, T{270 + 90i}};
    auto y_h_d = to_addr_space<AS3>(nda::vector<T>(3));
    nda::blas::gemv(1.0, nda::dagger(A_d), x_t_d, 0.0, y_h_d);
    EXPECT_ARRAY_NEAR(nda::to_host(y_h_d), exp_y_h, fp_tol<T>);
  } else {
    // contiguous matrix view * strided vector view
    auto x_v                 = nda::vector<T>(6);
    x_v(nda::range(0, 6, 2)) = x;
    auto x_v_d               = to_addr_space<AS2>(x_v);
    auto y_v_d               = to_addr_space<AS3>(nda::vector<T>(4));
    nda::blas::gemv(1, A_d(nda::range(2), nda::range::all), x_v_d(nda::range(0, 6, 2)), 0, y_v_d(nda::range(0, 4, 2)));
    EXPECT_ARRAY_NEAR(nda::to_host(y_v_d)(nda::range(0, 4, 2)), exp_y(nda::range(2)), fp_tol<T>);
  }
}

template <typename T, typename Layout>
void test_gemv_address_spaces() {
  test_gemv<T, Layout, Device, Device, Device>();
  test_gemv<T, Layout, Device, Unified, Device>();
  test_gemv<T, Layout, Unified, Unified, Unified>();
  test_gemv<T, Layout, Host, Host, Unified>();
}

template <typename T>
void test_gemv_layouts() {
  test_gemv_address_spaces<T, C_layout>();
  test_gemv_address_spaces<T, F_layout>();
}

TEST(NDA, CUBLASGemv) {
  test_gemv_layouts<float>();
  test_gemv_layouts<std::complex<float>>();
  test_gemv_layouts<double>();
  test_gemv_layouts<std::complex<double>>();
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

// Test the cuBLAS synchronize toggle API.
TEST(NDA, CUBLASSynchronizeToggle) {
  EXPECT_TRUE(nda::blas::device::get_synchronization());
  nda::blas::device::set_synchronization(false);
  EXPECT_FALSE(nda::blas::device::get_synchronization());

  // a small gemm should still succeed with sync disabled
  auto a = nda::cumatrix<double, F_layout>{nda::matrix<double, F_layout>{{{1, 2}, {3, 4}}}};
  auto b = nda::cumatrix<double, F_layout>{nda::matrix<double, F_layout>{{{5, 6}, {7, 8}}}};
  auto c = nda::cumatrix<double, F_layout>{nda::matrix<double, F_layout>::zeros({2, 2})};
  nda::blas::gemm(1.0, a, b, 0.0, c);

  // synchronize manually and check result
  nda::cuda_device_sync();
  EXPECT_ARRAY_NEAR(nda::to_host(c), nda::matrix<double, F_layout>{{{19, 22}, {43, 50}}}, 1e-12);

  // restore default
  nda::blas::device::set_synchronization(true);
  EXPECT_TRUE(nda::blas::device::get_synchronization());
}

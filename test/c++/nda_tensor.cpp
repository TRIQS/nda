// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;
using nda::mem::Host, nda::mem::Device, nda::mem::Unified;

TEST(NDA, TensorDefaultIndices) {
  EXPECT_EQ(nda::tensor::default_index<0>(), "");
  EXPECT_EQ(nda::tensor::default_index<1>(), "a");
  EXPECT_EQ(nda::tensor::default_index<2>(), "ab");
  EXPECT_EQ(nda::tensor::default_index<5>(), "abcde");
  EXPECT_EQ(nda::tensor::default_index<10>(), "abcdefghij");
  EXPECT_EQ(nda::tensor::default_index<26>(), "abcdefghijklmnopqrstuvwxyz");
}

// Test the generic tensor add function.
template <typename T, typename Layout1, typename Layout2, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_add() {
  constexpr bool can_permute = (AS1 != nda::mem::Host && AS2 != nda::mem::Host) || nda::tensor::have_tblis;
  constexpr bool can_reduce  = nda::tensor::have_tblis && (AS1 == nda::mem::Host || AS2 == nda::mem::Host);

  T alpha = T{3};
  T beta  = T{2};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 2 + 1i;
    beta *= 1 - 1i;
  }

  // vector addition: B_i = alpha * A_i + beta * B_i
  auto A1   = nda::vector<T>::rand({5});
  auto B1   = nda::vector<T>::rand({5});
  auto exp1 = nda::make_regular(alpha * A1 + beta * B1);
  auto B1_d = to_addr_space<AS2>(B1);
  nda::tensor::add(alpha, to_addr_space<AS1>(A1), "i", beta, B1_d, "i");
  EXPECT_ARRAY_NEAR(nda::to_host(B1_d), exp1, fp_tol<T>);

  // matrix addition: B_ij = alpha * A_ij + beta * B_ij
  auto A2   = nda::matrix<T, Layout1>::rand({3, 4});
  auto B2   = nda::matrix<T, Layout2>::rand({3, 4});
  auto exp2 = nda::make_regular(alpha * A2 + beta * B2);
  auto B2_d = to_addr_space<AS2>(B2);
  nda::tensor::add(alpha, to_addr_space<AS1>(A2), "ij", beta, B2_d, "ij");
  EXPECT_ARRAY_NEAR(nda::to_host(B2_d), exp2, fp_tol<T>);

  // rank-3 tensor addition: B_ijk = alpha * A_ijk + beta * B_ijk
  auto A4   = nda::array<T, 3, Layout1>::rand({2, 3, 4});
  auto B4   = nda::array<T, 3, Layout2>::rand({2, 3, 4});
  auto exp4 = nda::make_regular(alpha * A4 + beta * B4);
  auto B4_d = to_addr_space<AS2>(B4);
  nda::tensor::add(alpha, to_addr_space<AS1>(A4), "ijk", beta, B4_d, "ijk");
  EXPECT_ARRAY_NEAR(nda::to_host(B4_d), exp4, fp_tol<T>);

  // matrix addition with views: B_ij = alpha * A_ij + beta * B_ij
  auto A6   = nda::array<T, 2, Layout1>::rand({5, 5});
  auto B6   = nda::array<T, 2, Layout2>::rand({5, 5});
  auto exp6 = nda::make_regular(alpha * A6(nda::range(1, 4), nda::range(2, 5)) + beta * B6(nda::range(0, 5, 2), nda::range(1, 4)));
  auto A6_d = to_addr_space<AS1>(A6);
  auto B6_d = to_addr_space<AS2>(B6);
  nda::tensor::add(alpha, A6_d(nda::range(1, 4), nda::range(2, 5)), "ij", beta, B6_d(nda::range(0, 5, 2), nda::range(1, 4)), "ij");
  EXPECT_ARRAY_NEAR(nda::to_host(B6_d)(nda::range(0, 5, 2), nda::range(1, 4)), exp6, fp_tol<T>);

  // matrix addition with default indices: B = alpha * A + beta * B
  auto A8   = nda::matrix<T, Layout1>::rand({3, 4});
  auto B8   = nda::matrix<T, Layout2>::rand({3, 4});
  auto exp8 = nda::make_regular(alpha * A8 + beta * B8);
  auto B8_d = to_addr_space<AS2>(B8);
  nda::tensor::add(alpha, to_addr_space<AS1>(A8), beta, B8_d);
  EXPECT_ARRAY_NEAR(nda::to_host(B8_d), exp8, fp_tol<T>);

  // matrix addition with default indices and default factors: B = A
  auto A9   = nda::matrix<T, Layout1>::rand({3, 4});
  auto B9   = nda::matrix<T, Layout2>::rand({3, 4});
  auto exp9 = nda::make_regular(A9);
  auto B9_d = to_addr_space<AS2>(B9);
  nda::tensor::add(to_addr_space<AS1>(A9), B9_d);
  EXPECT_ARRAY_NEAR(nda::to_host(B9_d), exp9, fp_tol<T>);

  // addition involving conjugate expressions (complex types only)
  if constexpr (nda::is_complex_v<T>) {
    // B_ij = alpha * conj(A_ij) + beta * B_ij
    auto A7   = nda::matrix<T, Layout1>::rand({3, 4});
    auto B7   = nda::matrix<T, Layout2>::rand({3, 4});
    auto exp7 = nda::make_regular(alpha * nda::conj(A7) + beta * B7);
    auto B7_d = to_addr_space<AS2>(B7);
    nda::tensor::add(alpha, nda::conj(to_addr_space<AS1>(A7)), "ij", beta, B7_d, "ij");
    EXPECT_ARRAY_NEAR(nda::to_host(B7_d), exp7, fp_tol<T>);
  }

  // out-of-place addition: C_ijk = alpha * A_ijk + beta * B_ijk
  auto A10   = nda::array<T, 3, Layout1>::rand({2, 3, 4});
  auto B10   = nda::array<T, 3, Layout2>::rand({2, 3, 4});
  auto C10   = nda::array<T, 3, Layout2>::rand({2, 3, 4});
  auto exp10 = nda::make_regular(alpha * A10 + beta * B10);
  auto A10_d = to_addr_space<AS1>(A10);
  auto B10_d = to_addr_space<AS2>(B10);
  auto C10_d = to_addr_space<AS2>(C10);
  nda::tensor::add(alpha, A10_d, "ijk", beta, B10_d, "ijk", C10_d, "ijk");
  EXPECT_ARRAY_NEAR(nda::to_host(C10_d), exp10, fp_tol<T>);

  // out-of-place addition with alpha = beta = 1 convenience overload
  auto A11   = nda::array<T, 3, Layout1>::rand({2, 3, 4});
  auto B11   = nda::array<T, 3, Layout2>::rand({2, 3, 4});
  auto C11   = nda::array<T, 3, Layout2>::zeros({2, 3, 4});
  auto exp11 = nda::make_regular(A11 + B11);
  auto A11_d = to_addr_space<AS1>(A11);
  auto B11_d = to_addr_space<AS2>(B11);
  auto C11_d = to_addr_space<AS2>(C11);
  nda::tensor::add(A11_d, "ijk", B11_d, "ijk", C11_d, "ijk");
  EXPECT_ARRAY_NEAR(nda::to_host(C11_d), exp11, fp_tol<T>);

  // permutation and different-rank cases require cuTENSOR or TBLIS
  if constexpr (can_permute) {
    // matrix + transpose matrix: B_ji = alpha * A_ij + beta * B_ji
    auto A3   = nda::matrix<T, Layout1>::rand({3, 4});
    auto B3   = nda::matrix<T, Layout2>::rand({4, 3});
    auto exp3 = nda::make_regular(alpha * nda::transpose(A3) + beta * B3);
    auto B3_d = to_addr_space<AS2>(B3);
    nda::tensor::add(alpha, to_addr_space<AS1>(A3), "ij", beta, B3_d, "ji");
    EXPECT_ARRAY_NEAR(nda::to_host(B3_d), exp3, fp_tol<T>);

    // rank-3 tensor addition + permuted indices: B_kij = alpha * A_ijk + beta * B_kij
    auto A5   = nda::array<T, 3, Layout1>::rand({2, 3, 4});
    auto B5   = nda::array<T, 3, Layout2>::rand({4, 2, 3});
    auto exp5 = nda::array<T, 3, Layout2>::zeros({4, 2, 3});
    nda::for_each(exp5.shape(), [&](auto k, auto i, auto j) { exp5(k, i, j) = alpha * A5(i, j, k) + beta * B5(k, i, j); });
    auto B5_d = to_addr_space<AS2>(B5);
    nda::tensor::add(alpha, to_addr_space<AS1>(A5), "ijk", beta, B5_d, "kij");
    EXPECT_ARRAY_NEAR(nda::to_host(B5_d), exp5, fp_tol<T>);

    // different rank — reduction: B_i = beta * B_i + alpha * sum_j A_ij (only supported on the TBLIS host path)
    if constexpr (can_reduce) {
      auto A12   = nda::array<T, 2, Layout1>::rand({3, 4});
      auto B12   = nda::array<T, 1>::rand({3});
      auto exp12 = nda::array<T, 1>::zeros({3});
      nda::for_each(exp12.shape(), [&](auto i) {
        T s = 0;
        for (long j = 0; j < 4; ++j) s += A12(i, j);
        exp12(i) = beta * B12(i) + alpha * s;
      });
      auto B12_d = to_addr_space<AS2>(B12);
      nda::tensor::add(alpha, to_addr_space<AS1>(A12), "ij", beta, B12_d, "i");
      EXPECT_ARRAY_NEAR(nda::to_host(B12_d), exp12, fp_tol<T>);
    }

    // different rank — broadcast via default indices: B_ijk <- B_ijk + A_ij
    auto A13   = nda::array<T, 2, Layout1>::rand({3, 4});
    auto B13   = nda::array<T, 3, Layout2>::rand({3, 4, 5});
    auto exp13 = nda::array<T, 3, Layout2>::zeros({3, 4, 5});
    nda::for_each(exp13.shape(), [&](auto i, auto j, auto k) { exp13(i, j, k) = B13(i, j, k) + A13(i, j); });
    auto B13_d = to_addr_space<AS2>(B13);
    nda::tensor::add(T{1}, to_addr_space<AS1>(A13), T{1}, B13_d);
    EXPECT_ARRAY_NEAR(nda::to_host(B13_d), exp13, fp_tol<T>);
  }
}

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_add_layouts() {
  test_add<T, C_layout, C_layout, AS1, AS2>();
  test_add<T, C_layout, F_layout, AS1, AS2>();
  test_add<T, F_layout, C_layout, AS1, AS2>();
  test_add<T, F_layout, F_layout, AS1, AS2>();
}

template <typename T>
void test_add_on_device() {
  test_add_layouts<T, Device, Device>();
  test_add_layouts<T, Device, Unified>();
  test_add_layouts<T, Unified, Device>();
  test_add_layouts<T, Unified, Unified>();
}

template <typename T>
void test_add_on_host() {
  test_add_layouts<T, Host, Host>();
#ifdef NDA_HAVE_CUDA
  test_add_layouts<T, Host, Unified>();
  test_add_layouts<T, Unified, Host>();
#endif // NDA_HAVE_CUDA
}

#ifdef NDA_HAVE_CUTENSOR
TEST(NDA, TensorAddOnDevice) {
  test_add_on_device<float>();
  test_add_on_device<std::complex<float>>();
  test_add_on_device<double>();
  test_add_on_device<std::complex<double>>();
}
#endif // NDA_HAVE_CUTENSOR

TEST(NDA, TensorAddOnHost) {
  test_add_on_host<float>();
  test_add_on_host<std::complex<float>>();
  test_add_on_host<double>();
  test_add_on_host<std::complex<double>>();
}

#ifndef NDA_HAVE_TBLIS
// The nda fallback rejects mismatched index strings at runtime via require_equal_indices.
TEST(NDA, TensorAddOnHostFallbackMismatchedIndicesThrows) {
  auto A = nda::matrix<double>::rand({3, 3});
  auto B = nda::matrix<double>::rand({3, 3});
  EXPECT_THROW(nda::tensor::add(1.0, A, "ij", 1.0, B, "ji"), nda::runtime_error);
}
#endif // NDA_HAVE_TBLIS

// Test the generic tensor assign function.
template <typename T, typename Layout1, typename Layout2, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_assign() {
  // rank-2, default indices: B = A — supported by every backend
  auto A1   = nda::matrix<T, Layout1>::rand({3, 4});
  auto B1_d = to_addr_space<AS2>(nda::matrix<T, Layout2>::rand({3, 4}));
  nda::tensor::assign(to_addr_space<AS1>(A1), B1_d);
  EXPECT_ARRAY_EQ(nda::to_host(B1_d), A1);

  // rank-3, default indices: B = A — also supported by every backend
  auto A2   = nda::array<T, 3, Layout1>::rand({2, 3, 4});
  auto B2_d = to_addr_space<AS2>(nda::array<T, 3, Layout2>::rand({2, 3, 4}));
  nda::tensor::assign(to_addr_space<AS1>(A2), B2_d);
  EXPECT_ARRAY_EQ(nda::to_host(B2_d), A2);

  // permutation and different-rank cases require cuTENSOR or TBLIS
  if constexpr (nda::tensor::have_tblis || nda::tensor::have_cutensor) {
    // rank-2 permutation: B_ji = A_ij
    auto A3   = nda::matrix<T, Layout1>::rand({3, 4});
    auto exp3 = nda::make_regular(nda::transpose(A3));
    auto B3_d = to_addr_space<AS2>(nda::matrix<T, Layout2>::zeros({4, 3}));
    nda::tensor::assign(to_addr_space<AS1>(A3), "ij", B3_d, "ji");
    EXPECT_ARRAY_EQ(nda::to_host(B3_d), exp3);

    // rank-3 permutation: B_kij = A_ijk
    auto A4   = nda::array<T, 3, Layout1>::rand({2, 3, 4});
    auto exp4 = nda::array<T, 3, Layout2>::zeros({4, 2, 3});
    nda::for_each(exp4.shape(), [&](auto k, auto i, auto j) { exp4(k, i, j) = A4(i, j, k); });
    auto B4_d = to_addr_space<AS2>(nda::array<T, 3, Layout2>::zeros({4, 2, 3}));
    nda::tensor::assign(to_addr_space<AS1>(A4), "ijk", B4_d, "kij");
    EXPECT_ARRAY_EQ(nda::to_host(B4_d), exp4);

    // different rank — broadcast rank-2 A along the third axis of B: B_ijk = A_ij
    auto A5   = nda::matrix<T, Layout1>::rand({3, 4});
    auto exp5 = nda::array<T, 3, Layout2>::zeros({3, 4, 5});
    nda::for_each(exp5.shape(), [&](auto i, auto j, auto k) { exp5(i, j, k) = A5(i, j); });
    auto B5_d = to_addr_space<AS2>(nda::array<T, 3, Layout2>::zeros({3, 4, 5}));
    nda::tensor::assign(to_addr_space<AS1>(A5), "ij", B5_d, "ijk");
    EXPECT_ARRAY_EQ(nda::to_host(B5_d), exp5);
  }
}

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_assign_layouts() {
  test_assign<T, C_layout, C_layout, AS1, AS2>();
  test_assign<T, C_layout, F_layout, AS1, AS2>();
  test_assign<T, F_layout, C_layout, AS1, AS2>();
  test_assign<T, F_layout, F_layout, AS1, AS2>();
}

template <typename T>
void test_assign_on_device() {
  test_assign_layouts<T, Device, Device>();
  test_assign_layouts<T, Device, Unified>();
  test_assign_layouts<T, Unified, Device>();
  test_assign_layouts<T, Unified, Unified>();
}

template <typename T>
void test_assign_on_host() {
  test_assign_layouts<T, Host, Host>();
#ifdef NDA_HAVE_CUDA
  test_assign_layouts<T, Host, Unified>();
  test_assign_layouts<T, Unified, Host>();
#endif // NDA_HAVE_CUDA
}

#ifdef NDA_HAVE_CUTENSOR
TEST(NDA, TensorAssignOnDevice) {
  test_assign_on_device<float>();
  test_assign_on_device<std::complex<float>>();
  test_assign_on_device<double>();
  test_assign_on_device<std::complex<double>>();
}
#endif // NDA_HAVE_CUTENSOR

TEST(NDA, TensorAssignOnHost) {
  test_assign_on_host<float>();
  test_assign_on_host<std::complex<float>>();
  test_assign_on_host<double>();
  test_assign_on_host<std::complex<double>>();
}

#ifdef NDA_HAVE_CUDA
// Test the cross-memory recursive copy path of nda::tensor::assign.
template <typename T, typename Layout1, typename Layout2, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_assign_cross_memory() {
  // rank-1: always a leaf in rec_copy
  auto A1 = nda::vector<T>::zeros({7});
  for (long c = 0; auto &x : A1) x = static_cast<T>(c++);
  auto A1_d = to_addr_space<AS1>(A1);
  auto B1_d = to_addr_space<AS2>(nda::vector<T>::zeros({7}));
  nda::tensor::assign(A1_d, B1_d);
  EXPECT_ARRAY_EQ(nda::to_host(B1_d), A1);

  // rank-3 contiguous: leaf-at-top when Layout1 == Layout2, else descent to rank-1
  auto A2 = nda::array<T, 3, Layout1>::zeros({2, 3, 4});
  for (long c = 0; auto &x : A2) x = static_cast<T>(c++);
  auto A2_d = to_addr_space<AS1>(A2);
  auto B2_d = to_addr_space<AS2>(nda::array<T, 3, Layout2>::zeros({2, 3, 4}));
  nda::tensor::assign(A2_d, B2_d);
  EXPECT_ARRAY_EQ(nda::to_host(B2_d), A2);

  // rank-3 non-contiguous source view: forces recursive descent along the slowest-varying axis
  auto A3 = nda::array<T, 3, Layout1>::zeros({4, 6, 8});
  for (long c = 0; auto &x : A3) x = static_cast<T>(c++);
  auto A3_d      = to_addr_space<AS1>(A3);
  auto A3_view_d = A3_d(nda::range(0, 4, 2), nda::range(1, 5), nda::range(0, 8, 2));
  auto B3_d      = to_addr_space<AS2>(nda::array<T, 3, Layout2>::zeros({2, 4, 4}));
  nda::tensor::assign(A3_view_d, B3_d);
  auto expected = nda::make_regular(A3(nda::range(0, 4, 2), nda::range(1, 5), nda::range(0, 8, 2)));
  EXPECT_ARRAY_EQ(nda::to_host(B3_d), expected);
}

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_assign_cross_memory_layouts() {
  test_assign_cross_memory<T, C_layout, C_layout, AS1, AS2>();
  test_assign_cross_memory<T, C_layout, F_layout, AS1, AS2>();
  test_assign_cross_memory<T, F_layout, C_layout, AS1, AS2>();
  test_assign_cross_memory<T, F_layout, F_layout, AS1, AS2>();
}

template <typename T>
void test_assign_cross_memory_address_spaces() {
  test_assign_cross_memory_layouts<T, Host, Device>();
  test_assign_cross_memory_layouts<T, Device, Host>();
  test_assign_cross_memory_layouts<T, Host, Unified>();
  test_assign_cross_memory_layouts<T, Unified, Host>();
  test_assign_cross_memory_layouts<T, Device, Unified>();
  test_assign_cross_memory_layouts<T, Unified, Device>();
  test_assign_cross_memory_layouts<T, Device, Device>();
}

TEST(NDA, TensorAssignCrossMemory) {
  test_assign_cross_memory_address_spaces<float>();
  test_assign_cross_memory_address_spaces<double>();
  test_assign_cross_memory_address_spaces<std::complex<float>>();
  test_assign_cross_memory_address_spaces<std::complex<double>>();
  test_assign_cross_memory_address_spaces<int>();
}

// Test that permutations in the cross-memory fallback path throw.
TEST(NDA, TensorAssignCrossMemoryPermutationThrows) {
  auto A_h = nda::array<double, 2>::rand({3, 4});
  auto B_d = to_addr_space<Device>(nda::array<double, 2>::zeros({4, 3}));
  EXPECT_THROW(nda::tensor::assign(A_h, "ij", B_d, "ji"), nda::runtime_error);
}
#endif // NDA_HAVE_CUDA

// Test the generic tensor contract function.
template <typename T, typename Layout1, typename Layout2, typename Layout3, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2,
          nda::mem::AddressSpace AS3>
void test_contract() {
  auto const _ = nda::range::all;

  T alpha = T{3};
  T beta  = T{2};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 2 + 1i;
    beta *= 1 - 1i;
  }

  // matrix-matrix multiplication: C_ik = alpha * A_ij * B_jk + beta * C_ik
  auto A1   = nda::matrix<T, Layout1>::rand({3, 4});
  auto B1   = nda::matrix<T, Layout2>::rand({4, 5});
  auto C1   = nda::matrix<T, Layout3>::rand({3, 5});
  auto exp1 = nda::make_regular(alpha * nda::linalg::matmul(A1, B1) + beta * C1);
  auto C1_d = to_addr_space<AS3>(C1);
  nda::tensor::contract(alpha, to_addr_space<AS1>(A1), "ij", to_addr_space<AS2>(B1), "jk", beta, C1_d, "ik");
  EXPECT_ARRAY_NEAR(nda::to_host(C1_d), exp1, fp_tol<T>);

  // matrix-vector multiplication: C_i = alpha * A_ij * B_j + beta * C_i
  auto A2   = nda::matrix<T, Layout1>::rand({4, 5});
  auto B2   = nda::vector<T>::rand({5});
  auto C2   = nda::vector<T>::rand({4});
  auto exp2 = nda::make_regular(alpha * nda::linalg::matvecmul(A2, B2) + beta * C2);
  auto C2_d = to_addr_space<AS3>(C2);
  nda::tensor::contract(alpha, to_addr_space<AS1>(A2), "ij", to_addr_space<AS2>(B2), "j", beta, C2_d, "i");
  EXPECT_ARRAY_NEAR(nda::to_host(C2_d), exp2, fp_tol<T>);

  // outer product: C_ij = alpha * A_i * B_j
  auto A3   = nda::vector<T>::rand({3});
  auto B3   = nda::vector<T>::rand({4});
  auto C3   = nda::array<T, 2, Layout1>::zeros({3, 4});
  auto exp3 = nda::make_regular(alpha * nda::linalg::outer_product(A3, B3));
  auto C3_d = to_addr_space<AS3>(C3);
  nda::tensor::contract(alpha, to_addr_space<AS1>(A3), "i", to_addr_space<AS2>(B3), "j", T{0}, C3_d, "ij");
  EXPECT_ARRAY_NEAR(nda::to_host(C3_d), exp3, fp_tol<T>);

  // tensor contraction: C_il = alpha * A_ijk * B_jkl + beta * C_il
  auto A4   = nda::array<T, 3, Layout1>::rand({2, 3, 4});
  auto B4   = nda::array<T, 3, Layout2>::rand({3, 4, 5});
  auto C4   = nda::array<T, 2, Layout3>::rand({2, 5});
  auto exp4 = nda::array<T, 2, Layout3>::zeros({2, 5});
  nda::for_each(exp4.shape(), [&](auto i, auto l) {
    T sum = 0;
    for (auto j : nda::range(3))
      for (auto k : nda::range(4)) sum += A4(i, j, k) * B4(j, k, l);
    exp4(i, l) = alpha * sum + beta * C4(i, l);
  });
  auto C4_d = to_addr_space<AS3>(C4);
  nda::tensor::contract(alpha, to_addr_space<AS1>(A4), "ijk", to_addr_space<AS2>(B4), "jkl", beta, C4_d, "il");
  EXPECT_ARRAY_NEAR(nda::to_host(C4_d), exp4, fp_tol<T> * 10);

  // matrix-matrix multiplication with views: C_ik = alpha * A_ij * B_jk + beta * C_ik
  auto A5   = nda::array<T, 2, Layout1>::rand({5, 6});
  auto B5   = nda::array<T, 2, Layout2>::rand({6, 7});
  auto C5   = nda::array<T, 2, Layout3>::rand({3, 4});
  auto exp5 = nda::make_regular(alpha * nda::linalg::matmul(A5(nda::range(1, 4), _), B5(_, nda::range(2, 6))) + beta * C5);
  auto A5_d = to_addr_space<AS1>(A5);
  auto B5_d = to_addr_space<AS2>(B5);
  auto C5_d = to_addr_space<AS3>(C5);
  nda::tensor::contract(alpha, to_addr_space<AS1>(A5_d(nda::range(1, 4), _)), "ij", B5_d(_, nda::range(2, 6)), "jk", beta, C5_d, "ik");
  EXPECT_ARRAY_NEAR(nda::to_host(C5_d), exp5, fp_tol<T>);

  // batched matmul: C_ikl = alpha * A_ijl * B_jkl + beta * C_ikl
  auto A6   = nda::array<T, 3, Layout1>::rand({3, 4, 2});
  auto B6   = nda::array<T, 3, Layout2>::rand({4, 5, 2});
  auto C6   = nda::array<T, 3, Layout3>::rand({3, 5, 2});
  auto exp6 = nda::array<T, 3, F_layout>{C6};
  nda::blas::gemm_batch_strided(alpha, nda::array<T, 3, F_layout>{A6}, nda::array<T, 3, F_layout>{B6}, beta, exp6);
  auto C6_d = to_addr_space<AS3>(C6);
  nda::tensor::contract(alpha, to_addr_space<AS1>(A6), "ijl", to_addr_space<AS2>(B6), "jkl", beta, C6_d, "ikl");
  EXPECT_ARRAY_NEAR(nda::to_host(C6_d), exp6, fp_tol<T>);

  // contractions involving conjugate expressions (complex types only)
  if constexpr (nda::is_complex_v<T>) {
    // C_ik = alpha * conj(A_ij) * B_jk + beta * C_ik
    auto A7   = nda::matrix<T, Layout1>::rand({3, 4});
    auto B7   = nda::matrix<T, Layout2>::rand({4, 5});
    auto C7   = nda::matrix<T, Layout3>::rand({3, 5});
    auto exp7 = nda::make_regular(alpha * nda::linalg::matmul(nda::conj(A7), B7) + beta * C7);
    auto C7_d = to_addr_space<AS3>(C7);
    nda::tensor::contract(alpha, nda::conj(to_addr_space<AS1>(A7)), "ij", to_addr_space<AS2>(B7), "jk", beta, C7_d, "ik");
    EXPECT_ARRAY_NEAR(nda::to_host(C7_d), exp7, fp_tol<T>);

    // C_ik = alpha * conj(A_ij) * conj(B_jk) + beta * C_ik
    auto A8   = nda::matrix<T, Layout1>::rand({3, 4});
    auto B8   = nda::matrix<T, Layout2>::rand({4, 5});
    auto C8   = nda::matrix<T, Layout3>::rand({3, 5});
    auto exp8 = nda::make_regular(alpha * nda::linalg::matmul(nda::conj(A8), nda::conj(B8)) + beta * C8);
    auto C8_d = to_addr_space<AS3>(C8);
    nda::tensor::contract(alpha, nda::conj(to_addr_space<AS1>(A8)), "ij", nda::conj(to_addr_space<AS2>(B8)), "jk", beta, C8_d, "ik");
    EXPECT_ARRAY_NEAR(nda::to_host(C8_d), exp8, fp_tol<T>);
  }
}

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2, nda::mem::AddressSpace AS3>
void test_contract_layouts() {
  test_contract<T, C_layout, C_layout, C_layout, AS1, AS2, AS3>();
  test_contract<T, C_layout, C_layout, F_layout, AS1, AS2, AS3>();
  test_contract<T, C_layout, F_layout, C_layout, AS1, AS2, AS3>();
  test_contract<T, C_layout, F_layout, F_layout, AS1, AS2, AS3>();
  test_contract<T, F_layout, C_layout, C_layout, AS1, AS2, AS3>();
  test_contract<T, F_layout, C_layout, F_layout, AS1, AS2, AS3>();
  test_contract<T, F_layout, F_layout, C_layout, AS1, AS2, AS3>();
  test_contract<T, F_layout, F_layout, F_layout, AS1, AS2, AS3>();
}

template <typename T>
void test_contract_on_device() {
  test_contract_layouts<T, Device, Device, Device>();
  test_contract_layouts<T, Device, Device, Unified>();
  test_contract_layouts<T, Device, Unified, Device>();
  test_contract_layouts<T, Unified, Device, Device>();
  test_contract_layouts<T, Device, Unified, Unified>();
  test_contract_layouts<T, Unified, Device, Unified>();
  test_contract_layouts<T, Unified, Unified, Device>();
  test_contract_layouts<T, Unified, Unified, Unified>();
}

template <typename T>
void test_contract_on_host() {
  test_contract_layouts<T, Host, Host, Host>();
#ifdef NDA_HAVE_CUDA
  test_contract_layouts<T, Host, Host, Unified>();
  test_contract_layouts<T, Host, Unified, Host>();
  test_contract_layouts<T, Unified, Host, Host>();
  test_contract_layouts<T, Host, Unified, Unified>();
  test_contract_layouts<T, Unified, Host, Unified>();
  test_contract_layouts<T, Unified, Unified, Host>();
#endif // NDA_HAVE_CUDA
}

#ifdef NDA_HAVE_CUTENSOR
TEST(NDA, TensorContractOnDevice) {
  test_contract_on_device<float>();
  test_contract_on_device<std::complex<float>>();
  test_contract_on_device<double>();
  test_contract_on_device<std::complex<double>>();
}
#endif // NDA_HAVE_CUTENSOR

#ifdef NDA_HAVE_TBLIS
TEST(NDA, TensorContractOnHost) {
  test_contract_on_host<float>();
  test_contract_on_host<std::complex<float>>();
  test_contract_on_host<double>();
  test_contract_on_host<std::complex<double>>();
}
#endif // NDA_HAVE_TBLIS

// Test the generic tensor dot function.
template <typename T, typename Layout1, typename Layout2, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_dot() {
  // A = [[1, 2], [3, 4]],  B = [[5, 6], [7, 8]]
  auto A   = nda::matrix<T, Layout1>{{1, 2}, {3, 4}};
  auto B   = nda::matrix<T, Layout2>{{5, 6}, {7, 8}};
  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);

  // sum_{i,j} A(i,j) * B(i,j) = 1*5 + 2*6 + 3*7 + 4*8 = 70
  EXPECT_COMPLEX_NEAR(nda::tensor::dot(A_d, "ab", B_d, "ab"), T{70});
  // overload uses the rank-2 default indices ("ab"/"ab")
  EXPECT_COMPLEX_NEAR(nda::tensor::dot(A_d, B_d), T{70});
  // sum_{i,j} A(i,j) * B(j,i) = 1*5 + 2*7 + 3*6 + 4*8 = 69 — permuted indices require cuTENSOR or TBLIS
  if constexpr (nda::tensor::have_tblis || nda::tensor::have_cutensor) { EXPECT_COMPLEX_NEAR(nda::tensor::dot(A_d, "ab", B_d, "ba"), T{69}); }

  // complex types and conjugation
  // C = [[1+i, 2], [3, 4-i]],  D = [[5, 6+2i], [7, 8]]
  if constexpr (nda::is_complex_v<T>) {
    auto C   = nda::matrix<T, Layout1>{{T(1, 1), T(2, 0)}, {T(3, 0), T(4, -1)}};
    auto D   = nda::matrix<T, Layout2>{{T(5, 0), T(6, 2)}, {T(7, 0), T(8, 0)}};
    auto C_d = to_addr_space<AS1>(C);
    auto D_d = to_addr_space<AS2>(D);
    EXPECT_COMPLEX_NEAR(nda::tensor::dot(C_d, "ab", D_d, "ab"), T(70, 1));
    EXPECT_COMPLEX_NEAR(nda::tensor::dot(nda::conj(C_d), "ab", D_d, "ab"), T(70, 7));
    EXPECT_COMPLEX_NEAR(nda::tensor::dot(C_d, "ab", nda::conj(D_d), "ab"), T(70, -7));
    EXPECT_COMPLEX_NEAR(nda::tensor::dot(nda::conj(C_d), "ab", nda::conj(D_d), "ab"), T(70, -1));
    if constexpr (nda::tensor::have_tblis || nda::tensor::have_cutensor) {
      EXPECT_COMPLEX_NEAR(nda::tensor::dot(C_d, "ab", D_d, "ba"), T(69, 3));
      EXPECT_COMPLEX_NEAR(nda::tensor::dot(nda::conj(C_d), "ab", D_d, "ba"), T(69, 9));
      EXPECT_COMPLEX_NEAR(nda::tensor::dot(C_d, "ab", nda::conj(D_d), "ba"), T(69, -9));
      EXPECT_COMPLEX_NEAR(nda::tensor::dot(nda::conj(C_d), "ab", nda::conj(D_d), "ba"), T(69, -3));
    }
  }
}

template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_dot_layouts() {
  test_dot<T, C_layout, C_layout, AS1, AS2>();
  test_dot<T, C_layout, F_layout, AS1, AS2>();
  test_dot<T, F_layout, C_layout, AS1, AS2>();
  test_dot<T, F_layout, F_layout, AS1, AS2>();
}

#ifdef NDA_HAVE_CUTENSOR
TEST(NDA, TensorDotOnDevice) {
  test_dot_layouts<float, Device, Device>();
  test_dot_layouts<std::complex<float>, Device, Device>();
  test_dot_layouts<double, Device, Device>();
  test_dot_layouts<std::complex<double>, Device, Device>();
}
#endif // NDA_HAVE_CUTENSOR

TEST(NDA, TensorDotOnHost) {
  test_dot_layouts<float, Host, Host>();
  test_dot_layouts<std::complex<float>, Host, Host>();
  test_dot_layouts<double, Host, Host>();
  test_dot_layouts<std::complex<double>, Host, Host>();
}

#ifdef NDA_HAVE_TBLIS
// Different-rank dot via TBLIS einsum with a repeated index in A: trace of diag(A) against v, i.e. sum_i A_ii * v_i.
TEST(NDA, TensorDotDifferentRankOnHost) {
  using T    = double;
  auto A     = nda::array<T, 2>::rand({3, 3});
  auto v     = nda::array<T, 1>::rand({3});
  T expected = 0;
  for (long i = 0; i < 3; ++i) expected += A(i, i) * v(i);
  auto result = nda::tensor::dot(A, "ii", v, "i");
  EXPECT_NEAR(result, expected, fp_tol<T>);
}
#else
// The nda host fallback rejects mismatched index strings.
TEST(NDA, TensorDotMismatchedIndicesOnHostFallbackThrows) {
  auto A = nda::matrix<double>{{1, 2}, {3, 4}};
  auto B = nda::matrix<double>{{5, 6}, {7, 8}};
  EXPECT_THROW((void)nda::tensor::dot(A, "ab", B, "ba"), nda::runtime_error);
}
#endif // NDA_HAVE_TBLIS

// Test the generic tensor elementwise trinary function.
template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2, nda::mem::AddressSpace AS3>
void test_elementwise_trinary() {
  using nda::tensor::binary_op;
  constexpr bool on_host = (AS1 == nda::mem::Host || AS2 == nda::mem::Host || AS3 == nda::mem::Host);

  T alpha = T{2};
  T beta  = T{3};
  T gamma = T{4};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 1 + 1i;
    beta *= 2 - 1i;
    gamma *= 1 - 2i;
  }

  auto A = nda::array<T, 3>::rand({2, 3, 4});
  auto B = nda::array<T, 3>::rand({2, 3, 4});
  auto C = nda::array<T, 3>::rand({2, 3, 4});

  auto check = [&](binary_op op_AB, binary_op op_ABC, auto const &expected) {
    auto C_d = to_addr_space<AS3>(C);
    nda::tensor::elementwise_trinary(alpha, to_addr_space<AS1>(A), "abc", beta, to_addr_space<AS2>(B), "abc", gamma, C_d, "abc", op_AB, op_ABC);
    EXPECT_ARRAY_NEAR(nda::to_host(C_d), expected, fp_tol<T> * 10);
  };

  // SUM/SUM, PROD/SUM, SUM/PROD, PROD/PROD
  check(binary_op::SUM, binary_op::SUM, (alpha * A + beta * B) + gamma * C);
  check(binary_op::PROD, binary_op::SUM, (alpha * A) * (beta * B) + gamma * C);
  check(binary_op::SUM, binary_op::PROD, (alpha * A + beta * B) * (gamma * C));
  check(binary_op::PROD, binary_op::PROD, (alpha * A) * (beta * B) * (gamma * C));

  // MAX/MIN-involved combos: real value types only
  if constexpr (!nda::is_complex_v<T>) {
    check(binary_op::SUM, binary_op::MAX, nda::max(alpha * A + beta * B, gamma * C));
    check(binary_op::SUM, binary_op::MIN, nda::min(alpha * A + beta * B, gamma * C));
    check(binary_op::MAX, binary_op::PROD, nda::max(alpha * A, beta * B) * (gamma * C));
    check(binary_op::MIN, binary_op::PROD, nda::min(alpha * A, beta * B) * (gamma * C));
  }

  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);
  auto C_d = to_addr_space<AS3>(C);
  if constexpr (on_host) {
    // Abs-family + NORM_2 (cuTENSOR rejects these as op_AB or op_ABC)
    nda::array<T, 3> exp(A.shape());

    // abs-family as op_AB
    exp = (nda::abs(alpha * A) + nda::abs(beta * B)) + gamma * C;
    check(binary_op::SUM_ABS, binary_op::SUM, exp);

    exp = nda::max(nda::abs(alpha * A), nda::abs(beta * B)) + gamma * C;
    check(binary_op::MAX_ABS, binary_op::SUM, exp);

    exp = nda::min(nda::abs(alpha * A), nda::abs(beta * B)) + gamma * C;
    check(binary_op::MIN_ABS, binary_op::SUM, exp);

    exp = nda::sqrt(nda::abs2(alpha * A) + nda::abs2(beta * B)) * (gamma * C);
    check(binary_op::NORM_2, binary_op::PROD, exp);

    // abs-family as op_ABC
    exp = nda::abs(alpha * A + beta * B) + nda::abs(gamma * C);
    check(binary_op::SUM, binary_op::SUM_ABS, exp);

    exp = nda::max(nda::abs((alpha * A) * (beta * B)), nda::abs(gamma * C));
    check(binary_op::PROD, binary_op::MAX_ABS, exp);

    exp = nda::sqrt(nda::abs2(alpha * A + beta * B) + nda::abs2(gamma * C));
    check(binary_op::SUM, binary_op::NORM_2, exp);

    // MAX/MIN on complex T throw on the nda host fallback
    if constexpr (nda::is_complex_v<T>) {
      EXPECT_THROW(nda::tensor::elementwise_trinary(alpha, A_d, "abc", beta, B_d, "abc", gamma, C_d, "abc", binary_op::SUM, binary_op::MAX),
                   nda::runtime_error);
      EXPECT_THROW(nda::tensor::elementwise_trinary(alpha, A_d, "abc", beta, B_d, "abc", gamma, C_d, "abc", binary_op::MAX, binary_op::SUM),
                   nda::runtime_error);
    }

    // mismatched indices throw on the nda fallback (both pairings: A/B and B/C)
    EXPECT_THROW(nda::tensor::elementwise_trinary(alpha, A_d, "abc", beta, B_d, "acb", gamma, C_d, "abc"), nda::runtime_error);
    EXPECT_THROW(nda::tensor::elementwise_trinary(alpha, A_d, "abc", beta, B_d, "abc", gamma, C_d, "acb"), nda::runtime_error);
  } else {
    // differing index strings work on cuTENSOR (idx_c is a permutation of idx_a/idx_b)
    auto C_perm = nda::array<T, 3>::rand({3, 4, 2}); // indexed "bca"
    auto exp    = nda::array<T, 3>::zeros({3, 4, 2});
    nda::for_each(exp.shape(), [&](auto b, auto c, auto a) { exp(b, c, a) = (alpha * A(a, b, c) + beta * B(a, b, c)) + gamma * C_perm(b, c, a); });
    auto C_perm_d = to_addr_space<AS3>(C_perm);
    nda::tensor::elementwise_trinary(alpha, to_addr_space<AS1>(A), "abc", beta, to_addr_space<AS2>(B), "abc", gamma, C_perm_d, "bca", binary_op::SUM,
                                     binary_op::SUM);
    EXPECT_ARRAY_NEAR(nda::to_host(C_perm_d), exp, fp_tol<T> * 10);

    // unsupported op on the device throws
    EXPECT_THROW(nda::tensor::elementwise_trinary(alpha, A_d, "abc", beta, B_d, "abc", gamma, C_d, "abc", binary_op::SUM_ABS, binary_op::SUM),
                 nda::runtime_error);
  }
}

template <typename T>
void test_elementwise_trinary_on_device() {
  test_elementwise_trinary<T, Device, Device, Device>();
  test_elementwise_trinary<T, Device, Device, Unified>();
  test_elementwise_trinary<T, Device, Unified, Device>();
  test_elementwise_trinary<T, Unified, Device, Device>();
  test_elementwise_trinary<T, Device, Unified, Unified>();
  test_elementwise_trinary<T, Unified, Device, Unified>();
  test_elementwise_trinary<T, Unified, Unified, Device>();
  test_elementwise_trinary<T, Unified, Unified, Unified>();
}

template <typename T>
void test_elementwise_trinary_on_host() {
  test_elementwise_trinary<T, Host, Host, Host>();
#ifdef NDA_HAVE_CUDA
  test_elementwise_trinary<T, Host, Host, Unified>();
  test_elementwise_trinary<T, Host, Unified, Host>();
  test_elementwise_trinary<T, Unified, Host, Host>();
  test_elementwise_trinary<T, Host, Unified, Unified>();
  test_elementwise_trinary<T, Unified, Host, Unified>();
  test_elementwise_trinary<T, Unified, Unified, Host>();
#endif // NDA_HAVE_CUDA
}

#ifdef NDA_HAVE_CUTENSOR
TEST(NDA, TensorElementwiseTrinaryOnDevice) {
  test_elementwise_trinary_on_device<float>();
  test_elementwise_trinary_on_device<std::complex<float>>();
  test_elementwise_trinary_on_device<double>();
  test_elementwise_trinary_on_device<std::complex<double>>();
}
#endif // NDA_HAVE_CUTENSOR

TEST(NDA, TensorElementwiseTrinaryOnHost) {
  test_elementwise_trinary_on_host<float>();
  test_elementwise_trinary_on_host<std::complex<float>>();
  test_elementwise_trinary_on_host<double>();
  test_elementwise_trinary_on_host<std::complex<double>>();
}

// Test the generic tensor elementwise binary function.
template <typename T, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_elementwise() {
  using nda::tensor::binary_op;
  constexpr bool on_host = (AS1 == nda::mem::Host || AS2 == nda::mem::Host);

  T alpha = T{2};
  T beta  = T{3};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 1 + 1i;
    beta *= 2 - 1i;
  }

  auto A = nda::array<T, 3>::rand({2, 3, 4});
  auto B = nda::array<T, 3>::rand({2, 3, 4});

  // exercise both the full-signature and the default-indices overload of elementwise
  auto check = [&](binary_op op, auto const &expected) {
    auto B_d1 = to_addr_space<AS2>(B);
    nda::tensor::elementwise(alpha, to_addr_space<AS1>(A), "abc", beta, B_d1, "abc", op);
    EXPECT_ARRAY_NEAR(nda::to_host(B_d1), expected, fp_tol<T>);

    auto B_d2 = to_addr_space<AS2>(B);
    nda::tensor::elementwise(alpha, to_addr_space<AS1>(A), beta, B_d2, op);
    EXPECT_ARRAY_NEAR(nda::to_host(B_d2), expected, fp_tol<T>);
  };

  // SUM, PROD
  check(binary_op::SUM, alpha * A + beta * B);
  check(binary_op::PROD, (alpha * A) * (beta * B));

  // MAX/MIN: real value types only
  if constexpr (!nda::is_complex_v<T>) {
    check(binary_op::MAX, nda::max(alpha * A, beta * B));
    check(binary_op::MIN, nda::min(alpha * A, beta * B));
  }

  auto A_d = to_addr_space<AS1>(A);
  auto B_d = to_addr_space<AS2>(B);
  if constexpr (on_host) {
    // Abs-family + NORM_2
    check(binary_op::SUM_ABS, nda::abs(alpha * A) + nda::abs(beta * B));
    check(binary_op::MAX_ABS, nda::max(nda::abs(alpha * A), nda::abs(beta * B)));
    check(binary_op::MIN_ABS, nda::min(nda::abs(alpha * A), nda::abs(beta * B)));
    check(binary_op::NORM_2, nda::sqrt(nda::abs2(alpha * A) + nda::abs2(beta * B)));

    // MAX/MIN on complex T throw on the nda host fallback
    if constexpr (nda::is_complex_v<T>) {
      EXPECT_THROW(nda::tensor::elementwise(alpha, A_d, "abc", beta, B_d, "abc", binary_op::MAX), nda::runtime_error);
      EXPECT_THROW(nda::tensor::elementwise(alpha, A_d, "abc", beta, B_d, "abc", binary_op::MIN), nda::runtime_error);
    }

    // mismatched indices throw on the nda fallback
    EXPECT_THROW(nda::tensor::elementwise(alpha, A_d, "abc", beta, B_d, "acb", binary_op::PROD), nda::runtime_error);
  } else {
    // differing index strings work (idx_b is a permutation of idx_a)
    auto B_perm = nda::array<T, 3>::rand({3, 4, 2}); // indexed "bca"
    auto exp    = nda::array<T, 3>::zeros({3, 4, 2});
    nda::for_each(exp.shape(), [&](auto b, auto c, auto a) { exp(b, c, a) = alpha * A(a, b, c) + beta * B_perm(b, c, a); });
    auto B_perm_d = to_addr_space<AS2>(B_perm);
    nda::tensor::elementwise(alpha, to_addr_space<AS1>(A), "abc", beta, B_perm_d, "bca", binary_op::SUM);
    EXPECT_ARRAY_NEAR(nda::to_host(B_perm_d), exp, fp_tol<T>);

    // unsupported op on the device throws
    EXPECT_THROW(nda::tensor::elementwise(alpha, A_d, "abc", beta, B_d, "abc", binary_op::SUM_ABS), nda::runtime_error);
  }
}

template <typename T>
void test_elementwise_on_device() {
  test_elementwise<T, Device, Device>();
  test_elementwise<T, Device, Unified>();
  test_elementwise<T, Unified, Device>();
  test_elementwise<T, Unified, Unified>();
}

template <typename T>
void test_elementwise_on_host() {
  test_elementwise<T, Host, Host>();
#ifdef NDA_HAVE_CUDA
  test_elementwise<T, Host, Unified>();
  test_elementwise<T, Unified, Host>();
#endif // NDA_HAVE_CUDA
}

#ifdef NDA_HAVE_CUTENSOR
TEST(NDA, TensorElementwiseOnDevice) {
  test_elementwise_on_device<float>();
  test_elementwise_on_device<std::complex<float>>();
  test_elementwise_on_device<double>();
  test_elementwise_on_device<std::complex<double>>();
}
#endif // NDA_HAVE_CUTENSOR

TEST(NDA, TensorElementwiseOnHost) {
  test_elementwise_on_host<float>();
  test_elementwise_on_host<std::complex<float>>();
  test_elementwise_on_host<double>();
  test_elementwise_on_host<std::complex<double>>();
}

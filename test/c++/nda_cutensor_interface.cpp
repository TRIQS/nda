// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <cutensor.h>

#include <array>
#include <complex>
#include <iostream>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;

// Test successful linking against cuTENSOR.
TEST(NDA, CUTENSORLinking) {
  auto version = cutensorGetVersion();
  std::cout << "cuTENSOR version: " << version << std::endl;
}

// Test the cuTENSOR permute operation.
template <typename T>
void test_permute() {
  using namespace nda::tensor;

  T alpha = T{3};
  if constexpr (nda::is_complex_v<T>) alpha *= 2 + 1i;

  // identity: B_ij = A_ij
  auto A1_h = nda::array<T, 2>::rand({3, 4});
  auto A1_d = nda::to_device(A1_h);
  auto B1_d = nda::to_device(nda::array<T, 2>::zeros({3, 4}));
  device::permute(T{1}, A1_d, "ij", B1_d, "ij");
  EXPECT_ARRAY_EQ(nda::to_host(B1_d), A1_h);

  // transpose: B_ji = A_ij
  auto B2_d = nda::to_device(nda::array<T, 2>::zeros({4, 3}));
  device::permute(T{1}, A1_d, "ij", B2_d, "ji");
  EXPECT_ARRAY_EQ(nda::to_host(B2_d), nda::transpose(A1_h));

  // scaling: B_ij = alpha * A_ij
  auto B3_d = nda::to_device(nda::array<T, 2, F_layout>::zeros({3, 4}));
  device::permute(alpha, A1_d, "ij", B3_d, "ij");
  EXPECT_ARRAY_NEAR(nda::to_host(B3_d), alpha * A1_h, fp_tol<T>);

  // higher rank permutation: B_kji = A_ijk
  auto A4_h = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto A4_d = nda::to_device(A4_h);
  auto B4_d = nda::to_device(nda::array<T, 3, F_layout>::zeros({4, 2, 3}));
  device::permute(T{1}, A4_d, "ijk", B4_d, "kij");
  auto exp4 = nda::permuted_indices_view<nda::encode(std::array{1, 2, 0})>(A4_h);
  EXPECT_ARRAY_EQ(nda::to_host(B4_d), exp4);

  // higher rank permutation + scaling + unary op
  if constexpr (nda::is_complex_v<T>) {
    // B_jik = alpha * conj(A_ijk)
    auto B5_d = nda::to_device(nda::array<T, 3>::zeros({3, 2, 4}));
    device::permute(alpha, {A4_d, unary_op::CONJ}, "ijk", B5_d, "jik");
    auto exp5 = alpha * nda::conj(nda::permuted_indices_view<nda::encode(std::array{1, 0, 2})>(A4_h));
    EXPECT_ARRAY_NEAR(nda::to_host(B5_d), exp5, fp_tol<T>);
  } else {
    // B_jik = alpha * sqrt(A_ijk)
    auto B5_d = nda::to_device(nda::array<T, 3>::zeros({3, 2, 4}));
    device::permute(alpha, {A4_d, unary_op::SQRT}, "ijk", B5_d, "jik");
    auto exp5 = alpha * nda::sqrt(nda::permuted_indices_view<nda::encode(std::array{1, 0, 2})>(A4_h));
    EXPECT_ARRAY_NEAR(nda::to_host(B5_d), exp5, fp_tol<T>);
  }
}

TEST(NDA, CUTENSORPermute) {
  test_permute<float>();
  test_permute<double>();
  test_permute<std::complex<float>>();
  test_permute<std::complex<double>>();
}

// Test the cuTENSOR elementwise_binary operation.
template <typename T>
void test_elementwise_binary() {
  using namespace nda::tensor;

  T alpha = T{2};
  T gamma = T{3};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 1 + 2i;
    gamma *= 2 - 1i;
  }

  // sum: D_ij = A_ij + C_ij
  auto A1_h = nda::array<T, 2>::rand({3, 4});
  auto C1_h = nda::array<T, 2>::rand({3, 4});
  auto A1_d = nda::to_device(A1_h);
  auto C1_d = nda::to_device(C1_h);
  auto D1_d = nda::to_device(nda::array<T, 2>::zeros({3, 4}));
  device::elementwise_binary(T{1}, A1_d, "ij", T{1}, C1_d, "ij", D1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(D1_d), A1_h + C1_h, fp_tol<T>);

  // in-place sum: C_ij = A_ij + C_ij
  device::elementwise_binary(T{1}, A1_d, "ij", T{1}, C1_d, "ij", C1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C1_d), A1_h + C1_h, fp_tol<T>);

  // weighted sum + broadcast: D_ijk = alpha * A_ij + gamma * C_ijk
  auto A2_h = nda::array<T, 2, F_layout>::rand({3, 4});
  auto C2_h = nda::array<T, 3, F_layout>::rand({3, 4, 2});
  auto A2_d = nda::to_device(A2_h);
  auto C2_d = nda::to_device(C2_h);
  auto D2_d = nda::to_device(nda::array<T, 3, F_layout>::zeros({3, 4, 2}));
  device::elementwise_binary(alpha, A2_d, "ij", gamma, C2_d, "ijk", D2_d);
  auto exp2 = nda::array<T, 3, F_layout>({3, 4, 2});
  for (auto i : nda::range(2)) { exp2(nda::ellipsis{}, i) = alpha * A2_h + gamma * C2_h(nda::ellipsis{}, i); }
  EXPECT_ARRAY_NEAR(nda::to_host(D2_d), exp2, fp_tol<T>);

  // in-place weighted sum + broadcast: C_ijk = alpha * A_ij + gamma * C_ijk
  device::elementwise_binary(alpha, A2_d, "ij", gamma, C2_d, "ijk", C2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C2_d), exp2, fp_tol<T>);

  // weighted product + pertmutation: D_kij = alpha * A_ijk * gamma * C_kij
  auto A3_h = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto C3_h = nda::array<T, 3, F_layout>::rand({4, 2, 3});
  auto A3_d = nda::to_device(A3_h);
  auto C3_d = nda::to_device(C3_h);
  auto D3_d = nda::to_device(nda::array<T, 3, F_layout>::zeros({4, 2, 3}));
  device::elementwise_binary(alpha, A3_d, "ijk", gamma, C3_d, "kij", D3_d, binary_op::PROD);
  auto exp3 = nda::array<T, 3, F_layout>({4, 2, 3});
  nda::for_each(A3_h.shape(),
                [&exp3, &A3_h, &C3_h, alpha, gamma](auto i, auto j, auto k) { exp3(k, i, j) = alpha * A3_h(i, j, k) * gamma * C3_h(k, i, j); });
  EXPECT_ARRAY_NEAR(nda::to_host(D3_d), exp3, fp_tol<T>);

  // in-place weighted product + permutation: C_kij = alpha * A_ijk * gamma * C_kij
  device::elementwise_binary(alpha, A3_d, "ijk", gamma, C3_d, "kij", C3_d, binary_op::PROD);
  EXPECT_ARRAY_NEAR(nda::to_host(C3_d), exp3, fp_tol<T>);

  // with unary ops
  if constexpr (nda::is_complex_v<T>) {
    // out-of-place: D_ij = alpha * conj(A_ij) + gamma * C_ij
    auto A4_h = nda::array<T, 2>::rand({3, 4});
    auto C4_h = nda::array<T, 2>::rand({3, 4});
    auto A4_d = nda::to_device(A4_h);
    auto C4_d = nda::to_device(C4_h);
    auto D4_d = nda::to_device(nda::array<T, 2>::zeros({3, 4}));
    device::elementwise_binary(alpha, {A4_d, unary_op::CONJ}, "ij", gamma, C4_d, "ij", D4_d, binary_op::SUM);
    EXPECT_ARRAY_NEAR(nda::to_host(D4_d), alpha * nda::conj(A4_h) + gamma * C4_h, fp_tol<T>);
  } else {
    // in-place: C_ji = alpha * (-A_ij) + gamma * exp(C_ji)
    auto A4_h = nda::array<T, 2>::rand({3, 4});
    auto C4_h = nda::array<T, 2>::rand({4, 3});
    auto A4_d = nda::to_device(A4_h);
    auto C4_d = nda::to_device(C4_h);
    device::elementwise_binary(alpha, {A4_d, unary_op::NEG}, "ij", gamma, {C4_d, unary_op::EXP}, "ji", C4_d, binary_op::SUM);
    auto exp4 = nda::make_regular(alpha * (-A4_h) + gamma * nda::exp(nda::transpose(C4_h)));
    EXPECT_ARRAY_NEAR(nda::to_host(C4_d), nda::transpose(exp4), fp_tol<T>);
  }
}

TEST(NDA, CUTENSORElementwiseBinary) {
  test_elementwise_binary<float>();
  test_elementwise_binary<double>();
  test_elementwise_binary<std::complex<float>>();
  test_elementwise_binary<std::complex<double>>();
}

// Test the cuTENSOR elementwise_trinary operation.
template <typename T>
void test_elementwise_trinary() {
  using namespace nda::tensor;

  T alpha = T{2};
  T beta  = T{3};
  T gamma = T{4};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 1 + 2i;
    beta *= 2 - 1i;
    gamma *= 1 + 1i;
  }

  // sum: D_ij = A_ij + B_ij + C_ij
  auto A1_h = nda::array<T, 2>::rand({3, 4});
  auto B1_h = nda::array<T, 2>::rand({3, 4});
  auto C1_h = nda::array<T, 2>::rand({3, 4});
  auto A1_d = nda::to_device(A1_h);
  auto B1_d = nda::to_device(B1_h);
  auto C1_d = nda::to_device(C1_h);
  auto D1_d = nda::to_device(nda::array<T, 2>::zeros({3, 4}));
  device::elementwise_trinary(T{1}, A1_d, "ij", T{1}, B1_d, "ij", T{1}, C1_d, "ij", D1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(D1_d), A1_h + B1_h + C1_h, fp_tol<T>);

  // in-place sum: C_ij = A_ij + B_ij + C_ij
  device::elementwise_trinary(T{1}, A1_d, "ij", T{1}, B1_d, "ij", T{1}, C1_d, "ij", C1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C1_d), A1_h + B1_h + C1_h, fp_tol<T>);

  // weighted sum + broadcast: D_ijk = alpha * A_ij + beta * B_ij + gamma * C_ijk
  auto A2_h = nda::array<T, 2, F_layout>::rand({3, 4});
  auto B2_h = nda::array<T, 2, F_layout>::rand({3, 4});
  auto C2_h = nda::array<T, 3, F_layout>::rand({3, 4, 2});
  auto A2_d = nda::to_device(A2_h);
  auto B2_d = nda::to_device(B2_h);
  auto C2_d = nda::to_device(C2_h);
  auto D2_d = nda::to_device(nda::array<T, 3, F_layout>::zeros({3, 4, 2}));
  device::elementwise_trinary(alpha, A2_d, "ij", beta, B2_d, "ij", gamma, C2_d, "ijk", D2_d);
  auto exp2 = nda::array<T, 3, F_layout>({3, 4, 2});
  for (auto i : nda::range(2)) { exp2(nda::ellipsis{}, i) = alpha * A2_h + beta * B2_h + gamma * C2_h(nda::ellipsis{}, i); }
  EXPECT_ARRAY_NEAR(nda::to_host(D2_d), exp2, fp_tol<T>);

  // in-place weighted sum + broadcast: C_ijk = alpha * A_ij + beta * B_ij + gamma * C_ijk
  device::elementwise_trinary(alpha, A2_d, "ij", beta, B2_d, "ij", gamma, C2_d, "ijk", C2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C2_d), exp2, fp_tol<T>);

  // op_AB = "*", op_ABC = "+" + permutation: D_kij = (alpha * A_ijk * beta * B_ijk) + gamma * C_kij
  auto A3_h = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto B3_h = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto C3_h = nda::array<T, 3, F_layout>::rand({4, 2, 3});
  auto A3_d = nda::to_device(A3_h);
  auto B3_d = nda::to_device(B3_h);
  auto C3_d = nda::to_device(C3_h);
  auto D3_d = nda::to_device(nda::array<T, 3, F_layout>::zeros({4, 2, 3}));
  device::elementwise_trinary(alpha, A3_d, "ijk", beta, B3_d, "ijk", gamma, C3_d, "kij", D3_d, binary_op::PROD, binary_op::SUM);
  auto exp3 = nda::array<T, 3, F_layout>({4, 2, 3});
  nda::for_each(A3_h.shape(), [&exp3, &A3_h, &B3_h, &C3_h, alpha, beta, gamma](auto i, auto j, auto k) {
    exp3(k, i, j) = alpha * A3_h(i, j, k) * beta * B3_h(i, j, k) + gamma * C3_h(k, i, j);
  });
  EXPECT_ARRAY_NEAR(nda::to_host(D3_d), exp3, fp_tol<T>);

  // in-place op_AB = "*", op_ABC = "+" + permutation: C_kij = (alpha * A_ijk * beta * B_ijk) + gamma * C_kij
  device::elementwise_trinary(alpha, A3_d, "ijk", beta, B3_d, "ijk", gamma, C3_d, "kij", C3_d, binary_op::PROD, binary_op::SUM);
  EXPECT_ARRAY_NEAR(nda::to_host(C3_d), exp3, fp_tol<T>);

  // with unary ops
  if constexpr (nda::is_complex_v<T>) {
    // out-of-place: D_ij = (alpha * conj(A_ij) + beta * conj(B_ij)) * gamma * C_ij
    auto A4_h = nda::array<T, 2>::rand({3, 4});
    auto B4_h = nda::array<T, 2>::rand({3, 4});
    auto C4_h = nda::array<T, 2>::rand({3, 4});
    auto A4_d = nda::to_device(A4_h);
    auto B4_d = nda::to_device(B4_h);
    auto C4_d = nda::to_device(C4_h);
    auto D4_d = nda::to_device(nda::array<T, 2>::zeros({3, 4}));
    device::elementwise_trinary(alpha, {A4_d, unary_op::CONJ}, "ij", beta, {B4_d, unary_op::CONJ}, "ij", gamma, C4_d, "ij", D4_d, binary_op::SUM,
                                binary_op::PROD);
    auto exp4 = nda::make_regular((alpha * nda::conj(A4_h) + beta * nda::conj(B4_h)) * gamma * C4_h);
    // alpha * beta * gamma inflates these values to ~70, where the absolute fp_tol is under 2 ULP
    auto tol4 = 10 * std::numeric_limits<nda::remove_complex_t<T>>::epsilon() * max_element(abs(exp4));
    EXPECT_ARRAY_NEAR(nda::to_host(D4_d), exp4, tol4);
  } else {
    // in-place: C_ji = (alpha * (-A_ij) + beta * sqrt(B_ij)) * gamma * sin(C_ji)
    auto A4_h = nda::array<T, 2>::rand({3, 4});
    auto B4_h = nda::array<T, 2>::rand({3, 4});
    auto C4_h = nda::array<T, 2>::rand({4, 3});
    auto A4_d = nda::to_device(A4_h);
    auto B4_d = nda::to_device(B4_h);
    auto C4_d = nda::to_device(C4_h);
    device::elementwise_trinary(alpha, {A4_d, unary_op::NEG}, "ij", beta, {B4_d, unary_op::SQRT}, "ij", gamma, {C4_d, unary_op::SIN}, "ji", C4_d,
                                binary_op::SUM, binary_op::PROD);
    auto exp4 = nda::make_regular((alpha * (-A4_h) + beta * nda::sqrt(B4_h)) * gamma * nda::sin(nda::transpose(C4_h)));
    EXPECT_ARRAY_NEAR(nda::to_host(C4_d), nda::transpose(exp4), fp_tol<T>);
  }
}

TEST(NDA, CUTENSORElementwiseTrinary) {
  test_elementwise_trinary<float>();
  test_elementwise_trinary<double>();
  test_elementwise_trinary<std::complex<float>>();
  test_elementwise_trinary<std::complex<double>>();
}

// Test the cuTENSOR reduce operation.
template <typename T>
void test_reduce() {
  using namespace nda::tensor;

  T alpha = T{2};
  T beta  = T{3};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 1 + 2i;
    beta *= 2 - 1i;
  }

  // sum over one index: D_i = alpha * sum_j(A_ij) + beta * C_i
  auto A1_h = nda::array<T, 2>::rand({3, 4});
  auto C1_h = nda::array<T, 1>::rand({3});
  auto A1_d = nda::to_device(A1_h);
  auto C1_d = nda::to_device(C1_h);
  auto D1_d = nda::to_device(nda::array<T, 1>::zeros({3}));
  device::reduce(alpha, A1_d, "ij", beta, C1_d, "i", D1_d);
  auto exp1 = nda::array<T, 1>::zeros({3});
  nda::for_each(exp1.shape(), [&exp1, &A1_h, &C1_h, alpha, beta](auto i) { exp1(i) = alpha * nda::sum(A1_h(i, nda::range::all)) + beta * C1_h(i); });
  EXPECT_ARRAY_NEAR(nda::to_host(D1_d), exp1, fp_tol<T>);

  // in-place sum over one index: C_i = alpha * sum_j(A_ij) + beta * C_i
  device::reduce(alpha, A1_d, "ij", beta, C1_d, "i", C1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C1_d), exp1, fp_tol<T>);

  // sum over two indices: D_i = alpha * sum_jk(A_ijk) + beta * C_i
  auto A2_h = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto C2_h = nda::array<T, 1>::rand({2});
  auto A2_d = nda::to_device(A2_h);
  auto C2_d = nda::to_device(C2_h);
  auto D2_d = nda::to_device(nda::array<T, 1>::zeros({2}));
  device::reduce(alpha, A2_d, "ijk", beta, C2_d, "i", D2_d);
  auto exp2 = nda::array<T, 1>::zeros({2});
  nda::for_each(exp2.shape(), [&exp2, &A2_h, &C2_h, alpha, beta](auto i) { exp2(i) = alpha * nda::sum(A2_h(i, nda::ellipsis{})) + beta * C2_h(i); });
  EXPECT_ARRAY_NEAR(nda::to_host(D2_d), exp2, fp_tol<T>);

  // in-place sum over two indices: C_i = alpha * sum_jk(A_ijk) + beta * C_i
  device::reduce(alpha, A2_d, "ijk", beta, C2_d, "i", C2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C2_d), exp2, fp_tol<T>);

  // sum over one index, keeping two: D_ik = alpha * sum_j(A_ijk) + beta * C_ik
  auto A3_h = nda::array<T, 3>::rand({2, 3, 4});
  auto C3_h = nda::array<T, 2>::rand({2, 4});
  auto A3_d = nda::to_device(A3_h);
  auto C3_d = nda::to_device(C3_h);
  auto D3_d = nda::to_device(nda::array<T, 2>::zeros({2, 4}));
  device::reduce(alpha, A3_d, "ijk", beta, C3_d, "ik", D3_d);
  auto exp3 = nda::array<T, 2>::zeros({2, 4});
  nda::for_each(exp3.shape(), [&](auto i, auto k) { exp3(i, k) = alpha * nda::sum(A3_h(i, nda::range::all, k)) + beta * C3_h(i, k); });
  EXPECT_ARRAY_NEAR(nda::to_host(D3_d), exp3, fp_tol<T>);

  // in-place sum over one index, keeping two: C_ik = alpha * sum_j(A_ijk) + beta * C_ik
  device::reduce(alpha, A3_d, "ijk", beta, C3_d, "ik", C3_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C3_d), exp3, fp_tol<T>);

  // reductions involving unary ops
  if constexpr (nda::is_complex_v<T>) {
    // product + conj: D_i = alpha * prod_j(conj(A_ij)) + beta * C_i
    auto A4_h = nda::array<T, 2>::rand({3, 4});
    auto C4_h = nda::array<T, 1>::rand({3});
    auto A4_d = nda::to_device(A4_h);
    auto C4_d = nda::to_device(C4_h);
    auto D4_d = nda::to_device(nda::array<T, 1>::zeros({3}));
    device::reduce(alpha, {A4_d, unary_op::CONJ}, "ij", beta, C4_d, "i", D4_d, binary_op::PROD);
    auto exp4 = nda::array<T, 1>::zeros({3});
    nda::for_each(exp4.shape(), [&exp4, &A4_h, &C4_h, alpha, beta](auto i) {
      exp4(i) = alpha * nda::product(nda::conj(A4_h(i, nda::range::all))) + beta * C4_h(i);
    });
    EXPECT_ARRAY_NEAR(nda::to_host(D4_d), exp4, fp_tol<T>);
  } else {
    // max + negate: D_i = alpha * max_j(-A_ij) + beta * C_i
    auto A4_h = nda::array<T, 2>::rand({3, 4});
    auto C4_h = nda::array<T, 1>::rand({3});
    auto A4_d = nda::to_device(A4_h);
    auto C4_d = nda::to_device(C4_h);
    auto D4_d = nda::to_device(nda::array<T, 1>::zeros({3}));
    device::reduce(alpha, {A4_d, unary_op::NEG}, "ij", beta, C4_d, "i", D4_d, binary_op::MAX);
    auto exp4 = nda::array<T, 1>::zeros({3});
    nda::for_each(exp4.shape(),
                  [&exp4, &A4_h, &C4_h, alpha, beta](auto i) { exp4(i) = alpha * nda::max_element(-A4_h(i, nda::range::all)) + beta * C4_h(i); });
    EXPECT_ARRAY_NEAR(nda::to_host(D4_d), exp4, fp_tol<T>);
  }

  // full reduction into rank-0 (scalar): D = alpha * sum_ij(A_ij) + beta * C
  auto A5_h = nda::array<T, 2>::rand({3, 4});
  auto C5_h = nda::array<T, 1>::rand({1});
  auto A5_d = nda::to_device(A5_h);
  auto C5_d = nda::to_device(C5_h);
  auto D5_d = nda::to_device(nda::array<T, 1>({T{0}}));
  device::reduce(alpha, A5_d, "ij", beta, C5_d.data(), "", D5_d.data());
  T exp5 = alpha * nda::sum(A5_h) + beta * C5_h(0);
  EXPECT_COMPLEX_NEAR(nda::to_host(D5_d)(0), exp5, fp_tol<T>);

  // in-place full reduction into rank-0 (scalar): C = alpha * sum_ij(A_ij) + beta * C
  device::reduce(alpha, A5_d, "ij", beta, C5_d.data(), "", C5_d.data());
  EXPECT_COMPLEX_NEAR(nda::to_host(C5_d)(0), exp5, fp_tol<T>);
}

TEST(NDA, CUTENSORReduce) {
  test_reduce<float>();
  test_reduce<double>();
  test_reduce<std::complex<float>>();
  test_reduce<std::complex<double>>();
}

// Test the cuTENSOR contract operation.
template <typename T>
void test_contract() {
  using namespace nda::tensor;

  T alpha = T{2};
  T beta  = T{3};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 1 + 2i;
    beta *= 2 - 1i;
  }

  // matrix-matrix multiplication: D_ik = alpha * A_ij * B_jk + beta * C_ik
  auto A1_h = nda::matrix<T>::rand({3, 4});
  auto B1_h = nda::matrix<T>::rand({4, 5});
  auto C1_h = nda::matrix<T>::rand({3, 5});
  auto A1_d = nda::to_device(A1_h);
  auto B1_d = nda::to_device(B1_h);
  auto C1_d = nda::to_device(C1_h);
  auto D1_d = nda::to_device(nda::matrix<T>::zeros({3, 5}));
  device::contract(alpha, A1_d, "ij", B1_d, "jk", beta, C1_d, "ik", D1_d);
  auto exp1 = nda::make_regular(alpha * A1_h * B1_h + beta * C1_h);
  EXPECT_ARRAY_NEAR(nda::to_host(D1_d), exp1, fp_tol<T>);

  // in-place matrix-matrix multiplication: C_ik = alpha * A_ij * B_jk + beta * C_ik
  device::contract(alpha, A1_d, "ij", B1_d, "jk", beta, C1_d, "ik", C1_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C1_d), exp1, fp_tol<T>);

  // matrix-vector multiplication: D_i = alpha * A_ij * B_j + beta * C_i
  auto A2_h = nda::matrix<T>::rand({4, 5});
  auto B2_h = nda::vector<T>::rand({5});
  auto C2_h = nda::vector<T>::rand({4});
  auto A2_d = nda::to_device(A2_h);
  auto B2_d = nda::to_device(B2_h);
  auto C2_d = nda::to_device(C2_h);
  auto D2_d = nda::to_device(nda::vector<T>::zeros({4}));
  device::contract(alpha, A2_d, "ij", B2_d, "j", beta, C2_d, "i", D2_d);
  auto exp2 = nda::make_regular(alpha * A2_h * B2_h + beta * C2_h);
  EXPECT_ARRAY_NEAR(nda::to_host(D2_d), exp2, fp_tol<T>);

  // in-place matrix-vector multiplication: C_i = alpha * A_ij * B_j + beta * C_i
  device::contract(alpha, A2_d, "ij", B2_d, "j", beta, C2_d, "i", C2_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C2_d), exp2, fp_tol<T>);

  // outer product of two vectors: D_ij = alpha * A_i * B_j + beta * C_ij
  auto A3_h = nda::array<T, 1>::rand({3});
  auto B3_h = nda::array<T, 1>::rand({4});
  auto C3_h = nda::array<T, 2>::zeros({3, 4});
  auto A3_d = nda::to_device(A3_h);
  auto B3_d = nda::to_device(B3_h);
  auto C3_d = nda::to_device(C3_h);
  auto D3_d = nda::to_device(nda::array<T, 2>::zeros({3, 4}));
  device::contract(alpha, A3_d, "i", B3_d, "j", T{0}, C3_d, "ij", D3_d);
  auto exp3 = nda::make_regular(alpha * nda::linalg::outer_product(A3_h, B3_h));
  EXPECT_ARRAY_NEAR(nda::to_host(D3_d), exp3, fp_tol<T>);

  // in-place outer product of two vectors: C_ij = alpha * A_i * B_j + beta * C_ij
  device::contract(alpha, A3_d, "i", B3_d, "j", T{0}, C3_d, "ij", C3_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C3_d), exp3, fp_tol<T>);

  // tensor contraction: D_il = alpha * A_ijk * B_jkl + beta * C_il
  auto A4_h = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto B4_h = nda::array<T, 3, F_layout>::rand({3, 4, 5});
  auto C4_h = nda::array<T, 2, F_layout>::rand({2, 5});
  auto A4_d = nda::to_device(A4_h);
  auto B4_d = nda::to_device(B4_h);
  auto C4_d = nda::to_device(C4_h);
  auto D4_d = nda::to_device(nda::array<T, 2, F_layout>::zeros({2, 5}));
  device::contract(alpha, A4_d, "ijk", B4_d, "jkl", beta, C4_d, "il", D4_d);
  auto exp4 = nda::array<T, 2>::zeros({2, 5});
  nda::for_each(exp4.shape(),
                [&](auto i, auto l) { exp4(i, l) = alpha * nda::sum(A4_h(i, nda::ellipsis{}) * B4_h(nda::ellipsis{}, l)) + beta * C4_h(i, l); });
  EXPECT_ARRAY_NEAR(nda::to_host(D4_d), exp4, fp_tol<T>);

  // in-place tensor contraction: C_il = alpha * A_ijk * B_jkl + beta * C_il
  device::contract(alpha, A4_d, "ijk", B4_d, "jkl", beta, C4_d, "il", C4_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C4_d), exp4, fp_tol<T>);

  // batched matmul: D_ikl = alpha * A_ijl * B_jkl + beta * C_ikl
  auto A5_h = nda::array<T, 3, F_layout>::rand({3, 4, 2});
  auto B5_h = nda::array<T, 3, F_layout>::rand({4, 5, 2});
  auto C5_h = nda::array<T, 3, F_layout>::rand({3, 5, 2});
  auto A5_d = nda::to_device(A5_h);
  auto B5_d = nda::to_device(B5_h);
  auto C5_d = nda::to_device(C5_h);
  auto D5_d = nda::to_device(nda::array<T, 3, F_layout>::zeros({3, 5, 2}));
  device::contract(alpha, A5_d, "ijl", B5_d, "jkl", beta, C5_d, "ikl", D5_d);
  auto exp5 = C5_h;
  nda::blas::gemm_batch_strided(alpha, A5_h, B5_h, beta, exp5);
  EXPECT_ARRAY_NEAR(nda::to_host(D5_d), exp5, fp_tol<T> * 10);

  // in-place batched matmul: C_ikl = alpha * A_ijl * B_jkl + beta * C_ikl
  device::contract(alpha, A5_d, "ijl", B5_d, "jkl", beta, C5_d, "ikl", C5_d);
  EXPECT_ARRAY_NEAR(nda::to_host(C5_d), exp5, fp_tol<T> * 10);

  // contractions involving unary ops
  if constexpr (nda::is_complex_v<T>) {
    // D_ik = alpha * conj(A_ij) * nda::conj(B_jk) + beta * C_ik (conj(C_ik) is ignored by cuTENSOR)
    auto A6_h = nda::matrix<T>::rand({3, 4});
    auto B6_h = nda::matrix<T>::rand({4, 5});
    auto C6_h = nda::matrix<T>::rand({3, 5});
    auto A6_d = nda::to_device(A6_h);
    auto B6_d = nda::to_device(B6_h);
    auto C6_d = nda::to_device(C6_h);
    auto D6_d = nda::to_device(nda::matrix<T>::zeros({3, 5}));
    device::contract(alpha, {A6_d, unary_op::CONJ}, "ij", {B6_d, unary_op::CONJ}, "jk", beta, C6_d, "ik", D6_d);
    auto exp6 = nda::make_regular(alpha * nda::conj(A6_h) * nda::conj(B6_h) + beta * C6_h);
    EXPECT_ARRAY_NEAR(nda::to_host(D6_d), exp6, fp_tol<T>);

    // D_ik = conj(C_ik) (conj(C_ik) works here as expected)
    device::contract(T{0}, A6_d, "ij", B6_d, "jk", T{1}, {C6_d, unary_op::CONJ}, "ik", D6_d);
    EXPECT_ARRAY_NEAR(nda::to_host(D6_d), nda::conj(C6_h), fp_tol<T>);
  } else {
    // C_ik = alpha * A_ij * B_jk + beta * C_ik (only IDENTITY seems to be allowed)
    auto A6_h = nda::matrix<T>::rand({3, 4});
    auto B6_h = nda::matrix<T>::rand({4, 5});
    auto C6_h = nda::matrix<T>::rand({3, 5});
    auto A6_d = nda::to_device(A6_h);
    auto B6_d = nda::to_device(B6_h);
    auto C6_d = nda::to_device(C6_h);
    device::contract(alpha, A6_d, "ij", B6_d, "jk", beta, C6_d, "ik", C6_d);
    auto exp6 = nda::make_regular(alpha * A6_h * B6_h + beta * C6_h);
    EXPECT_ARRAY_NEAR(nda::to_host(C6_d), exp6, fp_tol<T>);
  }

  // full contraction into rank-0 (scalar): D = alpha * sum_ij(A_ij * B_ij) + beta * C
  auto A7_h = nda::array<T, 2>::rand({3, 4});
  auto B7_h = nda::array<T, 2>::rand({3, 4});
  auto C7_h = nda::array<T, 1>({T{5}});
  auto A7_d = nda::to_device(A7_h);
  auto B7_d = nda::to_device(B7_h);
  auto C7_d = nda::to_device(C7_h);
  auto D7_d = nda::to_device(nda::array<T, 1>::zeros({1}));
  device::contract(alpha, A7_d, "ij", B7_d, "ij", beta, C7_d.data(), "", D7_d.data());
  T exp7 = alpha * nda::sum(A7_h * B7_h) + beta * C7_h(0);
  EXPECT_COMPLEX_NEAR(nda::to_host(D7_d)(0), exp7, fp_tol<T>);

  // in-place full contraction into rank-0 (scalar): C = alpha * sum_ij(A_ij * B_ij) + beta * C
  device::contract(alpha, A7_d, "ij", B7_d, "ij", beta, C7_d.data(), "", C7_d.data());
  EXPECT_COMPLEX_NEAR(nda::to_host(C7_d)(0), exp7, fp_tol<T>);
}

TEST(NDA, CUTENSORContract) {
  test_contract<float>();
  test_contract<double>();
  test_contract<std::complex<float>>();
  test_contract<std::complex<double>>();
}

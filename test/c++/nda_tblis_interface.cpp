// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>
#include <tblis/tblis.h>

#include <cmath>
#include <complex>
#include <concepts>
#include <type_traits>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;

// Test various typedefs of TBLIS.
TEST(NDA, TBLISTypes) {
  static_assert(std::same_as<::tblis::len_type, long>);
  static_assert(std::same_as<::tblis::stride_type, long>);
  static_assert(std::same_as<::tblis::label_type, char>);
}

// Test the construction of tensor_view objects.
TEST(NDA, TensorViewConstruction) {
  using namespace nda::tensor;

  auto check_tensor_view = [](auto &&a, auto &&tv, unary_op exp_op = unary_op::IDENTITY) {
    using array_t = std::remove_cvref_t<decltype(a)>;
    static_assert(std::same_as<decltype(a.data()), decltype(tv.data)>);
    EXPECT_EQ(tv.ndim, nda::get_rank<array_t>);
    EXPECT_EQ(tv.data, a.data());
    EXPECT_EQ(tv.op, exp_op);
    for (int i = 0; i < nda::get_rank<array_t>; ++i) {
      EXPECT_EQ(tv.extents[i], a.extent(i));
      EXPECT_EQ(tv.strides[i], a.indexmap().strides()[i]);
    }
  };

  // C-layout double array (rank 3)
  auto A1 = nda::array<double, 3>{2, 3, 4};
  check_tensor_view(A1, tensor_view(A1));

  // F-layout float array (rank 2)
  auto A2 = nda::array<float, 2, F_layout>{3, 4};
  check_tensor_view(A2, tensor_view(A2));

  // non-contiguous complex<double> array view
  auto A3 = nda::array<std::complex<double>, 2>{4, 5};
  auto v3 = A3(nda::range::all, nda::range(1, 4));
  check_tensor_view(v3, tensor_view(v3));

  // complex<float> vector (rank 1)
  auto A4 = nda::vector<std::complex<float>>{10};
  check_tensor_view(A4, tensor_view(A4));

  // conjugate complex<double> expression
  auto A5 = nda::array<std::complex<double>, 2>{3, 4};
  check_tensor_view(A5, tensor_view(nda::conj(A5)), unary_op::CONJ);

  // conjugate complex<float> expression
  auto A6 = nda::array<std::complex<float>, 2>{3, 4};
  check_tensor_view(A6(nda::range::all, 0), tensor_view(nda::conj(A6(nda::range::all, 0))), unary_op::CONJ);

  // const array
  const auto A7 = nda::array<double, 2>{3, 4};
  check_tensor_view(A7, tensor_view(A7));

  // implicit conversion from array in a function call
  auto A8 = nda::array<double, 2>{3, 4};
  auto fn = [&check_tensor_view](tensor_view<double> tv, auto &&a_ref, unary_op op) { check_tensor_view(a_ref, tv, op); };
  fn(A8, A8, unary_op::IDENTITY);
  fn({A8, unary_op::COS}, A8, unary_op::COS);

  // rank-0 tensor_view from a scalar pointer
  double a9 = 42.0;
  auto tv9  = tensor_view(&a9);
  static_assert(std::same_as<decltype(tv9), tensor_view<double>>);
  EXPECT_EQ(tv9.data, &a9);
  EXPECT_EQ(tv9.ndim, 0);
  EXPECT_EQ(tv9.extents, nullptr);
  EXPECT_EQ(tv9.strides, nullptr);
  EXPECT_EQ(tv9.op, unary_op::IDENTITY);

  // rank-0 tensor_view from a const scalar pointer
  const float a10 = 3.14f;
  auto tv10       = tensor_view(&a10);
  static_assert(std::same_as<decltype(tv10), tensor_view<const float>>);
  EXPECT_EQ(tv10.data, &a10);
  EXPECT_EQ(tv10.ndim, 0);

  // rank-0 tensor_view from a complex scalar pointer
  std::complex<double> a11{1.0, 2.0};
  auto tv11 = tensor_view(&a11);
  static_assert(std::same_as<decltype(tv11), tensor_view<std::complex<double>>>);
  EXPECT_EQ(tv11.data, &a11);
  EXPECT_EQ(tv11.ndim, 0);
}

// Test constructibility and convertibility of tensor_view objects.
TEST(NDA, TensorViewConversions) {
  using namespace nda::tensor;

  // non-const array deduces tensor_view<T>
  auto A = nda::array<std::complex<double>, 2>{3, 4};
  static_assert(std::same_as<decltype(tensor_view(A)), tensor_view<std::complex<double>>>);
  static_assert(std::same_as<decltype(tensor_view(A, unary_op::ABS)), tensor_view<std::complex<double>>>);

  // const array deduces tensor_view<const T> = const_tensor_view<T>
  const auto cA = A();
  static_assert(std::same_as<decltype(tensor_view(cA)), tensor_view<const std::complex<double>>>);
  static_assert(std::same_as<decltype(tensor_view(cA)), const_tensor_view<std::complex<double>>>);
  static_assert(std::same_as<decltype(tensor_view(cA, unary_op::NEG)), tensor_view<const std::complex<double>>>);

  // conjugate of a non-const array still deduces tensor_view<T> (non-const data pointer)
  static_assert(std::same_as<decltype(tensor_view(nda::conj(A))), tensor_view<std::complex<double>>>);

  // conjugate of a const array deduces tensor_view<const T>
  static_assert(std::same_as<decltype(tensor_view(nda::conj(cA))), const_tensor_view<std::complex<double>>>);

  // tensor_view<T> is constructible from a non-const array
  static_assert(std::constructible_from<tensor_view<std::complex<double>>, decltype(A)>);
  static_assert(std::constructible_from<tensor_view<std::complex<double>>, decltype(nda::conj(A))>);

  // tensor_view<T> is not constructible from a const array
  static_assert(!std::constructible_from<tensor_view<std::complex<double>>, decltype(cA)>);
  static_assert(!std::constructible_from<tensor_view<std::complex<double>>, decltype(nda::conj(cA))>);

  // const_tensor_view<T> is constructible from both const and non-const arrays
  static_assert(std::constructible_from<const_tensor_view<std::complex<double>>, decltype(A)>);
  static_assert(std::constructible_from<const_tensor_view<std::complex<double>>, decltype(cA)>);
  static_assert(std::constructible_from<const_tensor_view<std::complex<double>>, decltype(nda::conj(A))>);
  static_assert(std::constructible_from<const_tensor_view<std::complex<double>>, decltype(nda::conj(cA))>);

  // tensor_view<T> is implicitly convertible to const_tensor_view<T>.
  static_assert(std::convertible_to<tensor_view<double>, const_tensor_view<double>>);
  static_assert(std::convertible_to<tensor_view<float>, const_tensor_view<float>>);
  static_assert(std::convertible_to<tensor_view<std::complex<double>>, const_tensor_view<std::complex<double>>>);
  static_assert(std::convertible_to<tensor_view<std::complex<float>>, const_tensor_view<std::complex<float>>>);

  // const_tensor_view<T> is not convertible back to tensor_view<T>.
  static_assert(!std::convertible_to<const_tensor_view<double>, tensor_view<double>>);

  // cross type conversions should not work
  static_assert(!std::convertible_to<tensor_view<float>, tensor_view<double>>);
  static_assert(!std::convertible_to<tensor_view<float>, const_tensor_view<double>>);
  static_assert(!std::convertible_to<tensor_view<double>, const_tensor_view<float>>);

  // verify data preservation through conversion
  auto tv  = tensor_view<std::complex<double>>(A, unary_op::FLOOR);
  auto ctv = const_tensor_view<std::complex<double>>(tv); // implicit conversion
  EXPECT_EQ(ctv.data, tv.data);
  EXPECT_EQ(ctv.extents, tv.extents);
  EXPECT_EQ(ctv.strides, tv.strides);
  EXPECT_EQ(ctv.ndim, tv.ndim);
  EXPECT_EQ(ctv.op, tv.op);

  // implicit conversion in function calls
  auto takes_const_tv = [](const_tensor_view<std::complex<double>> v) { return v.data; };
  EXPECT_EQ(takes_const_tv(tv), A.data());                    // tensor_view<T> -> const_tensor_view<T>
  EXPECT_EQ(takes_const_tv(A), A.data());                     // array -> const_tensor_view<T>
  EXPECT_EQ(takes_const_tv(cA), cA.data());                   // const array -> const_tensor_view<T>
  EXPECT_EQ(takes_const_tv(nda::conj(A)), A.data());          // conj(non-const array) -> const_tensor_view<T>
  EXPECT_EQ(takes_const_tv({A, unary_op::ACOSH}), cA.data()); // (array, op) -> const_tensor_view<T>

  // rank-0 (scalar) tensor_view construction and conversion via pointer
  double a = 1.0;

  // tensor_view<T> is constructible from a non-const scalar pointer
  static_assert(std::constructible_from<tensor_view<double>, double *>);
  static_assert(std::constructible_from<tensor_view<std::complex<float>>, std::complex<float> *>);

  // tensor_view<T> is not constructible from const scalar pointer
  static_assert(!std::constructible_from<tensor_view<float>, const float *>);
  static_assert(!std::constructible_from<tensor_view<std::complex<double>>, const std::complex<double> *>);

  // const_tensor_view<T> is constructible from both const and non-const scalar pointers
  static_assert(std::constructible_from<const_tensor_view<double>, double *>);
  static_assert(std::constructible_from<const_tensor_view<double>, const double *>);

  // scalar tensor_view<T> implicitly converts to const_tensor_view<T>
  auto takes_scalar_const_tv = [](const_tensor_view<double> v) { return v.data; };
  auto stv                   = tensor_view<double>(&a);
  EXPECT_EQ(takes_scalar_const_tv(stv), &a);
}

// Test the TBLIS set function.
template <typename T>
void test_set() {
  T alpha = T{3};
  if constexpr (nda::is_complex_v<T>) alpha *= 2 + 1i;

  // full matrix: A_ij = alpha
  auto A1   = nda::array<T, 2, C_layout>::zeros({3, 4});
  auto exp1 = A1;
  exp1      = alpha;
  nda::tensor::tblis::set(alpha, A1, "ij");
  EXPECT_ARRAY_EQ(A1, exp1);

  // diagonal of matrix: A_ii = alpha
  auto A2   = nda::matrix<T>::zeros({4, 4});
  auto exp2 = A2;
  exp2      = alpha;
  nda::tensor::tblis::set(alpha, A2, "ii");
  EXPECT_ARRAY_EQ(A2, exp2);

  // full rank-3 array: A_ijk = alpha
  auto A3   = nda::array<T, 3, F_layout>::zeros({2, 3, 4});
  auto exp3 = A3;
  exp3      = alpha;
  nda::tensor::tblis::set(alpha, A3, "ijk");
  EXPECT_ARRAY_EQ(A3, exp3);

  // non-contiguous nda view: A_ij = alpha for i=1..3, j=2..4
  auto A4     = nda::array<T, 2, C_layout>::zeros({5, 6});
  auto exp4   = A4;
  auto exp4_v = exp4(nda::range(1, 4), nda::range(2, 5));
  exp4_v      = alpha;
  nda::tensor::tblis::set(alpha, A4(nda::range(1, 4), nda::range(2, 5)), "ij");
  EXPECT_ARRAY_EQ(A4, exp4);

  // partial rank-3 array: A_iji = alpha
  auto A5   = nda::array<T, 3>::zeros({4, 2, 4});
  auto exp5 = A5;
  for (auto i : nda::range(exp5.extent(1))) {
    auto v = nda::matrix_view<T>(exp5(nda::range::all, i, nda::range::all));
    v      = alpha;
  }
  nda::tensor::tblis::set(alpha, A5, "iji");
  EXPECT_ARRAY_EQ(A5, exp5);

  // scalar: a = alpha
  T a6 = T(0);
  nda::tensor::tblis::set(alpha, &a6, "");
  EXPECT_EQ(a6, alpha);
}

TEST(NDA, TBLISSet) {
  test_set<float>();
  test_set<std::complex<float>>();
  test_set<double>();
  test_set<std::complex<double>>();
}

// Test the TBLIS scale function.
template <typename T>
void test_scale() {
  T alpha = T{3};
  if constexpr (nda::is_complex_v<T>) alpha *= 2 + 1i;

  // matrix: A_ij = alpha * A_ij
  auto A1   = nda::array<T, 2, C_layout>::rand({3, 4});
  auto exp1 = A1;
  nda::tensor::tblis::scale(alpha, A1, "ij");
  EXPECT_ARRAY_NEAR(A1, exp1 * alpha, fp_tol<T>);

  // rank-3 array: A_ijk = alpha * A_ijk
  auto A2   = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto exp2 = A2;
  nda::tensor::tblis::scale(alpha, A2, "ijk");
  EXPECT_ARRAY_NEAR(A2, exp2 * alpha, fp_tol<T>);

  // non-contiguous nda view: A_ij = alpha * A_ij for i=1..3, j=2..4
  auto A3     = nda::array<T, 2, C_layout>::rand({5, 6});
  auto exp3   = A3;
  auto exp3_v = exp3(nda::range(1, 4), nda::range(2, 5));
  exp3_v *= alpha;
  nda::tensor::tblis::scale(alpha, A3(nda::range(1, 4), nda::range(2, 5)), "ij");
  EXPECT_ARRAY_NEAR(A3, exp3, fp_tol<T>);

  // TBLIS view: A_ijkj = alpha * A_ijkj
  auto A4   = nda::array<T, 4>::rand({2, 4, 2, 4});
  auto exp4 = A4;
  for (auto i : nda::range(exp4.extent(0))) {
    for (auto j : nda::range(exp4.extent(2))) {
      auto v = nda::matrix_view<T>(exp4(i, nda::range::all, j, nda::range::all));
      nda::diagonal(v) *= alpha;
    }
  }
  nda::tensor::tblis::scale(alpha, A4, "ijkj");
  EXPECT_ARRAY_NEAR(A4, exp4, fp_tol<T>);

  // conjugate expression (only for complex types)
  if constexpr (nda::is_complex_v<T>) {
    // A_ij = alpha * conj(A_ij)
    auto A5   = nda::array<T, 2, C_layout>::rand({3, 4});
    auto exp5 = nda::make_regular(nda::conj(A5));
    exp5 *= alpha;
    nda::tensor::tblis::scale(alpha, nda::conj(A5), "ij");
    EXPECT_ARRAY_NEAR(A5, exp5, fp_tol<T>);

    // A_ij = alpha * conj(A_ij) for i=1..3, j=2..4
    auto A6     = nda::array<T, 2, C_layout>::rand({5, 6});
    auto exp6   = A6;
    auto exp6_v = exp6(nda::range(1, 4), nda::range(2, 5));
    exp6_v      = alpha * nda::conj(A6(nda::range(1, 4), nda::range(2, 5)));
    nda::tensor::tblis::scale(alpha, nda::conj(A6(nda::range(1, 4), nda::range(2, 5))), "ij");
    EXPECT_ARRAY_NEAR(A6, exp6, fp_tol<T>);
  }

  // scalar: a = alpha * a
  T a7      = T(7);
  auto exp7 = a7 * alpha;
  nda::tensor::tblis::scale(alpha, &a7, "");
  EXPECT_EQ(a7, exp7);
}

TEST(NDA, TBLISScale) {
  test_scale<float>();
  test_scale<std::complex<float>>();
  test_scale<double>();
  test_scale<std::complex<double>>();
}

// Test the TBLIS reduce function.
template <typename T>
void test_reduce() {
  using nda::tensor::binary_op;

  auto A = nda::array<T, 2, C_layout>::rand({4, 4});
  auto B = nda::array<T, 3, F_layout>::rand({4, 3, 4});

  // full matrix + sum: a = sum_ij A_ij
  auto res = nda::tensor::tblis::reduce(binary_op::SUM, A, "ij");
  EXPECT_COMPLEX_NEAR(res, nda::sum(A), fp_tol<T>);

  // full rank-3 array + abs sum = sum_ijk |B_ijk|
  res = nda::tensor::tblis::reduce(binary_op::SUM_ABS, B, "ijk");
  EXPECT_COMPLEX_NEAR(res, nda::sum(nda::abs(B)), fp_tol<T> * 10);

  // diagonal of matrix + max abs: a = max_i |A_ii|
  res = nda::tensor::tblis::reduce(binary_op::MAX_ABS, A, "ii");
  EXPECT_EQ(res, nda::max_element(nda::abs(nda::diagonal(A))));

  // TBLIS view + L2 norm: a = sqrt(sum_ij |B_iji|^2)
  res   = nda::tensor::tblis::reduce(binary_op::NORM_2, B, "iji");
  T exp = 0;
  for (auto i : nda::range(B.extent(1))) {
    auto v = B(nda::range::all, i, nda::range::all);
    for (auto x : nda::diagonal(v)) exp += std::abs(x) * std::abs(x);
  }
  EXPECT_COMPLEX_NEAR(res, std::sqrt(exp), fp_tol<T>);

  // special unary ops
  if constexpr (!nda::is_complex_v<T>) {
    // nda view + max: a = max_ij A_2ij
    res = nda::tensor::tblis::reduce(binary_op::MAX, A(2, nda::ellipsis{}), "ij");
    EXPECT_EQ(res, nda::max_element(A(2, nda::ellipsis{})));

    // rank-3 array + min: a = min_ijk B_ijk
    res = nda::tensor::tblis::reduce(binary_op::MIN, B, "ijk");
    EXPECT_EQ(res, nda::min_element(B));
  } else {
    // matrix view + sum + conj: a = sum_ij conj(A_ij)
    res = nda::tensor::tblis::reduce(binary_op::SUM, nda::conj(A(nda::range(1, 3), nda::ellipsis{})), "ijk");
    EXPECT_COMPLEX_NEAR(res, nda::sum(nda::conj(A(nda::range(1, 3), nda::ellipsis{}))), fp_tol<T>);
  }

  // scalar: a = c
  T c = T(42);
  res = nda::tensor::tblis::reduce(binary_op::SUM, &c, "");
  EXPECT_EQ(res, c);
}

TEST(NDA, TBLISReduce) {
  test_reduce<float>();
  test_reduce<std::complex<float>>();
  test_reduce<double>();
  test_reduce<std::complex<double>>();
}

// Test the TBLIS dot function.
template <typename T>
void test_dot() {
  // vector dot product: a = u_i v_i
  auto u1 = nda::vector<T>::rand({5});
  auto v1 = nda::vector<T>::rand({5});
  EXPECT_COMPLEX_NEAR(nda::tensor::tblis::dot(u1, "i", v1, "i"), nda::linalg::dot(u1, v1), fp_tol<T>);

  // contraction of matrices: a = A_ij B_ij
  auto A2 = nda::array<T, 2, C_layout>::rand({3, 4});
  auto B2 = nda::array<T, 2, C_layout>::rand({3, 4});
  T exp2  = 0;
  nda::for_each(A2.shape(), [&exp2, &A2, &B2](auto i, auto j) { exp2 += A2(i, j) * B2(i, j); });
  EXPECT_COMPLEX_NEAR(nda::tensor::tblis::dot(A2, "ij", B2, "ij"), exp2, fp_tol<T>);

  // contraction of matrices + permutation: a = A_ij B_ji
  auto A3 = nda::array<T, 2, C_layout>::rand({3, 4});
  auto B3 = nda::array<T, 2, F_layout>::rand({4, 3});
  T exp3  = 0;
  nda::for_each(A3.shape(), [&exp3, &A3, &B3](auto i, auto j) { exp3 += A3(i, j) * B3(j, i); });
  EXPECT_COMPLEX_NEAR(nda::tensor::tblis::dot(A3, "ij", B3, "ji"), exp3, fp_tol<T>);

  // contraction of rank-3 arrays: a = A_ijk B_ijk
  auto A4 = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto B4 = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  T exp4  = 0;
  nda::for_each(A4.shape(), [&exp4, &A4, &B4](auto i, auto j, auto k) { exp4 += A4(i, j, k) * B4(i, j, k); });
  EXPECT_COMPLEX_NEAR(nda::tensor::tblis::dot(A4, "ijk", B4, "ijk"), exp4, fp_tol<T>);

  // contraction of nda views: a = A_ij2 B_j4i
  auto A5   = nda::array<T, 3, C_layout>::rand({4, 5, 6});
  auto B5   = nda::array<T, 3, C_layout>::rand({5, 6, 4});
  auto A5_v = A5(nda::ellipsis{}, 2);
  auto B5_v = B5(nda::range::all, 4, nda::range::all);
  T exp5    = 0;
  nda::for_each(A5_v.shape(), [&exp5, &A5_v, &B5_v](auto i, auto j) { exp5 += A5_v(i, j) * B5_v(j, i); });
  auto res5 = nda::tensor::tblis::dot(A5(nda::ellipsis{}, 2), "ij", B5(nda::range::all, 4, nda::range::all), "ji");
  EXPECT_COMPLEX_NEAR(res5, exp5, fp_tol<T>);

  // contraction of TBLIS view and a vector: a = A_ii B_i
  auto A6 = nda::array<T, 2>::rand({4, 4});
  auto B6 = nda::vector<T>::rand({4});
  EXPECT_COMPLEX_NEAR(nda::tensor::tblis::dot(A6, "ii", B6, "i"), nda::linalg::dot(nda::diagonal(A6), B6), fp_tol<T>);

  // scalars: c = a * b
  T a7 = T(3);
  T b7 = T(7);
  EXPECT_EQ(nda::tensor::tblis::dot(&a7, "", &b7, ""), a7 * b7);

  // dot products involving conjugate expressions (only for complex types)
  if constexpr (nda::is_complex_v<T>) {
    // contraction of matrices + conj: a = conj(A_ij) B_ij
    auto A8 = nda::array<T, 2, C_layout>::rand({3, 4});
    auto B8 = nda::array<T, 2, C_layout>::rand({3, 4});
    T exp8  = 0;
    nda::for_each(A8.shape(), [&exp8, &A8, &B8](auto i, auto j) { exp8 += std::conj(A8(i, j)) * B8(i, j); });
    EXPECT_COMPLEX_NEAR(nda::tensor::tblis::dot(nda::conj(A8), "ij", B8, "ij"), exp8, fp_tol<T>);

    // vector dotc product: a = conj(u_i) v_i
    auto u9 = nda::vector<T>::rand({6});
    auto v9 = nda::vector<T>::rand({6});
    EXPECT_COMPLEX_NEAR(nda::tensor::tblis::dot(nda::conj(u9), "i", v9, "i"), nda::linalg::dotc(u9, v9), fp_tol<T>);
  }
}

TEST(NDA, TBLISDot) {
  test_dot<float>();
  test_dot<std::complex<float>>();
  test_dot<double>();
  test_dot<std::complex<double>>();
}

// Test the TBLIS add function.
template <typename T>
void test_add() {
  T alpha = T{3};
  T beta  = T{2};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 2 + 1i;
    beta *= 1 - 1i;
  }

  // matrix + matrix: B = alpha * A + beta * B
  auto A1   = nda::array<T, 2, C_layout>::rand({3, 4});
  auto B1   = nda::array<T, 2, C_layout>::rand({3, 4});
  auto exp1 = nda::make_regular(alpha * A1 + beta * B1);
  nda::tensor::tblis::add(alpha, A1, "ij", beta, B1, "ij");
  EXPECT_ARRAY_NEAR(B1, exp1, fp_tol<T>);

  // rank-3 array + rank-3 array: B = alpha * A + beta * B
  auto A2   = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto B2   = nda::array<T, 3, F_layout>::rand({2, 3, 4});
  auto exp2 = nda::make_regular(alpha * A2 + beta * B2);
  nda::tensor::tblis::add(alpha, A2, "ijk", beta, B2, "ijk");
  EXPECT_ARRAY_NEAR(B2, exp2, fp_tol<T>);

  // matrix + transpose matrix: B_ji = alpha * A_ij + beta * B_ji
  auto A3   = nda::array<T, 2, C_layout>::rand({3, 4});
  auto B3   = nda::array<T, 2, F_layout>::rand({4, 3});
  auto exp3 = nda::make_regular(alpha * nda::transpose(A3) + beta * B3);
  nda::tensor::tblis::add(alpha, A3, "ij", beta, B3, "ji");
  EXPECT_ARRAY_NEAR(B3, exp3, fp_tol<T>);

  // scale matrix (beta = 0): B = alpha * A
  auto A4   = nda::array<T, 2, F_layout>::rand({3, 4});
  auto B4   = nda::array<T, 2, C_layout>::rand({3, 4});
  auto exp4 = nda::make_regular(alpha * A4);
  nda::tensor::tblis::add(alpha, A4, "ij", T(0), B4, "ij");
  EXPECT_ARRAY_NEAR(B4, exp4, fp_tol<T>);

  // nda view + nda view: B_ij = alpha * A_ij + beta * B_ij for i=1..3, j=2..4
  auto A5     = nda::array<T, 2, C_layout>::rand({5, 6});
  auto B5     = nda::array<T, 2, C_layout>::rand({5, 6});
  auto exp5   = B5;
  auto exp5_v = exp5(nda::range(1, 4), nda::range(2, 5));
  exp5_v      = alpha * A5(nda::range(1, 4), nda::range(2, 5)) + beta * exp5_v;
  nda::tensor::tblis::add(alpha, A5(nda::range(1, 4), nda::range(2, 5)), "ij", beta, B5(nda::range(1, 4), nda::range(2, 5)), "ij");
  EXPECT_ARRAY_NEAR(B5, exp5, fp_tol<T>);

  // TBLIS view + TBLIS view: B_iij = alpha * A_iji + beta * B_iij
  auto A6   = nda::array<T, 3>::rand({4, 3, 4});
  auto B6   = nda::array<T, 3>::rand({4, 4, 3});
  auto exp6 = B6;
  for (auto i : nda::range(A6.extent(1))) {
    auto tmp_rhs = nda::make_regular(alpha * A6(nda::range::all, i, nda::range::all) + beta * B6(nda::ellipsis{}, i));
    auto tmp_lhs = nda::diagonal(exp6(nda::ellipsis{}, i));
    tmp_lhs      = nda::diagonal(tmp_rhs);
  }
  nda::tensor::tblis::add(alpha, A6, "iji", beta, B6, "iij");
  EXPECT_ARRAY_NEAR(B6, exp6, fp_tol<T>);

  // add involving conjugate expressions (only for complex types)
  if constexpr (nda::is_complex_v<T>) {
    // conj matrix + matrix: B = alpha * conj(A) + beta * B
    auto A7   = nda::array<T, 2, C_layout>::rand({3, 4});
    auto B7   = nda::array<T, 2, C_layout>::rand({3, 4});
    auto exp7 = nda::make_regular(alpha * nda::conj(A7) + beta * B7);
    nda::tensor::tblis::add(alpha, nda::conj(A7), "ij", beta, B7, "ij");
    EXPECT_ARRAY_NEAR(B7, exp7, fp_tol<T>);

    // add with conjugate destination: B = alpha * A + beta * conj(B)
    auto A8   = nda::array<T, 2, C_layout>::rand({3, 4});
    auto B8   = nda::array<T, 2, C_layout>::rand({3, 4});
    auto exp8 = nda::array<T, 2, C_layout>::zeros({3, 4});
    nda::for_each(exp8.shape(), [&](auto i, auto j) { exp8(i, j) = alpha * A8(i, j) + beta * std::conj(B8(i, j)); });
    nda::tensor::tblis::add(alpha, A8, "ij", beta, nda::conj(B8), "ij");
    EXPECT_ARRAY_NEAR(B8, exp8, fp_tol<T>);
  }

  // rank-0 (scalar) + rank-0 (scalar): b = alpha * a + beta * b
  T a9      = T(5);
  T b9      = T(7);
  auto exp9 = alpha * a9 + beta * b9;
  nda::tensor::tblis::add(alpha, &a9, "", beta, &b9, "");
  EXPECT_COMPLEX_NEAR(b9, exp9, fp_tol<T>);
}

TEST(NDA, TBLISAdd) {
  test_add<float>();
  test_add<std::complex<float>>();
  test_add<double>();
  test_add<std::complex<double>>();
}

// Test the TBLIS mult function.
template <typename T>
void test_mult() {
  T alpha = T{3};
  T beta  = T{2};
  if constexpr (nda::is_complex_v<T>) {
    alpha *= 2 + 1i;
    beta *= 1 - 1i;
  }

  // matrix-matrix multiplication: C_ik = alpha * A_ij * B_jk + beta * C_ik
  auto A1   = nda::matrix<T>::rand({3, 4});
  auto B1   = nda::matrix<T>::rand({4, 5});
  auto C1   = nda::matrix<T>::rand({3, 5});
  auto exp1 = nda::make_regular(alpha * A1 * B1 + beta * C1);
  nda::tensor::tblis::mult(alpha, A1, "ij", B1, "jk", beta, C1, "ik");
  EXPECT_ARRAY_NEAR(C1, exp1, fp_tol<T>);

  // matrix-vector multiplication: C_i = alpha * A_ij * B_j + beta * C_i
  auto A2   = nda::matrix<T, F_layout>::rand({4, 5});
  auto B2   = nda::vector<T>::rand({5});
  auto C2   = nda::vector<T>::rand({4});
  auto exp2 = nda::make_regular(alpha * A2 * B2 + beta * C2);
  nda::tensor::tblis::mult(alpha, A2, "ij", B2, "j", beta, C2, "i");
  EXPECT_ARRAY_NEAR(C2, exp2, fp_tol<T>);

  // outer product of two vectors: C_ij = alpha * A_i * B_j + beta * C_ij
  T beta3   = T(0);
  auto A3   = nda::vector<T>::rand({3});
  auto B3   = nda::vector<T>::rand({4});
  auto C3   = nda::array<T, 2, C_layout>::zeros({3, 4});
  auto exp3 = nda::make_regular(alpha * nda::linalg::outer_product(A3, B3));
  nda::tensor::tblis::mult(alpha, A3, "i", B3, "j", beta3, C3, "ij");
  EXPECT_ARRAY_NEAR(C3, exp3, fp_tol<T>);

  // tensor contraction: C_il = alpha * A_ijk * B_jkl + beta * C_il
  auto A4   = nda::array<T, 3, C_layout>::rand({2, 3, 4});
  auto B4   = nda::array<T, 3, C_layout>::rand({3, 4, 5});
  auto C4   = nda::array<T, 2, C_layout>::rand({2, 5});
  auto exp4 = nda::array<T, 2, C_layout>::zeros({2, 5});
  nda::for_each(exp4.shape(), [&](auto i, auto l) {
    T sum = 0;
    for (auto j : nda::range(3))
      for (auto k : nda::range(4)) sum += A4(i, j, k) * B4(j, k, l);
    exp4(i, l) = alpha * sum + beta * C4(i, l);
  });
  nda::tensor::tblis::mult(alpha, A4, "ijk", B4, "jkl", beta, C4, "il");
  EXPECT_ARRAY_NEAR(C4, exp4, fp_tol<T>);

  // matrix-matrix multiplication with nda views: C_ik = alpha * A_ij * B_jk + beta * C_ik
  auto A6_full = nda::array<T, 2, C_layout>::rand({5, 6});
  auto B6_full = nda::array<T, 2, C_layout>::rand({6, 7});
  auto C6      = nda::array<T, 2, C_layout>::rand({3, 4});
  auto A6      = A6_full(nda::range(1, 4), nda::range::all);
  auto B6      = B6_full(nda::range::all, nda::range(2, 6));
  auto exp6    = nda::make_regular(alpha * nda::linalg::matmul(A6, B6) + beta * C6);
  nda::tensor::tblis::mult(alpha, A6, "ij", B6, "jk", beta, C6, "ik");
  EXPECT_ARRAY_NEAR(C6, exp6, fp_tol<T> * 10);

  // batched matmul: C_ikl = alpha * A_ijl * B_jkl + beta * C_ikl
  auto A7   = nda::array<T, 3, F_layout>::rand({3, 4, 2});
  auto B7   = nda::array<T, 3, F_layout>::rand({4, 5, 2});
  auto C7   = nda::array<T, 3, F_layout>::rand({3, 5, 2});
  auto exp7 = C7;
  nda::blas::gemm_batch_strided(alpha, A7, B7, beta, exp7);
  nda::tensor::tblis::mult(alpha, A7, "ijl", B7, "jkl", beta, C7, "ikl");
  EXPECT_ARRAY_NEAR(C7, exp7, fp_tol<T> * 10);

  // contractions involving conjugate expressions (only for complex types)
  if constexpr (nda::is_complex_v<T>) {
    // C_ij = alpha * conj(A_ij) * B_ij + beta * C_ij
    auto A8   = nda::matrix<T>::rand({3, 4});
    auto B8   = nda::matrix<T>::rand({4, 5});
    auto C8   = nda::matrix<T, F_layout>::rand({3, 5});
    auto exp8 = nda::make_regular(alpha * nda::conj(A8) * B8 + beta * C8);
    nda::tensor::tblis::mult(alpha, nda::conj(A8), "ij", B8, "jk", beta, C8, "ik");
    EXPECT_ARRAY_NEAR(C8, exp8, fp_tol<T> * 10);

    // C_ij = alpha * conj(A_ij) * conj(B_ij) + beta * C_ij
    auto A9   = nda::matrix<T>::rand({3, 4});
    auto B9   = nda::matrix<T>::rand({4, 5});
    auto C9   = nda::matrix<T, F_layout>::rand({3, 5});
    auto exp9 = nda::make_regular(alpha * nda::conj(A9) * nda::conj(B9) + beta * C9);
    nda::tensor::tblis::mult(alpha, nda::conj(A9), "ij", nda::conj(B9), "jk", beta, C9, "ik");
    EXPECT_ARRAY_NEAR(C9, exp9, fp_tol<T> * 10);

    // C_ij = alpha * A_ij * B_ij + beta * conj(C_ij)
    auto A10   = nda::matrix<T>::rand({3, 4});
    auto B10   = nda::matrix<T>::rand({4, 5});
    auto C10   = nda::matrix<T>::rand({3, 5});
    auto exp10 = nda::make_regular(alpha * A10 * B10 + beta * nda::conj(C10));
    nda::tensor::tblis::mult(alpha, A10, "ij", B10, "jk", beta, nda::conj(C10), "ik");
    EXPECT_ARRAY_NEAR(C10, exp10, fp_tol<T> * 10);
  }

  // full contraction into rank-0 (scalar): c = alpha * sum_ij(A_ij * B_ij) + beta * c
  auto A11   = nda::array<T, 2, C_layout>::rand({3, 4});
  auto B11   = nda::array<T, 2, C_layout>::rand({3, 4});
  T c11      = T(5);
  auto exp11 = alpha * nda::sum(A11 * B11) + beta * c11;
  nda::tensor::tblis::mult(alpha, A11, "ij", B11, "ij", beta, &c11, "");
  EXPECT_COMPLEX_NEAR(c11, exp11, fp_tol<T>);
}

TEST(NDA, TBLISMult) {
  test_mult<float>();
  test_mult<std::complex<float>>();
  test_mult<double>();
  test_mult<std::complex<double>>();
}

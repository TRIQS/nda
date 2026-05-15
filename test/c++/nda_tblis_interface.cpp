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

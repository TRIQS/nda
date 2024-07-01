// Copyright (c) 2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <complex>
#include <functional>

TEST(NDA, ArrayOnesRandAndZeros) {
  auto const shape = shape_t<3>{2, 3, 4};
  using lay_t      = nda::contiguous_layout_with_stride_order<nda::encode(std::array{1, 2, 0})>;

  // zeros, ones and rand
  auto A_zero_i = nda::array<int, 3>::zeros(shape);
  auto A_zero_c = nda::array<std::complex<double>, 3, nda::F_layout>::zeros(shape);
  auto A_zero_f = nda::array<foo_def_ctor, 3, lay_t>::zeros(shape);
  auto A_ones_i = nda::array<int, 3>::ones(shape);
  auto A_ones_c = nda::array<std::complex<double>, 3, nda::F_layout>::ones(shape);
  auto A_ones_d = nda::array<double, 3, lay_t>::ones(shape);
  auto A_rand_c = nda::array<std::complex<double>, 3, nda::F_layout>::rand(shape);
  auto A_rand_d = nda::array<double, 3, lay_t>::rand(shape);
  nda::for_each(shape, [&](auto i, auto j, auto k) {
    EXPECT_EQ(A_zero_i(i, j, k), 0);
    EXPECT_EQ(A_zero_c(i, j, k), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(A_zero_f(i, j, k), foo_def_ctor(0));
    EXPECT_EQ(A_ones_i(i, j, k), 1);
    EXPECT_EQ(A_ones_c(i, j, k), std::complex<double>(1.0, 0.0));
    EXPECT_EQ(A_ones_d(i, j, k), 1.0);
    EXPECT_TRUE(A_rand_c(i, j, k).real() >= 0.0);
    EXPECT_TRUE(A_rand_c(i, j, k).real() < 1.0);
    EXPECT_TRUE(A_rand_c(i, j, k).imag() >= 0.0);
    EXPECT_TRUE(A_rand_c(i, j, k).imag() < 1.0);
    EXPECT_TRUE(A_rand_d(i, j, k) >= 0.0);
    EXPECT_TRUE(A_rand_d(i, j, k) < 1.0);
  });
}

TEST(NDA, ArrayResize) {
  // 3D array
  auto const shape_3d = shape_t<3>{2, 3, 4};
  auto const size_3d  = nda::stdutil::product(shape_3d);

  // resize to a new shape
  nda::array<int, 3> B;
  auto ptr_B = B.data();
  EXPECT_EQ(B.size(), 0);
  EXPECT_EQ(B.data(), ptr_B);
  B.resize(shape_3d);
  EXPECT_EQ(B.size(), size_3d);
  EXPECT_EQ(B.shape(), shape_3d);
  EXPECT_NE(B.data(), ptr_B);
  ptr_B = B.data();

  // resize to the same shape (no actual resizing is done)
  B.resize(shape_3d);
  EXPECT_EQ(B.data(), ptr_B);

  // 3D array
  auto const shape_1d = shape_t<1>{10};
  auto const size_1d  = shape_1d[0];

  // resize to a new shape
  nda::array<int, 1> A;
  auto ptr_A = A.data();
  EXPECT_EQ(A.size(), 0);
  EXPECT_EQ(A.data(), ptr_A);
  A.resize(shape_1d);
  EXPECT_EQ(A.size(), size_1d);
  EXPECT_EQ(A.shape(), shape_1d);
  EXPECT_NE(A.data(), ptr_A);
  ptr_A = A.data();

  // resize to the same shape (no actual resizing is done)
  A.resize(shape_1d);
  EXPECT_EQ(A.data(), ptr_A);
}

TEST(NDA, ArrayWithNonDefaultConstructibleType) {
  // use array_adapter to initialize an array with non-default-constructible type
  nda::array<foo_non_def_ctor, 2> A{nda::array_adapter{std::array{2, 2}, [](int i, int j) { return i + 10 * j; }}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) { EXPECT_EQ(A(i, j).x, i + 10 * j); }

  // copy construct a new array
  auto B = A;
  EXPECT_EQ(B, A);
  B = foo_non_def_ctor{0};
  EXPECT_NE(B, A);

  // copy assign a new array
  B = A;
  EXPECT_EQ(B, A);

  // taking a view
  auto A_v = A();
  EXPECT_EQ(A_v, A);

  // resizing does not compile
  // A.resize(3, 3);

  // construct an 1D array with an initializer list
  nda::array<foo_non_def_ctor, 1> C_1d{12, 3};
  EXPECT_EQ(C_1d(0).x, 12);
  EXPECT_EQ(C_1d(1).x, 3);

  // construct a 2d array with an initializer list
  nda::array<foo_non_def_ctor, 2> C_2d{{12, 3}, {34, 67}};
  EXPECT_EQ(C_2d(0, 0).x, 12);
  EXPECT_EQ(C_2d(0, 1).x, 3);
  EXPECT_EQ(C_2d(1, 0).x, 34);
  EXPECT_EQ(C_2d(1, 1).x, 67);
}

TEST(NDA, ArrayWithNonCopyableType) {
  nda::array<foo_non_copy, 2> A(2, 2), B(2, 2);
  nda::for_each(A.shape(), [&](auto i, auto j) {
    A(i, j).x = static_cast<int>(i * 2 + j);
    B(i, j).x = A(i, j).x;
  });
  EXPECT_EQ(A, B);

  // copy construction does not compile
  // auto A_copy = A;

  // move construction
  auto A_mv = std::move(A);
  EXPECT_EQ(A_mv, B);

  // resizing doest not compile
  // A_mv.resize(3, 3);

  // vector of array of non-copyable type
  std::vector<nda::array<foo_non_copy, 1>> vec(2);
  vec.emplace_back(2);
}

TEST(NDA, ArrayOfAnArray) {
  nda::array<nda::array<int, 1>, 2> A(2, 2);

  // set each element of A to a 1D array
  nda::array<int, 1> A_sub{1, 2, 3};
  A() = A_sub;

  // check the elements
  nda::for_each(A.shape(), [&](auto i, auto j) { EXPECT_EQ(A(i, j), A_sub); });
}

TEST(NDA, ArrayOfFunctions) {
  nda::array<std::function<double(double)>, 2> F(2, 2);

  // initialize the array
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      auto s  = i + j;
      F(i, j) = [s](int k) { return k + s; };
    }
  }

  // check the array
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(F(i, j)(0), i + j);
}

TEST(NDA, StdSwapArray) {
  auto V = nda::array<long, 1>{3, 3, 3};
  auto W = nda::array<long, 1>{4, 4, 4, 4};

  std::swap(V, W);

  EXPECT_EQ(V, (nda::array<long, 1>{4, 4, 4, 4}));
  EXPECT_EQ(W, (nda::array<long, 1>{3, 3, 3}));

  // std::swap for views does not compile
  // auto V_v = V(nda::range(0, 2));
  // auto W_v = W(nda::range(0, 2));
  // std::swap(V_v, W_v);
}

TEST(NDA, ArrayBoundsChecking) {
  nda::array<long, 2> A(2, 3);
  EXPECT_THROW(A(0, 3), std::runtime_error);
  EXPECT_THROW(A(nda::range(0, 4), 2), std::runtime_error);
  EXPECT_THROW(A(nda::range(10, 14), 2), std::runtime_error);
  EXPECT_THROW(A(nda::range::all, 5), std::runtime_error);
  EXPECT_THROW(A(0, 3), std::runtime_error);
}

TEST(NDA, ConstructArrayWithInitializerList) {
  auto pts = nda::array<std::array<double, 2>, 1>{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  EXPECT_EQ(pts.shape(), (std::array<long, 1>{4}));
}

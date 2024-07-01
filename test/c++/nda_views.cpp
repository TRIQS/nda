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

#include <vector>

TEST(NDA, RebindView) {
  nda::array<int, 1> A{1, 2, 3, 4};
  auto A_v = A.as_array_view();
  nda::array<int, 1> B{5, 6, 7, 8};
  auto B_v = B.as_array_view();

  // rebind
  EXPECT_NE(A_v, B_v);
  B_v.rebind(A_v);
  EXPECT_EQ(A_v, B_v);
  EXPECT_EQ(A_v.data(), B_v.data());
}

TEST(NDA, RebindConstView) {
  nda::array<long, 2> A{{1, 2}, {3, 4}};
  nda::array<long, 2> B{{10, 20}, {30, 40}};
  auto const &A_cref = A;

  auto A_v = A_cref();
  EXPECT_EQ_ARRAY(A_v, A);
  A_v.rebind(B());
  EXPECT_EQ_ARRAY(A_v, B);

  // rebinding a non-const view to a const view does not compile
  // auto A_v2 = A();
  // A_v2.rebind(A_v);
}

TEST(NDA, SwapAndDeepSwapViews) {
  nda::array<int, 1> A{1, 2, 3, 4};
  auto A_v = A.as_array_view();
  nda::array<int, 1> B{5, 6, 7, 8};
  auto B_v = B.as_array_view();

  // swap
  swap(A_v, B_v);
  EXPECT_EQ(A_v.data(), B.data());
  EXPECT_EQ(A_v, B);
  EXPECT_EQ(B_v.data(), A.data());
  EXPECT_EQ(B_v, A);

  // deep swap
  deep_swap(A_v, B_v);
  EXPECT_EQ(A, (nda::array<int, 1>{5, 6, 7, 8}));
  EXPECT_EQ(B_v, A);
  EXPECT_EQ(B, (nda::array<int, 1>{1, 2, 3, 4}));
  EXPECT_EQ(A_v, B);
  deep_swap(A_v(nda::range(2)), B_v(nda::range(1, 3)));
  EXPECT_EQ(A, (nda::array<int, 1>{5, 1, 2, 8}));
  EXPECT_EQ(B, (nda::array<int, 1>{6, 7, 3, 4}));
}

TEST(NDA, ViewFromRawPointer) {
  // create a view of a std::vector using a pointer to its data
  std::vector<long> vec(10, 3);
  nda::array_view<long const, 2> vec_v({3, 3}, vec.data());
  EXPECT_EQ(vec_v.shape(), (shape_t<2>{3, 3}));
  for (auto x : vec_v) EXPECT_EQ(x, 3);
  // vec_v(1,1) = 19; // does and should not compile
}

TEST(NDA, AddTwoStdVectorsUsingViews) {
  // add two vectors by using views
  std::vector<long> vec1(10), vec2(10), res(10, -1);
  for (int i = 0; i < 10; ++i) {
    vec1[i] = i;
    vec2[i] = 10l * i;
  }

  nda::array_view<long const, 2> vec1_v({3, 3}, vec1.data());
  nda::array_view<long const, 2> vec2_v({3, 3}, vec2.data());
  nda::array_view<long, 2> res_v({3, 3}, res.data());

  res_v = vec1_v + vec2_v;

  for (int i = 0; i < 9; ++i) EXPECT_EQ(res[i], 11 * i);
  EXPECT_EQ(res[9], -1);
}

TEST(NDA, ArithmeticOperationsOnViews) {
  nda::array<double, 1> A{1, 2, 3, 4, 5};

  A *= 2;
  EXPECT_ARRAY_NEAR(A, (nda::array<double, 1>{2, 4, 6, 8, 10}));

  A[nda::range(2, 4)] /= 2.0;
  EXPECT_ARRAY_NEAR(A, (nda::array<double, 1>{2, 4, 3, 4, 10}));

  A[nda::range(0, 5, 2)] *= 10;
  EXPECT_ARRAY_NEAR(A, (nda::array<double, 1>{20, 4, 30, 4, 100}));
}

TEST(NDA, ViewsWithEllipsis) {
  nda::array<long, 3> A(2, 3, 4);
  A() = 7;
  EXPECT_ARRAY_NEAR(A(0, nda::ellipsis{}), A(0, nda::range::all, nda::range::all), 1.e-15);

  nda::array<long, 4> B(2, 3, 4, 5);
  B() = 8;
  EXPECT_ARRAY_NEAR(B(0, nda::ellipsis{}, 3), B(0, nda::range::all, nda::range::all, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(0, nda::ellipsis{}, 2, 3), B(0, nda::range::all, 2, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(nda::ellipsis{}, 2, 3), B(nda::range::all, nda::range::all, 2, 3), 1.e-15);
}

TEST(NDA, ViewsWithEmptyEllipsis) {
  nda::array<long, 4> B(2, 3, 4, 5);
  B() = 8;

  EXPECT_ARRAY_NEAR(B(1, 2, 3, nda::range::all), B(1, 2, 3, nda::range::all, nda::ellipsis{}), 1.e-15);
  EXPECT_ARRAY_NEAR(B(1, 2, 3, nda::range::all), B(1, 2, nda::ellipsis{}, 3, nda::range::all), 1.e-15);
  EXPECT_ARRAY_NEAR(B(1, 2, 3, nda::range::all), B(1, nda::ellipsis{}, 2, 3, nda::range::all), 1.e-15);
  EXPECT_ARRAY_NEAR(B(1, 2, 3, nda::range::all), B(nda::ellipsis{}, 1, 2, 3, nda::range::all), 1.e-15);

  EXPECT_NEAR(B(1, 2, 3, 4), B(1, 2, 3, 4, nda::ellipsis{}), 1.e-15);
  EXPECT_NEAR(B(1, 2, 3, 4), B(nda::ellipsis{}, 1, 2, 3, 4), 1.e-15);
  EXPECT_NEAR(B(1, 2, 3, 4), B(1, nda::ellipsis{}, 2, 3, 4), 1.e-15);
}

TEST(NDA, ArithmeticWithEllipsis) {
  auto sum0 = [](auto const &A) {
    nda::array<nda::get_value_t<decltype(A)>, nda::get_rank<decltype(A)> - 1> res = A(0, nda::ellipsis{});
    for (size_t u = 1; u < A.shape()[0]; ++u) res += A(u, nda::ellipsis{});
    return res;
  };

  nda::array<double, 2> A(5, 2);
  A() = 2;
  nda::array<double, 3> B(5, 2, 3);
  B() = 3;
  EXPECT_ARRAY_NEAR(sum0(A), nda::array<double, 1>{10, 10}, 1.e-15);
  EXPECT_ARRAY_NEAR(sum0(B), nda::array<double, 2>{{15, 15, 15}, {15, 15, 15}}, 1.e-15);
}

TEST(NDA, SharedView) {
  using v_t = nda::basic_array_view<double, 2, nda::C_layout, 'A', nda::default_accessor, nda::shared>;

  // create array to be viewed
  nda::array<double, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = i * 8.1 + 2.31 * j;

  // create shared view and bind to an object that goes out of scope
  v_t A_v;
  {
    auto B = A;
    A_v.rebind(v_t{B});
  }

  // check that the view is still valid
  EXPECT_EQ_ARRAY(A_v, A);
}

TEST(NDA, ConstView) {
  nda::array<long, 2> A(2, 3);
  A() = 98;

  auto f2 = [](nda::array_view<long, 2> const &) {};
  f2(A());

  // assigning to a const view does not compile
  // nda::array_const_view<long, 2> B{A()};
  // B(0, 0) = 2;
  // B()(0, 0) = 2;
  // B(nda::range(0, 2), nda::range(0, 2)) = 2;

  // assigning to a const array does not compile
  // const nda::array<long, 1> C = {1, 2, 3, 4};
  // C(0) = 2; // this compiles but it shouldn't
  // C()(0) = 2;
  // C(nda::range(0, 2))(0) = 10; // this compiles but it shouldn't
}

TEST(NDA, ViewToStringstream) {
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;
  nda::array_view<long, 2> A_v(A);

  std::stringstream fs_A, fs_A_v;
  fs_A << A;
  fs_A_v << A_v;
  EXPECT_EQ(fs_A.str(), fs_A_v.str());
}

TEST(NDA, 1DSlicesOf2DView) {
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  nda::array_view<long, 1> slice1(A(0, nda::range(0, 3)));
  nda::array_view<long, 1> slice2(A(1, nda::range(0, 2)));
  nda::array_view<long, 1> slice3(A(1, nda::range(1, 3)));
  nda::array_view<long, 1> slice4(A(nda::range(0, 2), 0));
  nda::array_view<long, 1> slice5(A(nda::range(0, 2), 1));

  EXPECT_EQ_ARRAY(slice1, (nda::array<long, 1>{0, 1, 2}));
  EXPECT_EQ_ARRAY(slice2, (nda::array<long, 1>{10, 11}));
  EXPECT_EQ_ARRAY(slice3, (nda::array<long, 1>{11, 12}));
  EXPECT_EQ_ARRAY(slice4, (nda::array<long, 1>{0, 10}));
  EXPECT_EQ_ARRAY(slice5, (nda::array<long, 1>{1, 11}));
}

TEST(NDA, ViewsWithDifferentMemoryLayouts) {
  nda::array<long, 3> A_c(2, 3, 4);
  nda::array<long, 3, nda::F_layout> A_f(2, 3, 4);
  nda::array<long, 3, nda::basic_layout<0, nda::encode(std::array{0, 1, 2}), nda::layout_prop_e::contiguous>> A1(2, 3, 4);
  nda::array<long, 3, nda::basic_layout<0, nda::encode(std::array{2, 1, 0}), nda::layout_prop_e::contiguous>> A2(2, 3, 4);
  nda::array<long, 3, nda::basic_layout<0, nda::encode(std::array{2, 0, 1}), nda::layout_prop_e::contiguous>> A3(2, 3, 4);
  nda::array<long, 3, nda::basic_layout<0, nda::encode(std::array{1, 2, 0}), nda::layout_prop_e::contiguous>> A4(2, 3, 4);

  // check index maps of the arrays
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ((A_c.indexmap()(i, j, k)), (3 * 4) * i + 4 * j + k);
        EXPECT_EQ((A1.indexmap()(i, j, k)), (3 * 4) * i + 4 * j + k);
        EXPECT_EQ((A_f.indexmap()(i, j, k)), i + 2 * j + (2 * 3) * k);
        EXPECT_EQ((A2.indexmap()(i, j, k)), i + 2 * j + (2 * 3) * k);
        EXPECT_EQ((A3.indexmap()(i, j, k)), 3 * i + j + (2 * 3) * k);
        EXPECT_EQ((A4.indexmap()(i, j, k)), i + (2 * 4) * j + 2 * k);
      }
    }
  }

  // fill array and check different slices
  auto check_slices = [](auto &A) {
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k) A(i, j, k) = 100 * (i + 1) + 10 * (j + 1) + (k + 1);

    EXPECT_EQ_ARRAY(A(0, nda::range::all, nda::range::all), (nda::array<long, 2>{{111, 112, 113, 114}, {121, 122, 123, 124}, {131, 132, 133, 134}}));
    EXPECT_EQ_ARRAY(A(1, nda::range::all, nda::range::all), (nda::array<long, 2>{{211, 212, 213, 214}, {221, 222, 223, 224}, {231, 232, 233, 234}}));
    EXPECT_EQ_ARRAY(A(nda::range::all, 0, nda::range::all), (nda::array<long, 2>{{111, 112, 113, 114}, {211, 212, 213, 214}}));
    EXPECT_EQ_ARRAY(A(nda::range::all, nda::range::all, 1), (nda::array<long, 2>{{112, 122, 132}, {212, 222, 232}}));
    EXPECT_EQ_ARRAY(A(nda::range::all, 0, 1), (nda::array<long, 1>{112, 212}));
  };

  check_slices(A_c);
  check_slices(A2);
  check_slices(A3);
  check_slices(A4);
  check_slices(A1);
  check_slices(A_f);
}

// Copyright (c) 2020-2023 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <iterator>

// Checks the layout of a given slice.
void check_slice_layout(auto view) {
  // check compile-time layout guarantees
  bool is_contiguous          = (view.indexmap().layout_prop == nda::layout_prop_e::contiguous);
  bool is_strided_1d          = has_strided_1d(view.indexmap().layout_prop);
  bool smallest_stride_is_one = has_smallest_stride_is_one(view.indexmap().layout_prop);
  EXPECT_EQ(is_contiguous, (is_strided_1d and smallest_stride_is_one));

  // use a basic array iterator without any optimization
  using layout_t   = typename decltype(view)::layout_t;
  using iterator_t = nda::array_iterator<layout_t::rank(), long const, typename nda::default_accessor::template accessor<long>::pointer>;
  auto it          = iterator_t{nda::permutations::apply(layout_t::stride_order, view.indexmap().lengths()),
                       nda::permutations::apply(layout_t::stride_order, view.indexmap().strides()), view.data(), false};
  auto it_end      = iterator_t{nda::permutations::apply(layout_t::stride_order, view.indexmap().lengths()),
                           nda::permutations::apply(layout_t::stride_order, view.indexmap().strides()), view.data(), true};

  // loop over all elements and check the strides
  bool strided_1d  = true;
  auto current_val = *it;
  auto stride      = *(std::next(it)) - current_val;
  for (; it != it_end; ++it) {
    strided_1d &= (*it == current_val);
    current_val += stride;
  }
  bool contiguous = strided_1d and (stride == 1);

  EXPECT_EQ(strided_1d, is_strided_1d);
  EXPECT_EQ(contiguous, is_contiguous);
  EXPECT_EQ(view.indexmap().is_contiguous(), is_contiguous);
  EXPECT_EQ(view.indexmap().is_strided_1d(), is_strided_1d);
};

// Check the layout of slices taken from a 3D array with a given stride order.
template <auto StrideOrder>
void test3d() {
  using arr_t = nda::array<long, 3, nda::basic_layout<0, StrideOrder, nda::layout_prop_e::contiguous>>;

  int n0 = 3, n1 = 4, n2 = 5;
  arr_t A(n0, n1, n2);

  // initialize the array and check that it is traversed in the correct order
  for (long c = 0; auto &x : A) {
    EXPECT_EQ(std::distance(&(*std::begin(A)), &x), c);
    x = c++;
  }

  // 3D slice
  check_slice_layout(A(nda::range::all, nda::range::all, nda::range::all));

  // 2D slice
  check_slice_layout(A(nda::range::all, nda::range::all, 2));
  check_slice_layout(A(nda::range::all, 2, nda::range::all));
  check_slice_layout(A(2, nda::range::all, nda::range::all));

  // 1D slice
  check_slice_layout(A(nda::range::all, 2, 2));
  check_slice_layout(A(2, nda::range::all, 2));
  check_slice_layout(A(2, 2, nda::range::all));
}

// Check the layout of slices taken from a 4D array with a given stride order.
template <auto StrideOrder>
void test4d() {
  using arr_t = nda::array<long, 4, nda::basic_layout<0, StrideOrder, nda::layout_prop_e::contiguous>>;

  int n0 = 3, n1 = 4, n2 = 5, n3 = 6;
  arr_t A(n0, n1, n2, n3);

  // initialize the array and check that it is traversed in the correct order
  for (long c = 0; auto &x : A) {
    EXPECT_EQ(std::distance(&(*std::begin(A)), &x), c);
    x = c++;
  }

  check_slice_layout(A(nda::range::all, nda::range::all, nda::range::all, nda::range::all));

  check_slice_layout(A(nda::range::all, nda::range::all, nda::range::all, 2));
  check_slice_layout(A(nda::range::all, nda::range::all, 2, nda::range::all));
  check_slice_layout(A(nda::range::all, 2, nda::range::all, nda::range::all));
  check_slice_layout(A(1, nda::range::all, nda::range::all, nda::range::all));

  check_slice_layout(A(nda::range::all, nda::range::all, 2, 2));
  check_slice_layout(A(nda::range::all, 2, nda::range::all, 2));
  check_slice_layout(A(nda::range::all, 2, 2, nda::range::all));
  check_slice_layout(A(1, nda::range::all, nda::range::all, 2));
  check_slice_layout(A(1, nda::range::all, 2, nda::range::all));
  check_slice_layout(A(1, 2, nda::range::all, nda::range::all));

  check_slice_layout(A(nda::range::all, 2, 2, 2));
  check_slice_layout(A(1, nda::range::all, 2, 2));
  check_slice_layout(A(1, 2, nda::range::all, 2));
  check_slice_layout(A(1, 2, 2, nda::range::all));
}

// Check the layout of slices taken from a 5D array with a given stride order.
template <auto StrideOrder>
void test5d() {
  using arr_t = nda::array<long, 5, nda::basic_layout<0, StrideOrder, nda::layout_prop_e::contiguous>>;

  int n0 = 3, n1 = 4, n2 = 5, n3 = 6, n4 = 7;
  arr_t A(n0, n1, n2, n3, n4);

  // initialize the array and check that it is traversed in the correct order
  for (long c = 0; auto &x : A) {
    EXPECT_EQ(std::distance(&(*std::begin(A)), &x), c);
    x = c++;
  }

  check_slice_layout(A(nda::range::all, nda::range::all, nda::range::all, nda::range::all, nda::range::all));

  check_slice_layout(A(nda::range::all, nda::range::all, nda::range::all, nda::range::all, 2));
  check_slice_layout(A(nda::range::all, nda::range::all, nda::range::all, 2, nda::range::all));
  check_slice_layout(A(nda::range::all, nda::range::all, 2, nda::range::all, nda::range::all));
  check_slice_layout(A(nda::range::all, 2, nda::range::all, nda::range::all, nda::range::all));
  check_slice_layout(A(1, nda::range::all, nda::range::all, nda::range::all, nda::range::all));

  check_slice_layout(A(nda::range::all, nda::range::all, nda::range::all, 2, 2));
  check_slice_layout(A(nda::range::all, nda::range::all, 2, nda::range::all, 2));
  check_slice_layout(A(nda::range::all, nda::range::all, 2, 2, nda::range::all));
  check_slice_layout(A(nda::range::all, 2, nda::range::all, nda::range::all, 2));
  check_slice_layout(A(nda::range::all, 2, nda::range::all, 2, nda::range::all));
  check_slice_layout(A(nda::range::all, 2, 2, nda::range::all, nda::range::all));
  check_slice_layout(A(1, nda::range::all, nda::range::all, nda::range::all, 2));
  check_slice_layout(A(1, 2, nda::range::all, nda::range::all, nda::range::all));
  check_slice_layout(A(1, nda::range::all, 2, nda::range::all, nda::range::all));
  check_slice_layout(A(1, nda::range::all, nda::range::all, 2, nda::range::all));

  check_slice_layout(A(nda::range::all, nda::range::all, 2, 2, 2));
  check_slice_layout(A(nda::range::all, 2, nda::range::all, 2, 2));
  check_slice_layout(A(nda::range::all, 2, 2, nda::range::all, 2));
  check_slice_layout(A(nda::range::all, 2, 2, 2, nda::range::all));
  check_slice_layout(A(1, 2, 2, nda::range::all, nda::range::all));
  check_slice_layout(A(1, 2, nda::range::all, nda::range::all, 2));
  check_slice_layout(A(1, 2, nda::range::all, 2, nda::range::all));
  check_slice_layout(A(1, nda::range::all, 2, nda::range::all, 2));
  check_slice_layout(A(1, nda::range::all, 2, 2, nda::range::all));
  check_slice_layout(A(1, nda::range::all, nda::range::all, 2, 2));

  check_slice_layout(A(nda::range::all, 2, 2, 2, 2));
  check_slice_layout(A(1, 2, 2, nda::range::all, 2));
  check_slice_layout(A(1, 2, 2, 2, nda::range::all));
  check_slice_layout(A(1, 2, nda::range::all, 2, 2));
  check_slice_layout(A(1, nda::range::all, 2, 2, 2));
}

TEST(NDA, SliceLayoutOf3DArray) {
  // test all possible stride orders
  test3d<nda::encode(std::array{0, 1, 2})>();
  test3d<nda::encode(std::array{1, 0, 2})>();
  test3d<nda::encode(std::array{2, 0, 1})>();
  test3d<nda::encode(std::array{0, 2, 1})>();
  test3d<nda::encode(std::array{1, 2, 0})>();
  test3d<nda::encode(std::array{2, 1, 0})>();
}

TEST(NDA, SliceLayoutOf4DArray) {
  // test all possible stride orders
  test4d<nda::encode(std::array{0, 1, 2, 3})>();
  test4d<nda::encode(std::array{1, 0, 2, 3})>();
  test4d<nda::encode(std::array{2, 0, 1, 3})>();
  test4d<nda::encode(std::array{0, 2, 1, 3})>();
  test4d<nda::encode(std::array{1, 2, 0, 3})>();
  test4d<nda::encode(std::array{2, 1, 0, 3})>();
  test4d<nda::encode(std::array{2, 1, 3, 0})>();
  test4d<nda::encode(std::array{1, 2, 3, 0})>();
  test4d<nda::encode(std::array{3, 2, 1, 0})>();
  test4d<nda::encode(std::array{2, 3, 1, 0})>();
  test4d<nda::encode(std::array{1, 3, 2, 0})>();
  test4d<nda::encode(std::array{3, 1, 2, 0})>();
  test4d<nda::encode(std::array{3, 0, 2, 1})>();
  test4d<nda::encode(std::array{0, 3, 2, 1})>();
  test4d<nda::encode(std::array{2, 3, 0, 1})>();
  test4d<nda::encode(std::array{3, 2, 0, 1})>();
  test4d<nda::encode(std::array{0, 2, 3, 1})>();
  test4d<nda::encode(std::array{2, 0, 3, 1})>();
  test4d<nda::encode(std::array{1, 0, 3, 2})>();
  test4d<nda::encode(std::array{0, 1, 3, 2})>();
  test4d<nda::encode(std::array{3, 1, 0, 2})>();
  test4d<nda::encode(std::array{1, 3, 0, 2})>();
  test4d<nda::encode(std::array{0, 3, 1, 2})>();
  test4d<nda::encode(std::array{3, 0, 1, 2})>();
}

TEST(NDA, SliceLayoutOf5DArray) {
  // test only some stride orders
  test5d<nda::encode(std::array{0, 1, 2, 3, 4})>();
  test5d<nda::encode(std::array{4, 3, 2, 1, 0})>();
  test5d<nda::encode(std::array{0, 1, 4, 3, 2})>();
  test5d<nda::encode(std::array{2, 3, 4, 0, 1})>();
  test5d<nda::encode(std::array{2, 4, 3, 0, 1})>();
}

TEST(NDA, SliceAndAccess3DArray) {
  auto n0 = 9, n1 = 7, n2 = 8;
  nda::array<std::complex<double>, 3, nda::contiguous_layout_with_stride_order<nda::encode(std::array{1, 0, 2})>> A(n0, n1, n2);
  for (int i = 0; i < n0; ++i)
    for (int j = 0; j < n1; ++j)
      for (int k = 0; k < n2; ++k) A(i, j, k) = 100 * i + 10 * j + k;

  nda::array<std::complex<double>, 3> B;
  B = A;

  for (int i = 0; i < n0; ++i)
    for (int j = 0; j < n1; ++j)
      for (int k = 0; k < n2; ++k) {
        EXPECT_COMPLEX_NEAR(A(i, nda::range::all, nda::range::all)(j, k), B(i, j, k));
        EXPECT_COMPLEX_NEAR(A(i, j, k), B(i, j, k));
        EXPECT_COMPLEX_NEAR((A - B)(i, j, k), 0);
        EXPECT_COMPLEX_NEAR(A(n0 - 1 - i, j, n2 - 1 - k), A(nda::range(n0 - 1, -1, -1), nda::range::all, nda::range(n2 - 1, -1, -1))(i, j, k));
        EXPECT_COMPLEX_NEAR(A(n0 - 1 - i, j, n2 - 1 - k), A(nda::range(n0 - 1, -1, -1), j, nda::range(n2 - 1, -1, -1))(i, k));
        if (i < n0 / 2 and k < n2 / 3) {
          EXPECT_COMPLEX_NEAR(A(n0 - 1 - 2 * i, j, n2 - 1 - 3 * k), A(nda::range(n0 - 1, -1, -2), j, nda::range(n2 - 1, -1, -3))(i, k));
        }
      }

  for (int i = 0; i < 5; ++i) { EXPECT_ARRAY_NEAR(A(i, nda::range::all, nda::range::all), B(i, nda::range::all, nda::range::all)); }
}

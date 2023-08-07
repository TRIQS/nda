// Copyright (c) 2020 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

// additional tests on slice
// Some are already in basic, and view

// Check the calculation of layout_prop in a 5d slices

template <auto StrideOrder>
void test5d() {

  std::cerr << " ======  " << nda::decode<5>(StrideOrder) << "==========" << std::endl;
  int N0 = 3, N1 = 4, N2 = 5, N3 = 6, N4 = 7;
  range R0(1, N0 - 1);
  range R1(1, N1 - 1);
  range R2(1, N2 - 1);
  range R3(1, N3 - 1);
  range R4(1, N4 - 1);

  //using A = nda::array<long, 5, nda::basic_layout<0, nda::encode(std::array{0,1, 4, 3, 2}), nda::layout_prop_e::contiguous>>;
  using A = nda::array<long, 5, nda::basic_layout<0, StrideOrder, nda::layout_prop_e::contiguous>>;
  //using A = nda::array<long, 5>;
  A a(N0, N1, N2, N3, N4);

  {
    long c = 0;
    for (auto &x : a) {
      EXPECT_EQ(std::distance(&(*std::begin(a)), &x), c);
      x = c++; // in memory order, 0,1,2,3,4 ....
    }
  }

  auto check = [n = 0](auto v) mutable {
    bool is_contiguous          = (v.indexmap().layout_prop == nda::layout_prop_e::contiguous);
    bool is_strided_1d          = (has_strided_1d(v.indexmap().layout_prop));
    bool smallest_stride_is_one = (has_smallest_stride_is_one(v.indexmap().layout_prop));

    EXPECT_EQ(is_contiguous, (is_strided_1d and smallest_stride_is_one));

    std::cerr << "n = " << n << " contiguous " << is_contiguous << " is_strided_1d = " << is_strided_1d
              << " smallest_stride_is_one = " << smallest_stride_is_one << std::endl;

    bool check_strided_1d = true;

    // forcing the basic iterator in rank dimension, avoiding all optimization
    using layout_t = typename decltype(v)::layout_t;
    using Iterator = nda::array_iterator<layout_t::rank(), long const, typename nda::default_accessor::template accessor<long>::pointer>;
    auto it        = Iterator{nda::permutations::apply(layout_t::stride_order, v.indexmap().lengths()),
                       nda::permutations::apply(layout_t::stride_order, v.indexmap().strides()), v.data(), false};
    auto end       = Iterator{nda::permutations::apply(layout_t::stride_order, v.indexmap().lengths()),
                        nda::permutations::apply(layout_t::stride_order, v.indexmap().strides()), v.data(), true};

    auto it1 = it;
    auto c   = *it1; // first element
    ++it1;
    auto stri = (*it1) - c; // the stride
    for (; it != end; ++it) {
      check_strided_1d &= (*it == c);
      c += stri;
    }

    bool check_contiguous = check_strided_1d and (stri == 1);

    EXPECT_EQ(check_strided_1d, is_strided_1d);
    EXPECT_EQ(check_contiguous, is_contiguous);
    EXPECT_EQ(v.indexmap().is_contiguous(), is_contiguous);
    EXPECT_EQ(v.indexmap().is_strided_1d(), is_strided_1d);

    ++n;
  };

  check(a(_, _, _, _, _));
  check(a(_, _, _, _, 2));

  check(a(_, _, _, 2, _));
  check(a(_, _, _, 2, 2));

  check(a(_, _, 2, _, _));
  check(a(_, _, 2, _, 2));
  check(a(_, _, 2, 2, _));
  check(a(_, _, 2, 2, 2));

  check(a(_, 2, _, _, _));
  check(a(_, 2, _, _, 2));
  check(a(_, 2, _, 2, _));
  check(a(_, 2, _, 2, 2));
  check(a(_, 2, 2, _, _));
  check(a(_, 2, 2, _, 2));
  check(a(_, 2, 2, 2, _));
  check(a(_, 2, 2, 2, 2));

  check(a(1, _, _, _, _));
  check(a(1, _, _, _, 2));
  check(a(1, _, _, 2, _));
  check(a(1, _, _, 2, 2));
  check(a(1, _, 2, _, _));
  check(a(1, _, 2, _, 2));
  check(a(1, _, 2, 2, _));
  check(a(1, _, 2, 2, 2));
  check(a(1, 2, _, _, _));
  check(a(1, 2, _, _, 2));
  check(a(1, 2, _, 2, _));
  check(a(1, 2, _, 2, 2));
  check(a(1, 2, 2, _, _));
  check(a(1, 2, 2, _, 2));
  check(a(1, 2, 2, 2, _));
  //check(a(1, 2, 2, 2, 2)); // not a view

  //check(a(_, _, _, _, R4));
  //check(a(R0, _, _, _, R4));
  // SPECIAL CASE TO BE FIXED : it is actually contiguous but it is not detected
  //check(a(R0, _, _, _, _));
}

// Check the calculation of layout_prop in a 5d : only some  slices
TEST(Slice, ContiguityComputation5d) { //NOLINT

  test5d<nda::encode(std::array{0, 1, 2, 3, 4})>();
  test5d<nda::encode(std::array{4, 3, 2, 1, 0})>();

  test5d<nda::encode(std::array{0, 1, 4, 3, 2})>();
  test5d<nda::encode(std::array{2, 3, 4, 0, 1})>();
  test5d<nda::encode(std::array{2, 4, 3, 0, 1})>();
  // ...
}

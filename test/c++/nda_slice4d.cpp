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

#include "./test_common.hpp"

// additional tests on slice
// Some are already in basic, and view

// isolate : a bit long to compile with sanitizer
// Check the calculation of layout_prop in a 4d slices

template <auto StrideOrder>
void test4d() {

  std::cerr << " ======  " << nda::decode<4>(StrideOrder) << "==========" << std::endl;
  int N0 = 3, N1 = 4, N2 = 5, N3 = 6;
  range R0(1, N0 - 1);
  range R1(1, N1 - 1);
  range R2(1, N2 - 1);
  range R3(1, N3 - 1);

  using A = nda::array<long, 4, nda::basic_layout<0, StrideOrder, nda::layout_prop_e::contiguous>>;
  A a(N0, N1, N2, N3);

  {
    long c = 0;
    for (auto &x : a) {
      // Checks that the iterator in indeed going one by one in memory, even with a non trivial StrideOrder
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
                       nda::permutations::apply(layout_t::stride_order, v.indexmap().strides()), v.data_start(), false};
    auto end       = Iterator{nda::permutations::apply(layout_t::stride_order, v.indexmap().lengths()),
                        nda::permutations::apply(layout_t::stride_order, v.indexmap().strides()), v.data_start(), true};

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

  check(a(_, _, _, _));
  check(a(_, _, _, 2));

  check(a(_, _, 2, _));
  check(a(_, _, 2, 2));

  check(a(_, 2, _, _));
  check(a(_, 2, _, 2));
  check(a(_, 2, 2, _));
  check(a(_, 2, 2, 2));

  check(a(1, _, _, _));
  check(a(1, _, _, 2));
  check(a(1, _, 2, _));
  check(a(1, _, 2, 2));
  check(a(1, 2, _, _));
  check(a(1, 2, _, 2));
  check(a(1, 2, 2, _));
  // check(a(1, 2, 2, 2));
}

// Check the calculation of layout_prop in a 4d slices
TEST(Slice, ContiguityComputation4dfull) { //NOLINT

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

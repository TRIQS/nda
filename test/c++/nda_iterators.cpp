// Copyright (c) 2019-2020 Simons Foundation
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

// ===================   for_each ===========================================

TEST(for_each, Mutable) { //NOLINT
  nda::array<int, 3> a(3, 4, 5);
  nda::for_each(a.shape(), [&a, c2 = 0](auto... i) mutable { a(i...) = c2++; });

  auto check = a;
  int c      = 0;
  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k) check(i, j, k) = c++;

  EXPECT_ARRAY_EQ(a, check);
}

// ====================   iterator ==========================================

TEST(iterator, empty) { //NOLINT
  nda::array<int, 1> arr(0);
  int s = 0;
  for (auto i : arr) s += i;
  EXPECT_EQ(s, 0);
}

//-----------------------------

TEST(iterator, Contiguous1d) { //NOLINT
  nda::array<long, 1> a;
  for (int i = 0; i < a.extent(0); ++i) a(i) = 1 + i;

  long c = 1;
  for (auto x : a) { EXPECT_EQ(x, c++); }
}

//-----------------------------

TEST(iterator, Contiguous2d) { //NOLINT
  nda::array<long, 2> a(2, 3);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j) a(i, j) = 1 + i + 10 * j;

  auto it = a.begin();

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j) {
      EXPECT_EQ(*it, a(i, j));
      EXPECT_FALSE(it == a.end());
      ++it;
    }
}

//-----------------------------

TEST(iterator, Contiguous3d) { //NOLINT
  nda::array<long, 3> a(3, 5, 9);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k) a(i, j, k) = 1 + i + 10 * j + 100 * k;

  auto it = a.begin();

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k) {
        EXPECT_EQ(*it, a(i, j, k));
        EXPECT_TRUE(it != a.end());
        ++it;
      }
}

//-----------------------------

TEST(iterator, Strided3d) { //NOLINT
  nda::array<long, 3> a(3, 5, 9);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k) a(i, j, k) = 1 + i + 10 * j + 100 * k;

  auto v = a(range(0, a.extent(0), 2), range(0, a.extent(1), 2), range(0, a.extent(2), 2));

  auto it = v.begin();

  for (int i = 0; i < v.extent(0); ++i)
    for (int j = 0; j < v.extent(1); ++j)
      for (int k = 0; k < v.extent(2); ++k) {
        EXPECT_EQ(*it, v(i, j, k));
        EXPECT_TRUE(it != v.end());
        ++it;
      }
  EXPECT_TRUE(it == v.end());
}

//-----------------------------

TEST(iterator, BlockStrided2d) { //NOLINT
  auto a = nda::rand<>(3, 4);

  EXPECT_TRUE(get_block_layout(a(_, range(0, 4, 2))));
  EXPECT_TRUE(get_block_layout(a(_, range(2))));
  EXPECT_TRUE(!get_block_layout(a(_, range(0, 4, 3))));
  EXPECT_TRUE(!get_block_layout(a(_, range(3))));

  auto av                                = a(_, range(0, 4, 2));
  auto [n_blocks, block_size, block_str] = get_block_layout(av).value();
  EXPECT_EQ(n_blocks * block_size, av.size());

  // Compare loop over array with pointer arithmetic based on block_size and block_str
  auto *ptr = a.data();
  for (auto [n, val] : itertools::enumerate(av)) {
    auto [bl_idx, inner_idx] = std::ldiv(n, block_size);
    EXPECT_EQ(val, *(ptr + bl_idx * block_str + inner_idx));
  }
}

//-----------------------------

TEST(iterator, bug) { //NOLINT
  const int N1 = 1000, N2 = 1000;
  nda::array<double, 2> a(2 * N1, 2 * N2);
  auto v = a(range(0, -1, 2), range(0, -1, 2));
  for (auto &x : v) { x = 10; }
}

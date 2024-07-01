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

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <iostream>
#include <utility>

// Dummy type for constructing a bar object.
struct foo {
  int i = 0;
};

// Dummy type for testing move semantics.
struct bar {
  int j = 0;
  bar(foo &&f) : j(f.i) { f.i = 0; }
};

TEST(NDA, ArrayAdapterBasics) {
  // test some basic functionality of the array adapter
  auto f       = [](long i, long j) { return i + 2 * j; };
  auto adapter = nda::array_adapter{std::array{2, 2}, f};

  // check that the adapter satisfies the ArrayOfRank concept
  static_assert(nda::ArrayOfRank<decltype(adapter), 2>);

  // initialize an array from the adapter
  auto A = nda::array<long, 2>{adapter};
  EXPECT_EQ_ARRAY(A, (nda::array<long, 2>{{0, 2}, {1, 3}}));

  // use the adapter in an algorithm
  EXPECT_EQ(nda::sum(adapter), nda::sum(A));

  // output the adapter
  std::cout << adapter << std::endl;
}

TEST(NDA, ArrayAdapterMoveElements) {
  // test moving elements from one array to another using an array adapter
  nda::array<foo, 2> A_foo(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) A_foo(i, j).i = 1 + i + 10 * j;

  auto f     = [&A_foo](long i, long j) { return bar{std::move(A_foo(i, j))}; };
  auto A_bar = nda::array<bar, 2>{nda::array_adapter{std::array{2, 2}, f}};

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(A_foo(i, j).i, 0);
      EXPECT_EQ(A_bar(i, j).j, (1 + i + 10 * j));
    }
  }
}

TEST(NDA, ArrayAdapterMoveElements2) {
  // test moving elements from one array to another using an array adapter
  nda::array<foo, 2> A_foo(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) A_foo(i, j).i = 1 + i + 10 * j;

  auto f2    = [A_foo2 = std::move(A_foo)](long i, long j) { return bar{std::move(A_foo2(i, j))}; };
  auto A_bar = nda::array<bar, 2>{nda::array_adapter{std::array{2, 2}, f2}};

  EXPECT_TRUE(A_foo.empty());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) { EXPECT_EQ(A_bar(i, j).j, (1 + i + 10 * j)); }
  }
}

TEST(NDA, ArrayAdapterMoveElements3) {
  // test moving elements from one array to another using nda::map instead of an array adapter
  nda::array<foo, 2> A_foo(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) A_foo(i, j).i = 1 + i + 10 * j;

  nda::array<bar, 2> A_bar = nda::map([](auto &&a) { return bar{std::move(std::forward<foo>(a))}; })(A_foo);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(A_foo(i, j).i, 0);
      EXPECT_EQ(A_bar(i, j).j, (1 + i + 10 * j));
    }
  }
}

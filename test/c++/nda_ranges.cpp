// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

// Owning, contiguous 1-dimensional arrays/views model std::ranges::contiguous_range.
static_assert(std::ranges::contiguous_range<nda::array<double, 1>>);
static_assert(std::ranges::contiguous_range<nda::array<double, 1> const>);
static_assert(std::ranges::contiguous_range<nda::vector<double>>);

// Views with a compile-time contiguity guarantee model it as well.
static_assert(std::ranges::contiguous_range<nda::array_contiguous_view<double, 1>>);
static_assert(std::ranges::contiguous_range<nda::array_contiguous_const_view<double, 1>>);

// The corresponding begin() iterator is an nda::array_iterator that models std::contiguous_iterator (not a raw
// pointer).
using array_iter_t       = decltype(std::declval<nda::array<double, 1> &>().begin());
using array_const_iter_t = decltype(std::declval<nda::array<double, 1> const &>().begin());
static_assert(std::contiguous_iterator<array_iter_t>);
static_assert(std::contiguous_iterator<array_const_iter_t>);
static_assert(not std::is_pointer_v<array_iter_t>);

// A general view carries no compile-time contiguity guarantee (layout_prop_e::none), so it is deliberately
// not promoted to a contiguous_range (it might be a strided slice). Its iterator is still random-access.
static_assert(not std::ranges::contiguous_range<nda::array_view<double, 1>>);
static_assert(std::ranges::random_access_range<nda::array_view<double, 1>>);

// A strided 1-dimensional view is random-access but not contiguous.
using strided_view_t = decltype(std::declval<nda::array<double, 1>>()(nda::range(0, 10, 2)));
static_assert(not std::ranges::contiguous_range<strided_view_t>);
static_assert(std::ranges::random_access_range<strided_view_t>);
static_assert(std::random_access_iterator<std::ranges::iterator_t<strided_view_t>>);

// Multi-dimensional arrays must not be flattened into a contiguous_range (would corrupt get_rank).
static_assert(not std::ranges::contiguous_range<nda::array<double, 2>>);
static_assert(nda::get_rank<nda::array<double, 2>> == 2);
static_assert(nda::get_rank<nda::array<double, 1>> == 1);

TEST(NDARanges, ContiguousRangeData) {
  // std::ranges::data must agree with the array's own data pointer.
  auto a = nda::array<double, 1>{1.0, 2.0, 3.0, 4.0};
  EXPECT_EQ(std::ranges::data(a), a.data());
  EXPECT_EQ(std::ranges::size(a), 4);
  EXPECT_EQ(std::to_address(a.begin()), a.data());
  EXPECT_EQ(a.end() - a.begin(), 4);
}

TEST(NDARanges, StdRangesAlgorithms) {
  // A contiguous 1D array works with the std::ranges algorithms.
  auto a = nda::array<long, 1>(5);
  std::ranges::fill(a, 7l);
  EXPECT_TRUE(std::ranges::all_of(a, [](long x) { return x == 7; }));

  std::iota(a.begin(), a.end(), 0l);
  auto v = std::vector<long>(a.begin(), a.end());
  EXPECT_TRUE(std::ranges::equal(a, v));
  EXPECT_EQ(std::ranges::count(a, 2l), 1);
}

TEST(NDARanges, ContiguousView) {
  // A contiguity-guaranteed view onto a contiguous 1D array is itself a contiguous range.
  auto a = nda::array<int, 1>{0, 1, 2, 3, 4, 5};
  auto v = nda::array_contiguous_view<int, 1>{a};
  static_assert(std::ranges::contiguous_range<decltype(v)>);
  EXPECT_EQ(std::ranges::data(v), a.data());
  int expected = 0;
  for (auto x : v) EXPECT_EQ(x, expected++);
}

TEST(NDARanges, EqualityWithArraysStillWorks) {
  // Making 1D arrays a contiguous_range must not break array == array comparisons.
  auto a = nda::array<double, 1>{1.0, 2.0, 3.0};
  auto b = nda::array<double, 1>{1.0, 2.0, 3.0};
  auto c = nda::array<double, 1>{1.0, 2.0, 4.0};
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);

  // Comparison against a foreign contiguous range still works.
  auto vec = std::vector<double>{1.0, 2.0, 3.0};
  EXPECT_TRUE(a == vec);
  EXPECT_TRUE(vec == a);
}

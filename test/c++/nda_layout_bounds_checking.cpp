// Copyright (c) 2019-2023 Simons Foundation
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

#include <nda/layout/bound_check_worker.hpp>
#include <nda/layout/range.hpp>

#include <array>
#include <iostream>
#include <stdexcept>

using namespace nda;

TEST(NDA, BoundsCheckingWithLongOnly) {
  std::array<long, 4> shape{3, 4, 5, 6};
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), 0, 0, 0, 0));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), 2, 3, 4, 5));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), 1, 1, 1, 1));
  EXPECT_THROW(assert_in_bounds(4, shape.data(), 3, 0, 0, 0), std::runtime_error);
  EXPECT_THROW(assert_in_bounds(4, shape.data(), 1, -4, 0, 6), std::runtime_error);
  EXPECT_THROW(assert_in_bounds(4, shape.data(), 0, 2, 4, 7), std::runtime_error);
  try {
    assert_in_bounds(4, shape.data(), 4, 0, -1, 0);
  } catch (std::exception const &e) { std::cout << e.what() << std::endl; }
}

TEST(NDA, BoundsCheckingWithRangeAllAndEllipsis) {
  auto all      = nda::range::all;
  auto ellipsis = nda::ellipsis{};
  std::array<long, 4> shape{3, 4, 5, 6};
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), all, 0, 0, 0));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), 2, all, 4, all));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), all, all, all, all));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), ellipsis));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), all, 0, ellipsis));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), ellipsis, 3, all));
  EXPECT_THROW(assert_in_bounds(4, shape.data(), all, -1, 0, 0), std::runtime_error);
  EXPECT_THROW(assert_in_bounds(4, shape.data(), 1, all, all, 6), std::runtime_error);
  EXPECT_THROW(assert_in_bounds(4, shape.data(), ellipsis, 7), std::runtime_error);
  try {
    assert_in_bounds(4, shape.data(), all, ellipsis, -1);
  } catch (std::exception const &e) { std::cout << e.what() << std::endl; }
}

TEST(NDA, BoundsCheckingWithRange) {
  using rg      = nda::range;
  auto all      = nda::range::all;
  auto ellipsis = nda::ellipsis{};
  std::array<long, 4> shape{3, 4, 5, 6};
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), rg(0, 2), 0, 0, 0));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), 2, all, 4, rg(2, 5)));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), ellipsis, rg(1, 3), 5));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), rg(0, 3), rg(0, 4), rg(0, 5), rg(0, 6, 2)));
  EXPECT_NO_THROW(assert_in_bounds(4, shape.data(), ellipsis, rg(5, -1)));
  EXPECT_THROW(assert_in_bounds(4, shape.data(), all, rg(-1, 2), 0, 0), std::runtime_error);
  EXPECT_THROW(assert_in_bounds(4, shape.data(), 1, all, rg(3, 7), 6), std::runtime_error);
  try {
    assert_in_bounds(4, shape.data(), 0, rg(-1, 2), 0, rg(2, 9, 3));
  } catch (std::exception const &e) { std::cout << e.what() << std::endl; }
}

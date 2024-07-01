// Copyright (c) 2023 Simons Foundation
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

#include <nda/stdutil/array.hpp>

#include <array>
#include <iostream>

TEST(NDA, StdArrayToOstream) {
  auto arr = std::array{1, 2, 3, 4, 5};
  std::cout << arr << std::endl;
}

TEST(NDA, StdArrayOperatorOverloads) {
  auto arr = std::array{1, 2, 3};
  ASSERT_EQ(arr + arr, (std::array{2, 4, 6}));
  ASSERT_EQ(arr - arr, (std::array{0, 0, 0}));
  ASSERT_EQ(arr * arr, (std::array{1, 4, 9}));
  ASSERT_EQ(-arr, (std::array{-1, -2, -3}));
  ASSERT_EQ(2 * arr, (std::array{2, 4, 6}));
}

TEST(NDA, StdArrayConstructions) {
  auto arr_int = nda::stdutil::make_initialized_array<3>(3);
  ASSERT_EQ(arr_int, (std::array{3, 3, 3}));

  auto arr_long = nda::stdutil::make_std_array<long>(arr_int);
  ASSERT_EQ(arr_long, (std::array{3l, 3l, 3l}));

  auto vec_int = nda::stdutil::to_vector(arr_int);
  ASSERT_EQ(vec_int, (std::vector{3, 3, 3}));
}

TEST(NDA, StdArrayModifyingFunctions) {
  auto arr = std::array{1, 2, 3, 4};
  ASSERT_EQ(nda::stdutil::append(arr, 5l), (std::array{1, 2, 3, 4, 5}));
  ASSERT_EQ(nda::stdutil::front_append(arr, 0), (std::array{0, 1, 2, 3, 4}));
  ASSERT_EQ(nda::stdutil::pop(arr), (std::array{1, 2, 3}));
  ASSERT_EQ(nda::stdutil::mpop<2>(arr), (std::array{1, 2}));
  ASSERT_EQ(nda::stdutil::front_pop(arr), (std::array{2, 3, 4}));
  ASSERT_EQ(nda::stdutil::front_mpop<2>(arr), (std::array{3, 4}));
  ASSERT_EQ(nda::stdutil::pop(arr), (std::array{1, 2, 3}));
  ASSERT_EQ(nda::stdutil::join(arr, std::array{5, 6}), (std::array{1, 2, 3, 4, 5, 6}));
}

TEST(NDA, StdArrayReductions) {
  auto arr = std::array{1, 2, 3, 4};
  ASSERT_EQ(nda::stdutil::sum(arr), 10);
  ASSERT_EQ(nda::stdutil::product(arr), 24);
  ASSERT_EQ(nda::stdutil::dot_product(arr, arr), 30);
}

// Copyright (c) 2019 Simons Foundation
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

// ==================== ANY ALL ==========================================

TEST(NDA, any_all) { //NOLINT
  auto nan = std::numeric_limits<double>::quiet_NaN();

  nda::array<double, 2> A(2, 3);
  A() = 98;

  EXPECT_FALSE(any(isnan(A)));

  A() = nan;
  EXPECT_TRUE(all(isnan(A)));

  A()     = 0;
  A(0, 0) = nan;

  EXPECT_FALSE(all(isnan(A)));
  EXPECT_TRUE(any(isnan(A)));
}

// -----------------------------------------------------

TEST(NDA, any_all_c) { //NOLINT
  auto nan = std::numeric_limits<double>::quiet_NaN();

  nda::array<std::complex<double>, 2> A(2, 3);
  A() = 98;

  EXPECT_FALSE(any(isnan(A)));

  A() = nan;
  EXPECT_TRUE(all(isnan(A)));

  A()     = 0;
  A(0, 0) = nan;

  EXPECT_FALSE(all(isnan(A)));
  EXPECT_TRUE(any(isnan(A)));
}

// ==============================================================

TEST(NDA, Algo1) { //NOLINT
  nda::array<int, 2> A(3, 3), B(3, 3);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i - j;
    }

  EXPECT_EQ(max_element(A), 7);
  EXPECT_EQ(sum(A), 36);
  EXPECT_EQ(min_element(B), -2);
  EXPECT_EQ(sum(B), 0);
  EXPECT_EQ((nda::array<int, 2>{A + 10 * B}), (nda::array<int, 2>{{1, -7, -15}, {12, 4, -4}, {23, 15, 7}}));
  EXPECT_EQ(max_element(A + 10 * B), 23);
}

MAKE_MAIN

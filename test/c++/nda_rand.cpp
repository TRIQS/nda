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

// ==============================================================

constexpr long N = 100;

TEST(NDA, Rand) { //NOLINT

  auto a = nda::rand<>(N, N);

  EXPECT_EQ(a.shape(), (nda::shape_t<2>{N, N}));
  EXPECT_GE(max_element(abs(a)), 0);
  EXPECT_LT(max_element(abs(a)), 1);

  // --- sum of arrays

  auto b = make_regular(a + 3.0 * nda::rand<>(N, N) + 1.0);

  EXPECT_EQ(b.shape(), (nda::shape_t<2>{N, N}));
  EXPECT_GE(max_element(abs(b)), 1.0);
  EXPECT_LT(max_element(abs(b)), 5.0);

  // --- scalar

  auto s = 2.0 + 3.0 * nda::rand<>();
  EXPECT_GE(s, 2.0);
  EXPECT_LT(s, 5.0);
}

// ==============================================================

TEST(NDA, RandSTDArr) { //NOLINT

  auto a = nda::array<double, 3>::rand(std::array{3, 4, 5});

  EXPECT_EQ(a.shape(), (nda::shape_t<3>{3, 4, 5}));
  EXPECT_GE(max_element(abs(a)), 0);
  EXPECT_LT(max_element(abs(a)), 1);
}

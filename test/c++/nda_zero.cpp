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

TEST(NDA, Zeros) { //NOLINT

  auto a = nda::zeros<long>(3, 3);

  EXPECT_EQ(a.shape(), (nda::shape_t<2>{3, 3}));
  EXPECT_EQ(max_element(abs(a)), 0);
}

// ==============================================================

TEST(NDA, ZeroStaticFactory) { //NOLINT

  auto a1 = nda::array<long, 1>::zeros(std::array{3});
  auto a2 = nda::array<long, 2>::zeros(std::array{3, 4});
  auto a3 = nda::array<long, 3>::zeros(std::array{3, 4, 5});

  EXPECT_EQ(a1.shape(), (nda::shape_t<1>{3}));
  EXPECT_EQ(a2.shape(), (nda::shape_t<2>{3, 4}));
  EXPECT_EQ(a3.shape(), (nda::shape_t<3>{3, 4, 5}));

  EXPECT_EQ(max_element(abs(a1)), 0);
  EXPECT_EQ(max_element(abs(a2)), 0);
  EXPECT_EQ(max_element(abs(a3)), 0);
}

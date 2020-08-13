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

TEST(Reinterpret, add_N_one) { //NOLINT
  nda::array<long, 2> a(3, 3);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j) a(i, j) = 1 + i + 10 * j;

  // view form
  {
    auto v = nda::reinterpret_add_fast_dims_of_size_one<2>(a());

    EXPECT_EQ(v.shape(), (nda::shape_t<4>{3, 3, 1, 1}));

    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < a.extent(1); ++j) EXPECT_EQ(v(i, j, 0, 0), 1 + i + 10 * j);
  }

  {
    // array form
    auto b = nda::reinterpret_add_fast_dims_of_size_one<2>(std::move(a));

    EXPECT_EQ(b.shape(), (nda::shape_t<4>{3, 3, 1, 1}));

    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < a.extent(1); ++j) EXPECT_EQ(b(i, j, 0, 0), 1 + i + 10 * j);
  }
}

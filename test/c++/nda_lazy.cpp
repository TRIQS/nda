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

#include "./test_common.hpp"

using clef::placeholder;

// ==============================================================

TEST(Lazy, Fill) { //NOLINT

  placeholder<0> i_;
  placeholder<1> j_;

  nda::array<double, 2> A(2, 2);
  A(i_, j_) << i_ * 8.1 + 2.31 * j_;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(A(i, j), i * 8.1 + 2.31 * j);
}

// ==============================================================

TEST(Lazy, ArrayArray) { //NOLINT

  nda::array<nda::array<double, 1>, 2> a(2, 2);

  a = nda::array<double, 1>(3);

  placeholder<0> i_;
  placeholder<1> j_;
  placeholder<2> k_;

  a(i_, j_)(k_) << i_ + 8.1 * j_ + 100 * k_;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 3; ++k) EXPECT_EQ((a(i, j)(k)), i + 8.1 * j + 100 * k);
}

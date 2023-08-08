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
// Authors: Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

// ==============================================================

TEST(StackArray, create) { //NOLINT

  nda::stack_array<long, 3, 3> a;
  nda::array<long, 2> d(3, 3);

  a = 3;
  d = 3;
  EXPECT_ARRAY_NEAR(a, d);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      a(i, j) = i + 10 * j;
      d(i, j) = i + 10 * j;
    }

  auto ac = a;

  ac = a + d;

  NDA_PRINT(a.indexmap());
  //NDA_PRINT(ac);
  NDA_PRINT(ac.indexmap());

  EXPECT_ARRAY_NEAR(a, d);
}

// ==============================================================

TEST(StackArray, slice) { //NOLINT

  nda::stack_array<long, 3, 3> a;

  a = 3;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) { a(i, j) = i + 10 * j; }

  {
    auto v = a(_, 1);

    nda::array<long, 2> ad{a};
    nda::array<long, 1> vd{v};

    NDA_PRINT(v.indexmap());
    NDA_PRINT(a);
    NDA_PRINT(v);

    EXPECT_ARRAY_NEAR(a, ad);
    EXPECT_ARRAY_NEAR(v, vd);
  }

  {
    auto v = a(1, _);

    nda::array<long, 2> ad{a};
    nda::array<long, 1> vd{v};

    NDA_PRINT(v.indexmap());
    NDA_PRINT(a);
    NDA_PRINT(v);

    EXPECT_ARRAY_NEAR(a, ad);
    EXPECT_ARRAY_NEAR(v, vd);
  }
}

// ==============================================================

TEST(Loop, Static) { //NOLINT
  nda::array<long, 2> a(3, 3);

  nda::for_each_static<nda::encode(std::array{3, 3}), 0>(a.shape(), [&a](auto x0, auto x1) { a(x0, x1) = x0 + 10 * x1; });

  std::cout << a << std::endl;
}

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

TEST(NDA, ExprTemplate) { //NOLINT

  nda::array<long, 2> A(2, 3);

  for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  nda::array<long, 2> R;

  R = A + 10;
  EXPECT_EQ(R, (nda::array<long, 2>{{0 + 10, 1 + 10, 2 + 10}, {10 + 10, 11 + 10, 12 + 10}}));

  R = A - 10;
  EXPECT_EQ(R, (nda::array<long, 2>{{0 - 10, 1 - 10, 2 - 10}, {10 - 10, 11 - 10, 12 - 10}}));

  R = 2 * A;
  EXPECT_EQ(R, (nda::array<long, 2>{{0, 2, 4}, {20, 22, 24}}));

  long s = 2;
  R      = s * A;
  EXPECT_EQ(R, (nda::array<long, 2>{{0, 2, 4}, {20, 22, 24}}));
}

// ==============================================================

TEST(NDA, compound_ops) { //NOLINT

  nda::array<long, 2> A(2, 3);

  for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  auto A2 = A;

  A *= 2.0;
  EXPECT_EQ(A, (nda::array<long, 2>{{0, 2, 4}, {20, 22, 24}}));

  A2 /= 2.0;
  EXPECT_EQ(A2, (nda::array<long, 2>{{0, 0, 1}, {5, 5, 6}}));

  nda::array<double, 2> B(A);
  B /= 4;
  EXPECT_ARRAY_NEAR(B, (nda::array<double, 2>{{0.0, 0.5, 1.0}, {5.0, 5.5, 6.0}}));
}

// ==============================================================

TEST(Vector, Ops) { //NOLINT

  nda::array<double, 1> V{1, 2, 3, 4, 5};

  V *= 2;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{2, 4, 6, 8, 10}));

  V[range(2, 4)] /= 2.0;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{2, 4, 3, 4, 10}));

  V[range(0, 5, 2)] *= 10;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{20, 4, 30, 4, 100}));
}

// ==============================================================

TEST(Vector, Ops2) { //NOLINT

  nda::array<double, 1> V{1, 2, 3, 4, 5};
  auto W = V;

  W += V;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{1, 2, 3, 4, 5}));
  EXPECT_ARRAY_NEAR(W, (nda::array<double, 1>{2, 4, 6, 8, 10}));

  W -= V;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{1, 2, 3, 4, 5}));
  EXPECT_ARRAY_NEAR(W, (nda::array<double, 1>{1, 2, 3, 4, 5}));
}

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

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

TEST(NDA, StackArrayConstructionAndAssignment) {
  nda::stack_array<long, 3, 3> A;
  nda::array<long, 2> B(3, 3);

  A = 3;
  B = 3;
  EXPECT_ARRAY_NEAR(A, B);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 10 * j;
      B(i, j) = i + 10 * j;
    }
  }
  EXPECT_ARRAY_NEAR(A, B);

  auto C = A;
  C      = A + B;
  EXPECT_ARRAY_NEAR(C, 2 * B);
}

TEST(NDA, StackArraySlice) {
  nda::stack_array<long, 3, 3> A;
  A = 3;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { A(i, j) = i + 10 * j; }
  }

  auto A_v1 = A(nda::range::all, 1);
  nda::array<long, 2> B1{A};
  nda::array<long, 1> B2{A_v1};
  EXPECT_ARRAY_NEAR(A, B1);
  EXPECT_ARRAY_NEAR(A_v1, B2);

  auto A_v2 = A(1, nda::range::all);
  nda::array<long, 2> C1{A};
  nda::array<long, 1> C2{A_v2};
  EXPECT_ARRAY_NEAR(A, C1);
  EXPECT_ARRAY_NEAR(A_v2, C2);
}

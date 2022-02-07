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
//
// Authors: Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

// ==============================================================

TEST(FortranC, Assign) { //NOLINT
  nda::array<long, 2, F_layout> Af(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) Af(i, j) = 10 * i + j;

  // copy to C layout
  nda::array<long, 2> B(Af);

  // assign
  nda::array<long, 2> A;
  A = Af;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(A(i, j), 10 * i + j);
      EXPECT_EQ(B(i, j), 10 * i + j);
    }
}

// ===============================================================

TEST(Fortran, ScalarAssign) { //NOLINT

  int N = 5;
  matrix<int, F_layout> a(N, N);
  a() = 2;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) EXPECT_EQ(a(i, j), (i == j ? 2 : 0));

  nda::array_view<int, 2, F_layout> aa(a);
  aa = 2;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) EXPECT_EQ(a(i, j), 2);
}
// ===============================================================

TEST(Fortran, is_stride_order_Fortran) { //NOLINT

  int N = 5;
  nda::matrix<int> a(N, N);
  nda::matrix<int, F_layout> af(N, N);
  nda::array<int, 1> v(N);
  nda::array<int, 1, F_layout> vf(N);

  EXPECT_FALSE(a.indexmap().is_stride_order_Fortran());
  EXPECT_TRUE(af.indexmap().is_stride_order_Fortran());
  EXPECT_TRUE(v.indexmap().is_stride_order_Fortran());
  EXPECT_TRUE(vf.indexmap().is_stride_order_Fortran());

  EXPECT_TRUE(a.indexmap().is_stride_order_C());
  EXPECT_FALSE(af.indexmap().is_stride_order_C());
  EXPECT_TRUE(v.indexmap().is_stride_order_C());
  EXPECT_TRUE(vf.indexmap().is_stride_order_C());
}

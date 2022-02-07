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

// ====================== VIEW ========================================

TEST(NDA, ViewBasic) { //NOLINT
  nda::array<long, 3> a(3, 3, 4);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) a(i, j, k) = i + 10 * j + 100 * k;

  auto v = a(_, 1, 2);

  EXPECT_EQ(v.shape(), (nda::shape_t<1>{3}));

  EXPECT_EQ(a(1, 1, 2), 1 + 10 * 1 + 100 * 2);

  a(1, 1, 2) = -28;
  EXPECT_EQ(v(1), a(1, 1, 2));
}
// ----------------------------------

TEST(NDA, Ellipsis) { //NOLINT
  nda::array<long, 3> A(2, 3, 4);
  A() = 7;

  EXPECT_ARRAY_NEAR(A(0, ___), A(0, _, _), 1.e-15);

  nda::array<long, 4> B(2, 3, 4, 5);
  B() = 8;

  EXPECT_ARRAY_NEAR(B(0, ___, 3), B(0, _, _, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(0, ___, 2, 3), B(0, _, 2, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(___, 2, 3), B(_, _, 2, 3), 1.e-15);
}

// ----------------------------------

template <typename ArrayType>
auto sum0(ArrayType const &A) {
  nda::array<typename ArrayType::value_type, ArrayType::rank - 1> res = A(0, ___);
  for (size_t u = 1; u < A.shape()[0]; ++u) res += A(u, ___);
  return res;
}

TEST(NDA, Ellipsis2) { //NOLINT
  nda::array<double, 2> A(5, 2);
  A() = 2;
  nda::array<double, 3> B(5, 2, 3);
  B() = 3;
  EXPECT_ARRAY_NEAR(sum0(A), nda::array<double, 1>{10, 10}, 1.e-15);
  EXPECT_ARRAY_NEAR(sum0(B), nda::array<double, 2>{{15, 15, 15}, {15, 15, 15}}, 1.e-15);
}

// ==============================================================

TEST(NDA, ConstView) { //NOLINT

  nda::array<long, 2> A(2, 3);
  A() = 98;

  auto f2 = [](nda::array_view<long, 2> const &) {};

  f2(A());

  nda::array_const_view<long, 2>{A()};

//#define SHOULD_NOT_COMPILE
#ifdef SHOULD_NOT_COMPILE
  {
    const nda::array<long, 1> A = {1, 2, 3, 4};

    // None of this should compile
    A(0)              = 2;
    A()(0)            = 2;
    A(range(0, 2))(0) = 10;
  }
#endif
}

// ==============================================================

TEST(NDA, Bug2) { //NOLINT

  nda::array<double, 3> A(10, 2, 2);
  A() = 0;

  A(4, range(), range()) = 1;
  A(5, range(), range()) = 2;

  matrix_view<double> M1 = A(4, range(), range());
  matrix_view<double> M2 = A(5, range(), range());

  EXPECT_ARRAY_NEAR(M1, matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M2, matrix<double>{{2, 2}, {2, 2}});

  M1 = M2;

  EXPECT_ARRAY_NEAR(M1, matrix<double>{{2, 2}, {2, 2}});
  EXPECT_ARRAY_NEAR(M2, matrix<double>{{2, 2}, {2, 2}});
}

// ==============================================================

TEST(NDA, View) { //NOLINT

  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  nda::array_view<long, 2> AA(A);

  std::stringstream fs1, fs2;
  fs1 << A;
  fs2 << AA;
  EXPECT_EQ(fs1.str(), fs2.str());
  EXPECT_EQ(AA(0, 0), 0);

  nda::array_view<long, 1> SL1(A(0, range(0, 3)));
  nda::array_view<long, 1> SL2(A(1, range(0, 2)));
  nda::array_view<long, 1> SL3(A(1, range(1, 3)));
  nda::array_view<long, 1> SL4(A(range(0, 2), 0));
  nda::array_view<long, 1> SL5(A(range(0, 2), 1));

  EXPECT_EQ_ARRAY(SL1, (nda::array<long, 1>{0, 1, 2}));
  EXPECT_EQ_ARRAY(SL2, (nda::array<long, 1>{10, 11}));
  EXPECT_EQ_ARRAY(SL3, (nda::array<long, 1>{11, 12}));
  EXPECT_EQ_ARRAY(SL4, (nda::array<long, 1>{0, 10}));
  EXPECT_EQ_ARRAY(SL5, (nda::array<long, 1>{1, 11}));
}

// ==============================================================

TEST(NDA, View3) { //NOLINT

  using nda::encode;
  //-------------

  nda::array<long, 3> A0(2, 3, 4);
  nda::array<long, 3, F_layout> Af(2, 3, 4);
  nda::array<long, 3, nda::basic_layout<0, encode(std::array{0, 1, 2}), nda::layout_prop_e::contiguous>> Ac(2, 3, 4);
  nda::array<long, 3, nda::basic_layout<0, encode(std::array{2, 1, 0}), nda::layout_prop_e::contiguous>> A1(2, 3, 4);

  // non trivial permutation
  nda::array<long, 3, nda::basic_layout<0, encode(std::array{2, 0, 1}), nda::layout_prop_e::contiguous>> A2(2, 3, 4);
  nda::array<long, 3, nda::basic_layout<0, encode(std::array{1, 2, 0}), nda::layout_prop_e::contiguous>> A3(2, 3, 4);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ((A0.indexmap()(i, j, k)), (3 * 4) * i + 4 * j + k);
        EXPECT_EQ((A1.indexmap()(i, j, k)), i + 2 * j + (2 * 3) * k);
        EXPECT_EQ((Ac.indexmap()(i, j, k)), (3 * 4) * i + 4 * j + k);
        EXPECT_EQ((Af.indexmap()(i, j, k)), i + 2 * j + (2 * 3) * k);

        EXPECT_EQ((A2.indexmap()(i, j, k)), 3 * i + j + (2 * 3) * k);
        EXPECT_EQ((A3.indexmap()(i, j, k)), i + (2 * 4) * j + 2 * k);
      }

  //-------------
  auto f = [](auto &A) {
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k) A(i, j, k) = 100 * (i + 1) + 10 * (j + 1) + (k + 1);

    auto _ = range{};

    EXPECT_EQ_ARRAY(A(0, _, _), (nda::array<long, 2>{{111, 112, 113, 114}, {121, 122, 123, 124}, {131, 132, 133, 134}}));
    EXPECT_EQ_ARRAY(A(1, _, _), (nda::array<long, 2>{{211, 212, 213, 214}, {221, 222, 223, 224}, {231, 232, 233, 234}}));
    EXPECT_EQ_ARRAY(A(_, 0, _), (nda::array<long, 2>{{111, 112, 113, 114}, {211, 212, 213, 214}}));
    EXPECT_EQ_ARRAY(A(_, _, 1), (nda::array<long, 2>{{112, 122, 132}, {212, 222, 232}}));
    EXPECT_EQ_ARRAY(A(_, 0, 1), (nda::array<long, 1>{112, 212}));
  };

  f(A0);
  f(A1);
  f(A2);
  f(A3);
  f(Ac);
  f(Af);
}

// ==============================================================

// old issue
TEST(NDA, IssueXXX) { //NOLINT

  nda::array<double, 3> A(10, 2, 2);
  A() = 0;

  A(4, range(), range()) = 1;
  A(5, range(), range()) = 2;

  matrix_view<double> M1 = A(4, range(), range());
  matrix_view<double> M2 = A(5, range(), range());

  EXPECT_ARRAY_NEAR(M1, matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M2, matrix<double>{{2, 2}, {2, 2}});

  M1 = M2;

  EXPECT_ARRAY_NEAR(M1, matrix<double>{{2, 2}, {2, 2}});
  EXPECT_ARRAY_NEAR(M2, matrix<double>{{2, 2}, {2, 2}});
}

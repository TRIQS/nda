// Copyright (c) 2019-2021 Simons Foundation
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
#include <nda/linalg/det_and_inverse.hpp>

using expr_1_m_mat = nda::expr<'-', long, nda::basic_array_view<long, 2, nda::C_layout, 'M', nda::default_accessor, nda::borrowed<>>>;

static_assert(expr_1_m_mat::algebra == 'M', "oops");
static_assert(nda::get_algebra<expr_1_m_mat> == 'M', "oops");

// ==============================================================

TEST(NDA, DanglingScalarProtection) { //NOLINT

  nda::array<long, 1> a{4, 2, 3}, b{8, 4, 6};

  auto f = [&a]() {
    double x = 2;
    return x * a;
  };

  static_assert(!std::is_reference_v<decltype(f().l)>, "Dangling !");
  EXPECT_EQ_ARRAY((nda::array<long, 1>{f()}), b);
}

// ==============================================================

TEST(NDA, Negate_Array) { //NOLINT

  static_assert(nda::Array<nda::array<double, 1>>, "EEE");

  nda::array<double, 1> A{4, 2, 3}, B{0, 0, 0};

  B = -A;

  EXPECT_ARRAY_NEAR(B, (nda::array<double, 1>{-4, -2, -3}), 1.e-12);
}

// ----------------------------------------------------

TEST(NDA, Negate_Matrix) { //NOLINT

  matrix<double> A{{1, 2}, {3, 4}}, B(2, 2);
  B() = 0;
  B   = -A;
  EXPECT_ARRAY_NEAR(B, (matrix<double>{{-1, -2}, {-3, -4}}));
}

// ==============================================================

TEST(NDA, ScalarDivMatrix) { //NOLINT

  auto a    = nda::matrix<double>(2, 2);
  a         = 0;
  auto b    = a;
  double l1 = 0.2, l2 = 1.4;
  a(0, 0) = 2;
  a(1, 1) = 4;
  b(0, 0) = l1 / (l2 - a(0, 0));
  b(1, 1) = l1 / (l2 - a(1, 1));

  EXPECT_ARRAY_NEAR(b, (l1 / (l2 - a)));
}

// ==============================================================

TEST(NDA, ExprTemplateMatrix) { //NOLINT

  matrix<long> A(2, 2), B(2, 2), C(2, 2);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      A(i, j) = 10 * i + j;
      B(i, j) = i + 2 * j;
    }

  //FIXME : does not compile. TO BE CLEANED when cleaning expression tempate
  EXPECT_EQ((A + 2 * B).shape(), (std::array<long, 2>{2, 2}));

  C = A + 2 * B;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(C(i, j), A(i, j) + 2 * B(i, j));

  C = std::plus<>{}(A, B);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(C(i, j), A(i, j) + B(i, j));

  //
  EXPECT_EQ(matrix<long>(2 * A), (matrix<long>{{0, 2}, {20, 22}}));
  EXPECT_EQ(matrix<long>(A + 2), (matrix<long>{{2, 1}, {10, 13}}));
  EXPECT_EQ(matrix<long>(1 + A), (matrix<long>{{1, 1}, {10, 12}}));
}

// ----------------------------------------------------

TEST(NDA, ExprTemplateMatrixMult) { //NOLINT

  matrix<long> A(2, 2), B(2, 2), C(2, 2);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      A(i, j) = 10 * i + j;
      B(i, j) = i + 2 * j;
    }

  // matrix multiplication
  matrix<double> Af(2, 2), Bf(2, 2), Cf(2, 2), id(2, 2);
  Af       = A;
  Bf       = B;
  Bf(0, 0) = 1;
  Cf()     = 0;
  id()     = 1;
  Cf       = Af * Bf;

  EXPECT_ARRAY_NEAR(Cf, (matrix<long>{{1, 3}, {21, 53}}));
  EXPECT_ARRAY_NEAR(matrix<double>(Af * Bf), (matrix<long>{{1, 3}, {21, 53}}));
  EXPECT_ARRAY_NEAR(matrix<double>(Af * (Bf + Cf)), (matrix<long>{{22, 56}, {262, 666}}));

  // test division
  // NB: SHOULD NOT COMPILE
  //EXPECT_ARRAY_NEAR(matrix<double>(2 / Af), (matrix<double>{{-2.2, 0.2}, {2.0, 0.0}}));

  EXPECT_ARRAY_NEAR(matrix<double>(2 * inverse(Af)), (matrix<double>{{-2.2, 0.2}, {2.0, 0.0}}));
  EXPECT_ARRAY_NEAR(matrix<double>(Af / 2), (matrix<double>{{0.0, 0.5}, {5.0, 5.5}}));
}

// ----------------------------------------------------

TEST(NDA, ExprTemplateArray) { //NOLINT

  static_assert(nda::get_algebra<nda::array<int, 1>> == 'A', "oops");

  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::r_is_scalar == true, "oops");
  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::l_is_scalar == false, "oops");
  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::algebra == 'A', "oops");

  nda::array<int, 1> A(3), B(3), C;
  nda::array<double, 1> D;
  B = 2;
  A = 3;

  C = A + B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{5, 5, 5});

  C = 2 * A + B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{8, 8, 8});

  C = A * B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{6, 6, 6});

  C = 2 * B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{4, 4, 4});

  C = 2 * B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{4, 4, 4});

  D = 2.3 * B;
  EXPECT_ARRAY_NEAR(D, nda::array<double, 1>{4.6, 4.6, 4.6});

  D = A + B / 1.2;
  EXPECT_ARRAY_NEAR(D, nda::array<double, 1>{4.66666666666667, 4.66666666666667, 4.66666666666667});

  //auto x = A + B + 2 * A;
  //EXPECT_PRINT("(([3,3,3] + [2,2,2]) + (2 * [3,3,3]))", x);

  C = A + 2 * A + 3 * A - 2 * A + A - A + A + A * 3 + A + A + A + A + A + A + A + A + A + A + A + A + A;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{63, 63, 63});
}

//================================================

clef::placeholder<0> i_;
clef::placeholder<1> j_;
clef::placeholder<2> k_;
clef::placeholder<3> l_;

//================================================

// inversion tensor (by element)
TEST(NDA, InverseTensor) { //NOLINT

  nda::array<dcomplex, 3> a(2, 2, 2);
  nda::array<dcomplex, 3> r(2, 2, 2);
  a(i_, j_, k_) << 1 + i_ + 10 * j_ + 100 * k_;
  r(i_, j_, k_) << 1 / (1.0 + i_ + 10 * j_ + 100 * k_);

  a = 1 / a; //inverse(a);

  EXPECT_ARRAY_NEAR(r, a);
}

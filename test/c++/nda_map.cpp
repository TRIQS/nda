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

TEST(NDA, MinMaxElement) { //NOLINT

  nda::array<int, 2> A(3, 3), B(3, 3), C;

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i - 3 * j;
    }

  C = A + B;

  EXPECT_EQ(max_element(A), 7);
  EXPECT_EQ(min_element(A), 1);
  EXPECT_EQ(max_element(B), 2);
  EXPECT_EQ(min_element(B), -6);
  EXPECT_EQ(max_element(A + B), 5);
  EXPECT_EQ(min_element(A + B), -1);
  EXPECT_EQ(sum(A), 36);
}

// ==============================================================

TEST(NDA, Map) { //NOLINT

  using arr_t = nda::array<double, 2>;
  arr_t A(3, 3), B(3, 3), Sqr_A(3, 3), abs_B_B(3, 3), A_10_m_B(3, 3), abs_A_10_m_B(3, 3), max_A_10_m_B(3, 3), pow_A(3, 3);

  using mat_t = nda::matrix<std::complex<double>>;
  mat_t C(3, 3), conj_C(3, 3), transp_C(3, 3);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i - 3 * j;
      C(i, j) = A(i, j) + 1i * B(i, j);

      pow_A(i, j)        = A(i, j) * A(i, j);
      Sqr_A(i, j)        = A(i, j) * A(i, j);
      abs_B_B(i, j)      = std::abs(2 * B(i, j));
      A_10_m_B(i, j)     = A(i, j) + 10 * B(i, j);
      abs_A_10_m_B(i, j) = std::abs(A(i, j) + 10 * B(i, j));
      max_A_10_m_B(i, j) = std::max(A(i, j), 10 * B(i, j));
      conj_C(i, j)       = A(i, j) - 1i * B(i, j);
      transp_C(j, i)     = A(i, j) + 1i * B(i, j);
    }

  auto Abs = nda::map([](double x) { return std::fabs(x); });
  auto Max = nda::map([](double x, double y) { return std::max(x, y); });
  auto sqr = nda::map([](double x) { return x * x; });

  EXPECT_ARRAY_NEAR(arr_t(pow(arr_t{A}, 2)), Sqr_A);
  EXPECT_ARRAY_NEAR(arr_t(sqr(A)), Sqr_A);
  EXPECT_ARRAY_NEAR(arr_t(Abs(B + B)), abs_B_B);
  EXPECT_ARRAY_NEAR(arr_t(A + 10 * B), A_10_m_B);
  EXPECT_ARRAY_NEAR(arr_t(Abs(A + 10 * B)), abs_A_10_m_B);
  EXPECT_ARRAY_NEAR(arr_t(Max(A, 10 * B)), max_A_10_m_B);
  EXPECT_ARRAY_NEAR(mat_t(conj(C)), conj_C);
  EXPECT_ARRAY_NEAR(mat_t(transpose(C)), transp_C);
  EXPECT_ARRAY_NEAR(mat_t(C * conj(transpose(C))), mat_t(transpose(conj(C) * transpose(C))));
}

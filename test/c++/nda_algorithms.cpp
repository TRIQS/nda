// Copyright (c) 2019-2022 Simons Foundation
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

#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <limits>

// Test fixture for testing various algorithms.
struct NDAAlgorithm : public ::testing::Test {
  protected:
  NDAAlgorithm() {
    A_d = nda::array<double, 3>::rand(shape) - 0.5;
    A_c = nda::array<std::complex<double>, 3>::rand(shape) - std::complex<double>{0.5, 0.5};
  }

  std::array<long, 3> shape{2, 3, 4};
  nda::array<double, 3> A_d;
  nda::array<std::complex<double>, 3> A_c;
  static auto constexpr nan = std::numeric_limits<double>::quiet_NaN();
};

TEST_F(NDAAlgorithm, Any) {
  auto greater1 = nda::map([](auto x) { return x > 1; })(A_d);
  EXPECT_FALSE(nda::any(greater1));

  A_d(1, 2, 2) = 2;
  EXPECT_TRUE(nda::any(greater1));

  EXPECT_FALSE(nda::any(nda::isnan(A_d)));

  A_d(1, 2, 2) = nan;
  EXPECT_TRUE(nda::any(nda::isnan(A_d)));
}

TEST_F(NDAAlgorithm, All) {
  auto greaterm05 = nda::map([](auto x) { return x > -0.5; })(A_d);
  EXPECT_TRUE(nda::all(greaterm05));

  A_d(0, 1, 3) = -1;
  EXPECT_FALSE(nda::all(greaterm05));

  EXPECT_FALSE(nda::all(nda::isnan(A_d)));

  A_d() = nan;
  EXPECT_TRUE(nda::all(nda::isnan(A_d)));
}

TEST_F(NDAAlgorithm, MaxElement) {
  EXPECT_EQ(nda::max_element(A_d), *std::max_element(A_d.begin(), A_d.end()));

  A_d(1, 1, 1) = 1;
  EXPECT_EQ(nda::max_element(A_d), 1);
}

TEST_F(NDAAlgorithm, MinElement) {
  EXPECT_EQ(nda::min_element(A_d), *std::min_element(A_d.begin(), A_d.end()));

  A_d(1, 1, 2) = -1;
  EXPECT_EQ(nda::min_element(A_d), -1);
}

TEST_F(NDAAlgorithm, Sum) {
  EXPECT_DOUBLE_EQ(nda::sum(A_d), std::accumulate(A_d.begin(), A_d.end(), 0.0));
  EXPECT_COMPLEX_NEAR(nda::sum(A_c), std::accumulate(A_c.begin(), A_c.end(), std::complex<double>{0.0, 0.0}));
}

TEST_F(NDAAlgorithm, Product) {
  EXPECT_DOUBLE_EQ(nda::product(A_d), std::accumulate(A_d.begin(), A_d.end(), 1.0, std::multiplies<>{}));
  EXPECT_COMPLEX_NEAR(nda::product(A_c), std::accumulate(A_c.begin(), A_c.end(), std::complex<double>{1.0, 0.0}, std::multiplies<>{}));
}

TEST_F(NDAAlgorithm, CustomFold) {
  auto minus_d = nda::fold([](auto r, auto x) { return r - x; }, A_d);
  EXPECT_DOUBLE_EQ(minus_d, std::accumulate(A_d.begin(), A_d.end(), 0.0, std::minus<>{}));

  auto minus_c = nda::fold([](auto r, auto x) { return r - x; }, A_c);
  EXPECT_COMPLEX_NEAR(minus_c, std::accumulate(A_c.begin(), A_c.end(), std::complex<double>{0.0, 0.0}, std::minus<>{}));
}

TEST_F(NDAAlgorithm, CombineAlgorithmsWithArithmeticOps) {
  nda::array<int, 2> A(3, 3), B(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i - 3 * j;
    }
  }

  EXPECT_EQ(nda::max_element(A), 7);
  EXPECT_EQ(nda::min_element(A), 1);
  EXPECT_EQ(nda::max_element(B), 2);
  EXPECT_EQ(nda::min_element(B), -6);
  EXPECT_EQ(nda::max_element(A + B), 5);
  EXPECT_EQ(nda::min_element(A + B), -1);
  EXPECT_EQ(nda::sum(A), 36);
  EXPECT_EQ(nda::sum(B), -18);
  EXPECT_EQ(nda::sum(A + B), 18);
}

TEST_F(NDAAlgorithm, HadamardProduct) {
  nda::array<int, 2> A(3, 3), B(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i - 3 * j;
    }
  }

  // test with nda::array
  auto Amat = nda::matrix<int>(A);
  auto Bmat = nda::matrix<int>(B);
  EXPECT_EQ(nda::hadamard(A, B), A * B);
  EXPECT_EQ(nda::hadamard(Amat, Bmat), A * B);
  EXPECT_EQ(nda::hadamard(Amat, B), A * B);

  // test with std::array
  auto arr1 = std::array<int, 3>{1, 2, 3};
  auto arr2 = std::array<long, 3>{4, 5, 6};
  EXPECT_EQ(nda::hadamard(arr1, arr2), arr1 * arr2);

  // test with std::vector
  auto vec1 = std::vector<long>{1, 2, 3};
  auto vec2 = std::vector<double>{4, 5, 6};
  auto vec3 = std::vector<double>{4, 10, 18};
  EXPECT_EQ(nda::hadamard(vec1, vec2), vec3);

  // test with arithmetic types or complex
  EXPECT_EQ(nda::hadamard(2, 3.0), 6.0);

  auto x = std::complex<long>{1, 2};
  auto y = x * 3;
  EXPECT_EQ(nda::hadamard(x, 3), y);
}
// Copyright (c) 2023 Simons Foundation
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
// Authors: Thomas Hahn

#include "./test_common.hpp"

#include <nda/nda.hpp>

#include <iostream>

// Test fixture for testing various array utilities.
struct NDAArrayUtilities : public ::testing::Test {
  protected:
  NDAArrayUtilities() {
    A_d = nda::array<double, 3>::rand(arr_shape) - 0.5;
    M_l = nda::matrix<long>::ones(mat_shape);
    V_c = nda::vector<std::complex<double>>::rand(vec_shape);
  }

  std::array<long, 3> arr_shape{2, 3, 4};
  nda::array<double, 3> A_d;
  std::array<long, 2> mat_shape{3, 5};
  nda::matrix<long> M_l;
  std::array<long, 1> vec_shape{2};
  nda::vector<std::complex<double>> V_c;
};

TEST_F(NDAArrayUtilities, FirstDimAndSecondDim) {
  EXPECT_EQ(nda::first_dim(A_d), arr_shape[0]);
  EXPECT_EQ(nda::second_dim(A_d), arr_shape[1]);
  EXPECT_EQ(nda::first_dim(M_l), mat_shape[0]);
  EXPECT_EQ(nda::second_dim(M_l), mat_shape[1]);
  EXPECT_EQ(nda::first_dim(V_c), vec_shape[0]);
}

TEST_F(NDAArrayUtilities, StreamOperator) {
  // array types
  std::cout << A_d << std::endl;
  std::cout << M_l << std::endl;
  std::cout << V_c << std::endl;

  // view types
  std::cout << A_d(0, nda::ellipsis{}) << std::endl;

  // binary expressions
  std::cout << V_c + V_c << std::endl;

  // function call expression
  std::cout << nda::cos(A_d) << std::endl;

  // unary expressions
  std::cout << -V_c << std::endl;
}

TEST_F(NDAArrayUtilities, EqualToOperator) {
  auto B_d = A_d;
  EXPECT_TRUE(A_d == B_d);
  EXPECT_FALSE(A_d != B_d);

  B_d(0, 0, 0) += 1.0;
  EXPECT_FALSE(A_d == B_d);
  EXPECT_TRUE(A_d != B_d);

  std::vector<int> vec{1, 2, 3};
  nda::vector<int> V_i(nda::vector_view<int>{vec});
  EXPECT_TRUE(V_i == vec);
  EXPECT_TRUE(vec == V_i);
  EXPECT_FALSE(V_i != vec);
  EXPECT_FALSE(vec != V_i);

  vec[1] = 4;
  EXPECT_FALSE(V_i == vec);
  EXPECT_FALSE(vec == V_i);
  EXPECT_TRUE(V_i != vec);
  EXPECT_TRUE(vec != V_i);
}

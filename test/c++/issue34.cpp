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

// Multiply mxm matrix A by mxnxp matrix B, combining last two indices of B
nda::array<double, 3> arraymult(nda::matrix<double, F_layout> a, nda::array<double, 3, F_layout> b) {

  auto [m, n, p] = b.shape();

  auto brs  = nda::reshape(b, std::array{m, n * p});
  auto bmat = nda::matrix_const_view<double, F_layout>(brs);
  return nda::reshape(a * bmat, std::array{m, n, p});
}

TEST(NDA, Issue34) { //NOLINT

  int m = 1;
  int n = 2;
  int p = 3;

  auto a = transpose(nda::eye<double>(m));

  auto b = nda::array<double, 3, F_layout>(m, n, p);
  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) { b(i, j, k) = double(j + n * k); }
    }
  }

  // Compute products the old-fashioned way
  auto ctrue = nda::array<double, 3, F_layout>(m, n, p);
  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < n; ++j) { ctrue(_, j, k) = a * nda::vector_const_view<double>(b(_, j, k)); }
  }

  // Compute products with my function
  auto c = arraymult(a, b);

  // Check agreement
  EXPECT_ARRAY_NEAR(c(0, _, _), ctrue(0, _, _));
}

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
// Authors: Nils Wentzell

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <array>

// Multiply an m-by-m matrix A by an m-by-n-by-p tensor B, combining the last two indices of B.
nda::array<double, 3> arraymult(nda::matrix<double, nda::F_layout> a, nda::array<double, 3, nda::F_layout> b) {
  auto [m, n, p] = b.shape();
  auto brs       = nda::reshape(b, std::array{m, n * p});
  auto bmat      = nda::matrix_const_view<double, nda::F_layout>(brs);
  return nda::reshape(a * bmat, std::array{m, n, p});
}

TEST(NDA, Issue34) {
  // issue concerning array/matrix/tensor multiplication
  int m = 1;
  int n = 2;
  int p = 3;

  auto A = transpose(nda::eye<double>(m));
  auto B = nda::array<double, 3, nda::F_layout>(m, n, p);
  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) { B(i, j, k) = double(j + n * k); }
    }
  }

  // compute the products the old-fashioned way
  auto C_exp = nda::array<double, 3, nda::F_layout>(m, n, p);
  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < n; ++j) { C_exp(nda::range::all, j, k) = A * nda::vector_const_view<double>(B(nda::range::all, j, k)); }
  }

  // compute products with the arraymult function
  auto C = arraymult(A, B);

  // check results
  EXPECT_ARRAY_NEAR(C(0, nda::ellipsis{}), C_exp(0, nda::ellipsis{}));
}

TEST(NDA, Bug2) {
  // bug concerning matrix views and their assignment
  nda::array<double, 3> A(10, 2, 2);
  A() = 0;

  A(4, nda::range::all, nda::range::all) = 1;
  A(5, nda::range::all, nda::range::all) = 2;

  nda::matrix_view<double> M1 = A(4, nda::range::all, nda::range::all);
  nda::matrix_view<double> M2 = A(5, nda::range::all, nda::range::all);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{2, 2}, {2, 2}});

  M1 = M2;

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{2, 2}, {2, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{2, 2}, {2, 2}});
}

TEST(NDA, DanglingScalarIssue) {
  // scalars should be stored as values not as references in lazy expressions
  nda::array<long, 1> a{4, 2, 3}, b{8, 4, 6};

  auto f = [&a]() {
    long x = 2;
    return x * a;
  };

  static_assert(!std::is_reference_v<decltype(f().l)>);
  EXPECT_EQ_ARRAY((nda::array<long, 1>{f()}), b);
}

TEST(NDA, LayoutInfoBug) {
  using expr_t = nda::expr<'+', nda::basic_array<std::complex<double>, 2, nda::C_layout, 'M', nda::heap<>> &,
                           const nda::basic_array_view<const std::complex<double>, 2, nda::basic_layout<0, 16, nda::layout_prop_e::none>, 'A',
                                                       nda::default_accessor, nda::borrowed<>> &>;

  EXPECT_EQ((nda::layout_prop_e::none & nda::layout_prop_e::contiguous), nda::layout_prop_e::none);
  EXPECT_EQ(expr_t::l_is_scalar, false);
  EXPECT_EQ(expr_t::r_is_scalar, false);
  EXPECT_EQ(expr_t::compute_layout_info().prop, nda::layout_prop_e::none);
  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop, nda::layout_prop_e::none);
  EXPECT_EQ(nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::contiguous);
  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop & nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::none);
}

TEST(NDA, STLAlgorithmBug) {
  nda::array<double, 1> A{1, 3, 2}, B{2, 3, 1};
  EXPECT_TRUE(std::lexicographical_compare(A.begin(), A.end(), B.begin(), B.end()));
  EXPECT_FALSE(std::lexicographical_compare(B.begin(), B.end(), A.begin(), A.end()));

  nda::array<double, 1> C = {1, 3, 2}, D = {1, 2, 10};
  EXPECT_FALSE(std::lexicographical_compare(C.begin(), C.end(), D.begin(), D.end()));
  EXPECT_TRUE(std::lexicographical_compare(D.begin(), D.end(), C.begin(), C.end()));
}

TEST(NDA, AmbiguousAssignmentOperatorIssue) {
  // issue concerning an ambiguous assignment operator overload when assigning a contiguous range
  nda::array<std::string, 1> A(3);
  A() = std::string("foo");
  for (int i = 0; i < 3; ++i) { EXPECT_EQ(A(i), std::string("foo")); }

  nda::array<std::array<int, 3>, 1> B(3);
  B = std::array{1, 2, 3};
  for (int i = 0; i < 3; ++i) { EXPECT_EQ(B(i), (std::array{1, 2, 3})); }
}

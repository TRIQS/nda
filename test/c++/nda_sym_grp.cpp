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
// Authors: Dominik Kiese

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>
#include <nda/sym_grp.hpp>

#include <array>
#include <complex>
#include <functional>
#include <tuple>
#include <vector>

// Consider the permutations:
// 1) {0, 1, 2, 3} -> {2, 1, 0, 3} and
// 2) {0, 1, 2, 3} -> {3, 2, 1, 0}.
// We assume a 4x4 matrix A is invariant when 1) is applied as a row and 2) as a column index transformation.
// Thus, there should be 6 symmetry groups (a to f) of equivalent elements in A:
//
//                    0   1   2   3
//                0   a   d   d   a
//                1   b   e   e   b
//                2   a   d   d   a
//                3   c   f   f   c
//
TEST(NDA, SymGrpMatrixPermutation) {
  using idx_t      = std::array<long, 2>;
  using sym_t      = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  // the 4x4 matrix, initialized such that it respects the symmetries
  nda::array<std::complex<double>, 2> A(4, 4);

  auto a  = nda::rand();
  A(0, 0) = a;
  A(2, 0) = a;
  A(0, 3) = a;
  A(2, 3) = a;

  auto b  = nda::rand();
  A(1, 0) = b;
  A(1, 3) = b;

  auto c  = nda::rand();
  A(3, 0) = c;
  A(3, 3) = c;

  auto d  = nda::rand();
  A(0, 1) = d;
  A(2, 1) = d;
  A(0, 2) = d;
  A(2, 2) = d;

  auto e  = nda::rand();
  A(1, 1) = e;
  A(1, 2) = e;

  auto f  = nda::rand();
  A(3, 1) = f;
  A(3, 2) = f;

  // 1) {0, 1, 2, 3} -> {2, 1, 0, 3}
  auto p0 = [](idx_t const &x) {
    auto p   = std::array<long, 4>{2, 1, 0, 3};
    idx_t xp = {p[x[0]], x[1]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // 2) {0, 1, 2, 3} -> {3, 2, 1, 0}
  auto p1 = [](idx_t const &x) {
    auto p   = std::array<long, 4>{3, 2, 1, 0};
    idx_t xp = {x[0], p[x[1]]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // compute the symmetry classes
  std::vector<sym_func_t> sym_list = {p0, p1};
  auto grp                         = nda::sym_grp{A, sym_list};

  // test if the number of classes matches our expectation
  EXPECT_EQ(grp.get_sym_classes().size(), 6);

  // init a second array from the symmetry group and test if it matches the input array
  nda::array<std::complex<double>, 2> B(4, 4);
  auto init_func = [&A](idx_t const &x) { return std::apply(A, x); };
  grp.init(B, init_func);
  EXPECT_EQ_ARRAY(A, B);

  // test symmetrization without breaking the symmetry
  auto const &[max_diff1, max_idx1] = grp.symmetrize(A);
  EXPECT_NEAR(max_diff1, 0.0, 1e-12);
  EXPECT_ARRAY_NEAR(A, B);

  // test symmetrization with breaking the symmetry
  A(0, 0) += 0.1;
  A(2, 0) -= 0.1;
  auto const [max_diff2, max_idx2] = grp.symmetrize(A);
  EXPECT_NEAR(max_diff2, 0.1, 1e-10);
  EXPECT_TRUE(max_idx2[0] == 0 || max_idx2[0] == 2);
  EXPECT_TRUE(max_idx2[1] == 0);
  EXPECT_ARRAY_NEAR(A, B);

  // test transformation to and initialization from representative data
  auto const vec = grp.get_representative_data(A);
  grp.init_from_representative_data(B, vec);
  EXPECT_EQ_ARRAY(A, B);
}

// Consider the transformations:
// 1) A_ij -> A_ji and
// 2) A_ij -> A_(i+1)j.
// If matrix A is supposed to be invariant under those transformations, then all its elements should fall into one
// symmetry class.
//
TEST(NDA, SymGrpMatrixFlipShift) {
  using idx_t      = std::array<long, 2>;
  using sym_t      = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  // the 4x4 matrix
  nda::array<std::complex<double>, 2> A(4, 4);
  A = std::complex<double>{1.1, 2.2};

  // 1) A_ij -> A_ji
  auto p0 = [](idx_t const &x) {
    idx_t xp = {x[1], x[0]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // 2) A_ij -> A_(i+1)j
  auto p1 = [](idx_t const &x) {
    idx_t xp = {x[0] + 1, x[1]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // compute the symmetry classes
  std::vector<sym_func_t> sym_list = {p0, p1};
  auto grp                         = nda::sym_grp{A, sym_list};

  // test if the number of classes matches our expectation
  EXPECT_EQ(grp.get_sym_classes().size(), 1);

  // init a second array from the symmetry group and test if it matches the input array
  nda::array<std::complex<double>, 2> B(4, 4);
  auto init_func = [&A](idx_t const &x) { return std::apply(A, x); };
  grp.init(B, init_func);
  EXPECT_EQ_ARRAY(A, B);

  // test symmetrization without breaking the symmetry
  auto const &[max_diff1, max_idx1] = grp.symmetrize(A);
  EXPECT_NEAR(max_diff1, 0.0, 1e-12);
  EXPECT_ARRAY_NEAR(A, B);

  // test symmetrization with breaking the symmetry
  A(1, 1) += 0.5;
  A(2, 3) -= 0.5;
  auto const [max_diff2, max_idx2] = grp.symmetrize(A);
  EXPECT_NEAR(max_diff2, 0.5, 1e-10);
  EXPECT_TRUE(max_idx2[0] == 1 || max_idx2[0] == 2);
  EXPECT_TRUE(max_idx2[1] == 1 || max_idx2[1] == 3);
  EXPECT_ARRAY_NEAR(A, B);

  // test transformation to and initialization from representative data
  auto const vec = grp.get_representative_data(A);
  grp.init_from_representative_data(B, vec);
  EXPECT_EQ_ARRAY(A, B);
}

// Consider the transformations:
// 1) A_ijklmn -> A_jkilmn and
// 2) A_ijklmn -> A_ijkmnl for a rank 6 tensor A.
// With N elements per dimension this should result in ((N^3 + 2N) / 3)^2 symmetry classes.
//
TEST(NDA, SymGrpTensorCylicTriplet) {
  using idx_t      = std::array<long, 6>;
  using sym_t      = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;
  auto constexpr n = 2;

  // the rank 6 tensor
  nda::array<std::complex<double>, 6> A(n, n, n, n, n, n);

  // 1) A_ijklmn -> A_jkilmn
  auto p0 = [](idx_t const &x) {
    idx_t xp = {x[1], x[2], x[0], x[3], x[4], x[5]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // 2) A_ijklmn -> A_ijkmnl
  auto p1 = [](idx_t const &x) {
    idx_t xp = {x[0], x[1], x[2], x[4], x[5], x[3]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // compute the symmetry classes
  std::vector<sym_func_t> sym_list = {p0, p1};
  auto grp                         = nda::sym_grp{A, sym_list};

  // test if the number of classes matches our expectation
  EXPECT_EQ(grp.get_sym_classes().size(), std::pow((std::pow(n, 3) + 2 * n) / 3, 2));

  // initialize the array with random values and symmetrize
  A                               = nda::array<std::complex<double>, 6>::rand(n, n, n, n, n, n);
  auto const &[max_diff, max_idx] = grp.symmetrize(A);
  EXPECT_LE(max_diff, 1.0);

  // init a second array from the symmetry group and test if it matches the input array
  nda::array<std::complex<double>, 6> B(n, n, n, n, n, n);
  auto init_func = [&A](idx_t const &x) { return std::apply(A, x); };
  grp.init(B, init_func);
  EXPECT_EQ_ARRAY(A, B);

  // test symmetrization without breaking the symmetry
  auto const &[max_diff1, max_idx1] = grp.symmetrize(A);
  EXPECT_NEAR(max_diff1, 0.0, 1e-12);
  EXPECT_ARRAY_NEAR(A, B);

  // test symmetrization with breaking the symmetry
  A(0, 1, 1, 0, 0, 0) += 0.1;
  A(1, 0, 1, 0, 0, 0) -= 0.1;
  auto const [max_diff2, max_idx2] = grp.symmetrize(A);
  EXPECT_NEAR(max_diff2, 0.1, 1e-10);
  EXPECT_ARRAY_NEAR(A, B);

  // test transformation to and initialization from representative data
  auto const vec = grp.get_representative_data(A);
  grp.init_from_representative_data(B, vec);
  EXPECT_EQ_ARRAY(A, B);
}

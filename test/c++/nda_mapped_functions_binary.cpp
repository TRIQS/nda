// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <cmath>
#include <complex>

// Constraint checks: confirm that disallowed forms do NOT satisfy the requires clause.
template <typename A, typename B>
concept has_nda_max = requires(A a, B b) { nda::max(a, b); };

static_assert(has_nda_max<nda::array<double, 2>, nda::array<double, 2>>);
static_assert(has_nda_max<double, double>);
static_assert(!has_nda_max<nda::array<double, 2>, nda::array<double, 3>>);
static_assert(!has_nda_max<nda::array<std::complex<double>, 2>, nda::array<std::complex<double>, 2>>);
static_assert(!has_nda_max<std::complex<double>, std::complex<double>>);
static_assert(!has_nda_max<nda::array<double, 2>, double>);
static_assert(!has_nda_max<double, nda::array<double, 2>>);

TEST(NDA, MapBinaryMaxMin) {
  // 2D double arrays
  nda::array<double, 2> A2{{1.0, 4.0}, {9.0, 16.0}};
  nda::array<double, 2> B2{{2.0, 3.0}, {5.0, 7.0}};
  EXPECT_ARRAY_EQ(nda::max(A2, B2), (nda::array<double, 2>{{2.0, 4.0}, {9.0, 16.0}}));
  EXPECT_ARRAY_EQ(nda::min(A2, B2), (nda::array<double, 2>{{1.0, 3.0}, {5.0, 7.0}}));

  // 3D double arrays (rank-generic path), cross-checked against std::max element-wise
  auto A3   = nda::array<double, 3>::rand({2, 3, 4});
  auto B3   = nda::array<double, 3>::rand({2, 3, 4});
  auto res3 = nda::max(A3, B3);
  nda::for_each(A3.shape(), [&A3, &B3, &res3](auto i, auto j, auto k) { EXPECT_EQ(res3(i, j, k), std::max(A3(i, j, k), B3(i, j, k))); });

  // Scalar form (mapped::operator() scalar specialization)
  EXPECT_EQ(nda::max(3.0, 4.0), 4.0);
  EXPECT_EQ(nda::min(3.0, 4.0), 3.0);
  EXPECT_EQ(nda::max(2, 5), 5);
  EXPECT_EQ(nda::min(2, 5), 2);

  // 1D int arrays
  nda::array<int, 1> Ai{1, 4, 9, 16};
  nda::array<int, 1> Bi{2, 3, 5, 7};
  EXPECT_EQ_ARRAY(nda::max(Ai, Bi), (nda::array<int, 1>{2, 4, 9, 16}));
  EXPECT_EQ_ARRAY(nda::min(Ai, Bi), (nda::array<int, 1>{1, 3, 5, 7}));
}

// Lazy composition: covers the MAX_ABS downstream use case.
TEST(NDA, MapBinaryMaxOfAbs) {
  nda::array<double, 2> A{{-3.0, 2.0}, {1.0, -7.0}};
  nda::array<double, 2> B{{1.0, -4.0}, {-5.0, 6.0}};
  EXPECT_ARRAY_EQ(nda::max(nda::abs(A), nda::abs(B)), (nda::array<double, 2>{{3.0, 4.0}, {5.0, 7.0}}));
}

// Lazy composition: covers the NORM_2 binary form (sqrt(|a|^2 + |b|^2)) without a dedicated helper.
TEST(NDA, MapBinaryHypotByCompositionReal) {
  nda::array<double, 1> A{3.0, 0.0, 1.0};
  nda::array<double, 1> B{4.0, 5.0, 0.0};
  EXPECT_ARRAY_NEAR(nda::sqrt(nda::abs2(A) + nda::abs2(B)), (nda::array<double, 1>{5.0, 5.0, 1.0}), fp_tol<double>);
}

TEST(NDA, MapBinaryHypotByCompositionComplex) {
  nda::vector<std::complex<double>> A{{3.0, 0.0}, {0.0, 0.0}, {1.0, 2.0}};
  nda::vector<std::complex<double>> B{{0.0, 4.0}, {5.0, 0.0}, {0.0, 0.0}};
  EXPECT_ARRAY_NEAR(nda::sqrt(nda::abs2(A) + nda::abs2(B)), nda::array<double, 1>{5.0, 5.0, std::sqrt(5.0)}, fp_tol<double>);
}

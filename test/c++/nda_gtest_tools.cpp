// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>

using namespace std::complex_literals;

// Relative comparison of real arrays spanning several orders of magnitude, where the
// relative criterion catches an error in a tiny element that an absolute one is blind to.
TEST(GTestTools, RelCloseReal) {
  nda::array<double, 1> a{1.0, 2.0, 1.0e-8};
  nda::array<double, 1> b{1.0, 2.0, 1.0001e-8};

  // the error is on the small element: |1e-8 - 1.0001e-8| / 1.0001e-8 ~ 1e-4
  EXPECT_ARRAY_REL_NEAR(a, b, 1.1e-4);
  EXPECT_FALSE(array_are_rel_close(a, b, 1.0e-5));

  // that same discrepancy is an absolute difference of only 1e-12, so the absolute
  // criterion with its default 1e-10 tolerance considers the arrays close -- exactly
  // the blind spot array_are_rel_close is meant to cover
  EXPECT_TRUE(array_are_close(a, b));
}

// Relative comparison of complex arrays: abs() must yield the magnitude.
TEST(GTestTools, RelCloseComplex) {
  nda::array<std::complex<double>, 1> a{1.0 + 0.0i, 2.0i};
  nda::array<std::complex<double>, 1> b{1.0 + 1.0e-6i, 2.0i};
  EXPECT_ARRAY_REL_NEAR(a, b, 1.0e-5);
  EXPECT_FALSE(array_are_rel_close(a, b, 1.0e-7));
}

// Exactly equal arrays and the vanishing-element convention (no division by zero).
TEST(GTestTools, RelCloseEqualAndZeros) {
  nda::array<double, 2> a{{1.0, 2.0}, {3.0, 4.0}};
  EXPECT_ARRAY_REL_NEAR(a, a);

  // two vanishing elements compare equal
  nda::array<double, 1> z0{0.0, 0.0};
  nda::array<double, 1> z1{0.0, 0.0};
  EXPECT_ARRAY_REL_NEAR(z0, z1);

  // empty arrays are considered equal
  nda::array<double, 1> e0(0), e1(0);
  EXPECT_ARRAY_REL_NEAR(e0, e1);
}

// Differing shapes must fail rather than throw or read out of bounds.
TEST(GTestTools, RelCloseShapeMismatch) {
  nda::array<double, 1> a{1.0, 2.0, 3.0};
  nda::array<double, 1> b{1.0, 2.0};
  EXPECT_FALSE(array_are_rel_close(a, b));
}

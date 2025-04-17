// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/stdutil/complex.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

// Generic function to test the complex arithmetic extensions.
template <typename X, typename Y, typename W>
  requires(std::is_floating_point_v<X> or std::is_floating_point_v<Y>)
void test_mixed_math() {
  W w        = 2.0;
  auto x     = std::complex<X>(2.0, 1.0);
  auto y     = std::complex<Y>(2.0, 2.0);
  double tol = std::pow(10.0, -std::numeric_limits<decltype((x + y).real())>::digits10);

  auto z = x + y;
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 3, tol);
  z = x - y;
  ASSERT_NEAR(z.real(), 0, tol);
  ASSERT_NEAR(z.imag(), -1, tol);
  z = x * y;
  ASSERT_NEAR(z.real(), 2, tol);
  ASSERT_NEAR(z.imag(), 6, tol);
  z = x / y;
  ASSERT_NEAR(z.real(), 0.75, tol);
  ASSERT_NEAR(z.imag(), -0.25, tol);

  z = y + x;
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 3, tol);
  z = y - x;
  ASSERT_NEAR(z.real(), 0, tol);
  ASSERT_NEAR(z.imag(), 1, tol);
  z = y * x;
  ASSERT_NEAR(z.real(), 2, tol);
  ASSERT_NEAR(z.imag(), 6, tol);
  z = y / x;
  ASSERT_NEAR(z.real(), 1.2, tol);
  ASSERT_NEAR(z.imag(), 0.4, tol);

  z = w + x;
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 1, tol);
  z = w - x;
  ASSERT_NEAR(z.real(), 0, tol);
  ASSERT_NEAR(z.imag(), -1, tol);
  z = w * x;
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 2, tol);
  z = w / x;
  ASSERT_NEAR(z.real(), 0.8, tol);
  ASSERT_NEAR(z.imag(), -0.4, tol);
}

TEST(NDA, ComplexMixedMathFloatDouble) { test_mixed_math<float, double, double>(); }

TEST(NDA, ComplexMixedMathFloatInt) { test_mixed_math<float, int, int>(); }

TEST(NDA, ComplexMixedMathIntFloat) { test_mixed_math<int, float, double>(); }

TEST(NDA, ComplexMixedMathLongDouble) { test_mixed_math<long, double, double>(); }

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
// Authors: Sergei Iskakov, Nils Wentzell

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

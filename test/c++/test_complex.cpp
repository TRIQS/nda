//
// Created by iskakoff on 7/24/23.
//
#include "test_common.hpp"
#include <nda/stdutil/complex.hpp>


template<typename X, typename Y, typename W>
  requires(std::is_floating_point_v<X> or std::is_floating_point_v<Y>)
void test_mixed_math(){
  std::complex<X> x;
  std::complex<Y> y;
  W w;
  x = std::complex(2.0, 1.0);
  y = std::complex(2.0, 2.0);
  w = 2.0;
  auto z = x + y;
  double tol = std::pow(10.0, -std::numeric_limits<decltype(z.real())>::digits10);
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 3, tol);
  z = x-y;
  ASSERT_NEAR(z.real(), 0, tol);
  ASSERT_NEAR(z.imag(), -1, tol);
  z = x*y;
  ASSERT_NEAR(z.real(), 2, tol);
  ASSERT_NEAR(z.imag(), 6, tol);
  z = x/y;
  ASSERT_NEAR(z.real(), 0.75, tol);
  ASSERT_NEAR(z.imag(), -0.25, tol);

  z = y+x;
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 3, tol);
  z = y-x;
  ASSERT_NEAR(z.real(), 0, tol);
  ASSERT_NEAR(z.imag(), 1, tol);
  z = y*x;
  ASSERT_NEAR(z.real(), 2, tol);
  ASSERT_NEAR(z.imag(), 6, tol);
  z = y/x;
  ASSERT_NEAR(z.real(), 1.2, tol);
  ASSERT_NEAR(z.imag(), 0.4, tol);

  z = w+x;
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 1, tol);
  z = w-x;
  ASSERT_NEAR(z.real(), 0, tol);
  ASSERT_NEAR(z.imag(), -1, tol);
  z = w*x;
  ASSERT_NEAR(z.real(), 4, tol);
  ASSERT_NEAR(z.imag(), 2, tol);
  z = w/x;
  ASSERT_NEAR(z.real(), 0.8, tol);
  ASSERT_NEAR(z.imag(), -0.4, tol);
}

TEST(complex_arithmetics_test, mixed_math_float_double) {
  test_mixed_math<float, double, double>();
}

TEST(complex_arithmetics_test, mixed_math_float_int) {
  test_mixed_math<float, int, int>();
}

TEST(complex_arithmetics_test, mixed_math_int_float) {
  test_mixed_math<int, float, double>();
}


TEST(complex_arithmetics_test, mixed_math_long_double) {
  test_mixed_math<long, double, double>();
}

MAKE_MAIN;

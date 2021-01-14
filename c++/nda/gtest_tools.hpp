// Copyright (c) 2019-2020 Simons Foundation
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

#pragma once

#ifndef NDA_DEBUG
#define NDA_DEBUG
#endif

#include "nda/basic_array.hpp"
#include <iostream>
#include <sstream>
#include <gtest/gtest.h> // NOLINT

/*#if H5_VERSION_GE(1, 8, 9)*/
//#include <h5/serialization.hpp>
//#endif

//using dcomplex = std::complex<double>;

// Complex are close
template <typename X, typename Y>
::testing::AssertionResult complex_are_close(X const &x, Y const &y, double precision = 1.e-10) {
  using std::abs;
  if (abs(x - y) < precision)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "abs(x-y) = " << abs(x - y) << "\n X = " << x << "\n Y = " << y;
}

#define EXPECT_COMPLEX_NEAR(X, ...) EXPECT_TRUE(complex_are_close(X, __VA_ARGS__))

// Arrays are equal
template <typename X, typename Y>
::testing::AssertionResult array_are_equal(X const &x, Y const &y) {
  if (x.shape() != y.shape())
    return ::testing::AssertionFailure() << "Comparing two arrays of different size "
                                         << "\n X = " << x << "\n Y = " << y;
  if (x == y)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "Arrays have different elements\n X = " << x << "\n Y = " << y;
}

#define EXPECT_EQ_ARRAY(X, Y) EXPECT_TRUE(array_are_equal(X, Y));
#define EXPECT_ARRAY_EQ(X, Y) EXPECT_TRUE(array_are_equal(X, Y));

// Arrays are close
template <typename X, typename Y>
::testing::AssertionResult array_are_close(X const &x1, Y const &y1, double precision = 1.e-10) {
  nda::array<nda::get_value_t<X>, nda::get_rank<X>> x = x1;
  nda::array<nda::get_value_t<X>, nda::get_rank<X>> y = y1;

  if (x.shape() != y.shape())
    return ::testing::AssertionFailure() << "Comparing two arrays of different size "
                                         << "\n X = " << x << "\n Y = " << y;

  // both x, y are contiguous, I check with basic tools instead of max_element(abs(x - y))
  if (x.size() == 0) return ::testing::AssertionSuccess();
  auto xx      = make_regular(x);
  auto yy      = make_regular(y);
  auto maxdiff = max_element(abs(make_regular(xx - yy)));
  if (maxdiff < precision)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "max_element(abs(x-y)) = " << maxdiff << "\n X = " << x << "\n Y = " << y;
}

#define EXPECT_ARRAY_NEAR(X, ...) EXPECT_TRUE(array_are_close(X, __VA_ARGS__))

// Arrays is almost 0
template <typename X>
::testing::AssertionResult array_almost_zero(X const &x1) {
  double precision                                    = 1.e-10;
  nda::array<nda::get_value_t<X>, nda::get_rank<X>> x = x1;

  if (x.size() == 0 || max_element(abs(x)) < precision)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "max_element(abs(x-y)) = " << max_element(abs(x)) << "\n X = " << x;
}

#define EXPECT_ARRAY_ZERO(X) EXPECT_TRUE(array_almost_zero(X))
//
template <typename X, typename Y>
::testing::AssertionResult generic_are_near(X const &x, Y const &y) {
  double precision = 1.e-12;
  using std::abs;
  if (abs(x - y) > precision)
    return ::testing::AssertionFailure() << "X = " << x << " and Y = " << y << " are different. \n Difference is : " << abs(x - y);
  return ::testing::AssertionSuccess();
}
#define EXPECT_CLOSE(X, Y) EXPECT_TRUE(generic_are_near(X, Y));

// ------------------  HDF5 --------------------
//
// We serialize to H5, deserialize, compare

//template <typename T> T rw_h5(T const &x, std::string filename = "ess", std::string name = "x") {

//namespace h5 = h5;
//T y; // must be default constructible

//{
//h5::file file(filename + ".h5", 'w');
//h5_write(file, name, x);
//}

//{
//h5::file file(filename + ".h5", 'r');
//h5_read(file, name, y);
//}

//#if H5_VERSION_GE(1, 8, 9)

////#define TRIQS_TEST_USE_H5_SERIA
//#ifdef TRIQS_TEST_USE_H5_SERIA

//std::cerr << "Checking H5 serialization/deserialization of \n " << triqs::utility::demangle(typeid(x).name()) << std::endl;
//auto s  = h5::serialize(x);
//T x2    = h5::deserialize<T>(s);
//auto s2 = h5::serialize(x);
//std::cerr << "Length of serialization string " << first_dim(s) << std::endl;
//EXPECT_EQ_ARRAY(s, s2); // << "Test h5 save, load, save, compare has failed !";
//#endif

//#endif

//return y;
//}

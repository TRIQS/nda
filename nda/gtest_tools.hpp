/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2015 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#define NDA_DEBUG
//#include <mpi/mpi.hpp>
#include <nda/array.hpp>
//#include <triqs/h5.hpp>
//#include <triqs/utility/typeid_name.hpp>
#include <iostream>
#include <sstream>
#include <gtest/gtest.h> // NOLINT

/*#if H5_VERSION_GE(1, 8, 9)*/
//#include <triqs/h5/serialization.hpp>
//#endif

using dcomplex = std::complex<double>;
//using triqs::clef::placeholder;

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

  if (x.size() == 0 || max_element(abs(x - y)) == 0)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "max_element(abs(x-y)) = " << max_element(abs(x - y)) << "\n X = " << x << "\n Y = " << y;
}

#define EXPECT_EQ_ARRAY(X, Y) EXPECT_TRUE(array_are_equal(X, Y));
#define EXPECT_ARRAY_EQ(X, Y) EXPECT_TRUE(array_are_equal(X, Y));

// Arrays are close
template <typename X, typename Y>
::testing::AssertionResult array_are_close(X const &x1, Y const &y1, double precision = 1.e-10) {
  nda::array<typename X::value_t, X::rank> x = x1;
  nda::array<typename X::value_t, X::rank> y = y1;
  if (x.shape() != y.shape())
    return ::testing::AssertionFailure() << "Comparing two arrays of different size "
                                         << "\n X = " << x << "\n Y = " << y;

  // both x, y are contiguous, I check with basic tools instead of max_element(abs(x - y))
  if (x.size() == 0) return ::testing::AssertionSuccess();
  using std::abs;
  using std::max;
  auto max_diff = abs(*x.data_start() - *y.data_start());
  for (long i = 0; i < x.size(); ++i) max_diff = max(max_diff, abs(x.data_start()[i] - y.data_start()[i]));
  if (max_diff < precision)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "max_element(abs(x-y)) = " << max_diff << "\n X = " << x << "\n Y = " << y;
}

#define EXPECT_ARRAY_NEAR(X, ...) EXPECT_TRUE(array_are_close(X, __VA_ARGS__))

// Arrays is almost 0
template <typename X>
::testing::AssertionResult array_almost_zero(X const &x1) {
  double precision                           = 1.e-10;
  nda::array<typename X::value_t, X::rank> x = x1;

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

//namespace h5 = triqs::h5;
//T y; // must be default constructible

//{
//h5::file file(filename + ".h5", H5F_ACC_TRUNC);
//h5_write(file, name, x);
//}

//{
//h5::file file(filename + ".h5", H5F_ACC_RDONLY);
//h5_read(file, name, y);
//}

//#if H5_VERSION_GE(1, 8, 9)

////#define TRIQS_TEST_USE_H5_SERIA
//#ifdef TRIQS_TEST_USE_H5_SERIA

//std::cerr << "Checking H5 serialization/deserialization of \n " << triqs::utility::demangle(typeid(x).name()) << std::endl;
//auto s  = triqs::h5::serialize(x);
//T x2    = triqs::h5::deserialize<T>(s);
//auto s2 = triqs::h5::serialize(x);
//std::cerr << "Length of serialization string " << first_dim(s) << std::endl;
//EXPECT_EQ_ARRAY(s, s2); // << "Test h5 save, load, save, compare has failed !";
//#endif

//#endif

//return y;
//}

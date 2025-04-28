// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides convenient tools for checking nda::basic_array and nda::basic_array_view objects with googletest.
 */

#pragma once

#ifndef NDA_DEBUG
#define NDA_DEBUG
#endif

#include "./nda.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <sstream>

/**
 * @addtogroup testing
 * @{
 */

/**
 * @brief Check the absolute difference of two (complex) numbers.
 *
 * @tparam X Type of the first number.
 * @tparam Y Type of the second number.
 * @param x First number.
 * @param y Second number.
 * @param precision Required precision for the comparison to be considered successful.
 * @return `::testing::AssertionSuccess()` if the absolute difference is less than the given precision,
 * `::testing::AssertionFailure()` otherwise.
 */
template <typename X, typename Y>
::testing::AssertionResult complex_are_close(X const &x, Y const &y, double precision = 1.e-10) {
  using std::abs;
  if (abs(x - y) < precision)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "abs(x - y) = " << abs(x - y) << "\n x = " << x << "\n y = " << y;
}

/// Macro that expects ::complex_are_close to return true.
#define EXPECT_COMPLEX_NEAR(X, ...) EXPECT_TRUE(complex_are_close(X, __VA_ARGS__))

/**
 * @brief Check that two arrays/views are equal, i.e. that they have the same shape and the same elements.
 *
 * @tparam X Type of the first array/view.
 * @tparam Y Type of the second array/view.
 * @param x First array/view.
 * @param y Second array/view.
 * @return `::testing::AssertionSuccess()` if the arrays/view have the same shape and the same elements,
 * `::testing::AssertionFailure()` otherwise.
 */
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

/// Macro that expects ::array_are_equal to return true.
#define EXPECT_EQ_ARRAY(X, Y) EXPECT_TRUE(array_are_equal(X, Y));

/// Macro that expects ::array_are_equal to return true.
#define EXPECT_ARRAY_EQ(X, Y) EXPECT_TRUE(array_are_equal(X, Y));

/**
 * @brief Check that two arrays/views are close, i.e. that they have the same shape and that the largest element of
 * their absolute difference is less than a given precision.
 *
 * @tparam X Type of the first array/view.
 * @tparam Y Type of the second array/view.
 * @param x First array/view.
 * @param y Second array/view.
 * @param precision Required precision for the comparison to be considered successful.
 * @return `::testing::AssertionSuccess()` if the arrays/view have the same shape and largest element of their absolute
 * difference is less than a given precision. `::testing::AssertionFailure()` otherwise.
 */
template <typename X, typename Y>
::testing::AssertionResult array_are_close(X const &x, Y const &y, double precision = 1.e-10) {
  nda::array<nda::get_value_t<X>, nda::get_rank<X>> x_reg = x;
  nda::array<nda::get_value_t<X>, nda::get_rank<X>> y_reg = y;

  // check their shapes
  if (x_reg.shape() != y_reg.shape())
    return ::testing::AssertionFailure() << "Comparing two arrays of different size "
                                         << "\n X = " << x_reg << "\n Y = " << y_reg;

  // empty arrays are considered equal
  if (x_reg.size() == 0) return ::testing::AssertionSuccess();

  // check their difference
  const auto maxdiff = max_element(abs(make_regular(x_reg - y_reg)));
  if (maxdiff < precision)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "max_element(abs(X - Y)) = " << maxdiff << "\n X = " << x_reg << "\n Y = " << y_reg;
}

/// Macro that expects ::array_are_close to return true.
#define EXPECT_ARRAY_NEAR(X, ...) EXPECT_TRUE(array_are_close(X, __VA_ARGS__))

/**
 * @brief Check that an array/view is close to zero, i.e. that its largest absolute element is less than 1e-10.
 *
 * @tparam X Type of the array/view.
 * @param x Array/View.
 * @return `::testing::AssertionSuccess()` if the absolute value of every element is less than 1e-10.
 * `::testing::AssertionFailure()` otherwise.
 */
template <typename X>
::testing::AssertionResult array_almost_zero(X const &x) {
  nda::array<nda::get_value_t<X>, nda::get_rank<X>> x_reg = x;

  constexpr double eps = 1.e-10;
  const auto max       = max_element(abs(x_reg));
  if (x_reg.size() == 0 || max < eps)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "max_element(abs(X)) = " << max << "\n X = " << x_reg;
}

/// Macro that expects ::array_almost_zero to return true.
#define EXPECT_ARRAY_ZERO(X) EXPECT_TRUE(array_almost_zero(X))

/**
 * @brief Check that that two generic objects are close, i.e. that their absolute difference is less than 1e-12.
 *
 * @tparam X Type of the first object.
 * @tparam Y Type of the second object.
 * @param x First object.
 * @param y Second object.
 * @return `::testing::AssertionSuccess()` if the absolute value of their difference is less than 1e-12.
 * `::testing::AssertionFailure()` otherwise.
 */
template <typename X, typename Y>
::testing::AssertionResult generic_are_near(X const &x, Y const &y) {
  double precision = 1.e-12;
  using std::abs;
  if (abs(x - y) > precision)
    return ::testing::AssertionFailure() << "X = " << x << " and Y = " << y << " are different. \n Difference is: " << abs(x - y);
  return ::testing::AssertionSuccess();
}

/// Macro that expects ::generic_are_near to return true.
#define EXPECT_CLOSE(X, Y) EXPECT_TRUE(generic_are_near(X, Y));

/** @} */

// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides automatic assignment for std::vector.
 */

#pragma once

#include "../clef.hpp"

#include <utility>
#include <vector>

namespace nda::clef {

  /**
   * @addtogroup clef_autoassign
   * @{
   */

  namespace detail {

    // Helper function to auto assign to a std::vector object.
    template <typename T, typename RHS>
    void clef_auto_assign_std_vector_impl(T &x, RHS &&rhs) {
      x = std::forward<RHS>(rhs);
    }

    // Helper function to auto assign to a std::vector object.
    template <typename Expr, int... Is, typename T>
    void clef_auto_assign_std_vector_impl(T &x, make_fun_impl<Expr, Is...> &&rhs) { // NOLINT (why rvalue reference?)
      clef_auto_assign_subscript(x, std::forward<make_fun_impl<Expr, Is...>>(rhs));
    }

  } // namespace detail

  /**
   * @brief Overload of `clef_auto_assign_subscript` function for std::vector.
   *
   * @tparam T Value type of the std::vector.
   * @tparam F Callable type.
   * @param v std::vector object.
   * @param f Callable object.
   */
  template <typename T, typename F>
  void clef_auto_assign_subscript(std::vector<T> &v, F f) {
    for (size_t i = 0; i < v.size(); ++i) detail::clef_auto_assign_std_vector_impl(v[i], f(i));
  }

  /** @} */

} // namespace nda::clef

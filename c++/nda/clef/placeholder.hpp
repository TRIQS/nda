// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides placeholders for the clef library.
 */

#pragma once

#include "./expression.hpp"
#include "./utils.hpp"

#include <type_traits>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_placeholders
   * @{
   */

  /// @cond
  // Forward declarations.
  template <int N, typename T>
  struct pair;
  /// @endcond

  /**
   * @brief A placeholder is an empty struct, labelled by an int.
   *
   * @details It is used in lazy expressions. For example:
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto expr = i_ + j_;
   * auto res = nda::clef::eval(expr, i_ = 1.0, j_ = 2.0); // double res = 3.0;
   * @endcode
   *
   * Here `expr` is a lazy binary nda::clef::expr with the nda::clef::tags::plus tag, which can be evaluated later on
   * with the nda::clef::eval function and by assigning values to the placeholders (see nda::clef::pair).
   *
   * @tparam N Integer label (must be < 64).
   */
  template <int N>
  struct placeholder {
    static_assert(N >= 0 && N < 64, "Placeholder index must be in {0, 1, ..., 63}");

    /// Integer label.
    static constexpr int index = N;

    /**
     * @brief Assign a value to the placeholder.
     *
     * @tparam RHS Type of the right-hand side.
     * @param rhs Right-hand side of the assignment.
     * @return An nda::clef::pair object, containing the integer label of the placeholder and the value assigned to it.
     */
    template <typename RHS>
    pair<N, RHS> operator=(RHS &&rhs) const { // NOLINT (we want to return a pair)
      return {std::forward<RHS>(rhs)};
    }

    /**
     * @brief Function call operator.
     *
     * @tparam Args Types of the function call arguments.
     * @param args Function call arguments.
     * @return An nda::clef::expr object with the nda::clef::tags::function tag containing the placeholder (for the
     * callable) and the given arguments.
     */
    template <typename... Args>
    auto operator()(Args &&...args) const {
      return expr<tags::function, placeholder, expr_storage_t<Args>...>{tags::function{}, *this, std::forward<Args>(args)...};
    }

    /**
     * @brief Subscript operator.
     *
     * @tparam T Type of the subscript argument.
     * @param t Subscript argument.
     * @return An nda::clef::expr object with the nda::clef::tags::subscript tag containing the placeholder (for the
     * object to be accessed) and the given argument.
     */
    template <typename T>
    auto operator[](T &&t) const {
      return expr<tags::subscript, placeholder, expr_storage_t<T>>{tags::subscript{}, *this, std::forward<T>(t)};
    }
  };

  /**
   * @brief A pair consisting of a placeholder and its assigned value.
   *
   * @details In most cases, the user should not have to explicitly create or handle pair objects. Instead, they are
   * constructed indirectly by assigning a value to a placeholder when calling nda::clef::eval
   * (see nda::clef::placeholder for an example).
   *
   * @tparam N Integer label of the placeholder.
   * @tparam T Value type.
   */
  template <int N, typename T>
  struct pair {
    /// Value assigned to the placeholder (can be an lvalue reference).
    T rhs;

    /// Integer label of the placeholder.
    static constexpr int p = N;

    /// Type of the value after applying std::decay.
    using value_type = std::decay_t<T>;
  };

  namespace detail {

    // Specialization of force_copy_in_expr_impl for nda::clef::placeholder types (always true).
    template <int N>
    constexpr bool force_copy_in_expr_impl<placeholder<N>> = true;

    // Specialization of ph_set for nda::clef::placeholder types.
    template <int N>
    struct ph_set<placeholder<N>> {
      static constexpr ull_t value = 1ull << N;
    };

    // Specialization of ph_set for nda::clef::pair types.
    template <int N, typename T>
    struct ph_set<pair<N, T>> : ph_set<placeholder<N>> {};

    // Specialization of is_lazy_impl for nda::clef::placeholder types.
    template <int N>
    constexpr bool is_lazy_impl<placeholder<N>> = true;

  } // namespace detail

  /** @} */

} // namespace nda::clef

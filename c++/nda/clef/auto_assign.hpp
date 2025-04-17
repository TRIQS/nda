// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functionality to assign values to lazy callable objects.
 */

#pragma once

#include "./expression.hpp"
#include "./function.hpp"
#include "./utils.hpp"
#include "./placeholder.hpp"
#include "../macros.hpp"

#include <functional>
#include <tuple>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_autoassign
   * @{
   */

  /**
   * @brief Overload of `clef_auto_assign` function for std::reference_wrapper objects.
   *
   * @details Simply calls the `clef_auto_assign` for the object contained in the wrapper.
   *
   * @tparam T Type of the object stored in the std::reference_wrapper.
   * @tparam RHS Type of the right-hand side.
   * @param wrapper std::reference_wrapper object.
   * @param rhs Right-hand side object.
   */
  template <typename T, typename RHS>
  FORCEINLINE void clef_auto_assign(std::reference_wrapper<T> wrapper, RHS &&rhs) {
    clef_auto_assign(wrapper.get(), std::forward<RHS>(rhs));
  }

  /**
   * @brief Overload of `clef_auto_assign` function for terminal expressions.
   *
   * @details Simply calls the `clef_auto_assign` for the child node of the expression.
   *
   * @tparam T Type of the expression's child node.
   * @tparam RHS Type of the right-hand side.
   * @param ex nda::clef::expr object with the nda::clef::tags::terminal tag.
   * @param rhs Right-hand side object.
   */
  template <typename T, typename RHS>
  FORCEINLINE void clef_auto_assign(expr<tags::terminal, T> const &ex, RHS &&rhs) {
    clef_auto_assign(std::get<0>(ex.childs), std::forward<RHS>(rhs));
  }

  /**
   * @brief Overload of `clef_auto_assign` function for generic expressions.
   *
   * @details It calls the specialized `operator<<` function for the given expression and right-hand side.
   *
   * @tparam Tag Tag of the expression.
   * @tparam Childs Types of the child nodes.
   * @tparam RHS Type of the right-hand side.
   * @param ex nda::clef::expr object.
   * @param rhs Right-hand side object.
   */
  template <typename Tag, typename... Childs, typename RHS>
  FORCEINLINE void clef_auto_assign(expr<Tag, Childs...> const &ex, RHS const &rhs) {
    ex << rhs;
  }

  // Overload of `clef_auto_assign` for rvalue references of generic expressions.
  template <typename Tag, typename... Childs, typename RHS>
  FORCEINLINE void clef_auto_assign(expr<Tag, Childs...> &&ex, RHS const &rhs) { // NOLINT (is the rvalue reference overload needed?)
    ex << rhs;
  }

  /**
   * @brief Assign values to the underlying object of a lazy function call expression.
   *
   * @details This calls the `clef_auto_assign` overload for the underlying object of the given expression using ADL.
   * It has to be implemented for all supported types.
   *
   * @tparam F Type of the callable object in the function call expression.
   * @tparam RHS Type of the right-hand side.
   * @tparam Is Integer labels of the placeholders in the function call expression.
   * @param ex nda::clef::expr object with the nda::clef::tags::function tag.
   * @param rhs Right-hand side object.
   */
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::function, F, placeholder<Is>...> const &ex, RHS &&rhs) {
    static_assert(detail::all_different(Is...), "Error in clef operator<<: Two of the placeholders on the LHS are the same");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }

  // Overload of nda::clef::operator<< for rvalue reference expressions.
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::function, F, placeholder<Is>...> &&ex, RHS &&rhs) { // NOLINT (is the rvalue reference overload needed?)
    static_assert(detail::all_different(Is...), "Error in clef operator<<: Two of the placeholders on the LHS are the same");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }

  // Overload of nda::clef::operator<< for lvalue reference expressions.
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::function, F, placeholder<Is>...> &ex, RHS &&rhs) {
    static_assert(detail::all_different(Is...), "Error in clef operator<<: Two of the placeholders on the LHS are the same");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }

  // Delete functions to avoid nonsensical cases, e.g. f(x_ + y_) << RHS.
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::function, F, T...> &&ex, RHS &&rhs) = delete; // NOLINT (no forwarding required here)
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::function, F, T...> &ex, RHS &&rhs) = delete; // NOLINT (no forwarding required here)
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::function, F, T...> const &ex, RHS &&rhs) = delete; // NOLINT (no forwarding required here)

  /** @} */

} // namespace nda::clef

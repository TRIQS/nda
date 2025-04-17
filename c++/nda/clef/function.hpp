// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functionality to turn lazy expressions into callable objects and to generally simplify the evaluation
 * of lazy expressions.
 */

#pragma once

#include "./eval.hpp"
#include "./utils.hpp"
#include "./placeholder.hpp"
#include "../macros.hpp"

#include <type_traits>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Helper struct to simplify calls to nda::clef::eval.
   *
   * @details It stores the object (usually a lazy expression) which we want to evaluate and takes integer labels of
   * placeholders as template arguments.
   *
   * See nda::clef::make_function for an example.
   *
   * @tparam T Type of the object.
   * @tparam Is Integer labels of the placeholders in the expression.
   */
  template <typename T, int... Is>
  struct make_fun_impl {
    /// Object to be evaluated.
    T obj;

    /**
     * @brief Function call operator.
     *
     * @details The arguments together with the integer labels (template parameters of the class) are used to construct
     * nda::clef::pair objects. The stored object and the pairs are then passed to the nda::clef::eval function to
     * perform the evaluation.
     *
     * @tparam Args Types of the function call arguments.
     * @param args Function call arguments.
     * @return Result of the evaluation.
     */
    template <typename... Args>
    FORCEINLINE decltype(auto) operator()(Args &&...args) const {
      return eval(obj, pair<Is, Args>{std::forward<Args>(args)}...);
    }
  };

  /**
   * @brief Factory function for nda::clef::make_fun_impl objects.
   *
   * @details The given arguments are used to construct a new nda::clef::make_fun_impl object. The first argument is
   * forwarded to its constructor and the integer labels of the remaining placeholder arguments are used in its template
   * argument list.
   *
   * The following example shows how to turn a binary lazy expression `ex` into a callable object `f` that takes two
   * arguments:
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto ex = i_ + j_;
   * auto f = nda::clef::make_function(ex, i_, j_);
   * auto res = f(1, 2);    // int res = 3;
   * auto res2 = f(1.5, 2); // double res2 = 3.5;
   * @endcode
   *
   * @note There is no check if the given placeholders actually match placeholders in the object to be evaluated.
   *
   * @tparam T Type of the object.
   * @tparam Phs Types of the placeholders.
   * @param obj Object to be stored in the nda::clef::make_fun_impl object.
   * @return A callable nda::clef::make_fun_impl object that takes as many arguments as placeholders were given.
   */
  template <typename T, typename... Phs>
  FORCEINLINE auto make_function(T &&obj, Phs...) {
    return make_fun_impl<std::decay_t<T>, Phs::index...>{std::forward<T>(obj)};
  }

  /** @} */

  /**
   * @ingroup clef_eval
   * @brief Specialization of nda::clef::evaluator for nda::clef::make_fun_impl types.
   *
   * @tparam T Type of the object stored in the nda::clef::make_fun_impl object.
   * @tparam Is Integer labels of the nda::clef::make_fun_impl type.
   * @tparam Pairs Types of the nda::clef::pair objects.
   */
  template <typename T, int... Is, typename... Pairs>
  struct evaluator<make_fun_impl<T, Is...>, Pairs...> {
    /// Type of the evaluator used.
    using e_t = evaluator<T, Pairs...>;

    /// Constexpr variable that is true if all the placeholders are assigned a value.
    static constexpr bool is_lazy = (detail::ph_set<make_fun_impl<T, Is...>>::value != detail::ph_set<Pairs...>::value);

    /**
     * @brief Evaluate the nda::clef::make_fun_impl object.
     *
     * @details It first evaluates the object stored in the given nda::clef::make_fun_impl object by applying the given
     * pairs. The result is then used to construct a new nda::clef::make_fun_impl instance together with the
     * placeholders from the original nda::clef::make_fun_impl object.
     *
     * @param f nda::clef::make_fun_impl object.
     * @param pairs nda::clef::pair objects.
     * @return nda::clef::make_fun_impl object containing the evaluated expression.
     */
    FORCEINLINE decltype(auto) operator()(make_fun_impl<T, Is...> const &f, Pairs &...pairs) const {
      return make_function(e_t()(f.obj, pairs...), placeholder<Is>()...);
    }
  };

  namespace detail {

    // Specialization of is_function_impl for nda::clef::make_fun_impl types.
    template <typename Expr, int... Is>
    inline constexpr bool is_function_impl<make_fun_impl<Expr, Is...>> = true;

    // Specialization of ph_set for nda::clef::make_fun_impl types.
    template <typename Expr, int... Is>
    struct ph_set<make_fun_impl<Expr, Is...>> {
      static constexpr ull_t value = ph_filter<ph_set<Expr>::value, Is...>::value;
    };

    // Specialization of is_lazy_impl for nda::clef::make_fun_impl types.
    template <typename Expr, int... Is>
    constexpr bool is_lazy_impl<make_fun_impl<Expr, Is...>> = (ph_set<make_fun_impl<Expr, Is...>>::value != 0);

    // Specialization of force_copy_in_expr_impl for nda::clef::make_fun_impl types (always true).
    template <typename Expr, int... Is>
    constexpr bool force_copy_in_expr_impl<make_fun_impl<Expr, Is...>> = true;

  } // namespace detail

} // namespace nda::clef

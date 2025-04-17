// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functionality to evaluate lazy expressions from the clef library.
 */

#pragma once

#include "./expression.hpp"
#include "./operation.hpp"
#include "./placeholder.hpp"
#include "./utils.hpp"
#include "../macros.hpp"

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_eval
   * @{
   */

  /// @cond
  // Forward declarations.
  template <typename T, typename... Pairs>
  decltype(auto) eval(T const &obj, Pairs &&...pairs);
  /// @endcond

  /**
   * @brief Generic evaluator for types which do not have a specialized evaluator.
   *
   * @tparam T Type to be evaluated.
   * @tparam Pairs Types of nda::clef::pair objects.
   */
  template <typename T, typename... Pairs>
  struct evaluator {
    /// Constexpr variable that is true if the type `T` is lazy.
    static constexpr bool is_lazy = is_any_lazy<T>;

    /**
     * @brief Evaluate the object and ignore all given nda::clef::pair objects.
     *
     * @param t Object to be evaluated.
     * @return Const reference to the object.
     */
    FORCEINLINE T const &operator()(T const &t, Pairs &...) const { return t; }
  };

  /**
   * @brief Specialization of nda::clef::evaluator for nda::clef::placeholder types.
   *
   * @tparam N Integer label of the placeholder.
   * @tparam Is Integer labels of the placeholders in the given pairs.
   * @tparam Ts Types of the values in the given pairs.
   */
  template <int N, int... Is, typename... Ts>
  struct evaluator<placeholder<N>, pair<Is, Ts>...> {
    private:
    // Helper function to determine the position of the nda::clef::pair that has the nda::clef::placeholder<N>.
    template <size_t... Ps>
    static constexpr int get_position_of_N(std::index_sequence<Ps...>) {
      return ((Is == N ? int(Ps) + 1 : 0) + ...) - 1;
    }

    // The position of the nda::clef::pair that has the nda::clef::placeholder<N> (-1 if no such pair was given).
    static constexpr int N_position = get_position_of_N(std::make_index_sequence<sizeof...(Is)>{});

    public:
    /**
     * @brief Constexpr variable that is true if the there is no nda::clef::pair containing an nda::clef::placeholder
     * with label `N`.
     */
    static constexpr bool is_lazy = (N_position == -1);

    /**
     * @brief Evaluate the placeholder.
     *
     * @param pairs Pack of nda::clef::pair objects.
     * @return Value stored in the pair with the same integer label as the placeholder or a placeholder with the label
     * `N` if no such pair was given.
     */
    FORCEINLINE decltype(auto) operator()(placeholder<N>, pair<Is, Ts> &...pairs) const {
      if constexpr (not is_lazy) { // N is one of the Is
        auto &pair_N = std::get<N_position>(std::tie(pairs...));
        // the pair is a temporary constructed for the time of the eval call
        // if it holds a reference, we return it, else we move the rhs object out of the pair
        if constexpr (std::is_lvalue_reference_v<decltype(pair_N.rhs)>) {
          return pair_N.rhs;
        } else {
          return std::move(pair_N.rhs);
        }
      } else { // N is not one of the Is
        return placeholder<N>{};
      }
    }
  };

  /**
   * @brief Specialization of nda::clef::evaluator for std::reference_wrapper types.
   *
   * @tparam T Value type of the std::reference_wrapper.
   * @tparam Pairs Types of the nda::clef::pair objects.
   */
  template <typename T, typename... Pairs>
  struct evaluator<std::reference_wrapper<T>, Pairs...> {
    /// Constexpr variable that is always false.
    static constexpr bool is_lazy = false;

    /**
     * @brief Evaluate the std::reference_wrapper by redirecting the evaluation to the object contained in the wrapper.
     *
     * @param wrapper std::reference_wrapper object.
     * @param pairs Pack of nda::clef::pair objects.
     * @return The result of evaluating the object contained in the wrapper together with the given pairs.
     */
    FORCEINLINE decltype(auto) operator()(std::reference_wrapper<T> const &wrapper, Pairs const &...pairs) const {
      return eval(wrapper.get(), pairs...);
    }
  };

  /**
   * @brief Specialization of nda::clef::evaluator for nda::clef::expr types.
   *
   * @tparam Tag Tag of the expression.
   * @tparam Childs Types of the child nodes.
   * @tparam Pairs Types of the nda::clef::pair objects.
   */
  template <typename Tag, typename... Childs, typename... Pairs>
  struct evaluator<expr<Tag, Childs...>, Pairs...> {
    /// Constexpr variable that is true if any of the evaluators of the child nodes is lazy.
    static constexpr bool is_lazy = (evaluator<Childs, Pairs...>::is_lazy or ...);

    private:
    // Helper function to evaluate the given expression.
    template <size_t... Is>
    [[nodiscard]] FORCEINLINE decltype(auto) eval_impl(std::index_sequence<Is...>, expr<Tag, Childs...> const &ex, Pairs &...pairs) const {
      return op_dispatch<Tag>(std::integral_constant<bool, is_lazy>{}, eval(std::get<Is>(ex.childs), pairs...)...);
    }

    public:
    /**
     * @brief Evaluate the given expression by applying the given nda::clef::pair objects.
     *
     * @note Depending on the given expression as well as the the given pairs, the result of the evaluation might be
     * again a lazy expression.
     *
     * @param ex Expression to be evaluated.
     * @param pairs nda::clef::pair objects to be applied to the expression.
     * @return The result of the evaluation (calls nda::clef::op_dispatch indirectly through a helper function).
     */
    [[nodiscard]] FORCEINLINE decltype(auto) operator()(expr<Tag, Childs...> const &ex, Pairs &...pairs) const {
      return eval_impl(std::make_index_sequence<sizeof...(Childs)>(), ex, pairs...);
    }
  };

  /**
   * @brief Generic function to evaluate expressions and other types.
   *
   * @details Calls the correct nda::clef::evaluator for the given expression/type.
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto ex = i_ + j_;
   * auto res = nda::clef::eval(ex, i_ = 1, j_ = 2); // int res = 3;
   * @endcode
   *
   * Here, `ex` is a binary expression with the `+` tag and the placeholder `i_` and `j_` as its child nodes. The `eval`
   * function calls the correct evaluator for the given expression and the given pairs.
   *
   * @tparam T Type of the expression/object.
   * @tparam Pairs Types of the nda::clef::pair objects.
   * @param obj Expression/object to be evaluated.
   * @param pairs nda::clef::pair objects to be applied to the expression.
   * @return The result of the evaluation depending on the type of the expression/object.
   */
  template <typename T, typename... Pairs>
  FORCEINLINE decltype(auto) eval(T const &obj, Pairs &&...pairs) { // NOLINT (we don't want to forward here)
    return evaluator<T, std::remove_reference_t<Pairs>...>()(obj, pairs...);
  }

  /** @} */

} // namespace nda::clef

// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides operations for the clef library.
 */

#pragma once

#include "./expression.hpp"
#include "./utils.hpp"
#include "../macros.hpp"

#include <functional>
#include <type_traits>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Controls evaluation behavior of function nodes during partial expression evaluation.
   *
   * When partially evaluating an expression tree, function nodes have two evaluation modes:
   * - false (default): Function is evaluated only if all arguments are non-lazy values.
   *   If any argument is lazy, the node is preserved with evaluated children but the
   *   function itself is NOT called.
   * - true: Function is ALWAYS called with all arguments (lazy or non-lazy).
   *   This requires the function to properly handle non-lazy arguments by moving them
   *   into new expression nodes using make_expr_call.
   *
   * @tparam F Function type to specialize for
   */
  template <typename F>
  constexpr bool supports_partial_eval_of_calls = false;

  /**
   * @brief Controls evaluation behavior of subscript operations during partial expression evaluation.
   *
   * Similar to supports_partial_eval_of_calls but specifically for the subscript operator[].
   * When true, the subscript operation will be called even with lazy arguments.
   *
   * @tparam T Type to specialize subscript evaluation for
   */
  template <typename T>
  constexpr bool supports_partial_eval_of_subscript = false;

  namespace detail {

    // Get the value from a std::reference_wrapper or simply forward the argument of any other type.
    template <typename U>
    FORCEINLINE U &&fget(U &&x) {
      return std::forward<U>(x);
    }

    // Specialization of fget for std::reference_wrapper.
    template <typename U>
    FORCEINLINE decltype(auto) fget(std::reference_wrapper<U> x) {
      return x.get();
    }

  } // namespace detail

  /**
   * @brief Generic operation performed on expression nodes.
   *
   * @details Specializations for the different operation tags provide an implementation of the function call operator
   * which performs the actual operation on the given operands.
   *
   * @tparam Tag Tag of the operation.
   */
  template <typename Tag>
  struct operation;

  /// Specialization of nda::clef::operation for nda::clef::tags::terminal.
  template <>
  struct operation<tags::terminal> {
    /**
     * @brief Perform a terminal operation.
     *
     * @tparam L Type of the operand.
     * @param l Operand.
     * @return Forwarded lvalue/rvalue reference of the argument.
     */
    template <typename L>
    FORCEINLINE L operator()(L &&l) const {
      return std::forward<L>(l);
    }
  };

  /// Specialization of nda::clef::operation for nda::clef::tags::function.
  template <>
  struct operation<tags::function> {
    /**
     * @brief Perform a function call operation.
     *
     * @tparam F Type of the callable.
     * @tparam Args Types of the function call arguments.
     * @param f Callable object.
     * @param args Function call arguments.
     * @return Result of the function call.
     */
    template <typename F, typename... Args>
    FORCEINLINE decltype(auto) operator()(F &&f, Args &&...args) const {
      return detail::fget(std::forward<F>(f))(detail::fget(std::forward<Args>(args))...);
    }
  };

  /// Specialization of nda::clef::operation for nda::clef::tags::subscript.
  template <>
  struct operation<tags::subscript> {
    /**
     * @brief Perform a subscript operation.
     *
     * @tparam F Type of the object to be subscripted.
     * @tparam Args Types of the subscript arguments.
     * @param f Object to be subscripted.
     * @param args Subscript arguments.
     * @return Result of the subscript operation.
     */
    template <typename F, typename... Args>
    FORCEINLINE decltype(auto) operator()(F &&f, Args &&...args) const {
      return detail::fget(std::forward<F>(f)).operator[](detail::fget(std::forward<Args>(args))...);
    }
  };

  // Define and implement all lazy binary operations.
#define CLEF_OPERATION(TAG, OP)                                                                                                                      \
  namespace tags {                                                                                                                                   \
    /** @brief Tag for binary `OP` expressions. */                                                                                                   \
    struct TAG : binary_op {                                                                                                                         \
      /** @brief String representation of the operation. */                                                                                          \
      static const char *name() { return AS_STRING(OP); }                                                                                            \
    };                                                                                                                                               \
  }                                                                                                                                                  \
  /** @brief Implementation of the lazy binary `OP` operation. */                                                                                    \
  template <typename L, typename R>                                                                                                                  \
  FORCEINLINE auto operator OP(L &&l, R &&r)                                                                                                         \
    requires(is_any_lazy<L, R>)                                                                                                                      \
  {                                                                                                                                                  \
    return expr<tags::TAG, expr_storage_t<L>, expr_storage_t<R>>{tags::TAG(), std::forward<L>(l), std::forward<R>(r)};                               \
  }                                                                                                                                                  \
  /** @brief Specialization of nda::clef::operation for nda::clef::tags::TAG. */                                                                     \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    /** @brief Function call operator to perform the actual binary `OP` operation. */                                                                \
    template <typename L, typename R>                                                                                                                \
    FORCEINLINE decltype(auto) operator()(L &&l, R &&r) const {                                                                                      \
      return detail::fget(std::forward<L>(l)) OP detail::fget(std::forward<R>(r));                                                                   \
    }                                                                                                                                                \
  }

  // clang-format off
  CLEF_OPERATION(plus, +);
  CLEF_OPERATION(minus, -);
  CLEF_OPERATION(multiplies, *);
  CLEF_OPERATION(divides, /);
  CLEF_OPERATION(greater, >);
  CLEF_OPERATION(less, <);
  CLEF_OPERATION(leq, <=);
  CLEF_OPERATION(geq, >=);
  CLEF_OPERATION(eq, ==);
  // clang-format on
#undef CLEF_OPERATION

  // Define and implement all lazy unary operations.
#define CLEF_OPERATION(TAG, OP)                                                                                                                      \
  namespace tags {                                                                                                                                   \
    /** @brief Tag for unary `OP` expressions. */                                                                                                    \
    struct TAG : unary_op {                                                                                                                          \
      /** @brief String representation of the operation. */                                                                                          \
      static const char *name() { return AS_STRING(OP); }                                                                                            \
    };                                                                                                                                               \
  }                                                                                                                                                  \
  /** @brief Implementation of the lazy unary `OP` operation. */                                                                                     \
  template <typename L>                                                                                                                              \
  FORCEINLINE auto operator OP(L &&l)                                                                                                                \
    requires(is_any_lazy<L>)                                                                                                                         \
  {                                                                                                                                                  \
    return expr<tags::TAG, expr_storage_t<L>>{tags::TAG(), std::forward<L>(l)};                                                                      \
  }                                                                                                                                                  \
  /** @brief Specialization of nda::clef::operation for nda::clef::tags::TAG. */                                                                     \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    /** @brief Function call operator to perform the actual unary `OP` operation. */                                                                 \
    template <typename L>                                                                                                                            \
    FORCEINLINE decltype(auto) operator()(L &&l) const {                                                                                             \
      return OP detail::fget(std::forward<L>(l));                                                                                                    \
    }                                                                                                                                                \
  }

  CLEF_OPERATION(unaryplus, +);
  CLEF_OPERATION(negate, -);
  CLEF_OPERATION(loginot, !);
#undef CLEF_OPERATION

  /// Specialization of nda::clef::operation for nda::clef::tags::if_else.
  template <>
  struct operation<tags::if_else> {
    /**
     * @brief Perform a ternary (if-else) operation.
     *
     * @tparam C Type of the condition.
     * @tparam A Type of the return type when the condition is true.
     * @tparam B Type of the return type when the condition is false.
     * @param c Condition convertible to bool.
     * @param a Return value when the condition is true.
     * @param b Return value when the condition is false (needs to be convertible to A).
     * @return Result of the ternary operation.
     */
    template <typename C, typename A, typename B>
    FORCEINLINE A operator()(C const &c, A const &a, B const &b) const {
      return detail::fget(c) ? detail::fget(a) : detail::fget(b);
    }
  };

  /**
   * @brief Create a lazy ternary (if-else) expression.
   *
   * @tparam C Type of the conditional expression.
   * @tparam A Type of the return expression when the condition is true.
   * @tparam B Type of the return expression when the condition is false.
   * @param c Conditional expression.
   * @param a Return expression when the condition is true.
   * @param b Return expression when the condition is false.
   * @return An nda::clef::expr object with the nda::clef::tags::ternary tag and the given
   * operands forwarded as its child nodes.
   */
  template <typename C, typename A, typename B>
  FORCEINLINE auto if_else(C &&c, A &&a, B &&b) {
    return expr<tags::if_else, expr_storage_t<C>, expr_storage_t<A>, expr_storage_t<B>>{tags::if_else(), std::forward<C>(c), std::forward<A>(a),
                                                                                        std::forward<B>(b)};
  }

  /**
   * @brief Dispatch operations containing at least one lazy operand.
   *
   * @details Since at least one operand is lazy, the operation is not performed immediately. Instead, a new
   * nda::clef::expr object is created with the given operation and operands.
   *
   * @tparam Tag Tag of the operation.
   * @tparam Args Types of the operands.
   * @param args Operands.
   * @return An nda::clef::expr for the given operation and operands.
   */

  template <typename Tag, typename... Args>
  FORCEINLINE auto op_dispatch(std::true_type, Args &&...args) {
    using Arg0 = std::decay_t<std::tuple_element_t<0, std::tuple<Args...>>>;
    if constexpr ((std::is_same_v<Tag, tags::function> and not supports_partial_eval_of_calls<Arg0>) or  //
                  (std::is_same_v<Tag, tags::subscript> and not supports_partial_eval_of_subscript<Arg0>) //
    )
      return expr<Tag, expr_storage_t<Args>...>{Tag(), std::forward<Args>(args)...};
     else
      return operation<Tag>()(std::forward<Args>(args)...);
  }

  /**
   * @brief Dispatch operations containing only non-lazy operands.
   *
   * @details Since all operands are non-lazy, the operation is performed immediately by calling the corresponding
   * nda::clef::operation with the forwarded operands.
   *
   * @tparam Tag Tag of the operation.
   * @tparam Args Types of the operands.
   * @param args Operands.
   * @return Result of actually performing the operation on the operands.
   */
  template <typename Tag, typename... Args>
  FORCEINLINE decltype(auto) op_dispatch(std::false_type, Args &&...args) {
    return operation<Tag>()(std::forward<Args>(args)...);
  }

  /** @} */

} // namespace nda::clef

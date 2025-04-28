// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functionality to make objects, functions and methods lazy.
 */

#pragma once

#include "./expression.hpp"
#include "./utils.hpp"

#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Create a terminal expression node of an object.
   *
   * @tparam T Type of the object.
   * @param t Given object.
   * @return An nda::clef::expr with the nda::clef::tags::terminal tag containing either a copy of the object itself
   * (if it is moved) or a reference to the object.
   */
  template <typename T>
  auto make_expr(T &&t) {
    return expr<tags::terminal, expr_storage_t<T>>{tags::terminal(), std::forward<T>(t)};
  }

  /**
   * @brief Create a terminal expression node of an object.
   *
   * @tparam T Type of the object.
   * @param t Given object.
   * @return An nda::clef::expr with the nda::clef::tags::terminal tag containing a copy of the object.
   */
  template <typename T>
  auto make_expr_from_clone(T &&t) {
    return expr<tags::terminal, std::decay_t<T>>{tags::terminal(), std::forward<T>(t)};
  }

  /**
   * @brief Create a function call expression from a callable object and a list of arguments.
   *
   * @details Note that this is equivalent to `nda::clef::make_expr(t)(args...)`.
   *
   * @tparam F Type of the callable object.
   * @tparam Args Types of the arguments.
   * @param f Callable object.
   * @param args Function arguments.
   * @return An nda::clef::expr with the nda::clef::tags::function tag containing the callable and the forwarded
   * arguments.
   */
  template <typename F, typename... Args>
  auto make_expr_call(F &&f, Args &&...args)
    requires(is_any_lazy<Args...>)
  {
    return expr<tags::function, expr_storage_t<F>, expr_storage_t<Args>...>{tags::function{}, std::forward<F>(f), std::forward<Args>(args)...};
  }

  /**
   * @brief Create a subscript expression from an object and a list of arguments.
   *
   * @details Note that this is equivalent to `nda::clef::make_expr(t)[args...]`.
   *
   * @tparam T Type of the object to be subscripted.
   * @tparam Args Types of the arguments.
   * @param t Object to be subscripted.
   * @param args Subscript arguments.
   * @return An nda::clef::expr with the nda::clef::tags::subscript tag containing the subscriptable object and the
   * forwarded arguments.
   */
  template <typename T, typename... Args>
  auto make_expr_subscript(T &&t, Args &&...args)
    requires(is_any_lazy<Args...>)
  {
    return expr<tags::subscript, expr_storage_t<T>, expr_storage_t<Args>...>{tags::subscript{}, std::forward<T>(t), std::forward<Args>(args)...};
  }

  /// Macro to make any function lazy, i.e. accept lazy arguments and return a function call expression node.
#define CLEF_MAKE_FNT_LAZY(name)                                                                                                                     \
  template <typename... A>                                                                                                                           \
  auto name(A &&...__a)                                                                                                                              \
    requires(nda::clef::is_any_lazy<A...>)                                                                                                           \
  {                                                                                                                                                  \
    return make_expr_call([](auto &&...__b) -> decltype(auto) { return name(std::forward<decltype(__b)>(__b)...); }, std::forward<A>(__a)...);       \
  }

  /// Macro to make any method lazy, i.e. accept lazy arguments and return a function call expression node.
#define CLEF_IMPLEMENT_LAZY_METHOD(TY, name)                                                                                                         \
  template <typename... A>                                                                                                                           \
  auto name(A &&...__a)                                                                                                                              \
    requires(nda::clef::is_any_lazy<A...>)                                                                                                           \
  {                                                                                                                                                  \
    return make_expr_call(                                                                                                                           \
       [](auto &&__obj, auto &&...__b) -> decltype(auto) { return std::forward<decltype(__obj)>(__obj).name(std::forward<decltype(__b)>(__b)...); }, \
       *this, std::forward<A>(__a)...);                                                                                                              \
  }

  /// Macro to make any function call operator lazy, i.e. accept lazy arguments and return a function call expression node.
#define CLEF_IMPLEMENT_LAZY_CALL(...)                                                                                                                \
  template <typename... Args>                                                                                                                        \
  auto operator()(Args &&...args) const &                                                                                                            \
    requires(nda::clef::is_any_lazy<Args...>)                                                                                                        \
  {                                                                                                                                                  \
    return make_expr_call(*this, std::forward<Args>(args)...);                                                                                       \
  }                                                                                                                                                  \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
  auto operator()(Args &&...args) &                                                                                                                  \
    requires(nda::clef::is_any_lazy<Args...>)                                                                                                        \
  {                                                                                                                                                  \
    return make_expr_call(*this, std::forward<Args>(args)...);                                                                                       \
  }                                                                                                                                                  \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
     auto operator()(Args &&...args)                                                                                                                 \
     && requires(nda::clef::is_any_lazy<Args...>) { return make_expr_call(std::move(*this), std::forward<Args>(args)...); }

  /** @} */

} // namespace nda::clef

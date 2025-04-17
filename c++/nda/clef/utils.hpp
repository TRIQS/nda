// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides some utility functions and type traits for the CLEF library.
 */

#pragma once

#include <functional>
#include <type_traits>

namespace nda::clef {

  /**
   * @addtogroup clef_utils
   * @{
   */

  namespace detail {

    // Helper variable to determine if an objects of type `T` should be forced to be copied into an expression tree.
    template <typename T>
    constexpr bool force_copy_in_expr_impl = false;

    // Helper struct to determine how a type should be stored in an expression tree.
    template <class T>
    struct expr_storage_impl {
      using type = T;
    };

    // Specialization of expr_storage_impl for lvalue references.
    template <class T>
    struct expr_storage_impl<T &>
       : std::conditional<force_copy_in_expr_impl<std::remove_const_t<T>>, std::remove_const_t<T>, std::reference_wrapper<T>> {};

    // Specialization of expr_storage_impl for rvalue references.
    template <class T>
    struct expr_storage_impl<T &&> {
      using type = T;
    };

    // Specialization of expr_storage_impl for const rvalue references.
    template <class T>
    struct expr_storage_impl<const T &&> {
      using type = T;
    };

    // Specialization of expr_storage_impl for const types.
    template <class T>
    struct expr_storage_impl<const T> {
      using type = T;
    };

    // Type used to encode which placeholder indices are used in an expression.
    using ull_t = unsigned long long;

    // Type trait that determines the set of placeholders in a parameter pack.
    template <typename... Ts>
    struct ph_set;

    // Specialization of ph_set for parameter packs containing at least one type.
    template <typename T0, typename... Ts>
    struct ph_set<T0, Ts...> {
      static constexpr ull_t value = ph_set<T0>::value | ph_set<Ts...>::value;
    };

    // Specialization of ph_set for a single non-lazy type.
    template <typename T>
    struct ph_set<T> {
      static constexpr ull_t value = 0;
    };

    // Filter out certain placeholders given an integer value encoded by ph_set.
    template <ull_t N, int... Is>
    struct ph_filter;

    // Specialization of ph_filter for at least one placeholder to filter out.
    template <ull_t N, int I0, int... Is>
    struct ph_filter<N, I0, Is...> {
      static constexpr ull_t value = ph_filter<N, Is...>::value & (~(1ull << I0));
    };

    // Specialization of ph_filter for no placeholders to filter out.
    template <ull_t N>
    struct ph_filter<N> {
      static constexpr ull_t value = N;
    };

    // Helper variable to determine if a type `T` is lazy.
    template <typename T>
    constexpr bool is_lazy_impl = false;

    // Specialization of is_lazy_impl for cvref types.
    template <typename T>
      requires(!std::is_same_v<T, std::remove_cvref_t<T>>)
    constexpr bool is_lazy_impl<T> = is_lazy_impl<std::remove_cvref_t<T>>;

    // Helper variable to determine if a type `T` is an make_fun_impl type.
    template <typename T>
    inline constexpr bool is_function_impl = false;

    // An erroneous diagnostics in gcc: i0 is indeed used. We silence it.
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif

    // Check if all given integers are different.
    template <typename... Is>
    constexpr bool all_different(int i0, Is... is) {
      return (((is - i0) * ... * 1) != 0);
    }

#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif

  } // namespace detail

  /// Constexpr variable that is true if objects of type `T` should be forced to be copied into an expression tree.
  template <typename T>
  constexpr bool force_copy_in_expr = detail::force_copy_in_expr_impl<T>;

  /**
   * @brief Type trait to determine how a type should be stored in an expression tree, i.e. either by reference or by
   * value?
   *
   * @details Rvalue references are copied/moved into the expression tree. Lvalue references are stored as a
   * std::reference_wrapper, unless the compile-time value of nda::clef::force_copy_in_expr is true.
   *
   * @tparam T Type to be stored.
   */
  template <class T>
  using expr_storage_t = typename detail::expr_storage_impl<T>::type;

  /// Constexpr variable that is true if the type `T` is a lazy type.
  template <typename T>
  constexpr bool is_lazy = detail::is_lazy_impl<T>;

  /// Constexpr variable that is true if any of the given types is lazy.
  template <typename... Ts>
  constexpr bool is_any_lazy = (is_lazy<Ts> or ...);

  /// Alias template for nda::clef::is_any_lazy.
  template <typename... Ts>
  constexpr bool is_clef_expression = is_any_lazy<Ts...>;

  /// Constexpr variable that is true if the type `T` is an nda::clef::make_fun_impl type.
  template <typename T>
  inline constexpr bool is_function = detail::is_function_impl<T>;

  /** @} */

} // namespace nda::clef

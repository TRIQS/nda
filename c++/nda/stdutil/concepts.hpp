#pragma once

// This file is a temporary replacement for <concepts> if not present (libc++ as 06/2020)
// to have a few basic concepts in std:: that we need.
// definitions are taken from cppreference

#if __cplusplus > 201703L

// Of course if implemented we use the real one
#if __has_include(<concepts>)
#include <concepts>

// Define them
#else
namespace std {

  // clang-format does not format well concepts yet
  // clang-format off

  template <class From, class To>
  concept convertible_to = std::is_convertible_v<From, To> and requires(std::add_rvalue_reference_t<From> (&f)()) {
    static_cast<To>(f());
  };

  namespace detail {
    template <class T, class U>
    concept SameHelper = std::is_same_v<T, U>;
  }

  template <class T, class U>
  concept same_as = detail::SameHelper<T, U> &&detail::SameHelper<U, T>;

  template <class T>
  concept integral = std::is_integral_v<T>;

  // clang-format on

} // namespace std

#endif
#endif

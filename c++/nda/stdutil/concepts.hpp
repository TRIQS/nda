// Copyright (c) 2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// This file is a temporary replacement for <concepts> if not present (libc++ as 06/2020)
// to have a few basic concepts in std:: that we need.
// definitions are taken from cppreference

#if __cplusplus > 201703L

#include <version>

// Use concepts header if available
// Skip inclusion for libc++ as header still incomplete
#ifndef _LIBCPP_VERSION
#if __has_include(<concepts>)
#include <concepts>
#endif

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

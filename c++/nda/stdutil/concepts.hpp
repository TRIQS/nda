// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides missing concepts for older compiler versions.
 */

#pragma once

#include <version>

// libcpp below 13 has incomplete <concepts>
#if defined(_LIBCPP_VERSION) and _LIBCPP_VERSION < 13000

#include <type_traits>

namespace std {

  template <class T>
  concept integral = std::is_integral_v<T>;

} // namespace std

#endif

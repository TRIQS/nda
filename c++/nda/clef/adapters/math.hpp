// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides lazy versions of various math functions of the standard library.
 */

#pragma once

#include "../clef.hpp"

#include <cmath>
#include <complex>

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

#define CLEF_MAKE_STD_FNT_LAZY(name)                                                                                                                 \
  using std::name;                                                                                                                                   \
  /** @brief Lazy version of std::name. */                                                                                                           \
  CLEF_MAKE_FNT_LAZY(name)

  // FIXME use vim generation
  CLEF_MAKE_STD_FNT_LAZY(cos)
  CLEF_MAKE_STD_FNT_LAZY(sin)
  CLEF_MAKE_STD_FNT_LAZY(tan)
  CLEF_MAKE_STD_FNT_LAZY(cosh)
  CLEF_MAKE_STD_FNT_LAZY(sinh)
  CLEF_MAKE_STD_FNT_LAZY(tanh)
  CLEF_MAKE_STD_FNT_LAZY(acos)
  CLEF_MAKE_STD_FNT_LAZY(asin)
  CLEF_MAKE_STD_FNT_LAZY(atan)
  CLEF_MAKE_STD_FNT_LAZY(exp)
  CLEF_MAKE_STD_FNT_LAZY(log)
  CLEF_MAKE_STD_FNT_LAZY(sqrt)
  CLEF_MAKE_STD_FNT_LAZY(abs)
  CLEF_MAKE_STD_FNT_LAZY(floor)
  CLEF_MAKE_STD_FNT_LAZY(pow)
  CLEF_MAKE_STD_FNT_LAZY(conj)

  /** @} */

} // namespace nda::clef

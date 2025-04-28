// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides various default nda::clef::placeholder objects.
 */

#pragma once

#include "./clef.hpp"

namespace nda::clef::literals {

  /**
   * @addtogroup clef_placeholders
   * @{
   */

  // Define literal placeholders starting from the end of the allowed index spectrum.
#define PH(I) (placeholder<63 - (I)>{})

  /// Generic placeholder #1.
  constexpr auto i_ = PH(0);

  /// Generic placeholder #2.
  constexpr auto j_ = PH(1);

  /// Generic placeholder #3.
  constexpr auto k_ = PH(2);

  /// Generic placeholder #4.
  constexpr auto l_ = PH(3);

  /// Placeholder for block indices.
  constexpr auto bl_ = PH(4);

  /// Placeholder for real fermionic frequencies.
  constexpr auto w_ = PH(5);

  /// Placeholder for imaginary fermionic frequencies.
  constexpr auto iw_ = PH(6);

  /// Placeholder for real bosonic frequencies.
  constexpr auto W_ = PH(7);

  /// Placeholder for imaginary bosonic frequencies.
  constexpr auto iW_ = PH(8);

  /// Placeholder for real times.
  constexpr auto t_ = PH(9);

  /// Placeholder for imaginary times.
  constexpr auto tau_ = PH(10);

#undef PH

  /** @} */

} // namespace nda::clef::literals

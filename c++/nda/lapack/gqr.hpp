// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK/cuSOLVER `orgqr` and `ungqr` routines.
 */

#pragma once

#include "./orgqr.hpp"
#include "./orgqr_batch.hpp"
#include "./ungqr.hpp"
#include "./ungqr_batch.hpp"
#include "../blas/tools.hpp"
#include "../traits.hpp"

#include <type_traits>
#include <utility>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Dispatcher to nda::lapack::orgqr for real value types and to nda::lapack::ungqr for complex value types.
   */
  template <BlasArray<2> A, BlasArrayFor<A, 1> TAU, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A>)
  int gqr(A &&a, TAU &&tau, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    if constexpr (std::is_same_v<value_type, float> or std::is_same_v<value_type, double>) {
      return orgqr(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
    } else {
      return ungqr(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
    }
  }

  /**
   * @ingroup linalg_lapack
   * @brief Dispatcher to nda::lapack::orgqr_batch for real value types and to nda::lapack::ungqr_batch for complex 
   * value types.
   */
  template <BlasArray<3> A, BlasArrayFor<A, 2> TAU, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A, TAU>)
  int gqr(A &&a, TAU &&tau, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    if constexpr (std::is_same_v<value_type, float> or std::is_same_v<value_type, double>) {
      return orgqr_batch(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
    } else {
      return ungqr_batch(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
    }
  }

} // namespace nda::lapack

// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `orgqr` and 'ungqr' routines.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `orgqr` and 'ungqr' routines.
   *
   * @details Generates an m-by-n real matrix \f$ \mathbf{Q} \f$ with orthonormal columns, which is defined as the first
   * n columns of a product of k elementary reflectors of order m:
   * \f[
   *   \mathbf{Q} = \mathbf{H}(1) \mathbf{H}(2) \ldots \mathbf{H}(k) \; ,
   * \f]
   * as returned by `geqrf`.
   * This routine calls orgqr or ungqr depending on the value type of the Matrix A. 
   *
   * @tparam A nda::MemoryMatrix 
   * @tparam TAU nda::MemoryVector 
   * @param a Input/output matrix. On entry, the i-th column must contain the vector which defines the elementary
   * reflector \f$ H(i) \; , i = 1,2,...,k \f$, as returned by `geqrf` in the first k columns. On exit, the m-by-n
   * matrix \f$ \mathbf{Q} \f$.
   * @param tau Input vector. `tau(i)` must contain the scalar factor of the elementary reflector \f$ \mathbf{H}(i) \f$,
   * as returned by `geqrf`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector TAU>
    requires(have_same_value_type_v<A, TAU> and is_blas_lapack_v<get_value_t<A>>)
  int gqr(A &&a, TAU &&tau) { // NOLINT (temporary views are allowed here)
    if constexpr (std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
      return orgqr(std::forward<A>(a), std::forward<TAU>(tau));
    else
      return ungqr(std::forward<A>(a), std::forward<TAU>(tau));
  }

  template <MemoryArrayOfRank<3> A, MemoryMatrix TAU>
    requires(have_same_value_type_v<A, TAU> and is_blas_lapack_v<get_value_t<A>>)
  int gqr(A &&a, TAU &&tau) { // NOLINT (temporary views are allowed here)
    if constexpr (std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
      return orgqr(std::forward<A>(a), std::forward<TAU>(tau));
    else
      return ungqr(std::forward<A>(a), std::forward<TAU>(tau));
  }

  template <MemoryMatrix A, MemoryVector TAU, MemoryVector W>
    requires(have_same_value_type_v<A, TAU, W> and is_blas_lapack_v<get_value_t<A>>)
  int gqr(A &&a, TAU &&tau, W && work) { // NOLINT (temporary views are allowed here)
    if constexpr (std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
      return orgqr(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
    else
      return ungqr(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
  }

  template <MemoryArrayOfRank<3> A, MemoryMatrix TAU, MemoryVector W>
    requires(have_same_value_type_v<A, TAU, W> and is_blas_lapack_v<get_value_t<A>>)
  int gqr(A &&a, TAU &&tau, W && work) { // NOLINT (temporary views are allowed here)
    if constexpr (std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
      return orgqr(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
    else
      return ungqr(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
  }

} // namespace nda::lapack

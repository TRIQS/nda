// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a dot product for two arrays, a scalar and an array, or two scalars.
 */

#pragma once

#include "../blas/dot.hpp"
#include "../declarations.hpp"
#include "../layout/policies.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"

#include <type_traits>

namespace nda {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Compute the dot product of two real arrays/views.
   *
   * @details It is generic in the sense that it allows the input arrays to belong to a different nda::mem::AddressSpace
   * (as long as they are compatible).
   *
   * If possible, it uses nda::blas::dot, otherwise it calls nda::blas::dot_generic.
   *
   * @tparam X Type of the left hand side array/view.
   * @tparam Y Type of the right hand side array/view.
   * @param x Left hand side array/view.
   * @param y Right hand side array/view.
   * @return The dot product of the two arrays/views.
   */
  template <typename X, typename Y>
  auto dot(X &&x, Y &&y) { // NOLINT (temporary views are allowed here)
    // check address space compatibility
    static constexpr auto L_adr_spc = mem::get_addr_space<X>;
    static constexpr auto R_adr_spc = mem::get_addr_space<Y>;
    static_assert(L_adr_spc != mem::None);
    static_assert(R_adr_spc != mem::None);

    // get resulting value type and vector type
    using value_t  = decltype(get_value_t<X>{} * get_value_t<Y>{});
    using vector_t = basic_array<value_t, 1, C_layout, 'V', nda::heap<mem::combine<L_adr_spc, R_adr_spc>>>;

    if constexpr (is_blas_lapack_v<value_t>) {
      // for double value types we use blas::dot
      // lambda to form a new vector with the correct value type if necessary
      auto as_container = []<typename A>(A const &a) -> decltype(auto) {
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, value_t>)
          return a;
        else
          return vector_t{a};
      };

      return blas::dot(as_container(x), as_container(y));
    } else {
      // for other value types we use a generic implementation
      return blas::dot_generic(x, y);
    }
  }

  /**
   * @brief Compute the dot product of two complex arrays/views.
   *
   * @details It is generic in the sense that it allows the input arrays to belong to a different nda::mem::AddressSpace
   * (as long as they are compatible).
   *
   * If possible, it uses nda::blas::dotc, otherwise it calls nda::blas::dotc_generic.
   *
   * @tparam X Type of the left hand side array/view.
   * @tparam Y Type of the right hand side array/view.
   * @param x Left hand side array/view.
   * @param y Right hand side array/view.
   * @return The dot product of the two arrays/views.
   */
  template <typename X, typename Y>
  auto dotc(X &&x, Y &&y) { // NOLINT (temporary views are allowed here)
    // check address space compatibility
    static constexpr auto L_adr_spc = mem::get_addr_space<X>;
    static constexpr auto R_adr_spc = mem::get_addr_space<Y>;
    static_assert(L_adr_spc != mem::None);
    static_assert(R_adr_spc != mem::None);

    // get resulting value type and vector type
    using value_t  = decltype(get_value_t<X>{} * get_value_t<Y>{});
    using vector_t = basic_array<value_t, 1, C_layout, 'V', nda::heap<mem::combine<L_adr_spc, R_adr_spc>>>;

    if constexpr (is_blas_lapack_v<value_t>) {
      // for double or complex value types we use blas::dotc
      // lambda to form a new vector with the correct value type if necessary
      auto as_container = []<typename A>(A const &a) -> decltype(auto) {
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, value_t>)
          return a;
        else
          return vector_t{a};
      };

      return blas::dotc(as_container(x), as_container(y));
    } else {
      // for other value types we use a generic implementation
      return blas::dotc_generic(x, y);
    }
  }

  /** @} */

} // namespace nda

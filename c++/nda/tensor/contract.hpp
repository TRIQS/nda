// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic tensor contraction with cuTENSOR/TBLIS dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./interface/tblis_interface.hpp"
#include "./tools.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <string_view>
#include <utility>

namespace nda::tensor {

  /**
   * @addtogroup tensor_ops
   * @{
   */

  /**
   * @brief Tensor contraction with cuTENSOR/TBLIS dispatch.
   *
   * @details This function performs a general tensor contraction of the form
   * \f[
   *   C_{\text{idx}_C} \leftarrow \alpha \, A_{\text{idx}_A} \cdot B_{\text{idx}_B} + \beta \, C_{\text{idx}_C} \;,
   * \f]
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, and \f$ A \f$, \f$ B \f$ and \f$ C \f$ are tensors of arbitrary
   * rank. The contraction pattern is specified via index strings (Einstein notation), where repeated indices between
   * \f$ A \f$ and \f$ B \f$ are summed over.
   * 
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, cuTENSOR's contraction operation is 
   * used.
   * - Otherwise, TBLIS support is required and `tblis_tensor_mult` is used.
   *
   * Index strings are forwarded as-is to the backend library (cuTENSOR or TBLIS) without any checks. It's the user's 
   * responsibility to ensure that they are valid and consistent with the shapes of the input tensors.
   * </details>
   * 
   * @note \f$ A \f$ and \f$ B \f$ are allowed to be lazy conjugate expressions (see 
   * nda::blas_lapack::is_conj_array_expr).
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj type.
   * @tparam B nda::blas_lapack::BlasArrayOrConjFor<A> type.
   * @tparam C nda::blas_lapack::BlasArrayFor<A> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input tensor \f$ A \f$.
   * @param idx_a Index string \f$ \text{idx}_A \f$ for tensor \f$ A \f$.
   * @param b Input tensor \f$ B \f$.
   * @param idx_b Index string \f$ \text{idx}_B \f$ for tensor \f$ B \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param c Input/Output tensor \f$ C \f$.
   * @param idx_c Index string \f$ \text{idx}_C \f$ for tensor \f$ C \f$.
   */
  template <BlasArrayOrConj A, BlasArrayOrConjFor<A> B, BlasArrayFor<A> C>
  void contract(get_value_t<A> alpha, A const &a, std::string_view idx_a, B const &b, std::string_view idx_b, get_value_t<A> beta, C &&c, // NOLINT
                std::string_view idx_c) {
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B, C>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::contract: cuTENSOR support is required");
    static_assert(run_on_device || have_tblis, "nda::tensor::contract: TBLIS support is required");

    // dispatch to backends
    if constexpr (run_on_device) {
      device::contract(alpha, a, idx_a, b, idx_b, beta, c, idx_c, c);
    } else {
      tblis::mult(alpha, a, idx_a, b, idx_b, beta, c, idx_c);
    }
  }

  /// Convenience overload of nda::tensor::contract with \f$ \alpha = 1 \f$ and \f$ \beta = 0 \f$.
  template <BlasArrayOrConj A, BlasArrayOrConjFor<A> B, BlasArrayFor<A> C>
  void contract(A const &a, std::string_view idx_a, B const &b, std::string_view idx_b, C &&c, std::string_view idx_c) { // NOLINT
    contract(get_value_t<A>{1}, a, idx_a, b, idx_b, get_value_t<A>{0}, std::forward<C>(c), idx_c);
  }

  /** @} */

} // namespace nda::tensor

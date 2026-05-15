// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic in-place elementwise trinary tensor operation with cuTENSOR/nda dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./tools.hpp"
#include "../exceptions.hpp"
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
   * @brief In-place elementwise trinary tensor operation with cuTENSOR/nda dispatch.
   *
   * @details This function performs an in-place elementwise trinary operation of the form
   * \f[
   *   C_{\text{idx}_C} \leftarrow \text{op}_{ABC} \bigl( \text{op}_{AB} \bigl( \alpha \, A_{\text{idx}_A}, \, \beta \, 
   *   B_{\text{idx}_B} \bigr), \, \gamma \, C_{\text{idx}_C} \bigr) \;,
   * \f]
   * where \f$ \alpha \f$, \f$ \beta \f$ and \f$ \gamma \f$ are scalars, \f$ A \f$, \f$ B \f$ and \f$ C \f$ are tensors
   * of arbitrary (possibly different) ranks, and \f$ \text{op}_{ABC} \f$ and \f$ \text{op}_{AB} \f$ are binary 
   * operations (see nda::tensor::binary_op). The index strings determine the einsum-style mapping between operands; 
   * indices absent from one operand drive broadcast/reduction on the backend.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, cuTENSOR's elementwise trinary operation
   * is used.
   * - Otherwise, fallback to nda expression assignment via nda::map.
   *
   * The supported binary operations depend on the library backend. The nda host fallback requires identical ranks
   * and identical index strings and supports all nda::tensor::binary_op values for both `op_AB` and `op_ABC`.
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
   * @param beta Input scalar \f$ \beta \f$.
   * @param b Input tensor \f$ B \f$.
   * @param idx_b Index string \f$ \text{idx}_B \f$ for tensor \f$ B \f$.
   * @param gamma Input scalar \f$ \gamma \f$.
   * @param c Input/Output tensor \f$ C \f$.
   * @param idx_c Index string \f$ \text{idx}_C \f$ for tensor \f$ C \f$.
   * @param op_AB Binary operation between \f$ A \f$ and \f$ B \f$ (default: `binary_op::SUM`).
   * @param op_ABC Binary operation between the result of \f$ \text{op}_{AB} \f$ and \f$ C \f$ (default: 
   * `binary_op::SUM`).
   */
  template <BlasArrayOrConj A, BlasArrayOrConjFor<A> B, BlasArrayFor<A> C>
  void elementwise_trinary(get_value_t<A> alpha, A const &a, std::string_view idx_a, get_value_t<A> beta, B const &b, std::string_view idx_b,
                           get_value_t<A> gamma, C &&c, std::string_view idx_c, binary_op op_AB = binary_op::SUM, // NOLINT
                           binary_op op_ABC = binary_op::SUM) {
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B, C>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::elementwise_trinary: cuTENSOR support is required");
    static_assert(run_on_device || (get_rank<A> == get_rank<B> && get_rank<A> == get_rank<C>),
                  "nda::tensor::elementwise_trinary: host fallback requires identical ranks");

    // dispatch to backends
    if constexpr (run_on_device) {
      device::elementwise_trinary(alpha, a, idx_a, beta, b, idx_b, gamma, c, idx_c, c, op_AB, op_ABC);
    } else {
      require_equal_indices(idx_a, idx_b, get_rank<A>, "elementwise_trinary");
      require_equal_indices(idx_b, idx_c, get_rank<A>, "elementwise_trinary");
      c = nda::map([alpha, beta, gamma, op_AB, op_ABC](auto x, auto y, auto z) {
        return detail::apply_binary(op_ABC, detail::apply_binary(op_AB, alpha * x, beta * y), gamma * z);
      })(a, b, c);
    }
  }

  /// Convenience overload of nda::tensor::elementwise_trinary with \f$ \alpha = \beta = 1 \f$ and \f$ \gamma = 0 \f$.
  template <BlasArrayOrConj A, BlasArrayOrConjFor<A> B, BlasArrayFor<A> C>
  void elementwise_trinary(A const &a, std::string_view idx_a, B const &b, std::string_view idx_b, C &&c, std::string_view idx_c, // NOLINT
                           binary_op op_AB = binary_op::SUM, binary_op op_ABC = binary_op::SUM) {
    elementwise_trinary(get_value_t<A>{1}, a, idx_a, get_value_t<A>{1}, b, idx_b, get_value_t<A>{0}, std::forward<C>(c), idx_c, op_AB, op_ABC);
  }

  /** @} */

} // namespace nda::tensor

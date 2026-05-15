// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic in-place elementwise binary tensor operation with cuTENSOR/nda dispatch.
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
   * @brief In-place elementwise binary tensor operation with cuTENSOR/nda dispatch.
   *
   * @details This function performs an in-place elementwise binary operation of the form
   * \f[
   *   B_{\text{idx}_B} \leftarrow \text{op}\bigl(\alpha \, A_{\text{idx}_A}, \, \beta \, B_{\text{idx}_B}\bigr) \;,
   * \f]
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, \f$ A \f$ and \f$ B \f$ are tensors, and \f$ \text{op} \f$ is a
   * binary operation (see nda::tensor::binary_op). The index strings specify how the dimensions of \f$ A \f$ map to
   * those of \f$ B \f$ (Einstein notation); when ranks differ on the cuTENSOR path, indices present in one tensor but
   * absent from the other drive broadcast/reduction on the backend.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, cuTENSOR's elementwise binary operation
   * is used.
   * - Otherwise, fallback to nda expression assignment via nda::map.
   *
   * The supported binary operations depend on the library backend. The nda host fallback requires identical ranks 
   * and identical index strings and supports all nda::tensor::binary_op values.
   * </details>
   *
   * @note \f$ A \f$ is allowed to be a lazy conjugate expression (see nda::blas_lapack::is_conj_array_expr).
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input tensor \f$ A \f$.
   * @param idx_a Index string \f$ \text{idx}_A \f$ for tensor \f$ A \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param b Input/Output tensor \f$ B \f$.
   * @param idx_b Index string \f$ \text{idx}_B \f$ for tensor \f$ B \f$.
   * @param op Binary operation (default: `binary_op::SUM`).
   */
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void elementwise(get_value_t<A> alpha, A const &a, std::string_view idx_a, get_value_t<A> beta, B &&b, std::string_view idx_b, // NOLINT
                   binary_op op = binary_op::SUM) {
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::elementwise: cuTENSOR support is required");
    static_assert(run_on_device || get_rank<A> == get_rank<B>, "nda::tensor::elementwise: host fallback requires identical ranks");

    // dispatch to backends
    if constexpr (run_on_device) {
      device::elementwise_binary(alpha, a, idx_a, beta, b, idx_b, b, op);
    } else {
      require_equal_indices(idx_a, idx_b, get_rank<A>, "elementwise");
      b = nda::map([alpha, beta, op](auto x, auto y) { return detail::apply_binary(op, alpha * x, beta * y); })(a, b);
    }
  }

  /// Convenience overload of nda::tensor::elementwise with \f$ \alpha = 1 \f$ and \f$ \beta = 0 \f$.
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void elementwise(A const &a, std::string_view idx_a, B &&b, std::string_view idx_b, binary_op op = binary_op::SUM) { // NOLINT
    elementwise(get_value_t<A>{1}, a, idx_a, get_value_t<A>{0}, std::forward<B>(b), idx_b, op);
  }

  /// Convenience overload of nda::tensor::elementwise with nda::tensor::default_index strings.
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void elementwise(get_value_t<A> alpha, A const &a, get_value_t<A> beta, B &&b, binary_op op = binary_op::SUM) { // NOLINT
    elementwise(alpha, a, default_index<get_rank<A>>(), beta, std::forward<B>(b), default_index<get_rank<B>>(), op);
  }

  /**
   * @brief Convenience overload of nda::tensor::elementwise with nda::tensor::default_index strings, \f$ \alpha = 1 \f$ 
   * and \f$ \beta = 0 \f$.
   */
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void elementwise(A const &a, B &&b, binary_op op = binary_op::SUM) { // NOLINT
    elementwise(get_value_t<A>{1}, a, default_index<get_rank<A>>(), get_value_t<A>{0}, std::forward<B>(b), default_index<get_rank<B>>(), op);
  }

  /** @} */

} // namespace nda::tensor

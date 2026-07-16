// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic tensor addition with cuTENSOR/TBLIS/nda dispatch.
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
   * @brief Tensor addition with cuTENSOR/TBLIS/nda dispatch.
   *
   * @details This function performs a general tensor addition of the form
   * \f[
   *   B_{\text{idx}_B} \leftarrow \alpha \, A_{\text{idx}_A} + \beta \, B_{\text{idx}_B} \;,
   * \f]
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, and \f$ A \f$ and \f$ B \f$ are tensors of arbitrary (possibly
   * different) rank. The index strings specify the einsum-style mapping between dimensions of \f$ A \f$ and \f$ B \f$;
   * indices present in one tensor but absent from the other drive broadcast/reduction on the backend.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, cuTENSOR's elementwise binary operation
   * is used.
   * - If TBLIS is available, `tblis_tensor_add` is used.
   * - Otherwise, fallback to nda expression assignment, i.e. `b = alpha * a + beta * b`.
   *
   * The nda host fallback requires identical ranks and identical index strings. For the other backend paths, all inputs
   * are forwarded as-is.
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
   */
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void add(get_value_t<A> alpha, A const &a, std::string_view idx_a, get_value_t<A> beta, B &&b, std::string_view idx_b) { // NOLINT
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::add: cuTENSOR support is required");
    static_assert(run_on_device || have_tblis || get_rank<A> == get_rank<B>, "nda::tensor::add: host fallback requires identical ranks");

    // dispatch to backends
    if constexpr (run_on_device) {
      device::elementwise_binary(alpha, a, idx_a, beta, b, idx_b, b);
    } else if constexpr (have_tblis) {
      tblis::add(alpha, a, idx_a, beta, b, idx_b);
    } else {
      require_equal_indices(idx_a, idx_b, get_rank<A>, "add");
      b = alpha * a + beta * b;
    }
  }

  /**
   * @brief Out-of-place tensor addition with cuTENSOR/TBLIS/nda dispatch.
   *
   * @details This function performs a general out-of-place tensor addition of the form
   * \f[
   *   C_{\text{idx}_C} \leftarrow \alpha \, A_{\text{idx}_A} + \beta \, B_{\text{idx}_B} \;,
   * \f]
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, and \f$ A \f$, \f$ B \f$ and \f$ C \f$ are tensors of arbitrary
   * (possibly different) rank. The result is written into the separate output tensor \f$ C \f$ (any prior contents of
   * \f$ C \f$ are overwritten). The index strings specify the einsum-style mapping between operand dimensions; indices
   * present in some operands but absent from others drive broadcast/reduction on the backend.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, cuTENSOR's elementwise binary operation
   * is used.
   * - If TBLIS is available, `tblis_tensor_add` is used twice.
   * - Otherwise, fallback to nda expression assignment, i.e. `c = alpha * a + beta * b`.
   *
   * The nda host fallback requires identical ranks and identical index strings. For the other backend paths, all inputs
   * are forwarded as-is.
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
   * @param c Output tensor \f$ C \f$.
   * @param idx_c Index string \f$ \text{idx}_C \f$ for tensor \f$ C \f$.
   */
  template <BlasArrayOrConj A, BlasArrayOrConjFor<A> B, BlasArrayFor<A> C>
  void add(get_value_t<A> alpha, A const &a, std::string_view idx_a, get_value_t<A> beta, B const &b, std::string_view idx_b, C &&c, // NOLINT
           std::string_view idx_c) {
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B, C>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::add: cuTENSOR support is required");
    static_assert(run_on_device || have_tblis || (get_rank<A> == get_rank<B> && get_rank<A> == get_rank<C>),
                  "nda::tensor::add: host fallback requires identical ranks");

    // dispatch to backends
    if constexpr (run_on_device) {
      device::elementwise_binary(alpha, a, idx_a, beta, b, idx_b, c, binary_op::SUM);
    } else if constexpr (have_tblis) {
      tblis::add(beta, b, idx_b, get_value_t<A>{0}, c, idx_c);
      tblis::add(alpha, a, idx_a, get_value_t<A>{1}, c, idx_c);
    } else {
      require_equal_indices(idx_a, idx_b, get_rank<A>, "add");
      require_equal_indices(idx_b, idx_c, get_rank<A>, "add");
      c = alpha * a + beta * b;
    }
  }

  /// Convenience overload of nda::tensor::add with nda::tensor::default_index strings.
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void add(get_value_t<A> alpha, A const &a, get_value_t<A> beta, B &&b) { // NOLINT
    add(alpha, a, default_index<get_rank<A>>(), beta, std::forward<B>(b), default_index<get_rank<B>>());
  }

  /** @} */

} // namespace nda::tensor

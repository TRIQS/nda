// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic full tensor dot product with cuTENSOR/TBLIS/nda dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./interface/tblis_interface.hpp"
#include "./tools.hpp"
#include "../algorithms.hpp"
#include "../basic_array.hpp"
#include "../declarations.hpp"
#include "../layout/policies.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <string_view>

namespace nda::tensor {

  /**
   * @addtogroup tensor_ops
   * @{
   */

  /**
   * @brief Full tensor dot product with cuTENSOR/TBLIS/nda dispatch.
   *
   * @details This function computes the full tensor dot product
   * \f[
   *   z \leftarrow \sum_{\text{idx}_A = \text{idx}_B} A_{\text{idx}_A} \cdot B_{\text{idx}_B} \;,
   * \f]
   * where \f$ z \f$ is a scalar of the same value type as \f$ A \f$ and \f$ B \f$. The tensors may have arbitrary
   * (possibly different) rank. The index strings define the einsum-style pairing of dimensions.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, cuTENSOR's contraction operation is
   * used.
   * - Otherwise, if TBLIS is available, `tblis_tensor_dot` is used.
   * - Otherwise, fallback to `nda::sum(nda::hadamard(a, b))`. The nda fallback requires identical ranks and identical
   * index strings; it does not support TBLIS's einsum semantics (repeated indices, broadcast).
   *
   * For the cuTENSOR and TBLIS dispatch paths, index strings are forwarded as-is to the backend library without any
   * checks. It's the user's responsibility to ensure that they are valid and consistent with the shapes of the input
   * tensors.
   * </details>
   *
   * @note \f$ A \f$ and \f$ B \f$ are allowed to be lazy conjugate expressions (see
   * nda::blas_lapack::is_conj_array_expr).
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj type.
   * @tparam B nda::blas_lapack::BlasArrayOrConjFor<A> type.
   * @param a Input tensor \f$ A \f$.
   * @param idx_a Index string \f$ \text{idx}_A \f$ for tensor \f$ A \f$.
   * @param b Input tensor \f$ B \f$.
   * @param idx_b Index string \f$ \text{idx}_B \f$ for tensor \f$ B \f$.
   * @return The scalar dot product.
   */
  template <BlasArrayOrConj A, BlasArrayOrConjFor<A> B>
  get_value_t<A> dot(A const &a, std::string_view idx_a, B const &b, std::string_view idx_b) {
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::dot: cuTENSOR support is required");
    static_assert(run_on_device || have_tblis || get_rank<A> == get_rank<B>, "nda::tensor::dot: nda host fallback requires identical ranks");

    // dispatch to backends
    if constexpr (run_on_device) {
      auto z = basic_array<get_value_t<A>, 1, C_layout, 'A', heap<mem::common_addr_space<A, B>>>::zeros({1});
      device::contract(get_value_t<A>{1}, a, idx_a, b, idx_b, get_value_t<A>{0}, z.data(), "", z.data());
      return nda::to_host(z)(0);
    } else if constexpr (have_tblis) {
      return tblis::dot(a, idx_a, b, idx_b);
    } else {
      require_equal_indices(idx_a, idx_b, get_rank<A>, "dot");
      return nda::sum(nda::hadamard(a, b));
    }
  }

  /// Convenience overload of nda::tensor::dot with nda::tensor::default_index strings.
  template <BlasArrayOrConj A, BlasArrayOrConjFor<A, get_rank<A>> B>
  get_value_t<A> dot(A const &a, B const &b) {
    auto idx = default_index<get_rank<A>>();
    return dot(a, idx, b, idx);
  }

  /** @} */

} // namespace nda::tensor

// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic full tensor reduction with cuTENSOR/TBLIS/nda dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./interface/tblis_interface.hpp"
#include "./tools.hpp"
#include "../algorithms.hpp"
#include "../basic_array.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mapped_functions.hpp"
#include "../mapped_functions.hxx"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <cmath>
#include <string_view>

namespace nda::tensor {

  /**
   * @addtogroup tensor_ops
   * @{
   */

  /**
   * @brief Full tensor reduction with cuTENSOR/TBLIS/nda dispatch.
   *
   * @details This function applies a binary reduction operation (sum, max, min, ...) over all elements of a tensor and
   * returns the resulting scalar
   * \f[
   *   z \leftarrow \text{op}_{\text{red}}(A) \;,
   * \f]
   * where \f$ A \f$ is a tensor of arbitrary rank and \f$ \text{op}_{\text{red}} \f$ is a binary reduction operation
   * (see nda::tensor::binary_op).
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input array satisfies nda::mem::have_device_compatible_addr_space, cuTENSOR's reduction operation is used.
   * - If TBLIS is available and `op_reduce != binary_op::PROD`, `tblis_tensor_reduce` is used.
   * - Otherwise, fallback to nda algorithms, e.g. nda::sum, nda::max_element, etc.
   *
   * The supported binary operations depend on the library backend. The nda fallback supports all 
   * nda::tensor::binary_op values.
   * </details>
   *
   * @note \f$ A \f$ is allowed to be a lazy conjugate expression (see nda::blas_lapack::is_conj_array_expr).
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj type.
   * @param a Input tensor \f$ A \f$.
   * @param op_reduce Binary reduction operation (default: `binary_op::SUM`).
   * @return The reduced scalar \f$ z \f$.
   */
  template <BlasArrayOrConj A>
  get_value_t<A> reduce(A const &a, binary_op op_reduce = binary_op::SUM) {
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::reduce: cuTENSOR support is required");

    // dispatch to backends
    if constexpr (run_on_device) {
      auto z = basic_array<get_value_t<A>, 1, C_layout, 'A', heap<mem::get_addr_space<A>>>::zeros({1});
      device::reduce(get_value_t<A>{1}, a, default_index<get_rank<A>>(), get_value_t<A>{0}, z.data(), "", z.data(), op_reduce);
      return nda::to_host(z)(0);
    } else {
      // TBLIS handles every op except PROD; fall through to the nda fallback for PROD
      if constexpr (have_tblis) {
        if (op_reduce != binary_op::PROD) return tblis::reduce(op_reduce, a, default_index<get_rank<A>>());
      }
      // MAX/MIN are not defined for complex value types
      auto max_min = [&](binary_op op) -> get_value_t<A> {
        if constexpr (is_complex_v<get_value_t<A>>) {
          NDA_RUNTIME_ERROR << "nda::tensor::reduce: binary_op::MAX/MIN are unsupported for complex value types";
        } else {
          return op == binary_op::MAX ? nda::max_element(a) : nda::min_element(a);
        }
      };
      switch (op_reduce) {
        case binary_op::SUM: return nda::sum(a);
        case binary_op::PROD: return nda::product(a);
        case binary_op::SUM_ABS: return nda::sum(nda::abs(a));
        case binary_op::MAX_ABS: return nda::max_element(nda::abs(a));
        case binary_op::MIN_ABS: return nda::min_element(nda::abs(a));
        case binary_op::NORM_2: return std::sqrt(nda::sum(nda::abs2(a)));
        case binary_op::MAX:
        case binary_op::MIN: return max_min(op_reduce);
        default: NDA_RUNTIME_ERROR << "nda::tensor::reduce: unknown binary_op on nda host fallback";
      }
    }
  }

  /** @} */

} // namespace nda::tensor

// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic in-place tensor constant fill with cuTENSOR/TBLIS/nda dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./interface/tblis_interface.hpp"
#include "./tools.hpp"
#include "../basic_array.hpp"
#include "../declarations.hpp"
#include "../layout/policies.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <array>

namespace nda::tensor {

  /**
   * @addtogroup tensor_ops
   * @{
   */

  /**
   * @brief In-place tensor constant fill with cuTENSOR/TBLIS/nda dispatch.
   *
   * @details This function sets every element of a tensor to a constant value
   * \f[
   *   A \leftarrow \alpha \;,
   * \f]
   * where \f$ \alpha \f$ is a scalar and \f$ A \f$ is a tensor of arbitrary rank.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input array satisfies nda::mem::have_device_compatible_addr_space, cuTENSOR's elementwise binary operation
   * is used with a temporary rank-0 tensor holding \f$ \alpha \f$.
   * - If TBLIS is available, `tblis_tensor_set` is used.
   * - Otherwise, fallback to nda expression assignment, i.e. `a = alpha`.
   * </details>
   *
   * @tparam A nda::blas_lapack::BlasArray type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Output tensor \f$ A \f$.
   */
  template <BlasArray A>
  void set(get_value_t<A> alpha, A &&a) { // NOLINT
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::set: cuTENSOR support is required");

    // dispatch to backends
    if constexpr (run_on_device) {
      auto tmp = basic_array<get_value_t<A>, 1, C_layout, 'A', heap<mem::get_addr_space<A>>>(1, alpha);
      device::elementwise_binary(get_value_t<A>{1}, tmp.data(), "", get_value_t<A>{0}, a, default_index<get_rank<A>>(), a, binary_op::SUM);
    } else if constexpr (have_tblis) {
      tblis::set(alpha, a, default_index<get_rank<A>>());
    } else {
      a = alpha;
    }
  }

  /** @} */

} // namespace nda::tensor

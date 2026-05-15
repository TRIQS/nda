// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic in-place tensor scaling with cuTENSOR/TBLIS/nda dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./interface/tblis_interface.hpp"
#include "./tools.hpp"
#include "../exceptions.hpp"
#include "../mapped_functions.hpp"
#include "../mapped_functions.hxx"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::tensor {

  /**
   * @addtogroup tensor_ops
   * @{
   */

  /**
   * @brief In-place tensor scaling with cuTENSOR/TBLIS/nda dispatch and optional element-wise unary operation.
   *
   * @details This function performs an in-place scaling of the form
   * \f[
   *   A \leftarrow \alpha \, \text{op}(A) \;,
   * \f]
   * where \f$ \alpha \f$ is a scalar, \f$ A \f$ is a tensor of arbitrary rank, and `op` is an element-wise unary
   * operation (see nda::tensor::unary_op).
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input array satisfies nda::mem::have_device_compatible_addr_space, cuTENSOR's permute operation is used.
   * - If TBLIS is available and `op` is `unary_op::IDENTITY` or `unary_op::CONJ`, `tblis_tensor_scale` is used.
   * - Otherwise, fallback to nda expression assignment, e.g. `a = alpha * nda::sqrt(a)`.
   *
   * The nda fallback supports `unary_op::IDENTITY`, `unary_op::CONJ`, `unary_op::SQRT`, `unary_op::ABS`, 
   * `unary_op::EXP`, `unary_op::LOG`, and `unary_op::RCP`. Other operations raise `NDA_RUNTIME_ERROR`.
   * </details>
   *
   * @tparam A nda::blas_lapack::BlasArray type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input/Output tensor \f$ A \f$.
   * @param op Unary operation to apply element-wise (default: nda::tensor::unary_op::IDENTITY).
   */
  template <BlasArray A>
  void scale(get_value_t<A> alpha, A &&a, unary_op op = unary_op::IDENTITY) { // NOLINT
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::scale: cuTENSOR support is required");

    // fold NEG into the scalar (alpha * (-x) == -alpha * x) so backends always see IDENTITY
    if (op == unary_op::NEG) {
      alpha = -alpha;
      op    = unary_op::IDENTITY;
    }

    // dispatch to backends
    if constexpr (run_on_device) {
      device::permute(alpha, tensor_view{a, op}, default_index<get_rank<A>>(), tensor_view{a, unary_op::IDENTITY}, default_index<get_rank<A>>());
    } else {
      // TBLIS only handles IDENTITY and CONJ; other ops fall through to the nda fallback below
      if constexpr (have_tblis) {
        if (op == unary_op::IDENTITY || op == unary_op::CONJ) {
          tblis::scale(alpha, tensor_view{a, op}, default_index<get_rank<A>>());
          return;
        }
      }
      switch (op) {
        case unary_op::IDENTITY: a = alpha * a; break;
        case unary_op::CONJ: a = alpha * nda::conj(a); break;
        case unary_op::ABS: a = alpha * nda::abs(a); break;
        case unary_op::SQRT: a = nda::map([alpha](auto x) { return alpha * std::sqrt(x); })(a); break;
        case unary_op::EXP: a = nda::map([alpha](auto x) { return alpha * std::exp(x); })(a); break;
        case unary_op::LOG: a = nda::map([alpha](auto x) { return alpha * std::log(x); })(a); break;
        case unary_op::RCP: a = nda::map([alpha](auto x) { return alpha / x; })(a); break;
        default:
          NDA_RUNTIME_ERROR << "nda::tensor::scale: unsupported unary_op on nda host fallback "
                               "(supported: IDENTITY, CONJ, NEG, SQRT, ABS, EXP, LOG, RCP)";
      }
    }
  }

  /** @} */

} // namespace nda::tensor

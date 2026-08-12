// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic tensor assignment with cuTENSOR/TBLIS/nda dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./interface/tblis_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../layout/range.hpp"
#include "../layout_transforms.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <string_view>
#include <type_traits>
#include <utility>

namespace nda::tensor {

  /**
   * @addtogroup tensor_ops
   * @{
   */

  namespace detail {

    /// Slice `arr` on the compile-time axis `Axis` at index `i`, padding the other axes with `range::all`.
    // `typename A` rather than `auto &&arr`: cudafe++ emits the malformed `template <int Axis, int Rank, >` for an
    // abbreviated function template whose body holds a lambda with an explicit template parameter list.
    template <int Axis, int Rank, typename A>
    decltype(auto) slice_axis(A &&arr, long i) {
      return [&]<size_t... Before, size_t... After>(std::index_sequence<Before...>, std::index_sequence<After...>) -> decltype(auto) {
        return arr(((void)Before, ::nda::range::all)..., i, ((void)After, ::nda::range::all)...);
      }(std::make_index_sequence<Axis>{}, std::make_index_sequence<Rank - Axis - 1>{});
    }

    // Recursively copy from \f$ A \f$ into \f$ B \f$ by descending A's slowest-varying axis until both operands have a
    // layout that nda::assign_from_ndarray can handle directly.
    template <MemoryArray A, MemoryArray B>
    void rec_copy(A const &a, B &&b) { // NOLINT(cppcoreguidelines-missing-std-forward)
      constexpr bool same_stride_order = get_layout_info<A>.stride_order == get_layout_info<B>.stride_order;
      if constexpr (get_rank<A> == 1 || (same_stride_order && has_layout_strided_1d<A> && has_layout_strided_1d<B>)) {
        b = a;
      } else {
        constexpr int rank = get_rank<A>;
        // slowest-varying axis of A — accessed at the type level to avoid taking constexpr of the runtime parameter
        constexpr int axis = decode<rank>(get_layout_info<A>.stride_order)[0];
        long n             = b.extent(axis);
        for (long i = 0; i < n; ++i) rec_copy(slice_axis<axis, rank>(a, i), slice_axis<axis, rank>(b, i));
      }
    }

  } // namespace detail

  /**
   * @brief Tensor assignment with cuTENSOR/TBLIS/nda dispatch.
   *
   * @details This function performs the assignment
   * \f[
   *   B_{\text{idx}_B} \leftarrow A_{\text{idx}_A} \;,
   * \f]
   * where \f$ A \f$ and \f$ B \f$ are tensors of arbitrary (possibly different) rank. The index strings specify the
   * einsum-style mapping between dimensions of \f$ A \f$ and \f$ B \f$, allowing for permutations of axes; when ranks
   * differ, indices present in one tensor but absent from the other drive broadcast/reduction on the backend.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If both arrays are in a device-compatible address space and cuTENSOR is available, cuTENSOR's permute operation
   * is used.
   * - If both arrays are in a host-compatible address space and TBLIS is available, `tblis_tensor_add` is used with
   * \f$ \alpha = 1 \f$, \f$ \beta = 0 \f$.
   * - If both arrays are in a host-compatible address space without TBLIS, fallback to nda expression assignment, i.e.
   * `b = a`.
   * - Otherwise (cross-memory: device-compatible without cuTENSOR or with unsupported value types, or mixed
   * host/device), fallback to a recursive cross-memory copy (no permutation).
   *
   * The nda host fallback and the cross-memory copy require identical ranks and identical index strings. The
   * cross-memory copy descends the leading axis until reaching a slice that nda can handle directly (rank-1,
   * contiguous, or strided-1d). For the cuTENSOR and TBLIS dispatch paths, all inputs are forwarded as-is to the
   * backend.
   * </details>
   *
   * @note This function can be used to assign across different address spaces, i.e. Host \f$ \to \f$ Device and Device 
   * \f$ \to \f$ Host. Use nda::tensor::add if you need to fuse a scalar factor or a conjugation with the copy.
   *
   * @tparam A nda::MemoryArray type.
   * @tparam B nda::MemoryArray type with the same value type as \f$ A \f$ (rank may differ from A).
   * @param a Input tensor \f$ A \f$.
   * @param idx_a Index string \f$ \text{idx}_A \f$ for tensor \f$ A \f$.
   * @param b Output tensor \f$ B \f$.
   * @param idx_b Index string \f$ \text{idx}_B \f$ for tensor \f$ B \f$.
   */
  template <MemoryArray A, MemoryArray B>
    requires(have_same_value_type_v<A, B>)
  void assign(A const &a, std::string_view idx_a, B &&b, std::string_view idx_b) { // NOLINT
    // compile-time checks
    constexpr bool device_compat = mem::have_device_compatible_addr_space<A, B>;
    constexpr bool host_compat   = mem::have_host_compatible_addr_space<A, B>;
    constexpr bool use_cutensor  = device_compat && have_cutensor && is_blas_lapack_v<get_value_t<A>>;
    constexpr bool use_tblis     = host_compat && have_tblis && is_blas_lapack_v<get_value_t<A>>;
    static_assert(use_cutensor || use_tblis || get_rank<A> == get_rank<B>,
                  "nda::tensor::assign: host/cross-memory fallback requires identical ranks");

    // dispatch to backends
    if constexpr (use_cutensor) {
      device::permute(get_value_t<A>{1}, a, idx_a, b, idx_b);
    } else if constexpr (use_tblis) {
      tblis::add(get_value_t<A>{1}, a, idx_a, get_value_t<A>{0}, b, idx_b);
    } else {
      require_equal_indices(idx_a, idx_b, get_rank<A>, "assign");
      if constexpr (host_compat) {
        b = a;
      } else {
        detail::rec_copy(a, b);
      }
    }
  }

  /// Convenience overload of nda::tensor::assign with nda::tensor::default_index strings.
  template <MemoryArray A, MemoryArray B>
    requires(have_same_value_type_v<A, B>)
  void assign(A const &a, B &&b) { // NOLINT
    assign(a, default_index<get_rank<A>>(), std::forward<B>(b), default_index<get_rank<B>>());
  }

  /** @} */

} // namespace nda::tensor

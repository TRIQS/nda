// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides definitions of various layout policies.
 */

#pragma once

#include "./idx_map.hpp"
#include "../traits.hpp"

#include <cstdint>
#include <type_traits>

namespace nda {

  /**
   * @addtogroup layout_pols
   * @{
   */

  /// @cond
  // Forward declarations.
  struct C_stride_layout;
  struct F_stride_layout;
  /// @endcond

  /**
   * @brief Contiguous layout policy with C-order (row-major order).
   * @details The last dimension varies the fastest, the first dimension varies the slowest.
   */
  struct C_layout {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = C_stride_layout;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = C_layout;
  };

  /**
   * @brief Contiguous layout policy with Fortran-order (column-major order).
   * @details The first dimension varies the fastest, the last dimension varies the slowest.
   */
  struct F_layout {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = F_stride_layout;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = F_layout;
  };

  /**
   * @brief Strided (non-contiguous) layout policy with C-order (row-major order).
   * @details The last dimension varies the fastest, the first dimension varies the slowest.
   */
  struct C_stride_layout {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = C_stride_layout;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = C_layout;
  };

  /**
   * @brief Strided (non-contiguous) layout policy with Fortran-order (column-major order).
   * @details The first dimension varies the fastest, the last dimension varies the slowest.
   */
  struct F_stride_layout {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = F_stride_layout;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = F_layout;
  };

  /**
   * @brief Generic layout policy with arbitrary order.
   *
   * @tparam StaticExtent Compile-time known shape.
   * @tparam StrideOrder Order in which the dimensions are stored in memory.
   * @tparam LayoutProp Compile-time guarantees about the layout of the data in memory.
   */
  template <uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  struct basic_layout {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = basic_layout<StaticExtents, StrideOrder, layout_prop_e::none>;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = basic_layout<StaticExtents, StrideOrder, layout_prop_e::contiguous>;
  };

  /**
   * @brief Contiguous layout policy with arbitrary stride order.
   * @tparam StrideOrder Order in which the dimensions are stored in memory.
   */
  template <uint64_t StrideOrder>
  using contiguous_layout_with_stride_order = basic_layout<0, StrideOrder, layout_prop_e::contiguous>;

  /**
   * @brief Get the contiguous layout policy for a given rank and stride order.
   *
   * @tparam Rank Rank of the layout.
   * @tparam StrideOrder Order in which the dimensions are stored in memory.
   */
  template <int Rank, uint64_t StrideOrder>
  using get_contiguous_layout_policy =
     std::conditional_t<StrideOrder == C_stride_order<Rank>, C_layout,
                        std::conditional_t<StrideOrder == Fortran_stride_order<Rank>, F_layout, contiguous_layout_with_stride_order<StrideOrder>>>;

  namespace detail {

    /// @cond
    // Forward declarations.
    template <typename L>
    struct layout_to_policy;
    /// @endcond

    // Get the correct layout policy given a general nda::idx_map.
    template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
    struct layout_to_policy<idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>> {
      using type = basic_layout<StaticExtents, StrideOrder, LayoutProp>;
    };

    // Get the correct layout policy given a contiguous nda::idx_map with C-order.
    template <int Rank>
    struct layout_to_policy<idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>> {
      using type = C_layout;
    };

    // Get the correct layout policy given a strided nda::idx_map with C-order.
    template <int Rank>
    struct layout_to_policy<idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>> {
      using type = C_stride_layout;
    };

    // Get the correct layout policy given a contiguous nda::idx_map with Fortran-order.
    template <int Rank>
      requires(Rank > 1)
    struct layout_to_policy<idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>> {
      using type = F_layout;
    };

    // Get the correct layout policy given a strided nda::idx_map with Fortran-order.
    template <int Rank>
      requires(Rank > 1)
    struct layout_to_policy<idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>> {
      using type = F_stride_layout;
    };

  } // namespace detail

  /** @} */

} // namespace nda

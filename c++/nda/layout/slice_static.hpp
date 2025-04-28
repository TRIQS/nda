// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides utilities that determine the resulting nda::idx_map when taking a slice of an nda::idx_map.
 */

#pragma once

#include "./range.hpp"
#include "../macros.hpp"
#include "../stdutil/array.hpp"
#include "../traits.hpp"

#include <array>
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef NDA_ENFORCE_BOUNDCHECK
#include "./bound_check_worker.hpp"
#endif

namespace nda {

  /// @cond
  // Forward declarations.
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map;
  /// @endcond

} // namespace nda

namespace nda::slice_static {

  /**
   * @addtogroup layout_idx
   * @{
   */

  // Notations for this file
  //
  // N: rank of the original idx_map
  // P: rank of the resulting idx_map
  // Q: number of arguments given when creating the slice
  //
  // n: 0, ..., N - 1: indexes the dimensions of the original idx_map
  // p: 0, ..., P - 1: indexes the dimensions of the resulting idx_map
  // q: 0, ..., Q - 1: indexes the arguments
  //
  // p are the indices of the non-long arguments (after ellipsis expansion)
  //
  // N - Q + e is the length of the ellipsis, i.e. the number of the dimensions covered by the
  // ellipsis, where e = +1 if there is an ellipsis and e = 0 otherwise.
  //
  // Let's assume the original idx_map is of rank N = 6 and the following slice arguments are given:
  //
  // Given args = long, long , ellipsis, long, range
  //  q             0     1       2        3     4
  //
  // Here the position of the ellipsis is 2 and its length is N - Q + 1 = 6 - 5 + 1 = 2.
  // After expanding the ellipsis, the arguments are:
  //
  // Expanded args = long, long, range::all, range::all, long, range
  //  q                0     1        2           2        3     4
  //  n                0     1        2           3        4     5
  //  p                -     -        0           1        -     2
  //
  // Now we can define the following (compile-time) maps:
  //
  // - q(n): n -> q
  // - n(p): p -> n
  // - q(p): p -> q

  namespace detail {

    // Helper function to get the position of an nda::ellipsis object in a given parameter pack.
    template <typename... Args, size_t... Is>
    constexpr int ellipsis_position_impl(std::index_sequence<Is...>) {
      // we know that there is at most one ellipsis
      int r = ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...);
      return (r == 0 ? 128 : r - 1);
    }

    // Get the position of an nda::ellipsis object in a given parameter pack (returns 128 if there is no nda::ellipsis).
    template <typename... Args>
    constexpr int ellipsis_position() {
      return detail::ellipsis_position_impl<Args...>(std::make_index_sequence<sizeof...(Args)>{});
    }

    // Map the original dimension n to the argument index q after ellipsis expansion.
    constexpr int q_of_n(int n, int e_pos, int e_len) {
      // n appears before the ellipsis or no ellipsis is present
      if (n < e_pos) return n;

      // n is covered by the ellipsis
      if (n < (e_pos + e_len)) return e_pos;

      // n appears after the ellipsis
      return n - (e_len - 1);
    }

    // Determine how the dimensions of the sliced index map are mapped to the dimensions of the original index map.
    template <int N, int P, size_t Q>
    constexpr std::array<int, P> n_of_p_map(std::array<bool, Q> const &args_is_range, int e_pos, int e_len) {
      auto result = stdutil::make_initialized_array<P>(0);
      // loop over all dimensions of the original index map
      for (int n = 0, p = 0; n < N; ++n) {
        // for each dimension n, determine the corresponding argument index q after ellipsis expansion
        int q = q_of_n(n, e_pos, e_len);
        // if q is a range, map the current p to the current n and increment p
        if (args_is_range[q]) result[p++] = n;
      }
      return result;
    }

    // Determine how the dimensions of the sliced index map are mapped to the arguments after ellipsis expansion.
    template <int N, int P, size_t Q>
    constexpr std::array<int, P> q_of_p_map(std::array<bool, Q> const &args_is_range, int e_pos, int e_len) {
      auto result = stdutil::make_initialized_array<P>(0);
      // loop over all dimensions of the original index map
      int p = 0;
      for (int n = 0; n < N; ++n) {
        // for each dimension n, determine the corresponding argument index q after ellipsis expansion
        int q = q_of_n(n, e_pos, e_len);
        // if q is a range, map the current p to the current q and increment p
        if (args_is_range[q]) result[p++] = q;
      }
      return result;
    }

    // Determine the pseudo inverse map p(n): n -> p. If an n has no corresponding p, the value is -1.
    template <size_t N, size_t P>
    constexpr std::array<int, N> p_of_n_map(std::array<int, P> const &n_of_p) {
      auto result = stdutil::make_initialized_array<N>(-1);
      for (size_t p = 0; p < P; ++p) result[n_of_p[p]] = p;
      return result;
    }

    // Determine the stride order of the sliced index map.
    template <size_t P, size_t N>
    constexpr std::array<int, P> slice_stride_order(std::array<int, N> const &orig_stride_order, std::array<int, P> const &n_of_p) {
      auto result = stdutil::make_initialized_array<P>(0);
      auto p_of_n = p_of_n_map<N>(n_of_p);
      for (int i = 0, ip = 0; i < N; ++i) {
        // n traverses the original stride order, slowest first
        int n = orig_stride_order[i];
        // get the corresponding dimension in the sliced index map
        int p = p_of_n[n];
        // if n maps to a p, add p to the stride order of the sliced index map
        if (p != -1) result[ip++] = p;
      }
      return result;
    }

    // Determine the compile-time layout properties of the sliced index map.
    template <size_t Q, size_t N>
    constexpr layout_prop_e slice_layout_prop(int P, bool has_only_rangeall_and_long, std::array<bool, Q> const &args_is_rangeall,
                                              std::array<int, N> const &orig_stride_order, layout_prop_e orig_layout_prop, int e_pos, int e_len) {
      // if there are ranges in the arguments
      if (not has_only_rangeall_and_long) {
        if (P == 1)
          // rank one is always at least strided_1d
          return layout_prop_e::strided_1d;
        else
          // otherwise we don't know
          return layout_prop_e::none;
      }

      // count the number of nda::range::all_t blocks in the argument list, e.g. long, range::all, range::all, long,
      // range::all -> 2 blocks
      int n_rangeall_blocks         = 0;
      bool previous_arg_is_rangeall = false;
      for (int i = 0; i < N; ++i) {
        int q                = q_of_n(orig_stride_order[i], e_pos, e_len);
        bool arg_is_rangeall = args_is_rangeall[q];
        if (arg_is_rangeall and (not previous_arg_is_rangeall)) ++n_rangeall_blocks;
        previous_arg_is_rangeall = arg_is_rangeall;
      }
      bool rangeall_are_grouped_in_memory = (n_rangeall_blocks <= 1);
      bool last_is_rangeall               = previous_arg_is_rangeall;

      // return the proper layout_prop_e
      if (has_contiguous(orig_layout_prop) and rangeall_are_grouped_in_memory and last_is_rangeall) return layout_prop_e::contiguous;
      if (has_strided_1d(orig_layout_prop) and rangeall_are_grouped_in_memory) return layout_prop_e::strided_1d;
      if (has_smallest_stride_is_one(orig_layout_prop) and last_is_rangeall) return layout_prop_e::smallest_stride_is_one;

      return layout_prop_e::none;
    }

    // Get the contribution to the flat index of the first element of the slice from a single dimension if the argument
    // is a long.
    FORCEINLINE long get_offset(long idx, long stride) { return idx * stride; }

    // Get the contribution to the flat index of the first element of the slice from a single dimension if the argument
    // is a range.
    FORCEINLINE long get_offset(range const &rg, long stride) { return rg.first() * stride; }

    // Get the contribution to the flat index of the first element of the slice from a single dimension if the argument
    // is a range::all_t or covered by an nda::ellipsis.
    FORCEINLINE long get_offset(range::all_t, long) { return 0; }

    // Get the length of the slice for a single dimension if the argument is a range.
    FORCEINLINE long get_length(range const &rg, long original_len) {
      auto last = (rg.last() == -1 and rg.step() > 0) ? original_len : rg.last();
      return range(rg.first(), last, rg.step()).size();
    }

    // Get the length of the slice for a single dimension if the argument is a range::all_t or covered by an
    // nda::ellipsis.
    FORCEINLINE long get_length(range::all_t, long original_len) { return original_len; }

    // Get the stride of the slice for a single dimension if the argument is a range.
    FORCEINLINE long get_stride(range const &rg, long original_str) { return original_str * rg.step(); }

    // Get the stride of the slice for a single dimension if the argument is a range::all_t or covered by an
    // nda::ellipsis..
    FORCEINLINE long get_stride(range::all_t, long original_str) { return original_str; }

    // Helper function to determine the resulting index map when taking a slice of a given index map.
    template <size_t... Ps, size_t... Ns, typename IdxMap, typename... Args>
    FORCEINLINE auto slice_idx_map_impl(std::index_sequence<Ps...>, std::index_sequence<Ns...>, IdxMap const &idxm, Args const &...args) {
      // optional bounds check
#ifdef NDA_ENFORCE_BOUNDCHECK
      nda::assert_in_bounds(idxm.rank(), idxm.lengths().data(), args...);
#endif
      // compile time check
      static_assert(IdxMap::rank() == sizeof...(Ns), "Internal error in slice_idx_map_impl: Rank and length of index sequence do not match");

      // rank of original and resulting idx_map, number of arguments, length of ellipsis, position of ellipsis in the
      // argument list
      static constexpr int N     = sizeof...(Ns);
      static constexpr int P     = sizeof...(Ps);
      static constexpr int Q     = sizeof...(Args);
      static constexpr int e_len = N - Q + 1;
      static constexpr int e_pos = ellipsis_position<Args...>();

      // is i-th argument a range/range::all_t/ellipsis?
      static constexpr std::array<bool, Q> args_is_range{(std::is_same_v<Args, range> or std::is_base_of_v<range::all_t, Args>)...};

      // is i-th argument a range::all_t/ellipsis?
      static constexpr std::array<bool, Q> args_is_rangeall{(std::is_base_of_v<range::all_t, Args>)...};

      // mapping between the dimensions of the resulting index map and the dimensions of the original index map
      static constexpr std::array<int, P> n_of_p = n_of_p_map<N, P>(args_is_range, e_pos, e_len);

      // mapping between the dimensions of the resulting index map and the given arguments
      static constexpr std::array<int, P> q_of_p = q_of_p_map<N, P>(args_is_range, e_pos, e_len);

      // more compile time checks
      static_assert(n_of_p.size() == P, "Internal error in slice_idx_map_impl: Size of the mapping n_of_p and P do not match");
      static_assert(q_of_p.size() == P, "Internal error in slice_idx_map_impl: Size of the mapping q_of_p and P do not match");

      // create tuple of arguments
      auto argstie = std::tie(args...);

      // flat index of the first element of the slice
      long offset = (get_offset(std::get<q_of_n(Ns, e_pos, e_len)>(argstie), std::get<Ns>(idxm.strides())) + ... + 0);

      // shape and strides of the resulting index map
      std::array<long, P> len{get_length(std::get<q_of_p[Ps]>(argstie), std::get<n_of_p[Ps]>(idxm.lengths()))...};
      std::array<long, P> str{get_stride(std::get<q_of_p[Ps]>(argstie), std::get<n_of_p[Ps]>(idxm.strides()))...};

      // static extents of the resulting index map: 0 (= dynamic extent) if the corresponding argument is not a
      // range/range::all_t/ellipsis
      static constexpr std::array<int, P> new_static_extents{(args_is_rangeall[q_of_p[Ps]] ? IdxMap::static_extents[n_of_p[Ps]] : 0)...};

      // stride order of the resulting index map
      static constexpr std::array<int, P> new_stride_order = slice_stride_order(IdxMap::stride_order, n_of_p);

      // compile-time layout properties of the resulting index map
      static constexpr bool has_only_rangeall_and_long = ((std::is_constructible_v<long, Args> or std::is_base_of_v<range::all_t, Args>) and ...);
      static constexpr layout_prop_e li =
         slice_layout_prop(P, has_only_rangeall_and_long, args_is_rangeall, IdxMap::stride_order, IdxMap::layout_prop, e_pos, e_len);

      // return the resulting index map
      static constexpr uint64_t new_static_extents_encoded = encode(new_static_extents);
      static constexpr uint64_t new_stride_order_encoded   = encode(new_stride_order);
      return std::make_pair(offset, idx_map<P, new_static_extents_encoded, new_stride_order_encoded, li>{len, str});
    }

  } // namespace detail

  /**
   * @brief Determine the resulting nda::idx_map when taking a slice of a given nda::idx_map.
   *
   * @details Let `n_args` be the number of given `long`, `nda::range`, `nda::range::all_t` or nda::ellipsis arguments.
   *
   * The rank ``R'`` of the resulting slice is determined by the rank `R` of the original nda::idx_map and the number
   * `n_long` of `long` arguments, i.e. ``R' = R - n_long``.
   *
   * The number of allowed nda::ellipsis objects is restricted to at most one. If an nda::ellipsis object is present and
   * `n_args <= R`, the ellipsis is expanded in terms of `nda::range::all_t` objects to cover the remaining
   * `R - n_args + 1` dimensions. Otherwise, the ellipsis is ignored.
   *
   * After ellipsis expansion, the only arguments contributing to the slice are `nda::range` and `nda::range::all_t`
   * objects. Together with the original nda::idx_map, they determine the resulting nda::idx_map.
   *
   * @tparam R Rank of the original nda::idx_map.
   * @tparam SE Static extents of the original nda::idx_map.
   * @tparam SO Stride order of the original nda::idx_map.
   * @tparam LP Layout properties of the original nda::idx_map.
   * @tparam Args Given argument types.
   * @param idxm Original nda::idx_map.
   * @param args Arguments consisting of `long`, `nda::range`, `nda::range::all_t` or nda::ellipsis objects.
   * @return Resulting nda::idx_map of the slice.
   */
  template <int R, uint64_t SE, uint64_t SO, layout_prop_e LP, typename... Args>
  FORCEINLINE decltype(auto) slice_idx_map(idx_map<R, SE, SO, LP> const &idxm, Args const &...args) {
    // number of ellipsis and long arguments
    static constexpr int n_args_ellipsis = ((std::is_same_v<Args, ellipsis>)+...);
    static constexpr int n_args_long     = (std::is_constructible_v<long, Args> + ...);

    // compile time checks
    static_assert(n_args_ellipsis <= 1, "Error in nda::slice_static::slice_idx_map: At most one ellipsis argument is allowed");
    static_assert((sizeof...(Args) <= R + 1), "Error in nda::slice_static::slice_idx_map: Incorrect number of arguments");
    static_assert((n_args_ellipsis == 1) or (sizeof...(Args) == R), "Error in nda::slice_static::slice_idx_map: Incorrect number of arguments");

    return detail::slice_idx_map_impl(std::make_index_sequence<R - n_args_long>{}, std::make_index_sequence<R>{}, idxm, args...);
  }

  /** @} */

} // namespace nda::slice_static

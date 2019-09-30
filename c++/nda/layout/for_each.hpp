/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2018 by the Simons Foundation
 * author : O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include "permutation.hpp"

namespace nda {

  // ----------------  for_each  -------------------------

  // C style
  template <int I, typename F, size_t R>
  FORCEINLINE void for_each_impl(std::array<long, R> idx_lengths, F &&f) {
    if constexpr (I == R)
      f();
    else {
      long imax = idx_lengths[I];
      for (int i = 0; i < imax; ++i) {
        for_each_impl<I + 1>(
           idx_lengths, [ i, f ](auto &&... x) __attribute__((always_inline)) { return f(i, x...); });
      }
    }
  }

  ///
  template <typename F, size_t R>
  FORCEINLINE void for_each2(std::array<long, R> idx_lengths, F &&f) {
    for_each_impl<0>(idx_lengths, f);
  }

  // ----------------  get_extent  -------------------------

  template <int I, int R, uint64_t StaticExtents>
  long get_extent(std::array<long, R> const &l) {
    if constexpr (StaticExtents == 0)
      return l[I]; // quick exit, no computation of
    else {
      constexpr auto static_extents = decode<R>(StaticExtents); // FIXME C++20
      if constexpr (static_extents[I] == 0)
        return l[I];
      else
        return static_extents[I];
    }
  }

  // ----------------  for_each  -------------------------

  // FIXME : See if we can merge with for_each ...
  // DEBUG

  template <int I, uint64_t StaticExtents, typename F, size_t R>
  FORCEINLINE void for_each_static_impl(std::array<long, R> idx_lengths, F &&f) {
    if constexpr (I == R)
      f();
    else {
      const long imax = get_extent<I, R, StaticExtents>(idx_lengths);
      for (int i = 0; i < imax; ++i) {
        for_each_impl<I + 1>(
           idx_lengths, [ i, f ](auto &&... x) __attribute__((always_inline)) { return f(i, x...); });
      }
    }
  }

  ///
  template <uint64_t StaticExtents, typename F, size_t R>
  FORCEINLINE void for_each_static(std::array<long, R> idx_lengths, F &&f) {
    for_each_static_impl<0, StaticExtents>(idx_lengths, f);
  }

  ///
  template <typename F, size_t R>
  FORCEINLINE void for_each(std::array<long, R> idx_lengths, F &&f) {
    for_each_static_impl<0, 0>(idx_lengths, f);
  }

} // namespace nda

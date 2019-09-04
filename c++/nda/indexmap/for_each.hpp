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

  //// Fortran style
  //template <int I, typename F, size_t R> FORCEINLINE void for_each_impl(std::array<long, R> idx_lengths, F &&f, traversal::Fortran_t) {
  //if constexpr (I == R)
  //f();
  //else {
  //for (int i = 0; i < idx_lengths[R - I - 1]; ++i) {
  //for_each_impl<I + 1>(idx_lengths, [ i, f ](auto &&... x) __attribute__((always_inline)) { return f(x..., i); }, traversal::Fortran);
  //}
  //}
  //}

  ///
  template <typename F, size_t R>
  FORCEINLINE void for_each(std::array<long, R> idx_lengths, F &&f) {
    for_each_impl<0>(idx_lengths, f);
  }

  ///
  //template <typename F, size_t R, typename Traversal> FORCEINLINE void for_each(std::array<long, R> idx_lengths, F &&f, Traversal tr) {
  //for_each_impl<0>(idx_lengths, f, tr);
  //}

} // namespace nda

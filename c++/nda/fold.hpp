/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012 by O. Parcollet
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

  /**
   * @tparam A
   * @tparam F is a function f(x, r)
   * @tparam R
   * @param f
   * @param a
   * @param r
   *
   * fold computes f(f(r, a(0,0)), a(0,1), ...)  etc
   */ 
  template <typename A, typename F, typename R = get_value_t<A>>
  auto fold(F f, A const&a, R r= R{}) {
    decltype(f(r, get_value_t<A>{})) r2 = r;
    // to take into account that f may be double,double -> double, while one passes 0 (an int...)
    // R = int, R2= double in such case, and the result will be a double, or narrowing will occur
    nda::for_each (a.shape(), [&a,&r2, &f](auto &&... args) { r2 = f(r2, a(args...)); });
    return r2;
  }

} // namespace nda

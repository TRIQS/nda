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
#ifndef TRIQS_ARRAYS_EXPRESSION_FOLD_H
#define TRIQS_ARRAYS_EXPRESSION_FOLD_H
#include "../array.hpp"

namespace triqs {
  namespace arrays {

    template <typename A, typename F, typename R> auto fold(F f, A &&a, R r) {
      decltype(f(r, typename A::value_type{})) r2 = r;
      // to take into account that f may be double,double -> double, while one passes 0 (an int...)
      // R = int, R2= double in such case, and the result will be a double, or narrowing will occur
      arrays::foreach (a, [&a](auto &&... args) { r2 = f(r2, a(args...)); });
      return r2;
    }

    template <typename A, typename F> R fold(F f, A &&a) { return fold(std::move(f), std::forward<A>(a), typename A::value_type{}); }


  } // namespace arrays
} // namespace triqs

#endif

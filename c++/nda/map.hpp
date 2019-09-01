/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011 by O. Parcollet
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

  // lazy expression for mapping a function on arrays A
  template <typename F, typename... A>
  struct expr_call;

  // impl details
  template <typename... Char>
  constexpr char _impl_find_common_algebra(char x0, Char... x) {
    return (((x == x0) && ...) ? x0 : 'N');
  }

  // algebra
  template <typename F, typename... A>
  constexpr char get_algebra<expr_call<F, A...>> = _impl_find_common_algebra(get_algebra<A>...);

  // NdArray concept
  template <typename F, typename... A>
  inline constexpr bool is_ndarray_v<expr_call<F, A...>> = true;

  //----------------------------

  // a lazy expression of f(A...) where f is the function to map and A the arrays
  // e.g. f is sqrt(x) and there is one A, or f is min(x,y) and there are 2 A
  template <typename F, typename... A>
  struct expr_call {

    F f;
    std::tuple<const A...> a; // tuple of array (ref) on which f is applied

    private: // FIXME C++20 lambda implementation details
    template <size_t... Is, typename... Args>
    [[gnu::always_inline]] auto _call(std::index_sequence<Is...>, Args const &... args) const {
      return f(std::get<Is>(a)(args...)...);
    }
    template <size_t... Is, typename Args>
    [[gnu::always_inline]] auto _call_bra(std::index_sequence<Is...>, Args const &args) const {
      return f(std::get<Is>(a)[args]...);
    }

    public:
    template <typename... Args>
    auto operator()(Args const &... args) const {
      return _call(std::make_index_sequence<sizeof...(A)>{}, args...);
    }

    // vector interface
    template <typename Args>
    auto operator[](Args &&args) const {
      return _call_bra(std::make_index_sequence<sizeof...(A)>{}, args);
    }

    // FIXME copy needed for the && case only. Overload ?
    auto shape() const { return std::get<0>(a).shape(); }

    long size() const { return a.size(); }
  };

  /*
   * The lambda which maps function F onto the array
   */
  template <class F>
  struct mapped {
    F f;

    template <typename A0, typename... A>
    expr_call<F, A0, A...> operator()(A0 &&a0, A &&... a) const {
      EXPECTS(((a.shape() == a0.shape()) && ...)); // same shape
      return {f, {std::forward<A0>(a0), std::forward<A>(a)...}};
    }
  };

  /**
  * 
  * Maps a function onto the array (elementwise)
  *
  * @tparam F A lambda (do not use function pointers here, make a small lambda it is easier)
  * @param f : function to be mapped
  * @return a lambda that accepts array(s) as argument and return a lazy call expressions.
  *
  * @example nda_map.cpp
  */
  template <class F>
  mapped<F> map(F f) {
    return {std::move(f)};
  }

} // namespace nda

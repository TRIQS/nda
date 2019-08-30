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

  template <typename F, typename... A> struct map_impl_result {

    static constexpr bool is_vec = (_detect_vector_v<A> and ...); // all A are vector like
    F f;
    std::tuple<const A...> tu;

    private: // implementation details
    template <size_t... Is, typename... Args>[[gnu::always_inline]] void _call(std::index_sequence<Is...>, Args const &... args) {
       return f(std::get<Is>(a(args...)...);
    }
    template <size_t... Is, typename Args>[[gnu::always_inline]] void _call_bra(std::index_sequence<Is...>, Args const &... args) REQUIRES(is_vec) {
       return f[std::get<Is>(a(args)...];
    }

    public:
    template <typename... Args> auto operator()(Args const &... args) const { return _call(std::index_sequence_for<A...>, args...); }

    // vector interface
    template <typename Args> auto operator[](Args &&args) const REQUIRES(is_vec) { return _call_bra(std::index_sequence_for<A...>, args); }
    long size() const REQUIRES(is_vec) { return a.size(); }
  };

  // -----------------------------------

  template <class F> struct map_impl {
    F f;
    template <typename... A> map_impl_result<F, A...> operator()(A &&... a) const { return {f, std::forward<A>(a)...}; }
  };

  /**
  * Given a function f : arg_type -> result_type, map(f) is the function promoted to arrays
  * map(f) : array<arg_type, N, Opt> --> array<result_type, N, Opt>
  */
  template <class F> map_impl<F> map(F f) { return {std::move(f)}; }

  // ---  implementation  print

  template <typename F, typename... A> std::ostream &operator<<(std::ostream &out, map_impl_result<F, A...> const &x) {
    return out << array<value_type, std::decay_t<A>::rank>(x);
  }

  /* template <class F> friend std::ostream &operator<<(std::ostream &out, map_impl<F> const &x) {
    return out << "map("
               << "F"
               << ")";
  }*/

  // propagate traits
  template <typename F, int arity, bool b, typename... A> struct ImmutableCuboidArray<map_impl_result<F, arity, b, A...>> : std::true_type {};

  template <typename F, int arity, bool b, typename... A>
  struct ImmutableArray<map_impl_result<F, arity, b, A...>> : _and<typename ImmutableArray<typename std::remove_reference<A>::type>::type...> {};
  template <typename F, int arity, bool b, typename... A>
  struct ImmutableMatrix<map_impl_result<F, arity, b, A...>> : _and<typename ImmutableMatrix<typename std::remove_reference<A>::type>::type...> {};
  template <typename F, int arity, bool b, typename... A>
  struct ImmutableVector<map_impl_result<F, arity, b, A...>> : _and<typename ImmutableVector<typename std::remove_reference<A>::type>::type...> {};

  //template<typename F, int arity, bool b, typename A> struct ImmutableArray <map_impl_result<F,arity,b,A>> : ImmutableArray <A>{};
  //template<typename F, int arity, bool b, typename A> struct ImmutableMatrix<map_impl_result<F,arity,b,A>> : ImmutableMatrix<A>{};
  //template<typename F, int arity, bool b, typename A> struct ImmutableVector<map_impl_result<F,arity,b,A>> : ImmutableVector<A>{};

} // namespace nda

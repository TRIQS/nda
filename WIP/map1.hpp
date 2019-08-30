
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
#include "../impl/common.hpp"
//#include <functional>

namespace nda {

  template <class F, int arity = F::arity> class map_impl;

  /**
  * Given a function f : arg_type -> result_type, map(f) is the function promoted to arrays
  * map(f) : array<arg_type, N, Opt> --> array<result_type, N, Opt>
  */
  //template<class F> map_impl<F,utility::function_arg_ret_type<F>::arity> map  (F f) { return {std::move(f),true}; }

  template <class F> map_impl<F, 1> map(F f) { return {std::move(f), true}; }
  template <class F> map_impl<F, 2> map2(F f) { return {std::move(f), true}; }

  // ----------- implementation  -------------------------------------

  template <typename F, int arity, bool is_vec, typename... A> struct map_impl_result;

  template <typename F, typename A> struct map_impl_result<F, 1, is_vec, A> {

    static constexpr bool is_vec = XXXX; // Put detection a[0] is ok.

    F f;
    const A a; // A can be a ref.

    template <typename... Args> auto operator()(Args &&... args) const { return f(a(std::forward<Args>(args)...)); }

    long size() const REQUIRES(is_vec) { return a.size(); }

    template <typename Args> auto operator[](Args &&args) const REQUIRES(is_vec) { return f(a[std::forward<Args>(args)]); }
  };

  template <typename F, typename... A> struct map_impl_result {

    static constexpr bool is_vec = XXXX; // Put detection a[0] is ok.

    F f;
    std::tuple<const A...> tu;

    template <size_t... Is, typename... Args>[[gnu::always_inline]] void _call(std::index_sequence<Is...>, Args const &... args) {
       return f(std::get<Is>(a(args...)...);
    }

    template <typename... Args> auto operator()(Args const &... args) const { return _call(std::index_sequence_for<A...>, args...); }

    friend std::ostream &operator<<(std::ostream &out, map_impl_result const &x);

    long size() const REQUIRES(is_vec) { return a.size(); }

    template <size_t... Is, typename Args>[[gnu::always_inline]] void _call_bra(std::index_sequence<Is...>, Args const &... args) REQUIRES(is_vec) {
       return f[std::get<Is>(a(args)...];
    }
    template <typename Args> auto operator[](Args &&args) const REQUIRES(is_vec) { return _call_bra(std::index_sequence_for<A...>, args); }
  };

  
  template <typename F, int arity, typename... A> std::ostream &operator<<(std::ostream &out, map_impl_result<F, arity, A...> const &x);
  friend std::ostream &operator<<(std::ostream &out, map_impl_result const &x) {
    return out << array<value_type, std14::decay_t<A>::domain_type::rank>(x);
  }

  // possible to generalize to N order using tuple techniques ...
  template <typename F, bool is_vec, typename A, typename B> struct map_impl_result<F, 2, is_vec, A, B> {
    using value_type  = typename std::result_of<F(typename remove_cv_ref<A>::type::value_type, typename remove_cv_ref<B>::type::value_type)>::type;
    using domain_type = typename remove_cv_ref<A>::type::domain_type;
    F f;
    typename std::add_const<A>::type a;
    typename std::add_const<B>::type b;
    domain_type domain() const { return a.domain(); }
    template <typename... Args> value_type operator()(Args &&... args) const {
      return f(a(std::forward<Args>(args)...), b(std::forward<Args>(args)...));
    }
    friend std::ostream &operator<<(std::ostream &out, map_impl_result const &x) {
      return out << array<value_type, std14::decay_t<A>::domain_type::rank>(x);
    }
    // rest is only for vector
    template <bool vec = is_vec> TYPE_ENABLE_IFC(size_t, vec) size() const { return a.size(); }
    template <typename Args, bool vec = is_vec> TYPE_ENABLE_IFC(value_type, vec) operator[](Args &&args) const {
      return f(a[std::forward<Args>(args)], b[std::forward<Args>(args)]);
    }
  };

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

  // NB The bool is to make constructor not ambiguous
  // clang on os X with lib++ has a pb otherwise (not clear what the pb is)
  template <class F, int arity> struct map_impl {
    F f;

    template <typename... A> map_impl_result<F, arity, _and<typename ImmutableVector<A>::type...>::value, A...> operator()(A &&... a) const {
      return {f, std::forward<A>(a)...};
    }

    friend std::ostream &operator<<(std::ostream &out, map_impl const &x) {
      return out << "map("
                 << "F"
                 << ")";
    }
  };
} // namespace nda
